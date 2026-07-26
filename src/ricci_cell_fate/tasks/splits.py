from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ricci_cell_fate.datasets.eligibility import write_task_eligibility
from ricci_cell_fate.datasets.registry import load_registry
from ricci_cell_fate.datasets.validate import find_first_existing
from ricci_cell_fate.io.anndata_io import read_h5ad
from ricci_cell_fate.utils.config import read_yaml
from ricci_cell_fate.utils.paths import ensure_dir, find_project_root
from ricci_cell_fate.utils.provenance import artifact_record, write_json, write_placeholder
from ricci_cell_fate.utils.seed import seed_everything


def _indices_to_names(names: list[str], indices: np.ndarray) -> list[str]:
    return [str(names[int(i)]) for i in indices]


def make_split_payload(
    *,
    train: list[str],
    validation: list[str],
    test: list[str],
    split_id: str,
    strategy: str,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "split_id": split_id,
        "strategy": strategy,
        "seed": seed,
        "train": train,
        "validation": validation,
        "test": test,
        "metadata": metadata or {},
    }


def random_split(names: list[str], seed: int, split_id: str, strategy: str = "random") -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(names))
    n = len(names)
    n_test = max(1, int(0.2 * n))
    n_val = max(1, int(0.1 * n))
    test = idx[:n_test]
    validation = idx[n_test : n_test + n_val]
    train = idx[n_test + n_val :]
    return make_split_payload(
        train=_indices_to_names(names, train),
        validation=_indices_to_names(names, validation),
        test=_indices_to_names(names, test),
        split_id=split_id,
        strategy=strategy,
        seed=seed,
    )


def time_holdout_split(
    names: list[str],
    values: pd.Series,
    seed: int,
    split_id: str,
    holdout_fraction: float,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().all():
        codes = pd.Categorical(values.astype(str)).codes
        numeric = pd.Series(codes, index=values.index)
    order = np.argsort(numeric.to_numpy())
    n = len(order)
    n_test = max(1, int(holdout_fraction * n))
    test = order[-n_test:]
    remaining = order[:-n_test]
    n_val = max(1, int(0.1 * len(remaining)))
    validation = remaining[-n_val:]
    train = remaining[:-n_val]
    return make_split_payload(
        train=_indices_to_names(names, train),
        validation=_indices_to_names(names, validation),
        test=_indices_to_names(names, test),
        split_id=split_id,
        strategy="time_based_holdout",
        seed=seed,
        metadata={"holdout_fraction": holdout_fraction},
    )


def category_holdout_split(
    names: list[str],
    values: pd.Series,
    seed: int,
    split_id: str,
    holdout_fraction: float,
    strategy: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    categories = np.array(sorted(map(str, pd.Series(values).dropna().unique())))
    if len(categories) < 2:
        return random_split(names, seed, split_id, strategy=f"{strategy}_fallback_random")
    rng.shuffle(categories)
    n_holdout = max(1, int(np.ceil(holdout_fraction * len(categories))))
    test_categories = set(categories[:n_holdout])
    val_categories = set(categories[n_holdout : n_holdout + max(1, min(1, len(categories) - n_holdout))])
    train, validation, test = [], [], []
    for name, value in zip(names, values.astype(str), strict=False):
        if value in test_categories:
            test.append(name)
        elif value in val_categories:
            validation.append(name)
        else:
            train.append(name)
    return make_split_payload(
        train=train,
        validation=validation,
        test=test,
        split_id=split_id,
        strategy=strategy,
        seed=seed,
        metadata={"test_categories": sorted(test_categories), "validation_categories": sorted(val_categories)},
    )


def graph_region_holdout_split(
    names: list[str],
    edge_table_path: Path,
    seed: int,
    split_id: str,
    holdout_fraction: float,
) -> dict[str, Any]:
    if not edge_table_path.exists():
        return random_split(names, seed, split_id, strategy="graph_region_fallback_random")
    edges = pd.read_csv(edge_table_path)
    graph = nx.Graph()
    graph.add_nodes_from(names)
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.source), str(row.target), weight=float(getattr(row, "weight", 1.0)))
    components = [list(c) for c in nx.connected_components(graph)]
    components.sort(key=len, reverse=True)
    n_target = max(1, int(holdout_fraction * len(names)))
    test: list[str] = []
    for comp in components:
        if len(test) >= n_target:
            break
        test.extend(map(str, comp[: max(1, min(len(comp), n_target - len(test)))]))
    remaining = [n for n in names if n not in set(test)]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    n_val = max(1, int(0.1 * len(names)))
    validation = remaining[:n_val]
    train = remaining[n_val:]
    return make_split_payload(
        train=train,
        validation=validation,
        test=test,
        split_id=split_id,
        strategy="graph_region_holdout",
        seed=seed,
        metadata={"holdout_fraction": holdout_fraction},
    )


def persist_split(payload: dict[str, Any], path: str | Path) -> None:
    write_json(payload, path)


def build_splits_for_dataset(
    dataset: str,
    *,
    root: Path | None = None,
    allow_placeholders: bool = False,
) -> list[dict[str, Any]]:
    root = root or find_project_root()
    config = read_yaml(root / "configs" / "tasks" / "default.yaml")
    seed = int(config.get("seed", 1729))
    seed_everything(seed)
    registry = load_registry(root / "configs" / "datasets" / "registry.yaml")
    spec = registry[dataset]
    processed_path = root / "data" / "processed" / dataset / spec.processed_filename
    placeholder = root / "data" / "splits" / dataset / "TODO_DATA_DEPENDENCY.md"
    if not processed_path.exists():
        msg = f"Missing processed dataset {processed_path}. Run preprocessing first."
        write_placeholder(placeholder, "TODO Data Dependency", msg)
        if allow_placeholders:
            return [{"dataset": dataset, "status": "placeholder", "path": str(placeholder)}]
        raise FileNotFoundError(msg)

    adata = read_h5ad(processed_path)
    names = list(map(str, adata.obs_names))
    columns = list(map(str, adata.obs.columns))
    time_key = find_first_existing(spec.time_key_candidates, columns)
    lineage_key = find_first_existing(spec.lineage_key_candidates, columns)
    batch_key = find_first_existing(spec.batch_key_candidates, columns)
    split_cfg = config.get("splits", {})
    repeats = int(split_cfg.get("repeats", 3))
    edge_table = root / "data" / "processed" / dataset / f"{dataset}_edge_weights.csv"
    tasks = [
        "branch_point_localization",
        "early_fate_prediction",
        "bottleneck_detection",
        "trajectory_segmentation",
        "cross_dataset_transfer",
    ]
    records = []
    for task in tasks:
        out_dir = ensure_dir(root / "data" / "splits" / dataset / task)
        for repeat in range(repeats):
            split_seed = seed + repeat
            if task == "early_fate_prediction" and time_key:
                payload = time_holdout_split(
                    names,
                    adata.obs[time_key],
                    split_seed,
                    f"{task}_time_{repeat}",
                    float(split_cfg.get("time_holdout_fraction", 0.25)),
                )
            elif task in {"early_fate_prediction", "cross_dataset_transfer"} and lineage_key:
                payload = category_holdout_split(
                    names,
                    adata.obs[lineage_key],
                    split_seed,
                    f"{task}_lineage_{repeat}",
                    float(split_cfg.get("lineage_holdout_fraction", 0.25)),
                    "lineage_holdout",
                )
            elif task == "cross_dataset_transfer" and batch_key:
                payload = category_holdout_split(
                    names,
                    adata.obs[batch_key],
                    split_seed,
                    f"{task}_batch_{repeat}",
                    0.25,
                    "donor_or_batch_holdout",
                )
            else:
                payload = graph_region_holdout_split(
                    names,
                    edge_table,
                    split_seed,
                    f"{task}_graph_region_{repeat}",
                    float(split_cfg.get("graph_region_holdout_fraction", 0.2)),
                )
            payload["dataset"] = dataset
            payload["task"] = task
            path = out_dir / f"{payload['split_id']}.json"
            persist_split(payload, path)
            records.append(
                artifact_record(
                    path,
                    dataset=dataset,
                    task=task,
                    split_id=payload["split_id"],
                    config_path="configs/tasks/default.yaml",
                    seed=split_seed,
                    root=root,
                )
            )
    return records


def build_splits_many(
    datasets: list[str],
    *,
    root: Path | None = None,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    root = root or find_project_root()
    records = []
    for dataset in datasets:
        records.extend(build_splits_for_dataset(dataset, root=root, allow_placeholders=allow_placeholders))
    write_task_eligibility(datasets, root=root)
    manifest = {"artifacts": records}
    write_json(manifest, root / "data" / "manifests" / "split_manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pancreas", "lung", "zebrafish", "paul15"])
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args(argv)
    build_splits_many(args.datasets, allow_placeholders=args.allow_placeholders)


if __name__ == "__main__":  # pragma: no cover
    main()

