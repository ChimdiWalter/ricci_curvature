from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ricci_cell_fate.baselines.external import record_external_baseline_deviations
from ricci_cell_fate.curvature.aggregate import curvature_feature_tables
from ricci_cell_fate.curvature.scoring import branch_bottleneck_scores
from ricci_cell_fate.datasets.registry import load_registry
from ricci_cell_fate.datasets.validate import find_first_existing
from ricci_cell_fate.graphs.construction import build_knn_graph, graph_from_edge_table
from ricci_cell_fate.graphs.features import degree_matched_rewire, graph_feature_table
from ricci_cell_fate.io.anndata_io import read_h5ad
from ricci_cell_fate.tasks.bottleneck import evaluate_bottleneck_scores
from ricci_cell_fate.tasks.branch import evaluate_branch_scores, top_k_predictions
from ricci_cell_fate.tasks.early_fate import run_early_fate_prediction
from ricci_cell_fate.utils.config import read_yaml
from ricci_cell_fate.utils.paths import ensure_dir, find_project_root
from ricci_cell_fate.utils.provenance import artifact_record, write_json, write_placeholder


GRAPH_FEATURE_COLUMNS = [
    "degree",
    "weighted_degree",
    "pagerank",
    "closeness",
    "betweenness",
    "diffusion_pseudotime",
]


def _load_first_split(root: Path, dataset: str, task: str) -> dict[str, Any] | None:
    split_dir = root / "data" / "splits" / dataset / task
    candidates = sorted(split_dir.glob("*.json"))
    if not candidates:
        return None
    return read_yaml(candidates[0]) if candidates[0].suffix in {".yaml", ".yml"} else _read_json(candidates[0])


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_graph(root: Path, dataset: str) -> tuple[Any, nx.Graph]:
    registry = load_registry(root / "configs" / "datasets" / "registry.yaml")
    spec = registry[dataset]
    processed_path = root / "data" / "processed" / dataset / spec.processed_filename
    edge_path = root / "data" / "processed" / dataset / f"{dataset}_edge_weights.csv"
    if not processed_path.exists() or not edge_path.exists():
        raise FileNotFoundError(f"Missing processed dataset or graph edge table for {dataset}")
    adata = read_h5ad(processed_path)
    edge_table = pd.read_csv(edge_path)
    graph = graph_from_edge_table(edge_table)
    graph.add_nodes_from(map(str, adata.obs_names))
    return adata, graph


def compute_node_feature_table(
    graph: nx.Graph,
    *,
    include_curvature: bool = True,
    include_ollivier: bool = True,
    max_ollivier_edges: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    graph_features = graph_feature_table(graph)
    edge_curvatures = None
    node_features = graph_features
    if include_curvature:
        edge_curvatures, curvature_features = curvature_feature_tables(
            graph,
            include_ollivier=include_ollivier,
            max_ollivier_edges=max_ollivier_edges,
        )
        node_features = node_features.join(curvature_features, how="left")
    scores = branch_bottleneck_scores(node_features)
    node_features = node_features.join(scores, how="left").fillna(0.0)
    return node_features, edge_curvatures


def add_expression_pcs(features: pd.DataFrame, adata: Any, n_pcs: int = 20) -> pd.DataFrame:
    if "X_pca" not in adata.obsm:
        return features
    pcs = np.asarray(adata.obsm["X_pca"])[:, :n_pcs]
    pc_df = pd.DataFrame(
        pcs,
        index=adata.obs_names.astype(str),
        columns=[f"PC{i + 1}" for i in range(pcs.shape[1])],
    )
    return features.join(pc_df, how="left")


def save_feature_artifacts(
    root: Path,
    dataset: str,
    node_features: pd.DataFrame,
    edge_curvatures: pd.DataFrame | None,
    *,
    mode: str,
) -> dict[str, Any]:
    out_dir = ensure_dir(root / "results" / "tables" / dataset)
    records: dict[str, Any] = {}
    node_path = out_dir / f"{dataset}_{mode}_node_features.csv"
    node_features.to_csv(node_path)
    records["node_features"] = artifact_record(node_path, dataset=dataset, task=mode, root=root)
    if edge_curvatures is not None:
        edge_path = out_dir / f"{dataset}_{mode}_edge_curvatures.csv"
        edge_curvatures.to_csv(edge_path, index=False)
        records["edge_curvatures"] = artifact_record(edge_path, dataset=dataset, task=mode, root=root)
    return records


def evaluate_dataset_tasks(
    root: Path,
    dataset: str,
    *,
    mode: str,
    feature_mode: str,
    include_curvature: bool,
    include_ollivier: bool,
    max_ollivier_edges: int | None,
    seed: int,
    graph_override: nx.Graph | None = None,
    early_fraction: float = 0.35,
    branch_quantile: float = 0.95,
    bottleneck_quantile: float = 0.95,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = load_registry(root / "configs" / "datasets" / "registry.yaml")
    spec = registry[dataset]
    adata, graph = load_dataset_graph(root, dataset)
    if graph_override is not None:
        graph = graph_override
    if feature_mode == "rewired_control":
        graph = degree_matched_rewire(graph, seed=seed)
    node_features, edge_curvatures = compute_node_feature_table(
        graph,
        include_curvature=include_curvature,
        include_ollivier=include_ollivier,
        max_ollivier_edges=max_ollivier_edges,
    )
    if feature_mode == "random_matched":
        rng = np.random.default_rng(seed)
        numeric = node_features.select_dtypes("number")
        node_features = pd.DataFrame(
            rng.normal(size=numeric.shape), index=numeric.index, columns=[f"random_{i}" for i in range(numeric.shape[1])]
        )
        node_features["branch_score"] = rng.normal(size=len(node_features))
        node_features["bottleneck_score"] = rng.normal(size=len(node_features))
    elif feature_mode == "pseudotime_only":
        node_features = node_features[["diffusion_pseudotime"]].copy()
        node_features["branch_score"] = -node_features["diffusion_pseudotime"]
        node_features["bottleneck_score"] = node_features["diffusion_pseudotime"]
    elif feature_mode == "graph_only":
        keep = [c for c in GRAPH_FEATURE_COLUMNS + ["branch_score", "bottleneck_score"] if c in node_features]
        node_features = node_features[keep].copy()
    else:
        node_features = add_expression_pcs(node_features, adata)
    if "diffusion_pseudotime" not in node_features:
        node_features["diffusion_pseudotime"] = np.linspace(0.0, 1.0, len(node_features))
    if "branch_score" not in node_features:
        node_features["branch_score"] = node_features.select_dtypes("number").mean(axis=1)
    if "bottleneck_score" not in node_features:
        node_features["bottleneck_score"] = node_features.select_dtypes("number").mean(axis=1)
    artifacts = save_feature_artifacts(root, dataset, node_features, edge_curvatures, mode=mode)

    rows: list[dict[str, Any]] = []
    branch = evaluate_branch_scores(
        node_features,
        node_features["branch_score"],
        proxy_quantile=branch_quantile,
    )
    rows.append(
        {
            "dataset": dataset,
            "task": "branch_point_localization",
            "mode": mode,
            "status_proxy_labels": 1.0,
            "label_source": "betweenness_proxy_quantile",
            "evaluation_scope": "full_graph_proxy_ranking",
            "split_id": "",
            "split_strategy": "",
            **branch,
        }
    )

    bottleneck = evaluate_bottleneck_scores(
        node_features,
        node_features["bottleneck_score"],
        proxy_quantile=bottleneck_quantile,
    )
    rows.append(
        {
            "dataset": dataset,
            "task": "bottleneck_detection",
            "mode": mode,
            "status_proxy_labels": 1.0,
            "label_source": "betweenness_plus_inverse_degree_proxy_quantile",
            "evaluation_scope": "full_graph_proxy_ranking",
            "split_id": "",
            "split_strategy": "",
            **bottleneck,
        }
    )

    split = _load_first_split(root, dataset, "early_fate_prediction")
    time_key = find_first_existing(spec.time_key_candidates, list(adata.obs.columns))
    lineage_key = find_first_existing(spec.lineage_key_candidates, list(adata.obs.columns))
    if split is not None:
        fate = run_early_fate_prediction(
            node_features,
            adata.obs.copy(),
            split,
            lineage_key_candidates=spec.lineage_key_candidates,
            time_key_candidates=spec.time_key_candidates,
            seed=seed,
            early_fraction=early_fraction,
        )
        rows.append(
            {
                "dataset": dataset,
                "task": "early_fate_prediction",
                "mode": mode,
                "split_id": str(split.get("split_id", "")),
                "split_strategy": str(split.get("strategy", "")),
                "time_key": time_key or "",
                "lineage_key": lineage_key or "",
                **fate,
            }
        )
    else:
        rows.append(
            {
                "dataset": dataset,
                "task": "early_fate_prediction",
                "mode": mode,
                "status_missing_split": 1.0,
                "time_key": time_key or "",
                "lineage_key": lineage_key or "",
                "split_id": "",
                "split_strategy": "",
            }
        )
    return rows, artifacts


def run_experiment_mode(
    datasets: list[str],
    *,
    mode: str,
    output_name: str,
    feature_mode: str = "curvature",
    include_curvature: bool = True,
    include_ollivier: bool = True,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    root = find_project_root()
    model_config = read_yaml(root / "configs" / "models" / "logistic.yaml")
    seed = int(model_config.get("seed", 1729))
    curvature_cfg = model_config.get("curvature", {})
    max_edges = curvature_cfg.get("max_ollivier_edges")
    max_edges = None if max_edges in {None, "null"} else int(max_edges)
    rows: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    for dataset in datasets:
        try:
            dataset_rows, artifacts = evaluate_dataset_tasks(
                root,
                dataset,
                mode=mode,
                feature_mode=feature_mode,
                include_curvature=include_curvature,
                include_ollivier=include_ollivier,
                max_ollivier_edges=max_edges,
                seed=seed,
            )
            rows.extend(dataset_rows)
            artifact_records.extend(artifacts.values())
            stale_placeholder = root / "results" / "metrics" / f"{dataset}_{mode}_MISSING_RESULT.md"
            if stale_placeholder.exists():
                stale_placeholder.unlink()
        except Exception as exc:
            placeholder = root / "results" / "metrics" / f"{dataset}_{mode}_MISSING_RESULT.md"
            write_placeholder(
                placeholder,
                "Missing Result",
                f"{mode} could not run for {dataset}: {type(exc).__name__}: {exc}",
            )
            if not allow_placeholders:
                raise
            rows.append({"dataset": dataset, "task": "all", "mode": mode, "status_missing_result": 1.0})
            artifact_records.append(artifact_record(placeholder, dataset=dataset, task=mode, root=root))
    out = root / "results" / "metrics" / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    manifest = {
        "mode": mode,
        "metrics": artifact_record(out, task=mode, root=root),
        "artifacts": artifact_records,
    }
    write_json(manifest, root / "results" / "manifests" / f"{mode}_manifest.json")
    return manifest


def run_baselines(datasets: list[str], allow_placeholders: bool = False) -> dict[str, Any]:
    root = find_project_root()
    record_external_baseline_deviations(root)
    return run_experiment_mode(
        datasets,
        mode="baselines",
        output_name="baseline_metrics.csv",
        feature_mode="graph_only",
        include_curvature=False,
        include_ollivier=False,
        allow_placeholders=allow_placeholders,
    )


def run_curvature_experiments(datasets: list[str], allow_placeholders: bool = False) -> dict[str, Any]:
    return run_experiment_mode(
        datasets,
        mode="curvature",
        output_name="curvature_metrics.csv",
        feature_mode="curvature",
        include_curvature=True,
        include_ollivier=True,
        allow_placeholders=allow_placeholders,
    )


def run_ablations(datasets: list[str], allow_placeholders: bool = False) -> dict[str, Any]:
    root = find_project_root()
    ablation_config = read_yaml(root / "configs" / "ablations" / "default.yaml")
    model_config = read_yaml(root / "configs" / "models" / "logistic.yaml")
    max_edges = model_config.get("curvature", {}).get("max_ollivier_edges")
    max_edges = None if max_edges in {None, "null"} else int(max_edges)
    all_rows = []
    all_records = []
    for feature_mode, include_curvature, include_ollivier in [
        ("remove_curvature", False, False),
        ("forman_only", True, False),
        ("rewired_control", True, False),
        ("random_matched", False, False),
    ]:
        manifest = run_experiment_mode(
            datasets,
            mode=f"ablation_{feature_mode}",
            output_name=f"ablation_{feature_mode}_metrics.csv",
            feature_mode=feature_mode,
            include_curvature=include_curvature,
            include_ollivier=include_ollivier,
            allow_placeholders=allow_placeholders,
        )
        metric_path = Path(manifest["metrics"]["path"])
        if metric_path.exists():
            df = pd.read_csv(metric_path)
            df["ablation"] = feature_mode
            all_rows.append(df)
        all_records.append(manifest["metrics"])
    registry = load_registry(root / "configs" / "datasets" / "registry.yaml")
    for seed in ablation_config.get("seed_sweep", [1729]):
        for k in ablation_config.get("k_values", [15]):
            for weighting in ablation_config.get("edge_weighting", ["heat_kernel"]):
                for dataset in datasets:
                    try:
                        spec = registry[dataset]
                        adata_path = root / "data" / "processed" / dataset / spec.processed_filename
                        if not adata_path.exists():
                            raise FileNotFoundError(f"Missing processed dataset {adata_path}")
                        adata = read_h5ad(adata_path)
                        graph, _ = build_knn_graph(
                            adata.obsm["X_pca"],
                            n_neighbors=int(k),
                            weight_scheme=str(weighting),
                            node_ids=list(map(str, adata.obs_names)),
                        )
                        rows, artifacts = evaluate_dataset_tasks(
                            root,
                            dataset,
                            mode=f"ablation_k{k}_{weighting}_seed{seed}",
                            feature_mode="curvature",
                            include_curvature=True,
                            include_ollivier=False,
                            max_ollivier_edges=max_edges,
                            seed=int(seed),
                            graph_override=graph,
                        )
                        df = pd.DataFrame(rows)
                        df["ablation"] = "vary_k_edge_weight_seed"
                        df["k"] = int(k)
                        df["edge_weighting"] = str(weighting)
                        df["seed"] = int(seed)
                        all_rows.append(df)
                        all_records.extend(artifacts.values())
                    except Exception as exc:
                        placeholder = root / "results" / "metrics" / f"{dataset}_ablation_k{k}_{weighting}_MISSING_RESULT.md"
                        write_placeholder(
                            placeholder,
                            "Missing Result",
                            f"k/edge-weight ablation failed for {dataset}: {type(exc).__name__}: {exc}",
                        )
                        all_records.append(artifact_record(placeholder, dataset=dataset, task="ablations", root=root))
                        if not allow_placeholders:
                            raise
    for early_fraction in ablation_config.get("early_cell_cutoffs", [0.35]):
        for threshold in ablation_config.get("threshold_quantiles", [0.05]):
            proxy_quantile = 1.0 - float(threshold)
            for dataset in datasets:
                try:
                    rows, artifacts = evaluate_dataset_tasks(
                        root,
                        dataset,
                        mode=f"ablation_early{early_fraction}_threshold{threshold}",
                        feature_mode="curvature",
                        include_curvature=True,
                        include_ollivier=False,
                        max_ollivier_edges=max_edges,
                        seed=int(ablation_config.get("seed", 1729)),
                        early_fraction=float(early_fraction),
                        branch_quantile=proxy_quantile,
                        bottleneck_quantile=proxy_quantile,
                    )
                    df = pd.DataFrame(rows)
                    df["ablation"] = "early_cutoff_threshold"
                    df["early_fraction"] = float(early_fraction)
                    df["threshold_quantile"] = float(threshold)
                    all_rows.append(df)
                    all_records.extend(artifacts.values())
                except Exception as exc:
                    placeholder = root / "results" / "metrics" / f"{dataset}_ablation_threshold_MISSING_RESULT.md"
                    write_placeholder(
                        placeholder,
                        "Missing Result",
                        f"early/threshold sensitivity failed for {dataset}: {type(exc).__name__}: {exc}",
                    )
                    all_records.append(artifact_record(placeholder, dataset=dataset, task="ablations", root=root))
                    if not allow_placeholders:
                        raise
    for graph_source in ablation_config.get("graph_sources", ["raw"]):
        if graph_source == "raw":
            continue
        for dataset in datasets:
            placeholder = root / "results" / "metrics" / f"{dataset}_ablation_{graph_source}_TODO_DATA_DEPENDENCY.md"
            write_placeholder(
                placeholder,
                "TODO Data Dependency",
                f"Graph-source ablation `{graph_source}` is configured but no `{graph_source}` graph "
                "artifact exists yet. Enable and run the denoising path before reporting this ablation.",
            )
            all_records.append(artifact_record(placeholder, dataset=dataset, task="ablations", root=root))
    out = root / "results" / "metrics" / "ablation_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
    else:
        pd.DataFrame([{"status_missing_result": 1.0}]).to_csv(out, index=False)
    payload = {"metrics": artifact_record(out, task="ablations", root=root), "inputs": all_records}
    write_json(payload, root / "results" / "manifests" / "ablation_manifest.json")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pancreas", "lung", "zebrafish", "paul15"])
    parser.add_argument("--mode", choices=["baselines", "curvature", "ablations"], default="curvature")
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "baselines":
        run_baselines(args.datasets, allow_placeholders=args.allow_placeholders)
    elif args.mode == "ablations":
        run_ablations(args.datasets, allow_placeholders=args.allow_placeholders)
    else:
        run_curvature_experiments(args.datasets, allow_placeholders=args.allow_placeholders)


if __name__ == "__main__":  # pragma: no cover
    main()
