from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

from ricci_cell_fate.datasets.registry import load_registry
from ricci_cell_fate.graphs.construction import build_knn_graph, save_graph_artifacts
from ricci_cell_fate.io.anndata_io import read_h5ad, write_h5ad
from ricci_cell_fate.utils.config import read_yaml
from ricci_cell_fate.utils.paths import ensure_dir, find_project_root
from ricci_cell_fate.utils.provenance import artifact_record, write_json, write_placeholder
from ricci_cell_fate.utils.seed import seed_everything


def _copy_adata(adata: Any) -> Any:
    return adata.copy()


def _counts_per_cell(x: Any) -> np.ndarray:
    return np.asarray(x.sum(axis=1)).ravel()


def _genes_per_cell(x: Any) -> np.ndarray:
    return np.asarray((x > 0).sum(axis=1)).ravel()


def _cells_per_gene(x: Any) -> np.ndarray:
    return np.asarray((x > 0).sum(axis=0)).ravel()


def qc_filter(adata: Any, config: dict[str, Any]) -> Any:
    qc = config.get("qc", {})
    out = _copy_adata(adata)
    x = out.X
    min_counts = float(qc.get("min_counts_per_cell", 0))
    min_genes = int(qc.get("min_genes_per_cell", 0))
    min_cells = int(qc.get("min_cells_per_gene", 1))
    cell_mask = (_counts_per_cell(x) >= min_counts) & (_genes_per_cell(x) >= min_genes)
    out = out[cell_mask].copy()
    gene_mask = _cells_per_gene(out.X) >= min_cells
    out = out[:, gene_mask].copy()
    out.obs["qc_total_counts"] = _counts_per_cell(out.X)
    out.obs["qc_n_genes"] = _genes_per_cell(out.X)
    return out


def normalize_log(adata: Any, config: dict[str, Any]) -> Any:
    norm = config.get("normalization", {})
    out = _copy_adata(adata)
    target_sum = float(norm.get("target_sum", 10000.0))
    totals = _counts_per_cell(out.X)
    scale = target_sum / np.maximum(totals, 1e-12)
    if sparse.issparse(out.X):
        out.X = sparse.diags(scale).dot(out.X).tocsr()
        if norm.get("log1p", True):
            out.X.data = np.log1p(out.X.data)
    else:
        out.X = np.asarray(out.X, dtype=float) * scale[:, None]
        if norm.get("log1p", True):
            out.X = np.log1p(out.X)
    return out


def select_hvg(adata: Any, config: dict[str, Any]) -> Any:
    n_top = int(config.get("hvg", {}).get("n_top_genes", 2000))
    if adata.n_vars <= n_top:
        adata.var["highly_variable"] = True
        return adata
    x = adata.X
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
        variance = mean_sq - mean**2
    else:
        variance = np.var(np.asarray(x), axis=0)
    top_idx = np.argsort(variance)[::-1][:n_top]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[top_idx] = True
    adata.var["highly_variable"] = mask
    return adata[:, mask].copy()


def compute_latent(adata: Any, config: dict[str, Any], seed: int) -> Any:
    latent = config.get("latent", {})
    n_components = int(latent.get("n_components", 50))
    if adata.n_obs < 2 or adata.n_vars < 2:
        x = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        adata.obsm["X_pca"] = np.asarray(x[:, :1], dtype=float)
        adata.uns["pca_variance_ratio"] = np.array([1.0])
        return adata
    n_components = max(1, min(n_components, adata.n_obs - 1, adata.n_vars - 1))
    if sparse.issparse(adata.X):
        model = TruncatedSVD(n_components=n_components, random_state=seed)
        coords = model.fit_transform(adata.X)
        variance_ratio = model.explained_variance_ratio_
    else:
        model = PCA(n_components=n_components, random_state=seed)
        coords = model.fit_transform(np.asarray(adata.X))
        variance_ratio = model.explained_variance_ratio_
    adata.obsm["X_pca"] = coords
    adata.uns["pca_variance_ratio"] = np.asarray(variance_ratio)
    return adata


def preprocess_anndata(adata: Any, config: dict[str, Any] | None = None) -> Any:
    config = config or {}
    seed = int(config.get("seed", 1729))
    seed_everything(seed)
    out = qc_filter(adata, config)
    out.layers["clean_counts"] = out.X.copy()
    out = normalize_log(out, config)
    out.layers["normalized_log"] = out.X.copy()
    out = select_hvg(out, config)
    out = compute_latent(out, config, seed)
    return out


def preprocess_dataset(
    dataset: str,
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    root = root or find_project_root()
    config_path = Path(config_path or root / "configs" / "preprocessing" / "default.yaml")
    config = read_yaml(config_path)
    registry = load_registry(root / "configs" / "datasets" / "registry.yaml")
    spec = registry[dataset]
    raw_path = root / "data" / "raw" / dataset / spec.raw_filename
    processed_dir = ensure_dir(root / "data" / "processed" / dataset)
    interim_dir = ensure_dir(root / "data" / "interim" / dataset)
    processed_path = processed_dir / spec.processed_filename
    placeholder = processed_dir / "TODO_DATA_DEPENDENCY.md"
    if not raw_path.exists():
        msg = f"Missing raw dataset {raw_path}. Run the official downloader first."
        write_placeholder(placeholder, "TODO Data Dependency", msg)
        if allow_placeholders:
            return {"dataset": dataset, "status": "placeholder", "path": str(placeholder)}
        raise FileNotFoundError(msg)

    seed = int(config.get("seed", 1729))
    adata = read_h5ad(raw_path)
    cleaned = qc_filter(adata, config)
    cleaned_path = interim_dir / f"{dataset}_cleaned.h5ad"
    write_h5ad(cleaned, cleaned_path)
    processed = preprocess_anndata(adata, config)
    write_h5ad(processed, processed_path)

    graph_cfg = config.get("graph", {})
    x = processed.obsm["X_pca"]
    node_ids = list(map(str, processed.obs_names))
    graph, edge_table = build_knn_graph(
        x,
        n_neighbors=int(graph_cfg.get("n_neighbors", 15)),
        metric=str(graph_cfg.get("metric", "euclidean")),
        weight_scheme=str(graph_cfg.get("weight_scheme", "heat_kernel")),
        heat_kernel_sigma=graph_cfg.get("heat_kernel_sigma", "auto"),
        node_ids=node_ids,
    )
    embedding_cols = [f"PC{i + 1}" for i in range(x.shape[1])]
    embedding = pd.DataFrame(x, index=processed.obs_names, columns=embedding_cols)
    artifact_paths = save_graph_artifacts(
        graph,
        edge_table,
        output_dir=processed_dir,
        prefix=dataset,
        node_metadata=processed.obs.copy(),
        embedding=embedding,
    )
    record = {
        "dataset": dataset,
        "status": "ok",
        "raw": artifact_record(raw_path, dataset=dataset, config_path=str(config_path), seed=seed, root=root),
        "cleaned": artifact_record(cleaned_path, dataset=dataset, config_path=str(config_path), seed=seed, root=root),
        "processed": artifact_record(processed_path, dataset=dataset, config_path=str(config_path), seed=seed, root=root),
        "graph_artifacts": {
            name: artifact_record(path, dataset=dataset, config_path=str(config_path), seed=seed, root=root)
            for name, path in artifact_paths.items()
        },
    }
    return record


def preprocess_many(
    datasets: list[str],
    *,
    root: Path | None = None,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    root = root or find_project_root()
    records = [
        preprocess_dataset(dataset, root=root, allow_placeholders=allow_placeholders)
        for dataset in datasets
    ]
    manifest = {"artifacts": records}
    write_json(manifest, root / "data" / "manifests" / "preprocess_manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pancreas", "lung", "zebrafish", "paul15"])
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args(argv)
    preprocess_many(args.datasets, allow_placeholders=args.allow_placeholders)


if __name__ == "__main__":  # pragma: no cover
    main()
