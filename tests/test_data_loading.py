from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from ricci_cell_fate.datasets.registry import load_registry
from ricci_cell_fate.datasets.validate import summarize_anndata
from ricci_cell_fate.io.anndata_io import read_h5ad, write_h5ad
from ricci_cell_fate.preprocessing.pipeline import preprocess_anndata


def test_dataset_registry_contains_official_routes():
    registry = load_registry()
    for name in ["pancreas", "lung", "zebrafish", "paul15"]:
        spec = registry[name]
        assert spec.official_api_url.startswith("https://")
        assert spec.loader


def test_h5ad_roundtrip_and_schema_summary(tmp_path):
    adata = ad.AnnData(
        X=np.ones((4, 3)),
        obs=pd.DataFrame({"time": [0, 1, 2, 3], "cell_type": ["a", "a", "b", "b"]}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    path = tmp_path / "tiny.h5ad"
    write_h5ad(adata, path)
    loaded = read_h5ad(path)
    summary = summarize_anndata(loaded)
    assert summary["n_obs"] == 4
    assert summary["n_vars"] == 3


def test_preprocessing_smoke_on_tiny_subset():
    rng = np.random.default_rng(5)
    adata = ad.AnnData(
        X=rng.poisson(2, size=(8, 6)).astype(float),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
    )
    processed = preprocess_anndata(
        adata,
        {
            "seed": 5,
            "qc": {"min_counts_per_cell": 0, "min_genes_per_cell": 0, "min_cells_per_gene": 1},
            "normalization": {"target_sum": 1000.0, "log1p": True},
            "hvg": {"n_top_genes": 4},
            "latent": {"n_components": 3},
        },
    )
    assert "X_pca" in processed.obsm
    assert processed.obsm["X_pca"].shape[1] == 3
