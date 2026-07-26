from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ricci_cell_fate.datasets.registry import DatasetSpec
from ricci_cell_fate.io.anndata_io import read_h5ad
from ricci_cell_fate.utils.provenance import write_json


def find_first_existing(keys: list[str] | tuple[str, ...], columns: list[str]) -> str | None:
    lower_to_real = {col.lower(): col for col in columns}
    for key in keys:
        if key in columns:
            return key
        if key.lower() in lower_to_real:
            return lower_to_real[key.lower()]
    return None


def summarize_anndata(adata: Any, spec: DatasetSpec | None = None) -> dict[str, Any]:
    obs_columns = list(map(str, adata.obs.columns))
    var_columns = list(map(str, adata.var.columns))
    uns_keys = list(map(str, getattr(adata, "uns", {}).keys()))
    obsm_keys = list(map(str, getattr(adata, "obsm", {}).keys()))
    summary = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": obs_columns,
        "var_columns": var_columns,
        "uns_keys": uns_keys,
        "obsm_keys": obsm_keys,
        "layers": list(map(str, getattr(adata, "layers", {}).keys())),
    }
    if spec is not None:
        summary.update(
            {
                "dataset": spec.name,
                "expected_shape": list(spec.expected_shape) if spec.expected_shape else None,
                "shape_matches_expected": (
                    tuple(adata.shape) == spec.expected_shape if spec.expected_shape else None
                ),
                "time_key": find_first_existing(spec.time_key_candidates, obs_columns),
                "lineage_key": find_first_existing(spec.lineage_key_candidates, obs_columns),
                "batch_key": find_first_existing(spec.batch_key_candidates, obs_columns),
            }
        )
    return summary


def validate_h5ad(path: str | Path, spec: DatasetSpec, output_json: str | Path | None = None) -> dict[str, Any]:
    adata = read_h5ad(path)
    summary = summarize_anndata(adata, spec)
    if spec.expected_shape and tuple(adata.shape) != spec.expected_shape:
        summary["warning"] = (
            f"Observed shape {tuple(adata.shape)} differs from registry expected "
            f"shape {spec.expected_shape}; continuing because upstream datasets may evolve."
        )
    if output_json:
        write_json(summary, output_json)
    return summary


def write_metadata_table(summary: dict[str, Any], output_csv: str | Path) -> None:
    rows = [
        {"field": key, "value": value if not isinstance(value, list) else ";".join(map(str, value))}
        for key, value in summary.items()
    ]
    p = Path(output_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)

