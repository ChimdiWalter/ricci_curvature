from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ricci_cell_fate.utils.paths import find_project_root


def dataset_summary_table(root: Path | None = None) -> pd.DataFrame:
    root = root or find_project_root()
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "data" / "manifests").glob("*_schema.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "dataset": payload.get("dataset", path.name.replace("_schema.json", "")),
                "n_cells": payload.get("n_obs"),
                "n_genes": payload.get("n_vars"),
                "time_key": payload.get("time_key"),
                "lineage_key": payload.get("lineage_key"),
                "batch_key": payload.get("batch_key"),
                "shape_matches_expected": payload.get("shape_matches_expected"),
            }
        )
    return pd.DataFrame(rows)


def split_summary_table(root: Path | None = None) -> pd.DataFrame:
    root = root or find_project_root()
    rows = []
    for path in sorted((root / "data" / "splits").glob("*/*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "dataset": payload.get("dataset"),
                "task": payload.get("task"),
                "split_id": payload.get("split_id"),
                "strategy": payload.get("strategy"),
                "n_train": len(payload.get("train", [])),
                "n_validation": len(payload.get("validation", [])),
                "n_test": len(payload.get("test", [])),
                "seed": payload.get("seed"),
            }
        )
    return pd.DataFrame(rows)


def ablation_summary_table(root: Path | None = None) -> pd.DataFrame:
    root = root or find_project_root()
    path = root / "results" / "metrics" / "ablation_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_reproducibility_table(root: Path | None = None) -> pd.DataFrame:
    root = root or find_project_root()
    manifests = sorted((root / "results" / "manifests").glob("*.json")) + sorted(
        (root / "data" / "manifests").glob("*.json")
    )
    return pd.DataFrame(
        [
            {
                "manifest": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "modified_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
            }
            for path in manifests
        ]
    )


def write_report_tables(root: Path | None = None) -> list[Path]:
    root = root or find_project_root()
    table_dir = root / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "dataset_summary.csv": dataset_summary_table(root),
        "split_summary.csv": split_summary_table(root),
        "ablation_table.csv": ablation_summary_table(root),
        "compute_reproducibility.csv": compute_reproducibility_table(root),
    }
    outputs = []
    for name, df in tables.items():
        out = table_dir / name
        df.to_csv(out, index=False)
        outputs.append(out)
    return outputs

