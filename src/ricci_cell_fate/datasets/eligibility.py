from __future__ import annotations

from pathlib import Path
from typing import Any

from ricci_cell_fate.datasets.registry import DatasetSpec, load_registry
from ricci_cell_fate.datasets.validate import find_first_existing, summarize_anndata
from ricci_cell_fate.io.anndata_io import read_h5ad
from ricci_cell_fate.utils.paths import find_project_root
from ricci_cell_fate.utils.provenance import write_json


TASKS = [
    "branch_point_localization",
    "early_fate_prediction",
    "bottleneck_detection",
    "trajectory_segmentation",
    "cross_dataset_transfer",
]


def infer_task_eligibility_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    has_time = summary.get("time_key") is not None
    has_lineage = summary.get("lineage_key") is not None
    has_embedding = any(str(k).lower().startswith("x_") for k in summary.get("obsm_keys", []))
    has_cluster = summary.get("lineage_key") is not None
    graph_available_after_preprocess = True

    return {
        "branch_point_localization": {
            "eligible": bool(has_cluster and graph_available_after_preprocess),
            "label_source": "annotation_or_proxy",
            "reason": "Requires cluster/lineage annotations or curvature-centrality proxy labels.",
        },
        "early_fate_prediction": {
            "eligible": bool(has_lineage and (has_time or has_embedding)),
            "label_source": "lineage annotations with early-cell restriction",
            "reason": "Requires fate labels and a time/pseudotime or embedding-derived early subset.",
        },
        "bottleneck_detection": {
            "eligible": bool(graph_available_after_preprocess),
            "label_source": "curvature/topology ranking with optional proxy evaluation",
            "reason": "Requires rebuilt graph; biological labels improve interpretability.",
        },
        "trajectory_segmentation": {
            "eligible": bool(graph_available_after_preprocess),
            "label_source": "graph communities plus metadata overlays",
            "reason": "Can run on any processed graph, with stronger evaluation if labels exist.",
        },
        "cross_dataset_transfer": {
            "eligible": bool(has_cluster),
            "label_source": "shared feature schema and compatible annotation diagnostics",
            "reason": "Requires comparable feature construction; label compatibility is checked later.",
        },
    }


def infer_task_eligibility(
    dataset_paths: dict[str, Path],
    registry: dict[str, DatasetSpec] | None = None,
) -> dict[str, Any]:
    registry = registry or load_registry()
    payload: dict[str, Any] = {"datasets": {}}
    for name, path in dataset_paths.items():
        spec = registry[name]
        if not path.exists():
            payload["datasets"][name] = {
                task: {"eligible": False, "reason": f"Missing processed file: {path}"}
                for task in TASKS
            }
            continue
        adata = read_h5ad(path)
        summary = summarize_anndata(adata, spec)
        payload["datasets"][name] = infer_task_eligibility_from_summary(summary)
        payload["datasets"][name]["schema"] = {
            "time_key": summary.get("time_key"),
            "lineage_key": summary.get("lineage_key"),
            "batch_key": summary.get("batch_key"),
            "n_obs": summary.get("n_obs"),
            "n_vars": summary.get("n_vars"),
        }
    return payload


def write_task_eligibility(datasets: list[str], root: Path | None = None) -> dict[str, Any]:
    root = root or find_project_root()
    dataset_paths = {
        name: root / "data" / "processed" / name / f"{name}_processed.h5ad" for name in datasets
    }
    payload = infer_task_eligibility(dataset_paths)
    out = root / "data" / "manifests" / "task_eligibility.json"
    write_json(payload, out)
    return payload

