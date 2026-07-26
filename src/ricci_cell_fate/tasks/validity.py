from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import networkx as nx
import pandas as pd

from ricci_cell_fate.tasks.bottleneck import proxy_bottleneck_labels
from ricci_cell_fate.tasks.branch import proxy_branch_labels
from ricci_cell_fate.tasks.early_fate import early_cell_mask, early_subset_source


def proxy_task_diagnostics(
    features: pd.DataFrame,
    *,
    branch_quantile: float = 0.95,
    bottleneck_quantile: float = 0.95,
) -> dict[str, float]:
    branch_labels = proxy_branch_labels(features, quantile=branch_quantile)
    bottleneck_labels = proxy_bottleneck_labels(features, quantile=bottleneck_quantile)
    overlap = int(((branch_labels == 1) & (bottleneck_labels == 1)).sum())
    union = int(((branch_labels == 1) | (bottleneck_labels == 1)).sum())

    def corr(left: str, right: str) -> float:
        if left not in features.columns or right not in features.columns:
            return float("nan")
        return float(features[left].astype(float).corr(features[right].astype(float)))

    payload = {
        "n_nodes": float(len(features)),
        "branch_positive_count": float(branch_labels.sum()),
        "bottleneck_positive_count": float(bottleneck_labels.sum()),
        "proxy_overlap_count": float(overlap),
        "proxy_jaccard": float(overlap / union) if union else 0.0,
        "branch_score_betweenness_corr": corr("branch_score", "betweenness"),
        "branch_score_degree_corr": corr("branch_score", "degree"),
        "bottleneck_score_betweenness_corr": corr("bottleneck_score", "betweenness"),
        "bottleneck_score_degree_corr": corr("bottleneck_score", "degree"),
    }
    curvature_mean = next((col for col in features.columns if col.endswith("_curvature_mean")), None)
    if curvature_mean is not None:
        payload["branch_score_curvature_corr"] = corr("branch_score", curvature_mean)
        payload["bottleneck_score_curvature_corr"] = corr("bottleneck_score", curvature_mean)
        payload["curvature_reference_column"] = curvature_mean
    return payload


def early_fate_split_diagnostics(
    features: pd.DataFrame,
    obs: pd.DataFrame,
    split: dict[str, Any],
    *,
    lineage_key: str,
    time_key: str | None,
    early_fraction: float = 0.35,
) -> dict[str, Any]:
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    fallback = features.loc[obs.index, "diffusion_pseudotime"]
    mask = early_cell_mask(obs, time_key, fallback, early_fraction)
    eligible = obs.index[mask.to_numpy()]
    eligible_set = set(eligible)
    train_ids = [item for item in split.get("train", []) if item in eligible_set]
    validation_ids = [item for item in split.get("validation", []) if item in eligible_set]
    test_ids = [item for item in split.get("test", []) if item in eligible_set]
    train_labels = obs.loc[train_ids, lineage_key].astype(str)
    validation_labels = obs.loc[validation_ids, lineage_key].astype(str)
    test_labels = obs.loc[test_ids, lineage_key].astype(str)
    train_classes = set(train_labels)
    test_classes = set(test_labels)
    unseen_test = sorted(test_classes - train_classes)
    return {
        "split_id": str(split.get("split_id", "")),
        "strategy": str(split.get("strategy", "")),
        "early_subset_source": early_subset_source(time_key),
        "n_early_cells": int(len(eligible)),
        "n_train": int(len(train_ids)),
        "n_validation": int(len(validation_ids)),
        "n_test": int(len(test_ids)),
        "n_train_classes": int(train_labels.nunique()),
        "n_validation_classes": int(validation_labels.nunique()),
        "n_test_classes": int(test_labels.nunique()),
        "n_unseen_test_classes": int(len(unseen_test)),
        "frac_unseen_test_cells": float((~test_labels.isin(train_classes)).mean()) if len(test_labels) else 0.0,
        "all_test_classes_unseen": bool(test_classes and test_classes.isdisjoint(train_classes)),
        "unseen_test_classes": ";".join(unseen_test),
    }


def graph_component_diagnostics(graph: nx.Graph) -> dict[str, Any]:
    component_sizes = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_components": len(component_sizes),
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "top_component_sizes": component_sizes[:10],
    }


def load_split(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
