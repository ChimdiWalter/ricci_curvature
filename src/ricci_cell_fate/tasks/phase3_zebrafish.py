from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ricci_cell_fate.tasks.phase3_pancreas import (
    BranchTargetArtifacts,
    build_branch_region_target,
    build_family_map,
    build_feature_families,
    evaluate_branch_feature_family,
    evaluate_multiclass_feature_family,
    full_graph_feature_signal,
    make_stratified_splits,
    rank_branch_nodes_from_scores,
    rank_fate_nodes_from_predictions,
)


@dataclass(frozen=True)
class EarlyLineageArtifacts:
    cell_labels: pd.DataFrame
    class_support: pd.DataFrame
    supported_labels: tuple[str, ...]


def build_early_lineage_task(
    obs: pd.DataFrame,
    *,
    stage_key: str,
    lineage_key: str,
    early_stages: list[str],
    supported_labels: list[str],
) -> EarlyLineageArtifacts:
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    stage_series = obs[stage_key].astype(str)
    lineage_series = obs[lineage_key].astype(str)
    eligible = stage_series.isin(list(map(str, early_stages))) & lineage_series.isin(list(map(str, supported_labels)))
    subset = obs.loc[eligible].copy()
    cell_labels = pd.DataFrame(
        {
            "node": subset.index,
            "stage": stage_series.loc[eligible].to_numpy(),
            "lineage_label": lineage_series.loc[eligible].to_numpy(),
        }
    ).set_index("node")
    support = (
        cell_labels["lineage_label"]
        .value_counts()
        .rename_axis("lineage_label")
        .reset_index(name="n_cells")
        .sort_values(["n_cells", "lineage_label"], ascending=[False, True])
    )
    return EarlyLineageArtifacts(
        cell_labels=cell_labels,
        class_support=support,
        supported_labels=tuple(support["lineage_label"].astype(str)),
    )


__all__ = [
    "BranchTargetArtifacts",
    "EarlyLineageArtifacts",
    "build_branch_region_target",
    "build_early_lineage_task",
    "build_family_map",
    "build_feature_families",
    "evaluate_branch_feature_family",
    "evaluate_multiclass_feature_family",
    "full_graph_feature_signal",
    "make_stratified_splits",
    "rank_branch_nodes_from_scores",
    "rank_fate_nodes_from_predictions",
]
