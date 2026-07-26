from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ricci_cell_fate.evaluation.metrics import classification_metrics, precision_recall_f1_at_k
from ricci_cell_fate.models.classifiers import make_logistic_classifier
from ricci_cell_fate.tasks.phase2_paul15 import (
    GRAPH_COLUMNS,
    build_feature_families,
    full_graph_feature_signal,
)


@dataclass(frozen=True)
class BranchTargetArtifacts:
    node_labels: pd.DataFrame
    cluster_summary: pd.DataFrame
    anchor_clusters: tuple[str, ...]


@dataclass(frozen=True)
class FateTaskArtifacts:
    cell_labels: pd.DataFrame
    class_support: pd.DataFrame
    supported_labels: tuple[str, ...]


@dataclass(frozen=True)
class PancreasTaskArtifacts:
    branch_labels: pd.DataFrame
    early_labels: pd.DataFrame
    early_class_support: pd.DataFrame
    cluster_progression_summary: pd.DataFrame


def build_family_map(mapping: dict[str, str]) -> dict[str, str]:
    return {str(key): str(value) for key, value in mapping.items()}


def build_cluster_transition_summary(
    obs: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    cluster_key: str,
    family_map: dict[str, str],
) -> pd.DataFrame:
    cluster_series = obs[cluster_key].astype(str)
    rows = []
    for cluster in sorted(cluster_series.unique()):
        nodes = set(cluster_series[cluster_series == cluster].index.astype(str))
        intra = 0.0
        external = 0.0
        neighbor_family_totals: dict[str, float] = {}
        neighbor_cluster_totals: dict[str, float] = {}
        for row in edge_table.itertuples(index=False):
            source = str(row.source)
            target = str(row.target)
            weight = float(getattr(row, "weight", 1.0))
            if source in nodes and target in nodes:
                intra += weight
            elif source in nodes:
                external += weight
                target_cluster = str(cluster_series[target])
                target_family = family_map[target_cluster]
                neighbor_cluster_totals[target_cluster] = neighbor_cluster_totals.get(target_cluster, 0.0) + weight
                neighbor_family_totals[target_family] = neighbor_family_totals.get(target_family, 0.0) + weight
            elif target in nodes:
                external += weight
                source_cluster = str(cluster_series[source])
                source_family = family_map[source_cluster]
                neighbor_cluster_totals[source_cluster] = neighbor_cluster_totals.get(source_cluster, 0.0) + weight
                neighbor_family_totals[source_family] = neighbor_family_totals.get(source_family, 0.0) + weight
        total = intra + external
        if external > 0:
            probs = np.array([value / external for value in neighbor_family_totals.values()], dtype=float)
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        else:
            entropy = 0.0
        rows.append(
            {
                "cluster": cluster,
                "family": family_map[cluster],
                "n_cells": int((cluster_series == cluster).sum()),
                "external_weight": external,
                "internal_weight": intra,
                "external_fraction": float(external / total) if total > 0 else 0.0,
                "neighbor_family_entropy": entropy,
                "n_neighbor_families": int(len(neighbor_family_totals)),
                "top_neighbor_clusters": "; ".join(
                    f"{name}:{value:.2f}"
                    for name, value in sorted(neighbor_cluster_totals.items(), key=lambda item: item[1], reverse=True)[:6]
                ),
                "top_neighbor_families": "; ".join(
                    f"{name}:{value:.2f}"
                    for name, value in sorted(neighbor_family_totals.items(), key=lambda item: item[1], reverse=True)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["neighbor_family_entropy", "external_fraction", "n_cells"],
        ascending=[False, False, False],
    )


def build_branch_region_target(
    obs: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    cluster_key: str,
    family_map: dict[str, str],
    anchor_clusters: list[str],
    tolerance_hops: int = 1,
) -> BranchTargetArtifacts:
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    graph = nx.Graph()
    for row in edge_table.itertuples(index=False):
        graph.add_edge(str(row.source), str(row.target), weight=float(getattr(row, "weight", 1.0)))
    graph.add_nodes_from(obs.index)

    cluster_summary = build_cluster_transition_summary(
        obs,
        edge_table,
        cluster_key=cluster_key,
        family_map=family_map,
    )
    cluster_summary["is_anchor_cluster"] = cluster_summary["cluster"].isin(anchor_clusters)

    cluster_series = obs[cluster_key].astype(str)
    exact_positive = cluster_series.isin(anchor_clusters)
    tolerance_positive = exact_positive.copy()
    if tolerance_hops > 0:
        exact_nodes = set(cluster_series.index[exact_positive.to_numpy()])
        tolerant_nodes = set(exact_nodes)
        frontier = set(exact_nodes)
        for _ in range(tolerance_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(str(neighbor) for neighbor in graph.neighbors(node))
            tolerant_nodes.update(next_frontier)
            frontier = next_frontier
        tolerance_positive = pd.Series(obs.index.isin(tolerant_nodes), index=obs.index)

    node_labels = pd.DataFrame(
        {
            "node": obs.index,
            "cluster": cluster_series.reindex(obs.index).to_numpy(),
            "family": [family_map[str(cluster)] for cluster in cluster_series.reindex(obs.index)],
            "branch_region_exact": exact_positive.astype(int).reindex(obs.index).to_numpy(),
            "branch_region_tolerant": tolerance_positive.astype(int).reindex(obs.index).to_numpy(),
            "anchor_cluster": cluster_series.isin(anchor_clusters).astype(int).reindex(obs.index).to_numpy(),
        }
    ).set_index("node")
    return BranchTargetArtifacts(
        node_labels=node_labels,
        cluster_summary=cluster_summary,
        anchor_clusters=tuple(anchor_clusters),
    )


def build_preterminal_fate_task(
    obs: pd.DataFrame,
    *,
    fine_key: str,
    cluster_key: str,
    pseudotime_key: str,
    label_map: dict[str, str],
) -> FateTaskArtifacts:
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    fine_labels = obs[fine_key].astype(str)
    eligible = fine_labels.isin(label_map)
    subset = obs.loc[eligible].copy()
    mapped = fine_labels.loc[eligible].map(label_map).astype(str)
    cell_labels = pd.DataFrame(
        {
            "node": subset.index,
            "cluster": subset[cluster_key].astype(str).to_numpy(),
            "cluster_fine": fine_labels.loc[eligible].to_numpy(),
            "fate_label": mapped.to_numpy(),
            "palantir_pseudotime": pd.to_numeric(subset[pseudotime_key], errors="coerce").to_numpy(),
        }
    ).set_index("node")
    support = (
        cell_labels["fate_label"]
        .value_counts()
        .rename_axis("fate_label")
        .reset_index(name="n_cells")
        .sort_values(["n_cells", "fate_label"], ascending=[False, True])
    )
    return FateTaskArtifacts(
        cell_labels=cell_labels,
        class_support=support,
        supported_labels=tuple(support["fate_label"].astype(str)),
    )


def make_stratified_splits(
    labels: pd.Series,
    *,
    split_seeds: list[int],
    test_size: float,
) -> list[dict[str, Any]]:
    y = labels.astype(str).to_numpy()
    index = labels.index.to_numpy()
    payload = []
    for seed in split_seeds:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
        payload.append(
            {
                "seed": int(seed),
                "train_nodes": [str(index[item]) for item in train_idx],
                "test_nodes": [str(index[item]) for item in test_idx],
            }
        )
    return payload


def evaluate_branch_feature_family(
    x: pd.DataFrame,
    node_labels: pd.DataFrame,
    split_payload: dict[str, Any],
    *,
    top_k: list[int],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_nodes = split_payload["train_nodes"]
    test_nodes = split_payload["test_nodes"]
    y_exact = node_labels["branch_region_exact"].astype(int)
    y_tolerant = node_labels["branch_region_tolerant"].astype(int)
    model = make_logistic_classifier(seed=seed)
    model.fit(x.loc[train_nodes], y_exact.loc[train_nodes])
    prob = model.predict_proba(x.loc[test_nodes])
    positive_scores = prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else prob.ravel()
    pred = model.predict(x.loc[test_nodes])
    exact_metrics = classification_metrics(y_exact.loc[test_nodes].to_numpy(), pred, prob)
    tolerant_metrics = classification_metrics(y_tolerant.loc[test_nodes].to_numpy(), pred, prob)
    row: dict[str, Any] = {
        "split_seed": int(split_payload["seed"]),
        "n_train": len(train_nodes),
        "n_test": len(test_nodes),
        "train_positive_exact": int(y_exact.loc[train_nodes].sum()),
        "test_positive_exact": int(y_exact.loc[test_nodes].sum()),
        "test_positive_tolerant": int(y_tolerant.loc[test_nodes].sum()),
        "auroc_exact": exact_metrics.get("auroc", float("nan")),
        "auprc_exact": exact_metrics.get("auprc", float("nan")),
        "macro_f1_exact": exact_metrics.get("macro_f1", float("nan")),
        "auroc_tolerant": tolerant_metrics.get("auroc", float("nan")),
        "auprc_tolerant": tolerant_metrics.get("auprc", float("nan")),
        "macro_f1_tolerant": tolerant_metrics.get("macro_f1", float("nan")),
    }
    for k in top_k:
        exact_top = precision_recall_f1_at_k(y_exact.loc[test_nodes].to_numpy(), positive_scores, k)
        tolerant_top = precision_recall_f1_at_k(y_tolerant.loc[test_nodes].to_numpy(), positive_scores, k)
        for name, value in exact_top.items():
            row[f"{name}_{k}_exact"] = value
        for name, value in tolerant_top.items():
            row[f"{name}_{k}_tolerant"] = value
    prediction_rows = pd.DataFrame(
        {
            "node": test_nodes,
            "score": positive_scores,
            "predicted_label": pred,
            "label_exact": y_exact.loc[test_nodes].to_numpy(),
            "label_tolerant": y_tolerant.loc[test_nodes].to_numpy(),
        }
    )
    return row, prediction_rows


def evaluate_multiclass_feature_family(
    x: pd.DataFrame,
    cell_labels: pd.DataFrame | pd.Series,
    split_payload: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_nodes = split_payload["train_nodes"]
    test_nodes = split_payload["test_nodes"]
    if isinstance(cell_labels, pd.Series):
        y = cell_labels.astype(str)
    else:
        label_col = "early_lineage_label" if "early_lineage_label" in cell_labels.columns else "fate_label"
        y = cell_labels[label_col].astype(str)
    model = make_logistic_classifier(seed=seed)
    model.fit(x.loc[train_nodes], y.loc[train_nodes])
    pred = model.predict(x.loc[test_nodes])
    prob = model.predict_proba(x.loc[test_nodes])
    metrics = classification_metrics(y.loc[test_nodes].to_numpy(), pred, prob)
    confidence = prob.max(axis=1) if prob.ndim == 2 else prob.ravel()
    row: dict[str, Any] = {
        "split_seed": int(split_payload["seed"]),
        "n_train": len(train_nodes),
        "n_test": len(test_nodes),
        "n_classes": int(y.loc[train_nodes].nunique()),
        "macro_f1": metrics.get("macro_f1", float("nan")),
        "auroc": metrics.get("auroc", float("nan")),
        "auprc": metrics.get("auprc", float("nan")),
    }
    class_order = [str(item) for item in getattr(model.named_steps["logistic"], "classes_", [])]
    prediction_rows = pd.DataFrame(
        {
            "node": test_nodes,
            "predicted_label": pred.astype(str),
            "true_label": y.loc[test_nodes].astype(str).to_numpy(),
            "confidence": confidence,
        }
    )
    for idx, label in enumerate(class_order):
        prediction_rows[f"prob_{label}"] = prob[:, idx]
    return row, prediction_rows


def rank_branch_nodes_from_scores(scores: pd.Series, labels: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    ordered = scores.sort_values(ascending=False).head(k)
    return pd.DataFrame(
        {
            "node": ordered.index.astype(str),
            "score": ordered.to_numpy(dtype=float),
            "cluster": labels.loc[ordered.index, "cluster"].to_numpy(),
            "family": labels.loc[ordered.index, "family"].to_numpy(),
            "label_exact": labels.loc[ordered.index, "branch_region_exact"].to_numpy(),
            "label_tolerant": labels.loc[ordered.index, "branch_region_tolerant"].to_numpy(),
        }
    )


def rank_fate_nodes_from_predictions(predictions: pd.DataFrame, cell_labels: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    ordered = predictions.sort_values("confidence", ascending=False).head(k).copy()
    ordered["cluster"] = cell_labels.loc[ordered["node"], "cluster"].to_numpy()
    ordered["cluster_fine"] = cell_labels.loc[ordered["node"], "cluster_fine"].to_numpy()
    if "palantir_pseudotime" in cell_labels.columns:
        ordered["palantir_pseudotime"] = cell_labels.loc[ordered["node"], "palantir_pseudotime"].to_numpy()
    return ordered


def build_pancreas_task_artifacts(
    obs: pd.DataFrame,
    config: dict[str, Any],
    *,
    edge_table: pd.DataFrame | None = None,
) -> PancreasTaskArtifacts:
    obs = obs.copy()
    obs.index = obs.index.astype(str)

    branch_cfg = dict(config.get("branch_region_task", {}))
    cluster_key = str(branch_cfg.get("cluster_key", "clusters"))
    exact_clusters = [str(item) for item in branch_cfg.get("exact_positive_clusters", [])]
    tolerant_clusters = [str(item) for item in branch_cfg.get("tolerant_positive_clusters", exact_clusters)]
    family_map = build_family_map(branch_cfg.get("family_map", {}))
    cluster_series = obs[cluster_key].astype(str)
    branch_labels = pd.DataFrame(
        {
            "cluster": cluster_series.to_numpy(),
            "family": [family_map.get(str(cluster), str(cluster)) for cluster in cluster_series],
            "branch_region_exact": cluster_series.isin(exact_clusters).astype(int).to_numpy(),
            "branch_region_tolerant": cluster_series.isin(tolerant_clusters).astype(int).to_numpy(),
        },
        index=obs.index,
    )

    early_cfg = dict(config.get("early_fate_task", {}))
    fine_key = str(early_cfg.get("cluster_key", "clusters_fine"))
    pseudotime_key = str(config.get("metadata", {}).get("pseudotime_key", "palantir_pseudotime"))
    eligible_clusters_by_label = dict(early_cfg.get("eligible_clusters_by_label", {}))
    label_map = {
        str(cluster): str(label)
        for label, clusters in eligible_clusters_by_label.items()
        for cluster in clusters
    }
    fine_labels = obs[fine_key].astype(str)
    eligible = fine_labels.isin(label_map)
    early_labels = pd.DataFrame(
        {
            "cluster": obs.get("clusters", pd.Series(index=obs.index, dtype="object")).astype(str).to_numpy(),
            "cluster_fine": fine_labels.to_numpy(),
            "early_committed": eligible.astype(int).to_numpy(),
            "early_lineage_label": fine_labels.map(label_map).where(eligible),
            pseudotime_key: pd.to_numeric(obs.get(pseudotime_key), errors="coerce"),
        },
        index=obs.index,
    )
    early_class_support = (
        early_labels.loc[early_labels["early_committed"] == 1, "early_lineage_label"]
        .value_counts()
        .rename_axis("early_lineage_label")
        .reset_index(name="n_cells")
        .sort_values(["n_cells", "early_lineage_label"], ascending=[False, True])
    )
    if edge_table is not None and not edge_table.empty:
        cluster_progression_summary = build_cluster_transition_summary(
            obs,
            edge_table,
            cluster_key=cluster_key,
            family_map=family_map,
        )
        cluster_progression_summary["is_exact_positive_cluster"] = cluster_progression_summary["cluster"].isin(
            exact_clusters
        )
        cluster_progression_summary["is_tolerant_positive_cluster"] = cluster_progression_summary["cluster"].isin(
            tolerant_clusters
        )
    else:
        cluster_progression_summary = (
            branch_labels.reset_index(names="node")
            .groupby(["cluster", "family"], as_index=False)
            .agg(
                n_cells=("node", "count"),
                branch_region_exact=("branch_region_exact", "sum"),
                branch_region_tolerant=("branch_region_tolerant", "sum"),
            )
            .sort_values(["branch_region_exact", "branch_region_tolerant", "n_cells"], ascending=[False, False, False])
        )
    return PancreasTaskArtifacts(
        branch_labels=branch_labels,
        early_labels=early_labels,
        early_class_support=early_class_support,
        cluster_progression_summary=cluster_progression_summary,
    )


__all__ = [
    "GRAPH_COLUMNS",
    "BranchTargetArtifacts",
    "FateTaskArtifacts",
    "PancreasTaskArtifacts",
    "build_branch_region_target",
    "build_cluster_transition_summary",
    "build_family_map",
    "build_feature_families",
    "build_pancreas_task_artifacts",
    "build_preterminal_fate_task",
    "evaluate_branch_feature_family",
    "evaluate_multiclass_feature_family",
    "full_graph_feature_signal",
    "make_stratified_splits",
    "rank_branch_nodes_from_scores",
    "rank_fate_nodes_from_predictions",
]
