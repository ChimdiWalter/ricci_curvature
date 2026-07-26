from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit

from ricci_cell_fate.evaluation.metrics import classification_metrics, precision_recall_f1_at_k
from ricci_cell_fate.models.classifiers import make_logistic_classifier


GRAPH_COLUMNS = [
    "degree",
    "weighted_degree",
    "pagerank",
    "closeness",
    "betweenness",
    "diffusion_pseudotime",
]


@dataclass(frozen=True)
class BranchTargetArtifacts:
    node_labels: pd.DataFrame
    cluster_summary: pd.DataFrame
    anchor_clusters: tuple[str, ...]


def build_family_map(mapping: dict[str, str]) -> dict[str, str]:
    return {str(cluster): str(family) for cluster, family in mapping.items()}


def build_cluster_transition_summary(
    obs: pd.DataFrame,
    edge_table: pd.DataFrame,
    family_map: dict[str, str],
) -> pd.DataFrame:
    cluster_series = obs["paul15_clusters"].astype(str)
    rows = []
    unique_clusters = sorted(cluster_series.unique())
    for cluster in unique_clusters:
        nodes = set(cluster_series[cluster_series == cluster].index)
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

    cluster_summary = build_cluster_transition_summary(obs, edge_table, family_map)
    cluster_summary["is_anchor_cluster"] = cluster_summary["cluster"].isin(anchor_clusters)

    cluster_series = obs["paul15_clusters"].astype(str)
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


def build_feature_families(
    *,
    canonical_features: pd.DataFrame,
    random_seed: int,
    rewired_graph_features: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    families: dict[str, pd.DataFrame] = {}
    features = canonical_features.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    forman_cols = [column for column in features.columns if column.startswith("forman_curvature_")]
    ollivier_cols = [column for column in features.columns if column.startswith("ollivier_curvature_")]
    graph_cols = [column for column in GRAPH_COLUMNS if column in features.columns]
    centrality_cols = [column for column in ["pagerank", "closeness"] if column in features.columns]
    covariate_cols = [column for column in ["degree", "betweenness"] if column in features.columns]

    families["degree_only"] = features[["degree"]].copy()
    families["weighted_degree_only"] = features[["weighted_degree"]].copy()
    families["centrality_only"] = features[centrality_cols].copy()
    families["betweenness_only"] = features[["betweenness"]].copy()
    families["diffusion_pseudotime_only"] = features[["diffusion_pseudotime"]].copy()
    families["graph_feature_stack"] = features[graph_cols].copy()
    families["forman_curvature_only"] = features[forman_cols].copy()
    if ollivier_cols:
        families["ollivier_curvature_only"] = features[ollivier_cols].copy()
    families["graph_plus_forman"] = features[graph_cols + forman_cols].copy()
    if ollivier_cols:
        families["graph_plus_ollivier"] = features[graph_cols + ollivier_cols].copy()

    if covariate_cols and forman_cols:
        families["residualized_forman"] = _residualize(features[forman_cols], features[covariate_cols])
    if covariate_cols and ollivier_cols:
        families["residualized_ollivier"] = _residualize(features[ollivier_cols], features[covariate_cols])

    rng = np.random.default_rng(random_seed)
    matched_dim = len(graph_cols + ollivier_cols) if ollivier_cols else len(graph_cols + forman_cols)
    matched_dim = max(1, matched_dim)
    families["random_matched_control"] = pd.DataFrame(
        rng.normal(size=(len(features), matched_dim)),
        index=features.index,
        columns=[f"random_feature_{index + 1}" for index in range(matched_dim)],
    )
    if rewired_graph_features is not None:
        rewired = rewired_graph_features.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rewired_cols = [column for column in GRAPH_COLUMNS if column in rewired.columns]
        families["rewired_graph_control"] = rewired[rewired_cols].copy()
    return families


def make_phase2_splits(
    labels: pd.Series,
    *,
    split_seeds: list[int],
    test_size: float,
) -> list[dict[str, Any]]:
    y = labels.to_numpy()
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


def evaluate_feature_family(
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
    if prob.ndim == 2 and prob.shape[1] > 1:
        positive_scores = prob[:, 1]
    else:
        positive_scores = prob.ravel()
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


def rank_nodes_from_scores(scores: pd.Series, labels: pd.DataFrame, k: int = 50) -> pd.DataFrame:
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


def full_graph_feature_signal(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    y_exact = labels["branch_region_exact"].astype(float)
    y_tolerant = labels["branch_region_tolerant"].astype(float)
    covariates = features[[column for column in ["degree", "betweenness"] if column in features.columns]].astype(float)
    for column in [item for item in features.columns if "curvature" in item]:
        series = features[column].astype(float)
        rows.append(
            {
                "feature": column,
                "analysis": "raw",
                "corr_exact": float(series.corr(y_exact)),
                "corr_tolerant": float(series.corr(y_tolerant)),
            }
        )
        if not covariates.empty:
            residual = _residualize(series.to_frame(name=column), covariates)[column]
            rows.append(
                {
                    "feature": column,
                    "analysis": "residualized_on_degree_betweenness",
                    "corr_exact": float(residual.corr(y_exact)),
                    "corr_tolerant": float(residual.corr(y_tolerant)),
                }
            )
    return pd.DataFrame(rows)


def _residualize(target_features: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    cov = covariates.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    output = pd.DataFrame(index=target_features.index)
    model = LinearRegression()
    for column in target_features.columns:
        y = target_features[column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        model.fit(cov, y)
        fitted = model.predict(cov)
        output[column] = y - fitted
    return output
