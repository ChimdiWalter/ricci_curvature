from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def parse_zebrafish_sample_tokens(nodes: pd.Index | pd.Series | list[str]) -> pd.Series:
    index = pd.Index(nodes).astype(str)
    tokens = index.to_series(index=index).str.extract(r"_(DS\d+)_")[0].fillna("NO_DS")
    return tokens.rename("sample_token")


def make_group_holdout_splits(labels: pd.Series, groups: pd.Series) -> list[dict[str, Any]]:
    labels = labels.astype(str)
    groups = groups.astype(str).reindex(labels.index)
    payload: list[dict[str, Any]] = []
    for group_name in sorted(groups.dropna().unique()):
        test_mask = groups.eq(group_name)
        train_mask = ~test_mask
        if labels.loc[test_mask].nunique() < 2 or labels.loc[train_mask].nunique() < 2:
            continue
        payload.append(
            {
                "group": str(group_name),
                "train_nodes": labels.index[train_mask].astype(str).tolist(),
                "test_nodes": labels.index[test_mask].astype(str).tolist(),
            }
        )
    return payload


def build_pancreas_bottleneck_proxy(
    obs: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    cluster_key: str = "clusters",
    pseudotime_key: str = "palantir_pseudotime",
    anchor_cluster: str = "Ngn3 high EP",
    late_quantile: float = 0.75,
    tolerance_hops: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    cluster_series = obs[cluster_key].astype(str)
    pseudotime = pd.to_numeric(obs[pseudotime_key], errors="coerce")

    anchor_mask = cluster_series.eq(str(anchor_cluster))
    anchor_pseudotime = pseudotime.loc[anchor_mask].dropna()
    if anchor_pseudotime.empty:
        raise ValueError(f"No valid pseudotime values found for anchor cluster {anchor_cluster!r}.")
    threshold = float(anchor_pseudotime.quantile(float(late_quantile)))
    exact_positive = anchor_mask & pseudotime.ge(threshold)

    graph = nx.Graph()
    for row in edge_table.itertuples(index=False):
        graph.add_edge(str(row.source), str(row.target), weight=float(getattr(row, "weight", 1.0)))
    graph.add_nodes_from(obs.index)

    tolerant_nodes = set(obs.index[exact_positive.to_numpy()])
    frontier = set(tolerant_nodes)
    for _ in range(int(tolerance_hops)):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(str(neighbor) for neighbor in graph.neighbors(node))
        tolerant_nodes.update(next_frontier)
        frontier = next_frontier
    tolerant_positive = pd.Series(obs.index.isin(tolerant_nodes), index=obs.index)

    anchor_rank = pd.Series(np.nan, index=obs.index, dtype=float)
    if anchor_mask.any():
        anchor_rank.loc[anchor_mask] = pseudotime.loc[anchor_mask].rank(pct=True, method="average")

    labels = pd.DataFrame(
        {
            "node": obs.index,
            "cluster": cluster_series.reindex(obs.index).to_numpy(),
            "palantir_pseudotime": pseudotime.reindex(obs.index).to_numpy(),
            "anchor_cluster": anchor_mask.astype(int).reindex(obs.index).to_numpy(),
            "anchor_pseudotime_rank": anchor_rank.reindex(obs.index).to_numpy(),
            "bottleneck_exact": exact_positive.astype(int).reindex(obs.index).to_numpy(),
            "bottleneck_tolerant": tolerant_positive.astype(int).reindex(obs.index).to_numpy(),
        }
    ).set_index("node")
    summary = {
        "anchor_cluster": str(anchor_cluster),
        "late_quantile": float(late_quantile),
        "late_quantile_threshold": threshold,
        "n_anchor_cluster": int(anchor_mask.sum()),
        "n_exact_positive": int(exact_positive.sum()),
        "n_tolerant_positive": int(tolerant_positive.sum()),
        "tolerance_hops": int(tolerance_hops),
    }
    return labels, summary


def binary_label_metrics(
    y_true: pd.Series,
    positive_scores: np.ndarray,
    *,
    positive_label: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = y_true.astype(str)
    y = y_true.eq(str(positive_label)).astype(int).to_numpy()
    scores = np.asarray(positive_scores, dtype=float)
    pred = (scores >= float(threshold)).astype(int)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
    }


def bootstrap_binary_metrics(
    y_true: pd.Series,
    positive_scores: np.ndarray,
    *,
    positive_label: str,
    n_bootstrap: int = 1000,
    seed: int = 1729,
) -> dict[str, float]:
    y_true = y_true.astype(str).reset_index(drop=True)
    scores = np.asarray(positive_scores, dtype=float)
    if len(y_true) != len(scores):
        raise ValueError("y_true and positive_scores must have the same length.")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_bootstrap)):
        sample_idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        sample_y = y_true.iloc[sample_idx]
        sample_scores = scores[sample_idx]
        if sample_y.nunique() < 2:
            continue
        draws.append(binary_label_metrics(sample_y, sample_scores, positive_label=positive_label))
    out: dict[str, float] = {}
    for metric in ["macro_f1", "auroc", "auprc"]:
        series = np.array([item[metric] for item in draws], dtype=float)
        if series.size == 0:
            out[f"{metric}_mean"] = float("nan")
            out[f"{metric}_ci_low"] = float("nan")
            out[f"{metric}_ci_high"] = float("nan")
            continue
        out[f"{metric}_mean"] = float(series.mean())
        out[f"{metric}_ci_low"] = float(np.quantile(series, 0.025))
        out[f"{metric}_ci_high"] = float(np.quantile(series, 0.975))
    return out
