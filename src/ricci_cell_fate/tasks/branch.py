from __future__ import annotations

import numpy as np
import pandas as pd

from ricci_cell_fate.evaluation.metrics import precision_recall_f1_at_k


def proxy_branch_labels(features: pd.DataFrame, quantile: float = 0.95) -> pd.Series:
    centrality = features.get("betweenness", pd.Series(0.0, index=features.index)).astype(float)
    threshold = float(centrality.quantile(quantile))
    return (centrality >= threshold).astype(int)


def evaluate_branch_scores(
    features: pd.DataFrame,
    scores: pd.Series,
    *,
    labels: pd.Series | None = None,
    top_k: list[int] | None = None,
    proxy_quantile: float = 0.95,
) -> dict[str, float]:
    labels = labels if labels is not None else proxy_branch_labels(features, quantile=proxy_quantile)
    labels = labels.reindex(features.index).fillna(0).astype(int)
    scores = scores.reindex(features.index).fillna(0.0).astype(float)
    metrics: dict[str, float] = {}
    for k in top_k or [10, 25, 50]:
        out = precision_recall_f1_at_k(labels.to_numpy(), scores.to_numpy(), k)
        metrics.update({f"{name}_{k}": value for name, value in out.items()})
    metrics["n_positive"] = float(labels.sum())
    metrics["n_nodes"] = float(len(labels))
    return metrics


def top_k_predictions(scores: pd.Series, k: int = 50) -> pd.DataFrame:
    ordered = scores.sort_values(ascending=False).head(k)
    return pd.DataFrame({"node": ordered.index.astype(str), "score": ordered.to_numpy(dtype=float)})
