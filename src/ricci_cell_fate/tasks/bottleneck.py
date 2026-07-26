from __future__ import annotations

import pandas as pd

from ricci_cell_fate.evaluation.metrics import precision_recall_f1_at_k


def proxy_bottleneck_labels(features: pd.DataFrame, quantile: float = 0.95) -> pd.Series:
    betweenness = features.get("betweenness", pd.Series(0.0, index=features.index)).astype(float)
    degree = features.get("degree", pd.Series(0.0, index=features.index)).astype(float)
    score = betweenness.rank(pct=True) + (1.0 - degree.rank(pct=True))
    threshold = float(score.quantile(quantile))
    return (score >= threshold).astype(int)


def evaluate_bottleneck_scores(
    features: pd.DataFrame,
    scores: pd.Series,
    *,
    labels: pd.Series | None = None,
    top_k: list[int] | None = None,
    proxy_quantile: float = 0.95,
) -> dict[str, float]:
    labels = labels if labels is not None else proxy_bottleneck_labels(features, quantile=proxy_quantile)
    labels = labels.reindex(features.index).fillna(0).astype(int)
    scores = scores.reindex(features.index).fillna(0.0).astype(float)
    metrics: dict[str, float] = {}
    for k in top_k or [10, 25, 50]:
        out = precision_recall_f1_at_k(labels.to_numpy(), scores.to_numpy(), k)
        metrics.update({f"{name}_{k}": value for name, value in out.items()})
    metrics["n_positive"] = float(labels.sum())
    metrics["n_nodes"] = float(len(labels))
    return metrics
