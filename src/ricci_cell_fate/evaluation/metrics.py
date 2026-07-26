from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ricci_cell_fate.utils.paths import find_project_root
from ricci_cell_fate.utils.provenance import write_json


def precision_recall_f1_at_k(
    y_true: np.ndarray | list[int], scores: np.ndarray | list[float], k: int
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    k = min(max(int(k), 1), len(y))
    idx = np.argsort(s)[::-1][:k]
    pred = np.zeros_like(y)
    pred[idx] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {"precision_at_k": float(precision), "recall_at_k": float(recall), "f1_at_k": float(f1)}


def expected_calibration_error(
    y_true: np.ndarray | list[int],
    prob: np.ndarray | list[float],
    *,
    n_bins: int = 10,
) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def classification_metrics(
    y_true: np.ndarray | list[Any],
    y_pred: np.ndarray | list[Any],
    y_prob: np.ndarray | list[float] | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    metrics = {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}
    if y_prob is not None:
        prob = np.asarray(y_prob)
        classes = np.unique(y)
        try:
            is_binary = len(classes) <= 2
            if prob.ndim == 1 or prob.shape[1] == 1:
                positive = prob.ravel()
                metrics["auroc"] = float(roc_auc_score(y, positive))
                metrics["auprc"] = float(average_precision_score(y, positive))
                if set(classes).issubset({0, 1}):
                    metrics["brier"] = float(brier_score_loss(y.astype(int), positive))
                    metrics["ece"] = expected_calibration_error(y.astype(int), positive)
            elif prob.ndim == 2 and prob.shape[1] == 2 and is_binary:
                positive = prob[:, 1]
                metrics["auroc"] = float(roc_auc_score(y, positive))
                metrics["auprc"] = float(average_precision_score(y, positive))
                if set(classes).issubset({0, 1}):
                    metrics["brier"] = float(brier_score_loss(y.astype(int), positive))
                    metrics["ece"] = expected_calibration_error(y.astype(int), positive)
            else:
                metrics["auroc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
                metrics["auprc"] = float(
                    average_precision_score(pd.get_dummies(y).to_numpy(), prob, average="macro")
                )
        except ValueError:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
    return metrics


def bootstrap_ci(
    values: np.ndarray | list[float],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 1729,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(n_bootstrap)]
    alpha = 1.0 - confidence_level
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2)),
    }


def summarize_metric_files(root: Path | None = None) -> dict[str, Any]:
    root = root or find_project_root()
    metric_dir = root / "results" / "metrics"
    records = []
    for path in sorted(metric_dir.glob("*_metrics.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        group_cols = [col for col in ["dataset", "task", "mode", "ablation"] if col in df.columns]
        grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
        for keys, group in grouped:
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            summary = {"file": str(path), "n_rows": int(len(group))}
            if group_cols:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                summary.update({col: key for col, key in zip(group_cols, keys, strict=False)})
            for col in numeric_cols:
                ci = bootstrap_ci(group[col].to_numpy(), n_bootstrap=200)
                summary[f"{col}_mean"] = ci["mean"]
                summary[f"{col}_ci_low"] = ci["ci_low"]
                summary[f"{col}_ci_high"] = ci["ci_high"]
            records.append(summary)
    payload = {"metric_summaries": records}
    write_json(payload, root / "results" / "metrics" / "summary_metrics.json")
    table_path = root / "results" / "tables" / "main_results.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(table_path, index=False)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    summarize_metric_files()


if __name__ == "__main__":  # pragma: no cover
    main()
