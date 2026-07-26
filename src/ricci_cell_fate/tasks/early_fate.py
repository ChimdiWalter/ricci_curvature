from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ricci_cell_fate.datasets.validate import find_first_existing
from ricci_cell_fate.evaluation.metrics import classification_metrics
from ricci_cell_fate.models.classifiers import make_logistic_classifier


def early_cell_mask(obs: pd.DataFrame, time_key: str | None, fallback_scores: pd.Series, early_fraction: float) -> pd.Series:
    if time_key and time_key in obs.columns:
        raw = pd.to_numeric(obs[time_key], errors="coerce")
        if raw.isna().all():
            raw = pd.Series(pd.Categorical(obs[time_key].astype(str)).codes, index=obs.index)
        threshold = float(raw.quantile(early_fraction))
        return pd.Series(raw <= threshold, index=obs.index)
    aligned = fallback_scores.reindex(obs.index.astype(str)).fillna(fallback_scores.median())
    threshold = float(aligned.quantile(early_fraction))
    return pd.Series(aligned <= threshold, index=obs.index)


def early_subset_source(time_key: str | None) -> str:
    if time_key:
        return f"time_key:{time_key}"
    return "diffusion_pseudotime_fallback"


def run_early_fate_prediction(
    features: pd.DataFrame,
    obs: pd.DataFrame,
    split: dict[str, Any],
    *,
    lineage_key_candidates: tuple[str, ...],
    time_key_candidates: tuple[str, ...],
    seed: int = 1729,
    early_fraction: float = 0.35,
) -> dict[str, float]:
    lineage_key = find_first_existing(lineage_key_candidates, list(obs.columns))
    time_key = find_first_existing(time_key_candidates, list(obs.columns))
    if lineage_key is None:
        return {"status_missing_label": 1.0, "early_subset_source": early_subset_source(time_key)}
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    common = features.index.intersection(obs.index.astype(str))
    feature_cols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    x = features.loc[common, feature_cols].replace([np.inf, -np.inf], np.nan)
    y = obs.loc[common, lineage_key].astype(str)
    mask = early_cell_mask(obs.loc[common], time_key, features.loc[common, "diffusion_pseudotime"], early_fraction)
    eligible = common[mask.to_numpy()]
    train_ids = [i for i in split.get("train", []) if i in set(eligible)]
    test_ids = [i for i in split.get("test", []) if i in set(eligible)]
    early_source = early_subset_source(time_key)
    if len(train_ids) < 5 or len(test_ids) < 2:
        return {
            "status_insufficient_split": 1.0,
            "n_train": float(len(train_ids)),
            "n_test": float(len(test_ids)),
            "early_subset_source": early_source,
        }
    if y.loc[train_ids].nunique() < 2 or y.loc[test_ids].nunique() < 2:
        return {
            "status_insufficient_classes": 1.0,
            "n_train": float(len(train_ids)),
            "n_test": float(len(test_ids)),
            "n_train_classes": float(y.loc[train_ids].nunique()),
            "n_test_classes": float(y.loc[test_ids].nunique()),
            "early_subset_source": early_source,
        }
    train_classes = set(y.loc[train_ids].astype(str))
    test_labels = y.loc[test_ids].astype(str)
    unseen_mask = ~test_labels.isin(train_classes)
    if unseen_mask.any():
        return {
            "status_unseen_test_classes": 1.0,
            "n_train": float(len(train_ids)),
            "n_test": float(len(test_ids)),
            "n_train_classes": float(len(train_classes)),
            "n_test_classes": float(test_labels.nunique()),
            "n_unseen_test_classes": float(test_labels[unseen_mask].nunique()),
            "frac_unseen_test_cells": float(unseen_mask.mean()),
            "early_subset_source": early_source,
        }
    model = make_logistic_classifier(seed=seed)
    model.fit(x.loc[train_ids], y.loc[train_ids])
    pred = model.predict(x.loc[test_ids])
    prob = model.predict_proba(x.loc[test_ids])
    metrics = classification_metrics(y.loc[test_ids].to_numpy(), pred, prob)
    metrics["n_train"] = float(len(train_ids))
    metrics["n_test"] = float(len(test_ids))
    metrics["n_classes"] = float(y.loc[train_ids].nunique())
    metrics["early_subset_source"] = early_source
    return metrics
