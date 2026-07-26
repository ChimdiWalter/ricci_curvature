from __future__ import annotations

import numpy as np
import pandas as pd

from ricci_cell_fate.evaluation.metrics import classification_metrics, expected_calibration_error, precision_recall_f1_at_k
from ricci_cell_fate.tasks.branch import evaluate_branch_scores


def test_classification_and_calibration_metrics():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    prob = np.array([0.1, 0.6, 0.8, 0.9])
    metrics = classification_metrics(y, pred, prob)
    assert metrics["macro_f1"] > 0
    assert 0 <= expected_calibration_error(y, prob) <= 1


def test_top_k_and_branch_metrics():
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8]
    out = precision_recall_f1_at_k(labels, scores, 2)
    assert out["precision_at_k"] == 1.0
    features = pd.DataFrame({"betweenness": scores, "degree": [1, 2, 1, 2]}, index=list("abcd"))
    branch = evaluate_branch_scores(features, pd.Series(scores, index=list("abcd")), top_k=[2])
    assert "f1_at_k_2" in branch

