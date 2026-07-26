from __future__ import annotations

import pandas as pd

from ricci_cell_fate.tasks.validity import early_fate_split_diagnostics, proxy_task_diagnostics


def test_proxy_task_diagnostics_smoke():
    features = pd.DataFrame(
        {
            "betweenness": [0.9, 0.7, 0.1, 0.0],
            "degree": [3.0, 2.0, 1.0, 1.0],
            "branch_score": [2.0, 1.0, -1.0, -2.0],
            "bottleneck_score": [1.5, 0.5, -0.5, -1.5],
            "forman_curvature_mean": [-1.0, -0.5, 0.2, 0.4],
        },
        index=list("abcd"),
    )
    diagnostics = proxy_task_diagnostics(features, branch_quantile=0.75, bottleneck_quantile=0.75)
    assert diagnostics["branch_positive_count"] >= 1
    assert diagnostics["bottleneck_positive_count"] >= 1
    assert "proxy_jaccard" in diagnostics


def test_early_fate_split_diagnostics_flags_unseen_test_classes():
    features = pd.DataFrame({"diffusion_pseudotime": [0.1, 0.2, 0.3, 0.4]}, index=list("abcd"))
    obs = pd.DataFrame({"lineage": ["A", "A", "B", "B"]}, index=list("abcd"))
    split = {"split_id": "toy", "strategy": "lineage_holdout", "train": ["a", "b"], "validation": [], "test": ["c", "d"]}
    report = early_fate_split_diagnostics(
        features,
        obs,
        split,
        lineage_key="lineage",
        time_key=None,
        early_fraction=1.0,
    )
    assert report["all_test_classes_unseen"] is True
    assert report["frac_unseen_test_cells"] == 1.0
