from __future__ import annotations

import pandas as pd

from ricci_cell_fate.tasks.phase2_paul15 import make_phase2_splits
from ricci_cell_fate.tasks.phase3_pancreas import build_pancreas_task_artifacts, evaluate_multiclass_feature_family


def _mock_config() -> dict[str, object]:
    return {
        "metadata": {"pseudotime_key": "palantir_pseudotime"},
        "branch_region_task": {
            "cluster_key": "clusters",
            "exact_positive_clusters": ["Fev+"],
            "tolerant_positive_clusters": ["Fev+", "Ngn3 high EP"],
            "family_map": {
                "Ngn3 low EP": "progenitor",
                "Ngn3 high EP": "progenitor",
                "Fev+": "branch",
                "Alpha": "alpha",
                "Beta": "beta",
                "Delta": "delta",
                "Epsilon": "epsilon",
            },
        },
        "early_fate_task": {
            "cluster_key": "clusters_fine",
            "eligible_clusters_by_label": {
                "alpha": ["Pre-Alpha", "Fev+ Alpha"],
                "beta": ["Pre-Beta", "Fev+ Beta"],
                "delta": ["Fev+ Delta"],
                "epsilon": ["Fev+ Epsilon"],
            },
        },
    }


def test_build_pancreas_task_artifacts_smoke():
    obs = pd.DataFrame(
        {
            "clusters": ["Fev+", "Ngn3 high EP", "Beta", "Alpha"],
            "clusters_fine": ["Fev+ Beta", "Ngn3 high EP", "Pre-Beta", "Pre-Alpha"],
            "clusters_coarse": ["Fev+", "Ngn3 high EP", "Endocrine", "Endocrine"],
            "palantir_pseudotime": [0.4, 0.2, 0.9, 0.85],
        },
        index=["a", "b", "c", "d"],
    )
    artifacts = build_pancreas_task_artifacts(obs, _mock_config())
    assert int(artifacts.branch_labels["branch_region_exact"].sum()) == 1
    assert int(artifacts.branch_labels["branch_region_tolerant"].sum()) == 2
    assert int(artifacts.early_labels["early_committed"].sum()) == 3
    assert set(artifacts.early_class_support["early_lineage_label"]) == {"alpha", "beta"}


def test_pancreas_early_splits_keep_all_classes():
    labels = pd.Series(
        ["alpha"] * 8 + ["beta"] * 8 + ["delta"] * 4 + ["epsilon"] * 4,
        index=[f"cell_{i}" for i in range(24)],
    )
    splits = make_phase2_splits(labels, split_seeds=[13, 17], test_size=0.25)
    for split in splits:
        train = labels.loc[split["train_nodes"]]
        test = labels.loc[split["test_nodes"]]
        assert set(train.unique()) == {"alpha", "beta", "delta", "epsilon"}
        assert set(test.unique()) == {"alpha", "beta", "delta", "epsilon"}


def test_multiclass_family_accepts_series_labels():
    x = pd.DataFrame(
        {
            "feature_1": [
                0.1, 0.2, 0.3, 0.4,
                1.0, 1.1, 1.2, 1.3,
                2.0, 2.1, 2.2, 2.3,
                3.0, 3.1, 3.2, 3.3,
            ],
            "feature_2": [
                0.0, 0.1, 0.2, 0.3,
                1.0, 1.1, 1.2, 1.3,
                2.0, 2.1, 2.2, 2.3,
                3.0, 3.1, 3.2, 3.3,
            ],
        },
        index=[f"cell_{i}" for i in range(16)],
    )
    labels = pd.Series(
        ["alpha"] * 4 + ["beta"] * 4 + ["delta"] * 4 + ["epsilon"] * 4,
        index=x.index,
    )
    split = make_phase2_splits(labels, split_seeds=[13], test_size=0.25)[0]
    metrics, predictions = evaluate_multiclass_feature_family(x, labels, split, seed=13)
    assert "macro_f1" in metrics
    assert not predictions.empty
