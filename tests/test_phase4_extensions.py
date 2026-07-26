from __future__ import annotations

import pandas as pd

from ricci_cell_fate.tasks.phase4_extensions import (
    binary_label_metrics,
    build_pancreas_bottleneck_proxy,
    make_group_holdout_splits,
    parse_zebrafish_sample_tokens,
)


def test_parse_zebrafish_sample_tokens_extracts_expected_groups():
    nodes = [
        "ZFDOME_WT_DS5_AAAATCAAGAGG",
        "ZF50_WT_DS3_AAACCAACGCTAT",
        "ZF30_WT_AAACCAACGCTAT",
    ]
    tokens = parse_zebrafish_sample_tokens(nodes)
    assert tokens.tolist() == ["DS5", "DS3", "NO_DS"]


def test_make_group_holdout_splits_filters_single_class_groups():
    labels = pd.Series(
        ["alpha", "beta", "alpha", "beta", "alpha", "alpha"],
        index=[f"cell_{i}" for i in range(6)],
    )
    groups = pd.Series(
        ["g1", "g1", "g2", "g2", "g3", "g3"],
        index=labels.index,
    )
    splits = make_group_holdout_splits(labels, groups)
    assert [item["group"] for item in splits] == ["g1", "g2"]


def test_build_pancreas_bottleneck_proxy_marks_late_anchor_cells():
    obs = pd.DataFrame(
        {
            "clusters": ["Ngn3 high EP", "Ngn3 high EP", "Ngn3 high EP", "Fev+", "Alpha"],
            "palantir_pseudotime": [0.20, 0.40, 0.60, 0.80, 0.90],
        },
        index=["a", "b", "c", "d", "e"],
    )
    edges = pd.DataFrame(
        {
            "source": ["a", "b", "c", "d"],
            "target": ["b", "c", "d", "e"],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    labels, summary = build_pancreas_bottleneck_proxy(obs, edges, late_quantile=0.5, tolerance_hops=1)
    assert summary["n_exact_positive"] == 2
    assert int(labels["bottleneck_exact"].sum()) == 2
    assert int(labels["bottleneck_tolerant"].sum()) >= 3


def test_binary_label_metrics_returns_finite_scores():
    labels = pd.Series(["Notochord", "Prechordal Plate", "Notochord", "Prechordal Plate"])
    metrics = binary_label_metrics(labels, [0.9, 0.1, 0.8, 0.2], positive_label="Notochord")
    assert metrics["macro_f1"] > 0.9
    assert metrics["auroc"] > 0.9
    assert metrics["auprc"] > 0.9
