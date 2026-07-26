from __future__ import annotations

import pandas as pd

from ricci_cell_fate.tasks.phase2_paul15 import build_feature_families, make_phase2_splits


def test_build_feature_families_smoke():
    features = pd.DataFrame(
        {
            "degree": [1.0, 2.0, 3.0, 4.0],
            "weighted_degree": [1.2, 2.1, 3.2, 4.1],
            "pagerank": [0.1, 0.2, 0.3, 0.4],
            "closeness": [0.3, 0.4, 0.2, 0.1],
            "betweenness": [0.0, 0.1, 0.4, 0.2],
            "diffusion_pseudotime": [0.1, 0.2, 0.3, 0.4],
            "forman_curvature_mean": [-1.0, -0.5, 0.2, 0.3],
            "forman_curvature_min": [-1.2, -0.8, 0.1, 0.2],
            "forman_curvature_max": [-0.7, -0.2, 0.4, 0.5],
            "forman_curvature_std": [0.1, 0.2, 0.1, 0.2],
            "ollivier_curvature_mean": [-0.8, -0.4, 0.3, 0.6],
            "ollivier_curvature_min": [-1.0, -0.5, 0.1, 0.3],
            "ollivier_curvature_max": [-0.4, -0.2, 0.5, 0.8],
            "ollivier_curvature_std": [0.2, 0.1, 0.2, 0.3],
        },
        index=list("abcd"),
    )
    families = build_feature_families(canonical_features=features, random_seed=13)
    assert "graph_feature_stack" in families
    assert "graph_plus_ollivier" in families
    assert "random_matched_control" in families


def test_make_phase2_splits_preserves_positive_support():
    labels = pd.Series([0, 0, 0, 1, 1, 1], index=list("abcdef"))
    splits = make_phase2_splits(labels, split_seeds=[13, 17], test_size=0.5)
    assert len(splits) == 2
    for split in splits:
        train = labels.loc[split["train_nodes"]]
        test = labels.loc[split["test_nodes"]]
        assert train.nunique() == 2
        assert test.nunique() == 2
