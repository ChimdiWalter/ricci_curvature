from __future__ import annotations

import pandas as pd


def compatibility_report(source_features: pd.DataFrame, target_features: pd.DataFrame) -> dict[str, object]:
    source_cols = set(source_features.select_dtypes("number").columns)
    target_cols = set(target_features.select_dtypes("number").columns)
    shared = sorted(source_cols & target_cols)
    return {
        "n_source_features": len(source_cols),
        "n_target_features": len(target_cols),
        "n_shared_features": len(shared),
        "shared_features": shared,
        "eligible": len(shared) >= 3,
    }

