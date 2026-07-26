from __future__ import annotations

import numpy as np
import pandas as pd


def branch_bottleneck_scores(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    out = pd.DataFrame(index=df.index)
    degree = df.get("degree", pd.Series(0.0, index=df.index)).astype(float)
    centrality = df.get("betweenness", pd.Series(0.0, index=df.index)).astype(float)
    curv_cols = [c for c in df.columns if c.endswith("_curvature_mean") or c.endswith("_curvature_min")]
    if curv_cols:
        curvature_signal = -df[curv_cols].astype(float).mean(axis=1)
    else:
        curvature_signal = pd.Series(0.0, index=df.index)
    def z(x: pd.Series) -> pd.Series:
        sd = float(x.std())
        return (x - float(x.mean())) / (sd if sd > 0 else 1.0)
    out["branch_score"] = z(centrality) + z(curvature_signal) + 0.25 * z(degree)
    out["bottleneck_score"] = z(curvature_signal) + z(centrality) - 0.25 * z(degree)
    out["curvature_extremum_score"] = z(curvature_signal)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

