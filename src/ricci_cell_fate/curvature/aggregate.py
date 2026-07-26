from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .forman import forman_edge_curvature
from .ollivier import ollivier_edge_curvature


def merge_edge_curvatures(
    graph: nx.Graph,
    include_ollivier: bool = True,
    max_ollivier_edges: int | None = None,
) -> pd.DataFrame:
    edge_df = forman_edge_curvature(graph)
    if include_ollivier:
        edge_df = edge_df.merge(
            ollivier_edge_curvature(graph, max_edges=max_ollivier_edges),
            on=["source", "target"],
            how="left",
        )
    return edge_df


def node_curvature_features(edge_curvatures: pd.DataFrame) -> pd.DataFrame:
    curvature_cols = [col for col in edge_curvatures.columns if col.endswith("_curvature")]
    rows = []
    nodes = sorted(set(edge_curvatures["source"]).union(set(edge_curvatures["target"])))
    for node in nodes:
        incident = edge_curvatures[
            (edge_curvatures["source"] == node) | (edge_curvatures["target"] == node)
        ]
        row: dict[str, float | str] = {"node": node}
        for col in curvature_cols:
            values = incident[col].dropna().to_numpy(dtype=float)
            if values.size == 0:
                row[f"{col}_mean"] = 0.0
                row[f"{col}_min"] = 0.0
                row[f"{col}_max"] = 0.0
                row[f"{col}_std"] = 0.0
            else:
                row[f"{col}_mean"] = float(np.mean(values))
                row[f"{col}_min"] = float(np.min(values))
                row[f"{col}_max"] = float(np.max(values))
                row[f"{col}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows).set_index("node")


def curvature_feature_tables(
    graph: nx.Graph,
    include_ollivier: bool = True,
    max_ollivier_edges: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_df = merge_edge_curvatures(
        graph,
        include_ollivier=include_ollivier,
        max_ollivier_edges=max_ollivier_edges,
    )
    node_df = node_curvature_features(edge_df)
    return edge_df, node_df
