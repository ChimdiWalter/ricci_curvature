from __future__ import annotations

import math

import networkx as nx
import pandas as pd


def forman_edge_curvature(graph: nx.Graph, node_weight: float = 1.0) -> pd.DataFrame:
    rows = []
    for u, v, attrs in graph.edges(data=True):
        w_uv = float(attrs.get("weight", 1.0))
        w_uv = max(w_uv, 1e-12)
        term = node_weight / w_uv + node_weight / w_uv
        for nbr in graph.neighbors(u):
            if nbr == v:
                continue
            w_ux = max(float(graph[u][nbr].get("weight", 1.0)), 1e-12)
            term -= node_weight / math.sqrt(w_uv * w_ux)
        for nbr in graph.neighbors(v):
            if nbr == u:
                continue
            w_vy = max(float(graph[v][nbr].get("weight", 1.0)), 1e-12)
            term -= node_weight / math.sqrt(w_uv * w_vy)
        rows.append({"source": str(u), "target": str(v), "forman_curvature": float(w_uv * term)})
    return pd.DataFrame(rows)

