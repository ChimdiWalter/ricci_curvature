from __future__ import annotations

from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def neighborhood_measure(
    graph: nx.Graph,
    node: str,
    *,
    alpha: float = 0.5,
    weight: str = "weight",
) -> dict[str, float]:
    neighbors = list(graph.neighbors(node))
    measure = {str(node): float(alpha)}
    if not neighbors:
        return measure
    weights = np.array([float(graph[node][nbr].get(weight, 1.0)) for nbr in neighbors], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        probs = np.repeat((1.0 - alpha) / len(neighbors), len(neighbors))
    else:
        probs = (1.0 - alpha) * weights / total
    for nbr, prob in zip(neighbors, probs, strict=False):
        measure[str(nbr)] = float(prob)
    return measure


def _earth_movers_distance(cost: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> float:
    n, m = cost.shape
    c = cost.reshape(-1)
    a_eq = []
    b_eq = []
    for i in range(n):
        row = np.zeros((n, m))
        row[i, :] = 1.0
        a_eq.append(row.reshape(-1))
        b_eq.append(supply[i])
    for j in range(m):
        col = np.zeros((n, m))
        col[:, j] = 1.0
        a_eq.append(col.reshape(-1))
        b_eq.append(demand[j])
    result = linprog(c, A_eq=np.vstack(a_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    if result.success:
        return float(result.fun)
    return float(np.sum(np.minimum.outer(supply, demand) * cost))


def ollivier_edge_curvature(
    graph: nx.Graph,
    *,
    alpha: float = 0.5,
    max_edges: int | None = None,
) -> pd.DataFrame:
    rows = []
    edges = list(graph.edges(data=True))
    if max_edges is not None:
        edges = edges[:max_edges]

    @lru_cache(maxsize=None)
    def distances_from(node: str) -> dict[str, float]:
        return {str(k): float(v) for k, v in nx.single_source_dijkstra_path_length(graph, node, weight="distance").items()}

    for u, v, attrs in edges:
        source = str(u)
        target = str(v)
        mu = neighborhood_measure(graph, source, alpha=alpha)
        mv = neighborhood_measure(graph, target, alpha=alpha)
        left = list(mu)
        right = list(mv)
        cost = np.zeros((len(left), len(right)), dtype=float)
        for i, a in enumerate(left):
            da = distances_from(a)
            for j, b in enumerate(right):
                cost[i, j] = da.get(b, float(attrs.get("distance", 1.0)))
        supply = np.array([mu[a] for a in left], dtype=float)
        demand = np.array([mv[b] for b in right], dtype=float)
        base_distance = max(float(attrs.get("distance", 1.0)), 1e-12)
        w1 = _earth_movers_distance(cost, supply, demand)
        rows.append(
            {
                "source": source,
                "target": target,
                "ollivier_curvature": float(1.0 - w1 / base_distance),
                "transport_distance": float(w1),
            }
        )
    return pd.DataFrame(rows)

