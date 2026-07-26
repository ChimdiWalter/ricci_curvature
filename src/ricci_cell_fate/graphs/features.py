from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def _safe_centrality(fn, graph: nx.Graph, default: float = 0.0) -> dict[str, float]:
    try:
        return {str(k): float(v) for k, v in fn(graph).items()}
    except Exception:
        return {str(node): default for node in graph.nodes}


def diffusion_pseudotime(graph: nx.Graph, root: str | None = None) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    if root is None:
        degree = dict(graph.degree(weight="weight"))
        root = min(degree, key=degree.get)
    lengths = nx.single_source_dijkstra_path_length(graph, root, weight="distance")
    max_len = max(lengths.values()) if lengths else 1.0
    max_len = max(max_len, np.finfo(float).eps)
    return {str(node): float(lengths.get(node, max_len) / max_len) for node in graph.nodes}


def graph_feature_table(
    graph: nx.Graph,
    root: str | None = None,
    *,
    betweenness_exact_limit: int = 500,
    betweenness_sample_size: int = 64,
    seed: int = 1729,
) -> pd.DataFrame:
    degree = {str(n): float(v) for n, v in graph.degree(weight=None)}
    weighted_degree = {str(n): float(v) for n, v in graph.degree(weight="weight")}
    pagerank = _safe_centrality(lambda g: nx.pagerank(g, weight="weight"), graph)
    closeness = _safe_centrality(lambda g: nx.closeness_centrality(g, distance="distance"), graph)
    if graph.number_of_nodes() <= betweenness_exact_limit:
        betweenness = _safe_centrality(
            lambda g: nx.betweenness_centrality(g, weight="distance", normalized=True), graph
        )
    else:
        k = min(betweenness_sample_size, graph.number_of_nodes())
        betweenness = _safe_centrality(
            lambda g: nx.betweenness_centrality(
                g,
                k=k,
                weight="distance",
                normalized=True,
                seed=seed,
            ),
            graph,
        )
    dpt = diffusion_pseudotime(graph, root=root)

    rows = []
    for node in graph.nodes:
        key = str(node)
        rows.append(
            {
                "node": key,
                "degree": degree.get(key, 0.0),
                "weighted_degree": weighted_degree.get(key, 0.0),
                "pagerank": pagerank.get(key, 0.0),
                "closeness": closeness.get(key, 0.0),
                "betweenness": betweenness.get(key, 0.0),
                "diffusion_pseudotime": dpt.get(key, 0.0),
            }
        )
    return pd.DataFrame(rows).set_index("node")


def degree_matched_rewire(graph: nx.Graph, seed: int = 1729, swaps_per_edge: int = 5) -> nx.Graph:
    rewired = graph.copy()
    nswap = max(1, swaps_per_edge * max(graph.number_of_edges(), 1))
    try:
        nx.double_edge_swap(rewired, nswap=nswap, max_tries=nswap * 20, seed=seed)
    except Exception:
        return graph.copy()
    for u, v in rewired.edges:
        if "weight" not in rewired[u][v]:
            rewired[u][v]["weight"] = 1.0
        if "distance" not in rewired[u][v]:
            rewired[u][v]["distance"] = 1.0 / max(float(rewired[u][v]["weight"]), 1e-12)
    return rewired
