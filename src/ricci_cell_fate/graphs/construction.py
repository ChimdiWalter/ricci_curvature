from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _as_array(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def heat_kernel_weights(distances: np.ndarray, sigma: float | str = "auto") -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    if sigma == "auto":
        positive = d[d > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
    else:
        scale = float(sigma)
    scale = max(scale, np.finfo(float).eps)
    return np.exp(-(d**2) / (2.0 * scale**2))


def compute_edge_weights(distances: np.ndarray, scheme: str, sigma: float | str = "auto") -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    if scheme == "binary":
        return np.ones_like(distances, dtype=float)
    if scheme == "distance":
        return 1.0 / (1.0 + distances)
    if scheme == "heat_kernel":
        return heat_kernel_weights(distances, sigma)
    raise ValueError(f"Unknown edge weighting scheme {scheme!r}")


def build_knn_graph(
    x: Any,
    *,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    weight_scheme: str = "heat_kernel",
    heat_kernel_sigma: float | str = "auto",
    node_ids: list[str] | None = None,
) -> tuple[nx.Graph, pd.DataFrame]:
    """Build a symmetric weighted kNN graph from a matrix."""
    arr = _as_array(x)
    if arr.ndim != 2:
        raise ValueError("Input matrix must be two-dimensional")
    n_obs = arr.shape[0]
    if n_obs < 2:
        raise ValueError("Need at least two observations to build a graph")
    k = min(max(1, n_neighbors), n_obs - 1)
    ids = node_ids or [str(i) for i in range(n_obs)]

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    weights = compute_edge_weights(distances, weight_scheme, heat_kernel_sigma)

    graph = nx.Graph()
    for node in ids:
        graph.add_node(node)
    rows = []
    for i in range(n_obs):
        for j_pos, j in enumerate(indices[i]):
            u, v = ids[i], ids[int(j)]
            distance = float(distances[i, j_pos])
            weight = float(weights[i, j_pos])
            if graph.has_edge(u, v):
                old = graph[u][v]
                old["weight"] = max(float(old["weight"]), weight)
                old["distance"] = min(float(old["distance"]), distance)
            else:
                graph.add_edge(u, v, weight=weight, distance=max(distance, 1e-12))
    for u, v, data in graph.edges(data=True):
        rows.append(
            {
                "source": u,
                "target": v,
                "weight": float(data.get("weight", 1.0)),
                "distance": float(data.get("distance", 1.0)),
            }
        )
    return graph, pd.DataFrame(rows)


def graph_to_sparse_adjacency(graph: nx.Graph, node_order: list[str] | None = None) -> sparse.csr_matrix:
    nodes = node_order or list(graph.nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    rows, cols, data = [], [], []
    for u, v, attrs in graph.edges(data=True):
        i, j = idx[u], idx[v]
        w = float(attrs.get("weight", 1.0))
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))


def save_graph_artifacts(
    graph: nx.Graph,
    edge_table: pd.DataFrame,
    *,
    output_dir: str | Path,
    prefix: str,
    node_metadata: pd.DataFrame | None = None,
    embedding: pd.DataFrame | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    node_order = list(graph.nodes)
    adjacency = graph_to_sparse_adjacency(graph, node_order)
    graph_path = out / f"{prefix}_graph.npz"
    sparse.save_npz(graph_path, adjacency)
    edges_path = out / f"{prefix}_edge_weights.csv"
    edge_table.to_csv(edges_path, index=False)
    paths = {"graph": str(graph_path), "edge_weights": str(edges_path)}
    if node_metadata is not None:
        node_path = out / f"{prefix}_node_metadata.csv"
        node_metadata.to_csv(node_path, index=True)
        paths["node_metadata"] = str(node_path)
    if embedding is not None:
        emb_path = out / f"{prefix}_embedding.csv"
        embedding.to_csv(emb_path, index=True)
        paths["embedding"] = str(emb_path)
    return paths


def graph_from_edge_table(edge_table: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in edge_table.itertuples(index=False):
        graph.add_edge(
            str(row.source),
            str(row.target),
            weight=float(getattr(row, "weight", 1.0)),
            distance=float(getattr(row, "distance", 1.0)),
        )
    return graph

