from __future__ import annotations

import json

import numpy as np

from ricci_cell_fate.graphs.construction import build_knn_graph, graph_to_sparse_adjacency
from ricci_cell_fate.tasks.splits import persist_split, random_split


def test_knn_graph_construction_and_adjacency():
    x = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], dtype=float)
    graph, edges = build_knn_graph(x, n_neighbors=2, node_ids=["a", "b", "c", "d"])
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() >= 3
    assert {"source", "target", "weight", "distance"}.issubset(edges.columns)
    adj = graph_to_sparse_adjacency(graph)
    assert adj.shape == (4, 4)
    assert adj.nnz > 0


def test_split_persistence(tmp_path):
    split = random_split(["a", "b", "c", "d", "e", "f"], seed=1, split_id="smoke")
    path = tmp_path / "split.json"
    persist_split(split, path)
    payload = json.loads(path.read_text())
    assert payload["split_id"] == "smoke"
    assert set(payload) >= {"train", "validation", "test", "seed"}

