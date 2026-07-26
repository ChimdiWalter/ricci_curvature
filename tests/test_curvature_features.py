from __future__ import annotations

import networkx as nx

from ricci_cell_fate.curvature.aggregate import curvature_feature_tables
from ricci_cell_fate.curvature.forman import forman_edge_curvature
from ricci_cell_fate.curvature.ollivier import ollivier_edge_curvature


def _path_graph():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0, distance=1.0)
    graph.add_edge("b", "c", weight=1.0, distance=1.0)
    graph.add_edge("c", "d", weight=1.0, distance=1.0)
    return graph


def test_forman_and_ollivier_curvature_smoke():
    graph = _path_graph()
    forman = forman_edge_curvature(graph)
    ollivier = ollivier_edge_curvature(graph)
    assert len(forman) == graph.number_of_edges()
    assert len(ollivier) == graph.number_of_edges()
    assert "forman_curvature" in forman.columns
    assert "ollivier_curvature" in ollivier.columns


def test_node_curvature_features():
    edge_df, node_df = curvature_feature_tables(_path_graph())
    assert not edge_df.empty
    assert not node_df.empty
    assert any(col.endswith("_mean") for col in node_df.columns)

