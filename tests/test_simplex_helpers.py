"""Tests for simplex_helpers: graph construction + triad geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wsinsight.insightlib.simplex_helpers import (
    build_dual_graph,
    build_line_graph,
    enumerate_edges_from_simplices,
    k_hop_on_adjacency,
    triad_geometry,
)


# ---------------------------------------------------------------------------
# Fixture: square with one diagonal splitting into two triangles.
#
#     3 ---- 2
#     |    / |
#     |   /  |
#     |  /   |
#     | /    |
#     0 ---- 1
#
# Vertices:  (0,0), (10,0), (10,10), (0,10)
# Simplices: [0,1,2] and [0,2,3]   (share edge 0-2)
# Undirected edges: {0-1, 1-2, 2-3, 0-3, 0-2}
# ---------------------------------------------------------------------------

CENTERS = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
SIMPLICES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def test_enumerate_edges_unique_and_sorted():
    edges = enumerate_edges_from_simplices(SIMPLICES)
    assert edges.shape == (5, 2)
    assert (edges[:, 0] < edges[:, 1]).all()
    edge_set = {tuple(e) for e in edges.tolist()}
    assert edge_set == {(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)}


def test_build_line_graph_shared_vertex_adjacency():
    edges = enumerate_edges_from_simplices(SIMPLICES)
    L = build_line_graph(edges, num_vertices=4)
    # Shape is (num_edges, num_edges)
    assert L.shape == (5, 5)
    # Each edge shares a vertex with several others; no self-loops.
    dense = L.toarray()
    assert (np.diag(dense) == 0).all()
    # Adjacency is symmetric.
    np.testing.assert_array_equal(dense, dense.T)
    # Edge 0-2 (the diagonal) touches all 4 other edges.
    # Find its row.
    diag_idx = None
    for i, e in enumerate(edges.tolist()):
        if tuple(e) == (0, 2):
            diag_idx = i
            break
    assert diag_idx is not None
    assert dense[diag_idx].sum() == 4


def test_build_dual_graph_two_triangles_share_vertices():
    D = build_dual_graph(SIMPLICES, num_vertices=4)
    dense = D.toarray()
    assert dense.shape == (2, 2)
    assert dense[0, 0] == 0 and dense[1, 1] == 0
    # They share ≥1 vertex → adjacent.
    assert dense[0, 1] == 1
    assert dense[1, 0] == 1


def test_k_hop_on_adjacency_excludes_self():
    D = build_dual_graph(SIMPLICES, num_vertices=4)
    nbrs = k_hop_on_adjacency(D, k=1)
    assert nbrs == [[1], [0]]
    # k=2 on 2-node graph still excludes self.
    nbrs2 = k_hop_on_adjacency(D, k=2)
    assert nbrs2 == [[1], [0]]


def test_triad_geometry_square_halves():
    geom = triad_geometry(CENTERS, SIMPLICES)
    # Both triangles are right isoceles with legs 10 → area 50.
    np.testing.assert_allclose(geom["area_px2"], [50.0, 50.0])
    # max edge is the hypotenuse 10√2 for each.
    np.testing.assert_allclose(
        geom["max_edge_px"], [10 * math.sqrt(2)] * 2, rtol=1e-6
    )
    # regularity in [0, 1]; not equilateral so strictly < 1.
    assert ((geom["regularity"] >= 0) & (geom["regularity"] <= 1)).all()
    assert (geom["regularity"] < 1.0).all()


def test_triad_geometry_equilateral_regularity_is_one():
    side = 10.0
    h = side * math.sqrt(3) / 2
    centers = np.array([[0, 0], [side, 0], [side / 2, h]], dtype=np.float64)
    simplices = np.array([[0, 1, 2]], dtype=np.int64)
    geom = triad_geometry(centers, simplices)
    assert math.isclose(float(geom["regularity"][0]), 1.0, rel_tol=1e-6)


def test_triad_geometry_sorted_edge_lengths():
    geom = triad_geometry(CENTERS, SIMPLICES)
    el = geom["edge_lengths_px"]
    # Each row sorted ascending.
    assert (el[:, 0] <= el[:, 1]).all()
    assert (el[:, 1] <= el[:, 2]).all()
