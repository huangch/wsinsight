"""Shared graph/geometry utilities for the simplicial composition family
(ncomp / ecomp / tcomp).

This module builds derived structures from a cached Delaunay triangulation:

* :func:`enumerate_edges_from_simplices` — unique undirected edges from ``(M, 3)``
  simplex indices.
* :func:`build_line_graph` — edge-edge adjacency (vertex-shared). Two Delaunay
  edges are neighbors iff they share a vertex. Used by ``ecomp``.
* :func:`build_dual_graph` — triad-triad adjacency (vertex-shared). Two Delaunay
  triangles are neighbors iff they share ≥ 1 vertex. Used by ``tcomp``.
* :func:`k_hop_on_adjacency` — generic k-hop reachability over an arbitrary
  sparse adjacency matrix, same pattern as :func:`insight_helpers.k_hop_neighbors`.
* :func:`triad_geometry` — vectorized per-triad edge lengths, area, perimeter,
  regularity, and max edge length in pixels.

All functions operate on numpy/scipy.sparse; no pandas or pandas-typed inputs.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import eye as speye

# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def enumerate_edges_from_simplices(simplices: np.ndarray) -> np.ndarray:
    """Return unique undirected edges derived from a ``(M, 3)`` simplex array.

    Parameters
    ----------
    simplices:
        ``(M, 3)`` integer array of triangle vertex indices.

    Returns
    -------
    np.ndarray
        ``(E, 2)`` int64 array of edges with ``edge[:, 0] < edge[:, 1]`` and
        rows sorted lexicographically.  ``E`` is the number of unique
        undirected edges.
    """
    if simplices.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    pairs = np.concatenate(
        [
            simplices[:, [0, 1]],
            simplices[:, [0, 2]],
            simplices[:, [1, 2]],
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs


# ---------------------------------------------------------------------------
# Line graph (edge-edge adjacency)
# ---------------------------------------------------------------------------


def build_line_graph(edges: np.ndarray, num_vertices: int) -> csr_matrix:
    """Return the line graph adjacency: two edges are neighbors iff they share a vertex.

    Uses incidence-matrix multiplication:  ``L = B @ B.T - 2*I``, binarised,
    where ``B`` is the ``(E, V)`` edge-vertex incidence matrix.  Two distinct
    Delaunay edges share at most one vertex, so off-diagonal entries of
    ``B @ B.T`` are either 0 or 1.

    Parameters
    ----------
    edges:
        ``(E, 2)`` int array of vertex indices per edge.
    num_vertices:
        ``V``: total number of vertices (cells), used to size the incidence
        matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(E, E)`` symmetric uint8 adjacency, zero diagonal.
    """
    E = edges.shape[0]
    if E == 0:
        return csr_matrix((0, 0), dtype=np.uint8)

    rows = np.repeat(np.arange(E, dtype=np.int64), 2)
    cols = edges.reshape(-1).astype(np.int64, copy=False)
    data = np.ones(rows.size, dtype=np.uint8)

    B = csr_matrix((data, (rows, cols)), shape=(E, num_vertices), dtype=np.uint8)
    L = (B @ B.T).tocsr()
    # Zero the diagonal. B@B.T diagonal is 2 (each edge shares both vertices with itself).
    L.setdiag(0)
    L.eliminate_zeros()
    # Binarise.
    L.data[:] = 1
    return L


# ---------------------------------------------------------------------------
# Dual graph (triad-triad adjacency)
# ---------------------------------------------------------------------------


def build_dual_graph(simplices: np.ndarray, num_vertices: int) -> csr_matrix:
    """Return the dual graph adjacency: two triads are neighbors iff they share ≥ 1 vertex.

    Uses ``D = C @ C.T - 3*I``, binarised, where ``C`` is the ``(T, V)``
    triad-vertex incidence matrix.  Off-diagonal ``C @ C.T`` entries are 0, 1,
    or 2 (two triads can share up to 2 vertices if they are face-adjacent).

    Parameters
    ----------
    simplices:
        ``(T, 3)`` int array of triangle vertex indices.
    num_vertices:
        Total number of vertices, used to size the incidence matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(T, T)`` symmetric uint8 adjacency, zero diagonal.
    """
    T = simplices.shape[0]
    if T == 0:
        return csr_matrix((0, 0), dtype=np.uint8)

    rows = np.repeat(np.arange(T, dtype=np.int64), 3)
    cols = simplices.reshape(-1).astype(np.int64, copy=False)
    data = np.ones(rows.size, dtype=np.uint8)

    C = csr_matrix((data, (rows, cols)), shape=(T, num_vertices), dtype=np.uint8)
    D = (C @ C.T).tocsr()
    D.setdiag(0)
    D.eliminate_zeros()
    D.data[:] = 1
    return D


# ---------------------------------------------------------------------------
# Generic k-hop reachability
# ---------------------------------------------------------------------------


def k_hop_on_adjacency(adj: csr_matrix, k: int) -> list[list[int]]:
    """Return the list of k-hop neighbors for each node of an adjacency matrix.

    Mirrors :func:`insight_helpers.k_hop_neighbors`:  builds ``M = A + I``,
    raises it to power ``k`` under boolean semiring (binarised at each step),
    and extracts per-row neighbor indices (excluding the node itself).

    Parameters
    ----------
    adj:
        ``(N, N)`` symmetric sparse adjacency.
    k:
        Number of hops; ``k >= 1``.

    Returns
    -------
    list[list[int]]
        ``neighbor_lists[i]`` is the sorted list of indices reachable from
        ``i`` within ``k`` hops, excluding ``i`` itself.  Isolated nodes map
        to an empty list.
    """
    Mk = k_hop_adjacency_matrix(adj, k)
    N = Mk.shape[0]
    indptr = Mk.indptr
    indices = Mk.indices
    neighbor_lists: list[list[int]] = []
    for i in range(N):
        row = indices[indptr[i] : indptr[i + 1]]
        nbrs = row[row != i].tolist()
        neighbor_lists.append(nbrs)
    return neighbor_lists


def k_hop_adjacency_matrix(adj: csr_matrix, k: int) -> csr_matrix:
    """Return the k-hop reachability adjacency matrix (zero diagonal, binarised).

    Builds ``M = A + I``, raises to power ``k`` under boolean semiring
    (binarising at each step), then zeros the diagonal.  The returned
    matrix ``A_k`` has ``A_k[i, j] == 1`` iff ``j`` is reachable from
    ``i`` within ``k`` hops and ``j != i``.

    This is the sparse-matrix form consumed directly by downstream
    aggregations (e.g. ``A_k @ one_hot_types`` yields per-node
    type counts) without materialising Python neighbor lists.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    N = adj.shape[0]
    if N == 0:
        return csr_matrix((0, 0), dtype=np.uint8)

    M = (adj + speye(N, dtype=np.uint8, format="csr")).tocsr()
    M.data[:] = 1

    Mk = M
    for _ in range(k - 1):
        Mk = (Mk @ M).tocsr()
        Mk.data[:] = 1

    # Exclude self-reachability so A_k behaves like an adjacency matrix.
    Mk.setdiag(0)
    Mk.eliminate_zeros()
    return Mk


# ---------------------------------------------------------------------------
# Triad geometry
# ---------------------------------------------------------------------------


def triad_geometry(centers: np.ndarray, simplices: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-triad geometry in pixel units.

    All quantities are returned in pixels (or pixels²); conversion to µm is
    the caller's responsibility.

    Parameters
    ----------
    centers:
        ``(N, 2)`` array of cell centres in pixels.
    simplices:
        ``(T, 3)`` int array of triangle vertex indices.

    Returns
    -------
    dict[str, np.ndarray]
        Keys (all length ``T`` except ``"edge_lengths_px"``):

        * ``"centroid_x"``, ``"centroid_y"`` — int32 triad centroids
        * ``"edge_lengths_px"`` — ``(T, 3)`` float64, sorted per-row ascending
        * ``"min_edge_px"``, ``"max_edge_px"``, ``"mean_edge_px"`` — float64
        * ``"perimeter_px"`` — float64
        * ``"area_px2"`` — float64, always non-negative (via shoelace)
        * ``"regularity"`` — float64 in ``[0, 1]``; 1 for equilateral triangles,
          using the formula ``12·√3·A / P²`` (ratio of area to that of the
          equilateral triangle with the same perimeter).
    """
    T = simplices.shape[0]
    if T == 0:
        return {
            "centroid_x": np.empty(0, dtype=np.int32),
            "centroid_y": np.empty(0, dtype=np.int32),
            "edge_lengths_px": np.empty((0, 3), dtype=np.float64),
            "min_edge_px": np.empty(0, dtype=np.float64),
            "max_edge_px": np.empty(0, dtype=np.float64),
            "mean_edge_px": np.empty(0, dtype=np.float64),
            "perimeter_px": np.empty(0, dtype=np.float64),
            "area_px2": np.empty(0, dtype=np.float64),
            "regularity": np.empty(0, dtype=np.float64),
        }

    pts = centers[simplices].astype(np.float64, copy=False)  # (T, 3, 2)

    # Edge vectors: v0v1, v0v2, v1v2
    e01 = pts[:, 1] - pts[:, 0]
    e02 = pts[:, 2] - pts[:, 0]
    e12 = pts[:, 2] - pts[:, 1]
    len01 = np.linalg.norm(e01, axis=1)
    len02 = np.linalg.norm(e02, axis=1)
    len12 = np.linalg.norm(e12, axis=1)
    edge_lengths = np.stack([len01, len02, len12], axis=1)  # (T, 3)
    edge_lengths_sorted = np.sort(edge_lengths, axis=1)

    perimeter = edge_lengths.sum(axis=1)

    # Shoelace area (always non-negative).
    cross = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]
    area = 0.5 * np.abs(cross)

    # Regularity: 12·√3·A / P². Equilateral → 1; degenerate (A=0) → 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        regularity = np.where(
            perimeter > 0,
            12.0 * np.sqrt(3.0) * area / (perimeter**2),
            0.0,
        )
    # Clip to [0, 1] to absorb floating-point noise.
    regularity = np.clip(regularity, 0.0, 1.0)

    centroid = pts.mean(axis=1)
    centroid_x = np.rint(centroid[:, 0]).astype(np.int32)
    centroid_y = np.rint(centroid[:, 1]).astype(np.int32)

    return {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "edge_lengths_px": edge_lengths_sorted,
        "min_edge_px": edge_lengths_sorted[:, 0],
        "max_edge_px": edge_lengths_sorted[:, 2],
        "mean_edge_px": edge_lengths.mean(axis=1),
        "perimeter_px": perimeter,
        "area_px2": area,
        "regularity": regularity,
    }
