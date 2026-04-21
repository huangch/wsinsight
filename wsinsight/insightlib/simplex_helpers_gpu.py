"""GPU-accelerated counterparts of a few :mod:`simplex_helpers` routines.

These functions mirror the CPU versions but operate on
:mod:`cupyx.scipy.sparse` matrices.  They are optional — callers should
import lazily and fall back to the CPU implementation if ``cupy`` is not
available or CUDA runtime initialisation fails.

EXPERIMENTAL.  Output values match the CPU implementations exactly for
boolean/integer adjacency work; float aggregation is done in float32 and
then materialised as dense ``numpy`` arrays on the host.
"""

from __future__ import annotations

import numpy as np


def _import_cupy():
    import cupy as cp  # noqa: F401
    import cupyx.scipy.sparse as cpsp  # noqa: F401

    return cp, cpsp


def build_line_graph_gpu(edges: np.ndarray, num_vertices: int):
    """GPU version of :func:`simplex_helpers.build_line_graph`.

    Returns a ``cupyx.scipy.sparse.csr_matrix`` of float32 with zero diagonal
    and binarised entries.  float32 is used because ``cupyx.scipy.sparse``
    only supports float/complex/bool dtypes.
    """
    cp, cpsp = _import_cupy()

    E = edges.shape[0]
    if E == 0:
        return cpsp.csr_matrix((0, 0), dtype=cp.float32)

    rows = cp.repeat(cp.arange(E, dtype=cp.int32), 2)
    cols = cp.asarray(edges.reshape(-1).astype(np.int32, copy=False))
    data = cp.ones(rows.size, dtype=cp.float32)

    B = cpsp.csr_matrix((data, (rows, cols)), shape=(E, num_vertices), dtype=cp.float32)
    L = (B @ B.T).tocsr()
    L.setdiag(cp.zeros(E, dtype=cp.float32))
    L.eliminate_zeros()
    L.data[:] = 1
    return L


def build_dual_graph_gpu(simplices: np.ndarray, num_vertices: int):
    """GPU version of :func:`simplex_helpers.build_dual_graph`."""
    cp, cpsp = _import_cupy()

    T = simplices.shape[0]
    if T == 0:
        return cpsp.csr_matrix((0, 0), dtype=cp.float32)

    rows = cp.repeat(cp.arange(T, dtype=cp.int32), 3)
    cols = cp.asarray(simplices.reshape(-1).astype(np.int32, copy=False))
    data = cp.ones(rows.size, dtype=cp.float32)

    C = cpsp.csr_matrix((data, (rows, cols)), shape=(T, num_vertices), dtype=cp.float32)
    D = (C @ C.T).tocsr()
    D.setdiag(cp.zeros(T, dtype=cp.float32))
    D.eliminate_zeros()
    D.data[:] = 1
    return D


def k_hop_adjacency_matrix_gpu(adj, k: int):
    """GPU version of :func:`simplex_helpers.k_hop_adjacency_matrix`.

    ``adj`` is a ``cupyx.scipy.sparse`` matrix.  Returns a CSR matrix with
    zero diagonal and binarised entries (float32).
    """
    cp, cpsp = _import_cupy()

    if k < 1:
        raise ValueError("k must be >= 1")

    N = adj.shape[0]
    if N == 0:
        return cpsp.csr_matrix((0, 0), dtype=cp.float32)

    I = cpsp.eye(N, dtype=cp.float32, format="csr")
    M = (adj + I).tocsr()
    M.data[:] = 1

    Mk = M
    for _ in range(k - 1):
        Mk = (Mk @ M).tocsr()
        Mk.data[:] = 1

    Mk.setdiag(cp.zeros(N, dtype=cp.float32))
    Mk.eliminate_zeros()
    return Mk
