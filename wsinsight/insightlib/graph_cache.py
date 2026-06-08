"""Delaunay graph cache backed by per-slide HDF5 files.

Layout of each ``graphs/{slide_id}.h5``::

    attrs:
        num_cells      int64
        mpp            float64
        centers_hash   bytes   (SHA-256 of cell_centers.tobytes())
    datasets:
        cell_centers   (N, 2)  int32
        simplices      (M, 3)  int32
        edges_source   (E,)    int32
        edges_target   (E,)    int32
        edges_length   (E,)    float64
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

from .insight_helpers import _delaunay_full, prune_edges

if TYPE_CHECKING:
    from ..uri_path import URIPath

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(graph_cache_dir: Path | URIPath, slide_id: str) -> Path:
    return Path(str(graph_cache_dir)) / f"{slide_id}.h5"


def _centers_hash(point2d_ary: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(point2d_ary).tobytes()).digest()


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

_REQUIRED_DATASETS = (
    "cell_centers",
    "simplices",
    "edges_source",
    "edges_target",
    "edges_length",
)


def _is_cache_valid(
    h5path: Path,
    num_cells: int,
    mpp: float,
    centers_hash: bytes,
) -> bool:
    """Return True if the cached graph matches the current slide data."""
    if not h5path.exists():
        return False
    try:
        with h5py.File(h5path, "r") as f:
            if int(f.attrs["num_cells"]) != num_cells:
                return False
            if float(f.attrs["mpp"]) != mpp:
                return False
            stored_hash = bytes(f.attrs["centers_hash"])
            if stored_hash != centers_hash:
                return False
            # A run interrupted mid-write can leave the attrs flushed but one
            # or more datasets missing/truncated; treat that as a cache miss so
            # the graph is rebuilt instead of crashing on read.
            for name in _REQUIRED_DATASETS:
                if name not in f:
                    return False
        return True
    except Exception:
        return False


def write_graph_cache(
    h5path: Path,
    cell_centers: np.ndarray,
    simplices: np.ndarray,
    edges_source: np.ndarray,
    edges_target: np.ndarray,
    edges_length: np.ndarray,
    mpp: float,
) -> None:
    """Write a Delaunay graph cache file."""
    h5path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5path, "w") as f:
        f.attrs["num_cells"] = len(cell_centers)
        f.attrs["mpp"] = mpp
        f.attrs["centers_hash"] = np.void(_centers_hash(cell_centers))
        f.create_dataset("cell_centers", data=cell_centers, dtype=np.int32)
        f.create_dataset("simplices", data=simplices, dtype=np.int32)
        f.create_dataset("edges_source", data=edges_source, dtype=np.int32)
        f.create_dataset("edges_target", data=edges_target, dtype=np.int32)
        f.create_dataset("edges_length", data=edges_length, dtype=np.float64)


def read_graph_cache(h5path: Path) -> dict:
    """Load all datasets and attrs from a graph cache file."""
    with h5py.File(h5path, "r") as f:
        return {
            "num_cells": int(f.attrs["num_cells"]),
            "mpp": float(f.attrs["mpp"]),
            "centers_hash": bytes(f.attrs["centers_hash"]),
            "cell_centers": f["cell_centers"][:],
            "simplices": f["simplices"][:],
            "edges_source": f["edges_source"][:],
            "edges_target": f["edges_target"][:],
            "edges_length": f["edges_length"][:],
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_or_build_delaunay(
    graph_cache_dir: Path | URIPath,
    slide_id: str,
    point2d_ary: np.ndarray,
    mpp: float,
    max_edge_length_px: float,
) -> pd.DataFrame:
    """Return a pruned edges DataFrame, using the graph cache when possible.

    If a valid cache exists, loads unpruned edges and prunes to
    *max_edge_length_px*.  Otherwise runs ``Delaunay``, writes the cache,
    then prunes.

    Parameters
    ----------
    graph_cache_dir:
        Directory for ``{slide_id}.h5`` files (typically ``results_dir / "graphs"``).
    slide_id:
        Stem of the WSI / CSV file.
    point2d_ary:
        (N, 2) int32 array of cell centres (center_x, center_y).
    mpp:
        Microns-per-pixel used to derive *point2d_ary*.
    max_edge_length_px:
        Distance threshold in pixels for edge pruning.

    Returns
    -------
    pd.DataFrame
        Columns ``source``, ``target``, ``length`` — pruned edges.
    """
    h5path = _cache_path(graph_cache_dir, slide_id)
    chash = _centers_hash(point2d_ary)

    if _is_cache_valid(h5path, len(point2d_ary), mpp, chash):
        _logger.debug("Graph cache HIT for %s", slide_id)
        try:
            data = read_graph_cache(h5path)
        except (KeyError, OSError) as exc:
            # Defence in depth: the validity check passed but the file is still
            # unreadable (e.g. truncated/corrupt). Drop it and rebuild rather
            # than letting the exception abort the whole hplot run.
            _logger.warning(
                "Graph cache for %s is unreadable (%s); rebuilding.", slide_id, exc
            )
            try:
                h5path.unlink()
            except OSError:
                pass
        else:
            return prune_edges(
                data["edges_source"], data["edges_target"],
                data["edges_length"], max_edge_length_px,
            )

    _logger.debug("Graph cache MISS for %s — building", slide_id)
    simplices, src, dst, lengths = _delaunay_full(point2d_ary)

    write_graph_cache(
        h5path,
        cell_centers=np.asarray(point2d_ary, dtype=np.int32),
        simplices=simplices,
        edges_source=src,
        edges_target=dst,
        edges_length=lengths,
        mpp=mpp,
    )

    return prune_edges(src, dst, lengths, max_edge_length_px)
