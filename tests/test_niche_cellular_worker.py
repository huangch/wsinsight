"""Tests for the per-cell niche CSV writer (Phase 4).

Covers the column contract that downstream stages depend on:
* ``niche_id`` is a single integer column, not a one-hot block
* cells dropped as isolated are NaN, which the voronoi helper reads as
  "unassigned"
* feature columns land on the right rows
* the frame is built in one concat rather than hundreds of ``.loc`` inserts
  (which pandas flags as "highly fragmented" and which is O(n_cols^2))
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_worker() -> dict:
    """Exec just the Phase-4 worker out of niche_generation.py."""
    src_path = (
        Path(__file__).resolve().parents[1]
        / "wsinsight" / "insightlib" / "niche_generation.py"
    )
    text = src_path.read_text()
    start = text.index("def _niche_cellular_worker(")
    end = text.index("def _niche_annotation_worker(")
    ns: dict = {"pd": pd, "np": np, "Path": Path, "os": os}
    exec(compile(text[start:end], str(src_path), "exec"), ns)
    return ns


WORKER = _load_worker()["_niche_cellular_worker"]

N_ROWS = 6
CLASSES = ["prob_tumor", "prob_immune"]
K_HOPS = 1
KHOP_DIM = (K_HOPS + 1) * len(CLASSES)      # 4


def _write_model_csv(path: Path) -> None:
    pd.DataFrame({
        "minx": np.arange(N_ROWS) * 10,
        "miny": np.arange(N_ROWS) * 10,
        "width": [8] * N_ROWS,
        "height": [8] * N_ROWS,
        "prob_tumor": np.linspace(0.1, 0.9, N_ROWS),
        "prob_immune": np.linspace(0.9, 0.1, N_ROWS),
    }).to_csv(path, index=False)


def _run(tmp_path: Path, *, kept, labels, hoptimus_only=False, n_feat=None):
    """Invoke the worker and return the resulting DataFrame."""
    model_csv = tmp_path / "model.csv"
    cell_csv = tmp_path / "cells.csv"
    _write_model_csv(model_csv)

    kept = np.asarray(kept, dtype=int)
    n_feat = n_feat if n_feat is not None else KHOP_DIM
    # Distinct value per (row, feature) so misplacement is detectable.
    X_raw = (np.arange(len(kept))[:, None] * 100
             + np.arange(n_feat)[None, :]).astype(np.float32)

    WORKER((
        "slide.svs", model_csv, kept, X_raw, CLASSES,
        np.asarray(labels, dtype=int), K_HOPS, 4, cell_csv, True,
        hoptimus_only, KHOP_DIM,
    ))
    return pd.read_csv(cell_csv), X_raw


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------

def test_writes_single_niche_id_column_not_one_hot(tmp_path):
    df, _ = _run(tmp_path, kept=range(N_ROWS), labels=[0, 1, 2, 3, 0, 1])

    assert "niche_id" in df.columns
    onehot = [c for c in df.columns if c.startswith("niche_") and c[6:].isdigit()]
    assert onehot == [], f"one-hot columns leaked back in: {onehot}"


def test_niche_id_values_match_labels(tmp_path):
    labels = [3, 1, 0, 2, 1, 0]
    df, _ = _run(tmp_path, kept=range(N_ROWS), labels=labels)
    assert df["niche_id"].tolist() == labels


def test_dropped_cells_get_nan_niche_id(tmp_path):
    """Isolated cells are excluded from the graph and must stay unassigned."""
    kept = [0, 2, 4]
    df, _ = _run(tmp_path, kept=kept, labels=[1, 2, 3])

    assert df.loc[kept, "niche_id"].tolist() == [1, 2, 3]
    dropped = [i for i in range(N_ROWS) if i not in kept]
    assert df.loc[dropped, "niche_id"].isna().all()


def test_original_columns_are_preserved(tmp_path):
    df, _ = _run(tmp_path, kept=range(N_ROWS), labels=[0] * N_ROWS)
    for col in ("minx", "miny", "width", "height", "prob_tumor", "prob_immune"):
        assert col in df.columns
    assert len(df) == N_ROWS


# ---------------------------------------------------------------------------
# Feature placement
# ---------------------------------------------------------------------------

def test_khop_features_land_on_the_right_rows(tmp_path):
    kept = [1, 3, 5]
    df, X_raw = _run(tmp_path, kept=kept, labels=[0, 1, 2])

    feature_cols = [f"feature_k{k}_{c.replace('prob_', '')}"
                    for k in range(K_HOPS + 1) for c in CLASSES]
    assert feature_cols == [c for c in df.columns if c.startswith("feature_k")]

    np.testing.assert_allclose(df.loc[kept, feature_cols].to_numpy(), X_raw)
    dropped = [i for i in range(N_ROWS) if i not in kept]
    assert df.loc[dropped, feature_cols].isna().all().all()


def test_hoptimus_only_writes_hoptimus_feature_columns(tmp_path):
    n_feat = 8
    df, X_raw = _run(
        tmp_path, kept=range(N_ROWS), labels=[0] * N_ROWS,
        hoptimus_only=True, n_feat=n_feat,
    )

    cols = [f"hoptimus_feature_{j}" for j in range(n_feat)]
    assert all(c in df.columns for c in cols)
    np.testing.assert_allclose(df[cols].to_numpy(), X_raw)
    # k-hop columns must not appear in hoptimus-only mode.
    assert [c for c in df.columns if c.startswith("feature_k")] == []


def test_khop_block_is_truncated_to_khop_dim(tmp_path):
    """With concatenated k-hop + H-Optimus training features, only the k-hop
    block is exported so the CSV contract stays stable."""
    df, X_raw = _run(
        tmp_path, kept=range(N_ROWS), labels=[0] * N_ROWS,
        n_feat=KHOP_DIM + 16,      # extra H-Optimus columns beyond khop_dim
    )
    feature_cols = [c for c in df.columns if c.startswith("feature_k")]
    assert len(feature_cols) == KHOP_DIM
    np.testing.assert_allclose(
        df[feature_cols].to_numpy(), X_raw[:, :KHOP_DIM]
    )


# ---------------------------------------------------------------------------
# The fragmentation regression.
# ---------------------------------------------------------------------------

def test_no_dataframe_fragmentation_warning(tmp_path):
    """Building 1536 columns via repeated .loc inserts triggers pandas'
    PerformanceWarning and is quadratic; the frame must be concatenated once."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        _run(
            tmp_path, kept=range(N_ROWS), labels=[0] * N_ROWS,
            hoptimus_only=True, n_feat=1536,
        )


def test_skips_existing_output_unless_overwrite(tmp_path):
    model_csv = tmp_path / "model.csv"
    cell_csv = tmp_path / "cells.csv"
    _write_model_csv(model_csv)
    cell_csv.write_text("sentinel\n")

    _, status = WORKER((
        "slide.svs", model_csv, np.arange(N_ROWS),
        np.zeros((N_ROWS, KHOP_DIM), dtype=np.float32), CLASSES,
        np.zeros(N_ROWS, dtype=int), K_HOPS, 4, cell_csv, False,
        False, KHOP_DIM,
    ))
    assert status == "skip"
    assert cell_csv.read_text() == "sentinel\n"
