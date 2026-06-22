"""Tests for the aggregate (``agg``) pipeline.

Covers the pure detection/contraction helpers in
``wsinsight.insightlib.aggregate`` and an end-to-end ``agg_generation`` run over
a tiny synthetic slide.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from wsinsight.insightlib.agg_generation import agg_generation
from wsinsight.insightlib.agg_generation import membership_column
from wsinsight.insightlib.aggregate import aggregate_features
from wsinsight.insightlib.aggregate import contract_to_quotient
from wsinsight.insightlib.aggregate import identify_aggregates
from wsinsight.insightlib.graph_cache import read_aggregate_cache
from wsinsight.insightlib.insight_helpers import compute_cell_center_points
from wsinsight.insightlib.insight_helpers import delaunay_triangulation

# ---------------------------------------------------------------------------
# Synthetic-data builders
# ---------------------------------------------------------------------------


def _grid_block(x0: int, y0: int, side: int, spacing: float, label: str):
    """Return (centers, labels) for a ``side``x``side`` grid block of one type."""
    xs = (np.tile(np.arange(side), side) * spacing + x0).astype(np.float64)
    ys = (np.repeat(np.arange(side), side) * spacing + y0).astype(np.float64)
    centers = np.stack([xs, ys], axis=1)
    labels = [label] * (side * side)
    return centers, labels


def _make_nodes_df(centers: np.ndarray, labels: list[str], classes: list[str]):
    """Build a model-output-shaped DataFrame with one-hot-ish prob_ columns."""
    n = len(centers)
    df = pd.DataFrame(
        {
            "minx": (centers[:, 0] - 2).astype(int),
            "miny": (centers[:, 1] - 2).astype(int),
            "width": np.full(n, 4, dtype=int),
            "height": np.full(n, 4, dtype=int),
            "cx": centers[:, 0],
            "cy": centers[:, 1],
        }
    )
    for c in classes:
        df[f"prob_{c}"] = 0.05
    for i, lab in enumerate(labels):
        df.at[i, f"prob_{lab}"] = 0.9
    return compute_cell_center_points(df)


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_membership_column_form():
    # Must ride the object_<name>_prob_<name> export-discovery path.
    assert membership_column("tls") == "object_tls_prob_tls"


def test_two_separated_blobs_give_two_aggregates():
    # Two dense ingredient blobs far apart, in a sparse sea of tumor cells.
    c1, l1 = _grid_block(0, 0, 6, 5.0, "t_cell")
    c2, l2 = _grid_block(500, 500, 6, 5.0, "b_cell")
    centers = np.concatenate([c1, c2], axis=0)
    labels = l1 + l2
    df = _make_nodes_df(centers, labels, ["t_cell", "b_cell", "tumor"])

    edges_df = delaunay_triangulation(
        df[["center_x", "center_y"]].to_numpy(), max_edge_length=25.0
    )
    agg_id = identify_aggregates(
        df,
        ["t_cell", "b_cell"],
        edges_df,
        k=2,
        N=8,
        R=0.5,
        min_size=10,
    )
    assert agg_id.max() == 1  # ids 0 and 1
    assert set(np.unique(agg_id[agg_id >= 0]).tolist()) == {0, 1}


def test_salt_and_pepper_yields_no_aggregate():
    # Sparse scattered ingredient cells: the density gate must reject them.
    rng = np.random.default_rng(0)
    centers = rng.uniform(0, 1000, size=(120, 2))
    labels = ["t_cell" if i % 2 == 0 else "tumor" for i in range(120)]
    df = _make_nodes_df(centers, labels, ["t_cell", "tumor"])

    edges_df = delaunay_triangulation(
        df[["center_x", "center_y"]].to_numpy(), max_edge_length=25.0
    )
    agg_id = identify_aggregates(
        df,
        ["t_cell"],
        edges_df,
        k=2,
        N=8,
        R=0.5,
        min_size=10,
    )
    assert (agg_id == -1).all()


def test_min_size_filter():
    # A 3x3 ingredient block (9 cells) is dropped when min_size=10.
    c1, l1 = _grid_block(0, 0, 3, 5.0, "t_cell")
    df = _make_nodes_df(c1, l1, ["t_cell", "tumor"])
    edges_df = delaunay_triangulation(
        df[["center_x", "center_y"]].to_numpy(), max_edge_length=25.0
    )
    agg_id = identify_aggregates(
        df,
        ["t_cell"],
        edges_df,
        k=1,
        N=1,
        R=0.1,
        min_size=10,
    )
    assert (agg_id == -1).all()


def test_contract_quotient_boundary_crossings():
    # Hand-built: cells 0,1 = aggregate 0; cells 2,3 = aggregate 1; cell 4 = none.
    # One edge (1-2) crosses the boundary -> exactly one quotient edge (0,1).
    agg_id = np.array([0, 0, 1, 1, -1], dtype=np.int64)
    centers = np.array([[0, 0], [1, 0], [10, 0], [11, 0], [100, 100]], dtype=np.float64)
    edges_df = pd.DataFrame(
        {
            "source": [0, 2, 1, 3],
            "target": [1, 3, 2, 4],  # 1-2 crosses; 3-4 touches a non-member
            "length": [1.0, 1.0, 9.0, 1.0],
        }
    )
    q = contract_to_quotient(agg_id, edges_df, centers)
    assert q["aggregate_sizes"].tolist() == [2, 2]
    src = q["quotient_edges_source"].tolist()
    dst = q["quotient_edges_target"].tolist()
    assert list(zip(src, dst, strict=True)) == [(0, 1)]


def test_aggregate_features_columns():
    c1, l1 = _grid_block(0, 0, 6, 5.0, "t_cell")
    df = _make_nodes_df(c1, l1, ["t_cell", "b_cell"])
    agg_id = np.zeros(len(df), dtype=np.int64)
    feats = aggregate_features(df, agg_id, slide_id="s", mpp=0.5)
    assert feats.loc[0, "n_cells"] == len(df)
    assert feats.loc[0, "slide_id"] == "s"
    assert "composition_t_cell_frac" in feats.columns
    assert feats.loc[0, "composition_t_cell_frac"] == pytest.approx(1.0)
    assert "aggregate_id" not in [] and feats.loc[0, "area_um2"] > 0


# ---------------------------------------------------------------------------
# End-to-end agg_generation
# ---------------------------------------------------------------------------


def _build_results_dir(tmp_path: Path) -> Path:
    # Two ingredient blobs in a sparse tumor background.
    c1, l1 = _grid_block(0, 0, 6, 5.0, "t_cell")
    c2, l2 = _grid_block(500, 500, 6, 5.0, "b_cell")
    rng = np.random.default_rng(1)
    bg = rng.uniform(0, 1000, size=(40, 2))
    centers = np.concatenate([c1, c2, bg], axis=0)
    labels = l1 + l2 + ["tumor"] * len(bg)
    df = _make_nodes_df(centers, labels, ["t_cell", "b_cell", "tumor"])
    # Drop the helper-added center columns; the worker recomputes them.
    df = df.drop(columns=["center_x", "center_y"])

    results_dir = tmp_path / "results"
    (results_dir / "model-outputs-csv").mkdir(parents=True)
    (results_dir / "graphs").mkdir(parents=True)
    df.to_csv(results_dir / "model-outputs-csv" / "synthetic.csv", index=False)
    return results_dir


def test_agg_end_to_end(tmp_path: Path):
    from wsinsight.uri_path import URIPath

    results_dir = _build_results_dir(tmp_path)
    fake_slide = tmp_path / "slides" / "synthetic.svs"
    fake_slide.parent.mkdir(parents=True)
    fake_slide.touch()

    with patch("wsinsight.insightlib.agg_generation.get_avg_mpp", return_value=1.0):
        failed = agg_generation(
            wsi_dir=URIPath(str(fake_slide.parent)),
            slide_paths=[URIPath(str(fake_slide))],
            results_dir=URIPath(str(results_dir)),
            name="tls",
            agg_types=["t_cell", "b_cell"],
            max_neighbor_distance_um=25.0,
            k=2,
            N=8,
            R=0.5,
            min_size=10,
            num_workers=1,
            overwrite=True,
        )

    assert failed == [], f"agg failed for: {failed}"

    # 1. membership column upserted, siblings preserved.
    model_csv = results_dir / "model-outputs-csv" / "synthetic.csv"
    out_df = pd.read_csv(model_csv)
    col = membership_column("tls")
    assert col in out_df.columns
    assert {"prob_t_cell", "prob_b_cell", "prob_tumor"}.issubset(out_df.columns)
    assert out_df[col].isin([0.0, 1.0]).all()
    assert out_df[col].sum() > 0  # some cells are members

    # 2. sidecar exists with one row per aggregate and prob_tls == 1.0.
    sidecar = results_dir / "agg-tls-outputs-csv" / "synthetic.csv"
    assert sidecar.exists()
    sdf = pd.read_csv(sidecar)
    assert len(sdf) == 2
    assert (sdf["prob_tls"] == 1.0).all()
    assert {"center_x", "center_y", "n_cells", "area_um2"}.issubset(sdf.columns)

    # 3. quotient graph subgroup written.
    cache = read_aggregate_cache(results_dir / "graphs" / "synthetic.h5", "tls")
    assert cache["aggregate_sizes"].shape[0] == 2
    assert cache["cell_to_aggregate"].shape[0] == len(out_df)


def test_agg_second_name_preserves_sibling(tmp_path: Path):
    from wsinsight.uri_path import URIPath

    results_dir = _build_results_dir(tmp_path)
    fake_slide = tmp_path / "slides" / "synthetic.svs"
    fake_slide.parent.mkdir(parents=True)
    fake_slide.touch()

    common = dict(
        wsi_dir=URIPath(str(fake_slide.parent)),
        slide_paths=[URIPath(str(fake_slide))],
        results_dir=URIPath(str(results_dir)),
        agg_types=["t_cell", "b_cell"],
        num_workers=1,
        overwrite=True,
    )
    with patch("wsinsight.insightlib.agg_generation.get_avg_mpp", return_value=1.0):
        assert agg_generation(name="tls", **common) == []
        assert agg_generation(name="follicle", **common) == []

    out_df = pd.read_csv(results_dir / "model-outputs-csv" / "synthetic.csv")
    # Both namespaced membership columns must coexist.
    assert membership_column("tls") in out_df.columns
    assert membership_column("follicle") in out_df.columns
