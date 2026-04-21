"""Tests for ecomp_generation: end-to-end over a tiny synthetic slide CSV."""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from wsinsight.insightlib.ecomp_generation import ecomp_generation


def _build_synthetic_cells(tmp_path: Path, n: int = 200, seed: int = 0) -> Path:
    """Write a small model-outputs-csv/synthetic.csv with 3 probability classes."""
    rng = np.random.default_rng(seed)
    # Place cells on a jittered grid so Delaunay edges are short (pixels here ~= µm).
    side = int(np.ceil(np.sqrt(n)))
    xs = np.tile(np.arange(side), side)[:n]
    ys = np.repeat(np.arange(side), side)[:n]
    jitter = rng.normal(0, 0.1, size=(n, 2))
    cx = (xs * 5.0 + jitter[:, 0]).astype(np.float64)
    cy = (ys * 5.0 + jitter[:, 1]).astype(np.float64)

    # Three-class probabilities via a soft label.
    labels = rng.integers(0, 3, size=n)
    probs = np.full((n, 3), 0.05)
    probs[np.arange(n), labels] = 0.9

    df = pd.DataFrame(
        {
            "minx": (cx - 2).astype(int),
            "miny": (cy - 2).astype(int),
            "width": np.full(n, 4, dtype=int),
            "height": np.full(n, 4, dtype=int),
            "cx": cx,
            "cy": cy,
            "prob_alpha": probs[:, 0],
            "prob_beta": probs[:, 1],
            "prob_gamma": probs[:, 2],
        }
    )

    results_dir = tmp_path / "results"
    (results_dir / "model-outputs-csv").mkdir(parents=True)
    (results_dir / "graphs").mkdir(parents=True)
    out = results_dir / "model-outputs-csv" / "synthetic.csv"
    df.to_csv(out, index=False)
    return results_dir


def test_ecomp_end_to_end(tmp_path: Path):
    results_dir = _build_synthetic_cells(tmp_path, n=200)

    # Bypass WSI reading by stubbing get_avg_mpp and providing a fake slide path.
    fake_slide = tmp_path / "slides" / "synthetic.svs"
    fake_slide.parent.mkdir(parents=True)
    fake_slide.touch()

    from wsinsight.uri_path import URIPath

    with patch(
        "wsinsight.insightlib.ecomp_generation.get_avg_mpp", return_value=1.0
    ):
        failed = ecomp_generation(
            wsi_dir=URIPath(str(fake_slide.parent)),
            slide_paths=[URIPath(str(fake_slide))],
            results_dir=URIPath(str(results_dir)),
            max_edge_um=50.0,
            ecomp_k=2,
            num_workers=1,
            overwrite=True,
        )

    assert failed == [], f"ecomp failed for: {failed}"

    out_csv = results_dir / "ecomp-outputs-csv" / "synthetic.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) > 0

    # Schema: C=3 classes → C*(C+1)/2 = 6 edge types.
    types = sorted(["alpha", "beta", "gamma"])
    expected_edge_types = {
        "__".join(p) for p in combinations_with_replacement(types, 2)
    }
    assert set(df["edge_type"].unique()).issubset(expected_edge_types)

    prop_cols = [c for c in df.columns if c.startswith("neighborhood_") and c.endswith("_prop")]
    count_cols = [c for c in df.columns if c.startswith("neighborhood_") and c.endswith("_count")]
    assert len(prop_cols) == 6
    assert len(count_cols) == 6

    # Invariant 1: sum(count columns) == neighborhood_size.
    counts_sum = df[count_cols].sum(axis=1)
    assert (counts_sum == df["neighborhood_size"]).all()

    # Invariant 2: proportions sum to ~1.0 where neighborhood_size > 0.
    mask = df["neighborhood_size"] > 0
    props_sum = df.loc[mask, prop_cols].sum(axis=1)
    np.testing.assert_allclose(props_sum, 1.0, atol=1e-9)

    # Invariant 3: cell_type_1 <= cell_type_2 alphabetically.
    assert (df["cell_type_1"] <= df["cell_type_2"]).all()

    # Invariant 4: edge_type is "__".join(sorted([cell_type_1, cell_type_2])).
    reconstructed = df["cell_type_1"].str.cat(df["cell_type_2"], sep="__")
    assert (reconstructed == df["edge_type"]).all()

    # Invariant 5: edge_length_um <= max_edge_um.
    assert (df["edge_length_um"] <= 50.0).all()
