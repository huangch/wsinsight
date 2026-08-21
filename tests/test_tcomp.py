"""Tests for tcomp_generation: end-to-end over a tiny synthetic slide CSV."""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from wsinsight.insightlib.tcomp_generation import tcomp_generation


def _build_synthetic_cells(tmp_path: Path, n: int = 200, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    xs = np.tile(np.arange(side), side)[:n]
    ys = np.repeat(np.arange(side), side)[:n]
    jitter = rng.normal(0, 0.1, size=(n, 2))
    cx = (xs * 5.0 + jitter[:, 0]).astype(np.float64)
    cy = (ys * 5.0 + jitter[:, 1]).astype(np.float64)

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


def test_tcomp_end_to_end(tmp_path: Path):
    results_dir = _build_synthetic_cells(tmp_path, n=200)

    fake_slide = tmp_path / "slides" / "synthetic.svs"
    fake_slide.parent.mkdir(parents=True)
    fake_slide.touch()

    from wsinsight.uri_path import URIPath

    with patch("wsinsight.insightlib.tcomp_generation.get_avg_mpp", return_value=1.0):
        failed = tcomp_generation(
            wsi_dir=URIPath(str(fake_slide.parent)),
            slide_paths=[URIPath(str(fake_slide))],
            results_dir=URIPath(str(results_dir)),
            max_edge_um=50.0,
            tcomp_k=2,
            num_workers=1,
            overwrite=True,
        )

    assert failed == [], f"tcomp failed for: {failed}"

    out_csv = results_dir / "tcomp-outputs-csv" / "synthetic.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) > 0

    # Schema: C=3 classes → C*(C+1)*(C+2)/6 = 10 triad types.
    types = sorted(["alpha", "beta", "gamma"])
    expected_triad_types = {
        "__".join(tri) for tri in combinations_with_replacement(types, 3)
    }
    assert set(df["triad_type"].unique()).issubset(expected_triad_types)

    prop_cols = [
        c for c in df.columns if c.startswith("neighborhood_") and c.endswith("_prop")
    ]
    count_cols = [
        c for c in df.columns if c.startswith("neighborhood_") and c.endswith("_count")
    ]
    assert len(prop_cols) == 10
    assert len(count_cols) == 10

    # Invariants 1 & 2.
    counts_sum = df[count_cols].sum(axis=1)
    assert (counts_sum == df["neighborhood_size"]).all()
    mask = df["neighborhood_size"] > 0
    props_sum = df.loc[mask, prop_cols].sum(axis=1)
    np.testing.assert_allclose(props_sum, 1.0, atol=1e-9)

    # Geometry invariants.
    assert (df["triad_area_um2"] > 0).all()
    assert (df["triad_max_edge_um"] <= 50.0).all()
    assert ((df["triad_regularity"] >= 0) & (df["triad_regularity"] <= 1)).all()

    # Vertex ids sorted per row.
    assert (df["vertex_1_id"] < df["vertex_2_id"]).all()
    assert (df["vertex_2_id"] < df["vertex_3_id"]).all()

    # cell_type columns alphabetically sorted per row.
    assert (df["cell_type_1"] <= df["cell_type_2"]).all()
    assert (df["cell_type_2"] <= df["cell_type_3"]).all()

    # triad_type reconstruction.
    reconstructed = (
        df["cell_type_1"] + "__" + df["cell_type_2"] + "__" + df["cell_type_3"]
    )
    assert (reconstructed == df["triad_type"]).all()
