"""Tests for the AnnData (.h5ad) per-cell exporter."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from wsinsight.write_h5ad import build_anndata_from_df
from wsinsight.write_h5ad import write_h5ads


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "minx": [0, 10, 20],
            "miny": [0, 10, 20],
            "width": [8, 8, 8],
            "height": [8, 8, 8],
            "center_x": [4, 14, 24],
            "center_y": [4, 14, 24],
            "polygon_wkt": ["POLYGON((0 0,8 0,8 8,0 8,0 0))"] * 3,
            "prob_tumor": [0.9, 0.1, 0.4],
            "prob_immune": [0.1, 0.9, 0.6],
            "hplot_layer": [-2, 0, 3],
            "ncomp_frac_tumor": [0.8, 0.2, 0.5],
        }
    )


def test_build_anndata_uses_prob_columns_as_X() -> None:
    adata = build_anndata_from_df(_sample_df(), prefix="prob", slide_id="slideA")
    assert adata.X.shape == (3, 2)
    assert list(adata.var_names) == ["tumor", "immune"]
    assert list(adata.obs["classification"]) == ["tumor", "immune", "immune"]


def test_build_anndata_stores_spatial_and_extra_features() -> None:
    adata = build_anndata_from_df(_sample_df(), slide_id="slideA")
    assert "spatial" in adata.obsm
    assert adata.obsm["spatial"].shape == (3, 2)
    # Non-probability numeric and geometry columns are preserved in obs.
    assert adata.obs["hplot_layer"].tolist() == [-2, 0, 3]
    assert "ncomp_frac_tumor" in adata.obs.columns
    assert adata.uns["wsinsight"]["classes"] == ["tumor", "immune"]


def test_build_anndata_falls_back_to_numeric_matrix() -> None:
    df = pd.DataFrame(
        {
            "center_x": [1, 2],
            "center_y": [3, 4],
            "feat_a": [0.5, 0.6],
            "feat_b": [1.0, 2.0],
        }
    )
    adata = build_anndata_from_df(df, prefix="prob", slide_id="s")
    # No prob_* columns → every numeric non-geometry column becomes X.
    assert list(adata.var_names) == ["feat_a", "feat_b"]
    assert adata.X.shape == (2, 2)
    assert "classification" not in adata.obs.columns


def test_write_h5ads_roundtrip(tmp_path: Path) -> None:
    csv_dir = tmp_path / "export-csv"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "slideA.csv"
    _sample_df().to_csv(csv_path, index=False)

    written = write_h5ads(
        [csv_path], results_dir=tmp_path, output_dir="export-h5ad", show_progress=False
    )
    assert len(written) == 1
    out = Path(written[0])
    assert out.exists() and out.suffix == ".h5ad"

    reloaded = ad.read_h5ad(out)
    assert reloaded.X.shape == (3, 2)
    assert "spatial" in reloaded.obsm
    assert reloaded.obs["hplot_layer"].tolist() == [-2, 0, 3]


def test_niche_id_becomes_categorical_obs_in_h5ad() -> None:
    """niche_id integer column must land as a Categorical in obs."""
    df = _sample_df().copy()
    df["niche_id"] = [2, 0, 3]

    adata = build_anndata_from_df(df, prefix="prob", slide_id="s")

    assert "niche_id" in adata.obs.columns
    assert list(adata.obs["niche_id"]) == [2, 0, 3]
    assert hasattr(adata.obs["niche_id"], "cat"), "niche_id must be Categorical"

    # prob_* columns still form X (unchanged behaviour)
    assert adata.X.shape == (3, 2)
    assert "classification" in adata.obs.columns


def test_write_h5ads_skips_existing_without_overwrite(tmp_path: Path) -> None:
    csv_dir = tmp_path / "export-csv"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "slideA.csv"
    _sample_df().to_csv(csv_path, index=False)

    write_h5ads([csv_path], results_dir=tmp_path, show_progress=False)
    # Second call without overwrite should skip the already-written slide.
    written = write_h5ads([csv_path], results_dir=tmp_path, show_progress=False)
    assert written == []
