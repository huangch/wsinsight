"""Unit tests for `wsinsight import` (cli.sptx_import) per-sample pipeline.

Builds a tiny synthetic Xenium sample (cells.parquet + 10x cell_feature_matrix.h5
+ registration_params.json) and a model-outputs-csv, runs the import, and checks
the resulting AnnData: shape, sparse X, expression aligned by barcode (not row
order), the model-output link columns, and the hit-rate QC.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("anndata")
pytest.importorskip("h5py")

from wsinsight.cli.sptx_import import _process_sample  # noqa: E402
from wsinsight.uri_path import URIPath  # noqa: E402


def _write_10x_h5(path: Path, barcodes, genes, dense_cells_by_genes, feature_types=None):
    import h5py
    from scipy.sparse import csc_matrix

    M = csc_matrix(np.asarray(dense_cells_by_genes).T)  # genes × cells (10x layout)
    if feature_types is None:
        feature_types = ["Gene Expression"] * len(genes)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=M.data.astype("float32"))
        g.create_dataset("indices", data=M.indices.astype("int64"))
        g.create_dataset("indptr", data=M.indptr.astype("int64"))
        g.create_dataset("shape", data=np.array(M.shape, dtype="int64"))  # (genes, cells)
        g.create_dataset("barcodes", data=np.array([b.encode() for b in barcodes]))
        fe = g.create_group("features")
        fe.create_dataset("name", data=np.array([x.encode() for x in genes]))
        fe.create_dataset("id", data=np.array([x.encode() for x in genes]))
        fe.create_dataset("feature_type", data=np.array([t.encode() for t in feature_types]))


def _write_params(path: Path):
    path.write_text(json.dumps({
        "xnumAnnotImgRegParamSiftMatrix": [1, 0, 0, 0, 1, 0],  # identity
        "xnumAnnotImgRegParamDapiImgPxlSize": 1.0,
        "xnumAnnotImgRegParamFlipHori": False,
        "xnumAnnotImgRegParamFlipVert": False,
        "xnumAnnotImgRegParamRotation": "0",
        "xnumAnnotImgRegParamSrcImgWidth": 1000,
        "xnumAnnotImgRegParamSrcImgHeight": 1000,
        "xnumAnnotImgRegParamSourceScale": 1,
        "xnumAnnotImgRegParamTargetScale": 1,
    }))


def _make_sample(tmp_path: Path):
    xdir = tmp_path / "xen"
    xdir.mkdir()
    # 4 cells at known µm positions; identity transform -> H&E px == µm
    ids = ["c0", "c1", "c2", "c3"]
    xy = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    pd.DataFrame({
        "cell_id": [i.encode() for i in ids],       # Xenium stores bytes
        "x_centroid": xy[:, 0], "y_centroid": xy[:, 1],
    }).to_parquet(xdir / "cells.parquet")

    # expression: 3 genes + 1 negative control; barcodes SHUFFLED vs cells.parquet
    genes = ["GENEA", "GENEB", "GENEC", "NegPrb1"]
    ftypes = ["Gene Expression"] * 3 + ["Negative Control Probe"]
    order = ["c2", "c0", "c3", "c1"]                 # deliberately different order
    expr_by_id = {"c0": [1, 0, 0, 9], "c1": [0, 2, 0, 9],
                  "c2": [0, 0, 3, 9], "c3": [4, 5, 6, 9]}
    dense = np.array([expr_by_id[b] for b in order], dtype=float)
    _write_10x_h5(xdir / "cell_feature_matrix.h5", order, genes, dense, ftypes)

    _write_params(xdir / "registration_params.json")

    # model-outputs: one detection box per cell (identity -> px == µm), plus probs
    md = pd.DataFrame({
        "minx": xy[:, 0] - 1, "miny": xy[:, 1] - 1,
        "width": [2.0] * 4, "height": [2.0] * 4,
        "prob_background": 0.0, "prob_neoplastic": [0.9, 0.1, 0.5, 0.2],
        "prob_inflammatory": 0.0, "prob_connective": 0.0,
        "prob_dead": 0.0, "prob_epithelial": 0.0,
    })
    mdir = tmp_path / "results" / "model-outputs-csv"
    mdir.mkdir(parents=True)
    md.to_csv(mdir / "S1.csv", index=False)

    outdir = tmp_path / "results" / "xenium-import"
    outdir.mkdir(parents=True)
    return xdir, URIPath(str(mdir / "S1.csv")), URIPath(str(outdir / "S1.h5ad")), ids


def test_import_affine_alignment_and_link(tmp_path):
    import anndata

    xdir, model_csv, out_path, ids = _make_sample(tmp_path)
    info = _process_sample("S1", xdir, None, model_csv, out_path,
                           transform="affine", want_genes=None, match_max_dist=0.0)

    assert info["n_cells"] == 4
    assert info["n_genes"] == 3           # negative-control probe dropped
    assert info["hit_rate_pct"] == 100.0  # every cell sits on its box

    a = anndata.read_h5ad(str(out_path.materialize()))
    assert a.shape == (4, 3)
    assert hasattr(a.X, "tocsr")          # sparse
    assert list(a.var_names) == ["GENEA", "GENEB", "GENEC"]

    # expression aligned by barcode despite shuffled matrix order
    X = a[a.obs_names.tolist().index("c3")].X
    X = np.asarray(X.todense()).ravel() if hasattr(X, "todense") else np.asarray(X).ravel()
    np.testing.assert_allclose(X, [4, 5, 6])

    # link columns present + EVERY model-output-csv column carried over (model_ prefix)
    for c in ["matched_box", "match_dist_px", "model_prob_neoplastic", "model_minx",
              "model_cell_id"]:
        assert c in a.obs.columns
    assert (a.obs["matched_box"] >= 0).all()
    # self-contained link id == "<sample_id>-<matched_box>"
    assert (a.obs["model_cell_id"] == "S1-" + a.obs["matched_box"].astype(int).astype(str)).all()
    assert a.uns["wsinsight_import"]["transform"] == "affine"
    assert list(a.uns["wsinsight_import"]["sources"]) == ["model"]


def test_import_gene_subset(tmp_path):
    import anndata

    xdir, model_csv, out_path, ids = _make_sample(tmp_path)
    info = _process_sample("S1", xdir, None, model_csv, out_path,
                           transform="affine", want_genes={"GENEB"}, match_max_dist=0.0)
    assert info["n_genes"] == 1
    a = anndata.read_h5ad(str(out_path.materialize()))
    assert list(a.var_names) == ["GENEB"]


def test_import_match_distance_cap(tmp_path):
    import anndata

    xdir, model_csv, out_path, ids = _make_sample(tmp_path)
    # cap of 0.5 px: cells sit exactly on box centres (dist 0), so all still match
    info = _process_sample("S1", xdir, None, model_csv, out_path,
                           transform="affine", want_genes=None, match_max_dist=0.5)
    assert info["hit_rate_pct"] == 100.0
    a = anndata.read_h5ad(str(out_path.materialize()))
    assert np.nanmax(a.obs["match_dist_px"].to_numpy()) <= 0.5


def test_import_unmatched_cell_leaves_wsi_fields_null(tmp_path):
    import anndata

    xdir, model_csv, out_path, ids = _make_sample(tmp_path)
    # Move c3 far from every detection box; with a tight cap it stays unmatched.
    cells = pd.read_parquet(xdir / "cells.parquet")
    cells.loc[cells.index[-1], ["x_centroid", "y_centroid"]] = [10_000.0, 10_000.0]
    cells.to_parquet(xdir / "cells.parquet")

    info = _process_sample("S1", xdir, None, model_csv, out_path,
                           transform="affine", want_genes=None, match_max_dist=5.0)
    assert info["hit_rate_pct"] == 75.0  # 3 of 4 cells matched

    a = anndata.read_h5ad(str(out_path.materialize()))
    row = a.obs.loc["c3"]
    assert int(row["matched_box"]) == -1
    assert np.isnan(row["match_dist_px"])
    # every carried model-output column is null for the unmatched cell
    assert np.isnan(row["model_minx"])
    assert np.isnan(row["model_prob_neoplastic"])
    assert pd.isna(row["model_cell_id"])
    # matched cells still carry a concrete link id
    assert (a.obs.loc[["c0", "c1", "c2"], "matched_box"] >= 0).all()


def test_import_include_niche_prefixes_and_dedup(tmp_path):
    import anndata

    xdir, model_csv, out_path, ids = _make_sample(tmp_path)
    # niche cells sidecar: row-aligned 1:1 with model-outputs-csv/S1.csv (4 rows),
    # echoing the model geometry (minx) plus genuinely new niche_* / feature_* cols.
    results_dir = URIPath(str(tmp_path / "results"))
    niche_cells = tmp_path / "results" / "niche-outputs-csv" / "cells"
    niche_cells.mkdir(parents=True)
    pd.DataFrame({
        "minx": [9.0, 19.0, 29.0, 39.0],          # echoes model geometry -> deduped
        "niche_0": [0.1, 0.2, 0.3, 0.4],
        "niche_1": [0.9, 0.8, 0.7, 0.6],            # already niche_-prefixed -> kept verbatim
        "feature_k0_x": [1.0, 2.0, 3.0, 4.0],  # -> niche_feature_k0_x
    }).to_csv(niche_cells / "S1.csv", index=False)

    _process_sample("S1", xdir, None, model_csv, out_path,
                    transform="affine", want_genes=None, match_max_dist=0.0,
                    include=("niche",), results_dir=results_dir)

    a = anndata.read_h5ad(str(out_path.materialize()))
    assert list(a.uns["wsinsight_import"]["sources"]) == ["model", "niche"]
    # new niche columns present under niche_ prefix (no double-prefix on niche_1)
    for c in ["niche_0", "niche_1", "niche_feature_k0_x"]:
        assert c in a.obs.columns
    # geometry echoed by niche is NOT duplicated: model owns minx, niche_minx absent
    assert "niche_minx" not in a.obs.columns
    assert "model_minx" in a.obs.columns
    # values aligned by matched row index (c0 -> row 0)
    assert float(a.obs.loc["c0", "niche_0"]) == 0.1
    assert float(a.obs.loc["c3", "niche_feature_k0_x"]) == 4.0

