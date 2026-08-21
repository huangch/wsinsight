"""Import spatial-transcriptomics (Xenium) gene expression onto WSInsight cells.

``wsinsight import`` maps each transcriptomics cell onto the registered H&E
image using the ST2WSI (SIFT affine + bUnwarpJ B-spline) transform, matches it
to the nearest per-cell model-output detection, and writes one AnnData
``.h5ad`` per slide under ``<results-dir>/imported-xenium/``. The existing
``model-outputs-csv/`` is never modified.

Inputs
------
* ``--wsi-dir``   H&E slides (used for exact target dimensions and matched to a
  sample by stem == manifest ``sample_id``).
* ``--sptx-dir``  a ``sptx-list://`` manifest
    (``path``<TAB>``sample_id``<TAB>``transform_dir``) whose
  first column points at either:
    - (platform=xenium)     each Xenium sample directory (``cells.parquet`` +
                            ``cell_feature_matrix.h5`` + ``registration_params.json``
                            + ``direct_transf.txt``), or
    - (platform=xenium-h5ad) each pre-annotated ``.h5ad`` produced by sptxinsight
                            (reads ``obsm["spatial"]`` for µm coordinates and ``X``
                            for expression; still uses the Xenium registration
                            transforms located at the 3rd column of the sptx-list manifest).
  The second column is the stable ``sample_id`` (must equal the H&E /
  model-output stem).
* ``--results-dir``  holds ``model-outputs-csv/`` and receives ``imported-xenium/``.
* 3rd column of sptx-list  per-sample directory holding the ST2WSI transform files.
  sub-directory per sample_id containing the ST2WSI registration files
  (``registration_params.json`` + optional ``direct_transf.txt``).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import click

from ..bunwarp import map_cells
from ..uri_path import URIPath
from ..uri_path import URIPathType
from ..wsi import get_avg_mpp
from ._meta import write_runtime_metadata
from ._paths import default_storage_kwargs
from ._paths import ensure_input_directory

_STORAGE_KWARGS = default_storage_kwargs()

_WSI_EXTS = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".scn",
    ".mrxs",
    ".vms",
    ".vmu",
    ".bif",
    ".dcm",
    ".qptiff",
)

# Optional per-cell add-on sources for ``--include``. Each maps to a sidecar CSV
# that is row-aligned 1:1 with ``model-outputs-csv/<sid>.csv`` (all are derived
# from the same model detections, in the same order), so it is joined onto each
# cell by the SAME matched row index used for the model output — no extra
# spatial join. Value = (obs prefix, relative directory parts under results-dir).
# ``model`` is always imported (mandatory, canonical owner of the geometry and
# ``prob_*`` columns) and is intentionally NOT listed here.
_ADDON_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "niche": ("niche_", ("niche-outputs-csv", "cells")),
    "hplot": ("hplot_", ("hplot-outputs-csv", "cells")),
    "ncomp": ("ncomp_", ("ncomp-outputs-csv",)),
}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _wsi_dims(wsi_path: URIPath) -> tuple[int, int]:
    """Full-resolution (width, height) of the H&E target image."""
    from ..wsi import get_wsi_cls

    slide = get_wsi_cls()(os.fspath(wsi_path))
    try:
        w, h = slide.dimensions
    finally:
        try:
            slide.close()
        except Exception:
            pass
    return int(w), int(h)


def _read_expression_h5(h5_path: Path, want_genes: Optional[set[str]]):
    """Read a 10x/Xenium ``cell_feature_matrix.h5`` -> (X csr cells×genes, var, barcodes).

    Keeps ``feature_type == 'Gene Expression'`` (drops negative-control / blank
    codewords) unless an explicit gene subset is requested.
    """
    import h5py
    import numpy as np
    import pandas as pd
    from scipy.sparse import csc_matrix

    with h5py.File(os.fspath(h5_path), "r") as f:
        if "matrix" not in f:
            raise ValueError(
                f"Unrecognized 10x matrix layout (no 'matrix' group): {h5_path}"
            )
        g = f["matrix"]
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(int(x) for x in g["shape"][:])  # (n_genes, n_cells)
        barcodes = [
            b.decode() if isinstance(b, (bytes, bytearray)) else str(b)
            for b in g["barcodes"][:]
        ]
        feats = g["features"]
        names = [
            b.decode() if isinstance(b, (bytes, bytearray)) else str(b)
            for b in feats["name"][:]
        ]
        ftype = None
        if "feature_type" in feats:
            ftype = [
                b.decode() if isinstance(b, (bytes, bytearray)) else str(b)
                for b in feats["feature_type"][:]
            ]

    M = csc_matrix((data, indices, indptr), shape=shape)  # genes × cells
    X = M.T.tocsr()  # cells × genes
    names = np.asarray(names, dtype=object)

    if want_genes is not None:
        keep = np.array([n in want_genes for n in names], dtype=bool)
    elif ftype is not None:
        keep = np.array([t == "Gene Expression" for t in ftype], dtype=bool)
    else:
        keep = np.ones(len(names), dtype=bool)

    X = X[:, keep]
    var = pd.DataFrame(index=pd.Index(names[keep], name="gene"))
    return X, var, barcodes


def _read_h5ad_minimal_compat(h5ad_path: Path, spatial_key: str):
    """Read only the pieces wsinsight import needs from an h5ad file.

    Some files written by newer toolchains contain optional encoded elements
    that older anndata versions cannot deserialize (for example
    IOSpec(encoding_type='null')). This fallback avoids whole-file deserialization
    and reads only X/obs/var/obsm[spatial_key].
    """
    import h5py
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    read_elem = None
    try:
        from anndata.experimental import read_elem as _read_elem

        read_elem = _read_elem
    except Exception:
        try:
            from anndata._io.specs import read_elem as _read_elem

            read_elem = _read_elem
        except Exception:
            read_elem = None

    if read_elem is None:
        raise click.ClickException(
            "Failed to import anndata read helpers for compatibility loading. "
            "Install a newer anndata, or regenerate the source h5ad with a "
            "version compatible with this environment."
        )

    with h5py.File(os.fspath(h5ad_path), "r") as f:
        if "X" not in f:
            raise click.ClickException(f"No 'X' dataset/group found in: {h5ad_path}")

        X = read_elem(f["X"])

        try:
            obs = read_elem(f["obs"])
        except Exception:
            # Minimal fallback: preserve cell ids even if full obs decoding fails.
            idx_key = f["obs"].attrs.get("_index", "_index")
            idx = read_elem(f["obs"][idx_key])
            obs = pd.DataFrame(index=pd.Index(np.asarray(idx, dtype=str)))

        try:
            var = read_elem(f["var"])
        except Exception:
            # Minimal fallback: preserve gene ids even if full var decoding fails.
            idx_key = f["var"].attrs.get("_index", "_index")
            idx = read_elem(f["var"][idx_key])
            var = pd.DataFrame(index=pd.Index(np.asarray(idx, dtype=str), name="gene"))

        if "obsm" not in f or spatial_key not in f["obsm"]:
            raise click.ClickException(
                f"obsm[{spatial_key!r}] not found in {h5ad_path}. "
                "sptxinsight annotated.h5ad must store cell centroids under this key."
            )
        spatial = np.asarray(read_elem(f["obsm"][spatial_key]), dtype=float)

    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm[spatial_key] = spatial
    return adata


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------


def _process_sample(
    sample_id: str,
    xdir: Path,
    wsi_path: Optional[URIPath],
    model_csv: URIPath,
    out_path: URIPath,
    transform: str,
    want_genes: Optional[set[str]],
    match_max_dist: float,
    dry_run: bool = False,
    *,
    include: tuple[str, ...] = (),
    results_dir: Optional[URIPath] = None,
) -> dict:
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from scipy.spatial import cKDTree

    # ---- Xenium centroids (µm) + cell ids ----
    cells = pd.read_parquet(xdir / "cells.parquet")
    cid = (
        cells["cell_id"]
        .map(lambda b: b.decode() if isinstance(b, (bytes, bytearray)) else str(b))
        .to_numpy()
    )
    xy_um = cells[["x_centroid", "y_centroid"]].to_numpy(float)

    # ---- transform µm -> full-res H&E px ----
    params = xdir / "registration_params.json"
    elastic = xdir / "direct_transf.txt"
    target_wh = None
    mpp_he = None
    if wsi_path is not None:
        mpp_he = get_avg_mpp(wsi_path)
    if transform == "affine+bspline":
        if wsi_path is None:
            raise click.ClickException(
                f"[{sample_id}] affine+bspline needs the H&E image (target dims); "
                "no matching --wsi-dir slide for this sample_id."
            )
        target_wh = _wsi_dims(wsi_path)
    xy_px = map_cells(
        xy_um,
        params,
        elastic if elastic.exists() else None,
        transform,
        target_wh=target_wh,
    )

    # ---- expression, aligned to cells.parquet order ----
    X, var, barcodes = _read_expression_h5(xdir / "cell_feature_matrix.h5", want_genes)
    bc_pos = {b: i for i, b in enumerate(barcodes)}
    rows = np.array([bc_pos.get(c, -1) for c in cid])
    keep = rows >= 0
    X = X[rows[keep]]
    xy_px = xy_px[keep]
    xy_um = xy_um[keep]
    cid = cid[keep]

    # ---- spatial join to model-output detections (nearest box centre) ----
    md = pd.read_csv(os.fspath(model_csv))
    cx = md["minx"].to_numpy(float) + md["width"].to_numpy(float) / 2.0
    cy = md["miny"].to_numpy(float) + md["height"].to_numpy(float) / 2.0
    tree = cKDTree(np.c_[cx, cy])
    dist, idx = tree.query(np.c_[xy_px[:, 0], xy_px[:, 1]], k=1)
    matched = idx.astype(int)
    if match_max_dist and match_max_dist > 0:
        over = dist > match_max_dist
        matched[over] = -1
        dist = dist.astype(float)
        dist[over] = np.nan
    hit_rate = float(np.mean(matched >= 0) * 100.0)

    if dry_run:
        # QC-only: report the match quality without building/writing the AnnData.
        return {
            "sample_id": sample_id,
            "n_cells": int(len(cid)),
            "n_genes": int(X.shape[1]),
            "hit_rate_pct": round(hit_rate, 2),
            "median_dist_px": round(float(np.nanmedian(dist)), 2)
            if len(dist)
            else None,
        }

    # ---- AnnData: obs=geometry/link, X=sparse expression, var=genes ----
    # x_um/y_um are H&E microns (x_px * mpp_he), not raw Xenium microns.
    # Original Xenium coordinates are stored as xenium_x_um/xenium_y_um.
    if mpp_he is not None:
        x_um_he = xy_px[:, 0] * mpp_he
        y_um_he = xy_px[:, 1] * mpp_he
    else:
        # Fallback: use raw Xenium microns if H&E MPP unavailable
        x_um_he = xy_um[:, 0]
        y_um_he = xy_um[:, 1]
    obs = pd.DataFrame(
        {
            "cell_id": cid.astype(str),
            "x_px": xy_px[:, 0],
            "y_px": xy_px[:, 1],
            "x_um": x_um_he,
            "y_um": y_um_he,
            "xenium_x_um": xy_um[:, 0],
            "xenium_y_um": xy_um[:, 1],
            "matched_box": matched,
            "match_dist_px": dist,
        }
    )
    # Carry EVERY model-output-csv column onto its matched cell so the h5ad is
    # self-contained (no need to re-open the CSV).  Columns are prefixed
    # ``model_`` — the prefix names the producing subcommand (``model`` is the
    # mandatory, canonical source that owns the geometry and ``prob_*`` columns).
    # ``md`` keeps its default 0..n-1 RangeIndex, so the positional ``matched``
    # index doubles as the row label; unmatched cells (matched_box == -1) look up
    # label -1, which is absent -> an all-NaN row for every model_ field.
    model_rows = md.reindex(matched)
    model_rows.columns = [f"model_{c}" for c in model_rows.columns]
    model_rows.index = obs.index
    obs = pd.concat([obs, model_rows], axis=1)
    # ``claimed`` tracks ORIGINAL (unprefixed) column names already emitted, so
    # optional add-on sources that echo the model geometry (niche/hplot re-list
    # ``minx``, ``prob_*`` …) don't duplicate them; model is the canonical owner.
    claimed: set[str] = set(md.columns)
    # Explicit link id == WSInsight's own export-h5ad obs index (<slide>-<row>);
    # None for cells with no matched detection.
    obs["model_cell_id"] = [
        f"{sample_id}-{int(b)}" if b >= 0 else None for b in matched
    ]

    # ---- optional per-cell add-on sources (--include) ----
    sources = ["model"]
    if include and results_dir is not None:
        for key in include:
            prefix, rel = _ADDON_SOURCES[key]
            src_csv = results_dir
            for part in rel:
                src_csv = src_csv / part
            src_csv = src_csv / f"{sample_id}.csv"
            if not src_csv.exists():
                continue
            src_df = pd.read_csv(os.fspath(src_csv))
            obs = _merge_source(obs, src_df, matched, prefix, claimed)
            sources.append(key)

    obs.index = obs["cell_id"].astype(str)

    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = xy_px
    adata.uns["wsinsight_import"] = {
        "sample_id": sample_id,
        "platform": "xenium",
        "transform": transform,
        "sources": sources,
        "target_wh": list(target_wh) if target_wh else None,
        "n_cells": int(len(obs)),
        "n_genes": int(adata.n_vars),
        "match_hit_rate_pct": round(hit_rate, 2),
        "median_match_dist_px": (float(np.nanmedian(dist)) if len(dist) else None),
    }
    _write_h5ad(adata, out_path)
    return {
        "sample_id": sample_id,
        "n_cells": int(len(obs)),
        "n_genes": int(adata.n_vars),
        "hit_rate_pct": round(hit_rate, 2),
        "median_dist_px": round(float(np.nanmedian(dist)), 2) if len(dist) else None,
    }


# ---------------------------------------------------------------------------
# Per-sample processing — xenium-h5ad platform
# ---------------------------------------------------------------------------


def _process_sample_h5ad(
    sample_id: str,
    h5ad_path: Path,
    reg_dir: Optional[Path],
    wsi_path: Optional[URIPath],
    model_csv: URIPath,
    out_path: URIPath,
    transform: str,
    want_genes: Optional[set[str]],
    match_max_dist: float,
    dry_run: bool = False,
    *,
    spatial_key: str = "spatial",
    include: tuple[str, ...] = (),
    results_dir: Optional[URIPath] = None,
) -> dict:
    """Process one pre-annotated ``.h5ad`` (sptxinsight output) for wsinsight import.

    Reads ``obsm[spatial_key]`` (µm) for cell coordinates and ``X`` for expression.
    The ST2WSI registration transform is sourced from ``reg_dir``
    (``registration_params.json`` + optional ``direct_transf.txt``), where
    ``reg_dir`` is provided per sample from sptx-list column 3.
    All downstream logic (spatial join, AnnData construction) is identical to the
    raw Xenium path.
    """
    import anndata
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from scipy.spatial import cKDTree

    # ---- Read pre-annotated AnnData ----
    try:
        adata_src = anndata.read_h5ad(os.fspath(h5ad_path))
    except Exception as exc:
        # Compatibility path for newer h5ad encodings unsupported by the
        # pinned anndata in this image (e.g., IOSpec encoding_type='null').
        msg = str(exc)
        if "IORegistryError" in type(exc).__name__ or "encoding_type='null'" in msg:
            adata_src = _read_h5ad_minimal_compat(h5ad_path, spatial_key)
        else:
            raise

    if spatial_key not in adata_src.obsm:
        raise click.ClickException(
            f"[{sample_id}] obsm[{spatial_key!r}] not found in {h5ad_path}. "
            "sptxinsight annotated.h5ad must store cell centroids (µm) under this key."
        )

    xy_um = np.asarray(adata_src.obsm[spatial_key], dtype=float)[:, :2]
    cid = np.asarray(adata_src.obs_names, dtype=str)

    # ---- expression ----
    from scipy import sparse

    X_src = adata_src.X
    if want_genes is not None:
        gene_names = list(adata_src.var_names)
        keep = np.array([g in want_genes for g in gene_names], dtype=bool)
        X_src = X_src[:, keep]
        var = pd.DataFrame(
            index=pd.Index(np.array(gene_names, dtype=object)[keep], name="gene")
        )
    else:
        var = pd.DataFrame(
            index=pd.Index(
                np.asarray(list(adata_src.var_names), dtype=object), name="gene"
            )
        )
    if not sparse.issparse(X_src):
        from scipy.sparse import csr_matrix

        X_src = csr_matrix(X_src)
    X = X_src.tocsr()

    # ---- transform µm -> full-res H&E px ----
    if reg_dir is None:
        raise click.ClickException(
            f"[{sample_id}] sptx-list column 3 (transform_dir) is required "
            "for platform=xenium-h5ad."
        )
    sample_reg_dir = reg_dir
    if not sample_reg_dir.is_dir():
        raise click.ClickException(
            f"[{sample_id}] registration directory not found: {sample_reg_dir}. "
            "Expected sptx-list column 3 to point at the sample transform folder "
            "containing registration_params.json."
        )
    params_file = sample_reg_dir / "registration_params.json"
    elastic_file = sample_reg_dir / "direct_transf.txt"

    target_wh = None
    mpp_he = None
    if wsi_path is not None:
        mpp_he = get_avg_mpp(wsi_path)
    if transform == "affine+bspline":
        if wsi_path is None:
            raise click.ClickException(
                f"[{sample_id}] affine+bspline needs the H&E image (target dims); "
                "no matching --wsi-dir slide for this sample_id."
            )
        target_wh = _wsi_dims(wsi_path)

    xy_px = map_cells(
        xy_um,
        params_file,
        elastic_file if elastic_file.exists() else None,
        transform,
        target_wh=target_wh,
    )

    # ---- spatial join to model-output detections (nearest box centre) ----
    md = pd.read_csv(os.fspath(model_csv))
    cx = md["minx"].to_numpy(float) + md["width"].to_numpy(float) / 2.0
    cy = md["miny"].to_numpy(float) + md["height"].to_numpy(float) / 2.0
    tree = cKDTree(np.c_[cx, cy])
    dist, idx = tree.query(np.c_[xy_px[:, 0], xy_px[:, 1]], k=1)
    matched = idx.astype(int)
    if match_max_dist and match_max_dist > 0:
        over = dist > match_max_dist
        matched[over] = -1
        dist = dist.astype(float)
        dist[over] = np.nan
    hit_rate = float(np.mean(matched >= 0) * 100.0)

    if dry_run:
        return {
            "sample_id": sample_id,
            "n_cells": int(len(cid)),
            "n_genes": int(X.shape[1]),
            "hit_rate_pct": round(hit_rate, 2),
            "median_dist_px": round(float(np.nanmedian(dist)), 2)
            if len(dist)
            else None,
        }

    # ---- AnnData: obs=geometry/link, X=sparse expression, var=genes ----
    if mpp_he is not None:
        x_um_he = xy_px[:, 0] * mpp_he
        y_um_he = xy_px[:, 1] * mpp_he
    else:
        x_um_he = xy_um[:, 0]
        y_um_he = xy_um[:, 1]
    obs = pd.DataFrame(
        {
            "cell_id": cid.astype(str),
            "x_px": xy_px[:, 0],
            "y_px": xy_px[:, 1],
            "x_um": x_um_he,
            "y_um": y_um_he,
            "xenium_x_um": xy_um[:, 0],
            "xenium_y_um": xy_um[:, 1],
            "matched_box": matched,
            "match_dist_px": dist,
        }
    )
    model_rows = md.reindex(matched)
    model_rows.columns = [f"model_{c}" for c in model_rows.columns]
    model_rows.index = obs.index
    obs = pd.concat([obs, model_rows], axis=1)
    claimed: set[str] = set(md.columns)
    obs["model_cell_id"] = [
        f"{sample_id}-{int(b)}" if b >= 0 else None for b in matched
    ]

    # Carry over sptxinsight obs columns (e.g. cell_type, leiden, etc.) under
    # the ``sptx_`` prefix so they are available in the resulting h5ad without
    # colliding with model_ / niche_ / hplot_ namespaces.
    sptx_obs_cols = [c for c in adata_src.obs.columns if c not in obs.columns]
    if sptx_obs_cols:
        sptx_extra = adata_src.obs[sptx_obs_cols].copy()
        sptx_extra.columns = [f"sptx_{c}" for c in sptx_obs_cols]
        sptx_extra.index = obs.index
        obs = pd.concat([obs, sptx_extra], axis=1)

    # ---- optional per-cell add-on sources (--include) ----
    sources = ["model"]
    if include and results_dir is not None:
        for key in include:
            prefix, rel = _ADDON_SOURCES[key]
            src_csv = results_dir
            for part in rel:
                src_csv = src_csv / part
            src_csv = src_csv / f"{sample_id}.csv"
            if not src_csv.exists():
                continue
            src_df = pd.read_csv(os.fspath(src_csv))
            obs = _merge_source(obs, src_df, matched, prefix, claimed)
            sources.append(key)

    obs.index = obs["cell_id"].astype(str)

    adata_out = AnnData(X=X, obs=obs, var=var)
    adata_out.obsm["spatial"] = xy_px
    adata_out.uns["wsinsight_import"] = {
        "sample_id": sample_id,
        "platform": "xenium-h5ad",
        "transform": transform,
        "sources": sources,
        "source_h5ad": os.fspath(h5ad_path),
        "target_wh": list(target_wh) if target_wh else None,
        "n_cells": int(len(obs)),
        "n_genes": int(adata_out.n_vars),
        "match_hit_rate_pct": round(hit_rate, 2),
        "median_match_dist_px": (float(np.nanmedian(dist)) if len(dist) else None),
    }
    _write_h5ad(adata_out, out_path)
    return {
        "sample_id": sample_id,
        "n_cells": int(len(obs)),
        "n_genes": int(adata_out.n_vars),
        "hit_rate_pct": round(hit_rate, 2),
        "median_dist_px": round(float(np.nanmedian(dist)), 2) if len(dist) else None,
    }


def _merge_source(obs, src_df, matched, prefix: str, claimed: set[str]):
    """Merge one row-aligned per-cell sidecar onto ``obs`` under ``prefix``.

    ``src_df`` is 1:1 with ``model-outputs-csv`` (default RangeIndex), so it is
    reindexed by the same positional ``matched`` array (unmatched -> NaN row).
    Only columns not already claimed by an earlier source are added; a column
    that already starts with ``prefix`` is kept verbatim (no double-prefix).
    """
    import pandas as pd

    src_rows = src_df.reindex(matched)
    src_rows.index = obs.index
    add: dict[str, object] = {}
    for c in src_df.columns:
        if c in claimed:
            continue
        col = c if c.startswith(prefix) else f"{prefix}{c}"
        add[col] = src_rows[c].to_numpy()
        claimed.add(c)
    if add:
        obs = pd.concat([obs, pd.DataFrame(add, index=obs.index)], axis=1)
    return obs


def _write_h5ad(adata, out_path: URIPath) -> None:
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tf:
        tmp = tf.name
    try:
        adata.write_h5ad(tmp)
        with open(tmp, "rb") as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command(name="import")
@click.option(
    "-i",
    "--wsi-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="Directory (or image-list://) of H&E slides. Matched to a sample by "
    "stem == manifest sample_id; provides the exact target dimensions used "
    "by the elastic (B-spline) transform.",
)
@click.option(
    "-s",
    "--sptx-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="A sptx-list:// manifest (path<TAB>sample_id<TAB>transform_dir per line). "
    "Column 2 (sample_id) and column 3 (transform_dir) are optional. "
    "For platform=xenium: path points at each raw Xenium sample directory. "
    "For platform=xenium-h5ad: path points at each sptxinsight annotated.h5ad file.",
)
@click.option(
    "-o",
    "--results-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="Directory holding WSInsight inference outputs (must contain "
    "model-outputs-csv/). imported-xenium/ is written here.",
)
@click.option(
    "--platform",
    type=click.Choice(["xenium", "xenium-h5ad"]),
    default="xenium",
    show_default=True,
    help="Spatial-transcriptomics platform to import. "
    "'xenium': read from raw Xenium output directories (cells.parquet + cell_feature_matrix.h5). "
    "'xenium-h5ad': read from sptxinsight annotated.h5ad files (obsm['spatial'] + X); "
    "the sptx-list 3rd column supplies the per-sample ST2WSI transform directory.",
)
@click.option(
    "--spatial-key",
    default="spatial",
    show_default=True,
    help="(platform=xenium-h5ad only) Key in adata.obsm holding cell centroids in microns.",
)
@click.option(
    "--transform",
    type=click.Choice(["affine", "affine+bspline", "none"]),
    default="affine+bspline",
    show_default=True,
    help="ST2WSI coordinate transform: SIFT affine only, or affine + bUnwarpJ "
    "elastic B-spline (default; requires the H&E target dimensions), or "
    "'none' to pass µm coordinates through unchanged (useful when the "
    "h5ad already contains pixel coordinates).",
)
@click.option(
    "--genes",
    default="all",
    show_default=True,
    help="'all' (full panel, Gene Expression features) or a comma-separated "
    "list of gene names to import.",
)
@click.option(
    "--include",
    default="",
    show_default=True,
    help="Comma-separated optional per-cell sources to merge into obs, each "
    "under its own prefix: niche (niche_), hplot (hplot_), ncomp (ncomp_). "
    "The mandatory 'model' source (model_*) is always imported and need "
    "not be listed. Empty = model only. Example: --include niche,hplot",
)
@click.option(
    "--match-max-dist",
    type=click.FloatRange(min=0),
    default=0.0,
    show_default=True,
    help="Maximum H&E-pixel distance for a cell↔detection match (0 = no cap; "
    "match_dist_px is always recorded so matches can be filtered later).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Recompute and overwrite existing per-slide imports.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    show_default=True,
    help="QC only: compute and report the cell↔detection match hit-rate and median "
    "match distance per sample, without reading expression or writing any h5ad. "
    "Use it to check a manifest pairing / registration before a full import.",
)
def sptx_import(
    *,
    wsi_dir: URIPath,
    sptx_dir: URIPath,
    results_dir: URIPath,
    platform: str = "xenium",
    spatial_key: str = "spatial",
    transform: str = "affine+bspline",
    genes: str = "all",
    include: str = "",
    match_max_dist: float = 0.0,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Import spatial-transcriptomics gene expression onto WSInsight cells.

    \b
    For each sample in the sptx-list manifest:
      • map cell centroids (µm) onto the H&E via the ST2WSI transform,
      • match each cell to the nearest model-output detection,
      • write one AnnData whose obs carries EVERY model-output-csv column of the
        matched detection (prefixed ``model_``, NaN when a cell has no match) plus
        ``model_cell_id`` (== WSInsight's export-h5ad obs index), so the h5ad is
        self-contained and needs no join back to the CSV. Optional per-cell
        sources requested via ``--include`` are merged the same way under their
        own prefixes (``niche_`` / ``hplot_`` / ``ncomp_``).

    \b
    platform=xenium (default):
      sptx-list paths point to raw Xenium output directories.

    \b
    platform=xenium-h5ad:
      sptx-list paths point to sptxinsight annotated.h5ad files.
      The 3rd column of the sptx-list manifest supplies the per-sample
      ST2WSI transform directory (registration_params.json etc.).
      sptxinsight obs columns are carried over under the ``sptx_`` prefix.

    \b
    Output written to <results-dir>/:
      imported-xenium/<sample_id>.h5ad
    """
    import numpy as np  # noqa: F401  (kept for parity / potential summaries)

    ensure_input_directory(wsi_dir, "--wsi-dir")
    ensure_input_directory(results_dir, "--results-dir")

    if platform == "xenium-h5ad" and transform != "none":
        pass  # per-sample transform_dir validated during iteration

    include_sources = tuple(
        s for s in (t.strip() for t in include.replace(",", " ").split()) if s
    )
    bad = [s for s in include_sources if s not in _ADDON_SOURCES]
    if bad:
        raise click.ClickException(
            f"--include: unknown source(s) {', '.join(bad)}. "
            f"Valid optional sources: {', '.join(_ADDON_SOURCES)} "
            "('model' is always included)."
        )
    # De-duplicate while preserving order; drop a stray 'model' (always on).
    seen: set[str] = set()
    include_sources = tuple(
        s for s in include_sources if s != "model" and not (s in seen or seen.add(s))
    )

    # ---- H&E slides indexed by stem ----
    wsi_dir = wsi_dir.coerce_image_list()
    slide_paths = [
        p
        for p in wsi_dir.iterdir()
        if getattr(wsi_dir, "scheme", "") in ("image-list", "sptx-list") or p.is_file()
    ]
    wsi_by_id = {p.stem: p for p in slide_paths}
    if not wsi_by_id:
        raise click.ClickException(f"No H&E slides found in: {wsi_dir}")

    model_output_dir = results_dir / "model-outputs-csv"
    if not model_output_dir.exists():
        raise click.ClickException(
            "model-outputs-csv/ not found under --results-dir. "
            "Run 'wsinsight infer' or 'wsinsight run' first."
        )

    out_dir = results_dir / "imported-xenium"
    out_dir.mkdir(parents=True, exist_ok=True)

    want_genes = None
    if genes and genes.strip().lower() != "all":
        want_genes = {g.strip() for g in genes.replace(",", " ").split() if g.strip()}

    # ---- iterate the sptx-list manifest ----
    samples = list(sptx_dir.iterdir())
    if not samples:
        raise click.ClickException(
            f"No samples found in --sptx-dir manifest: {sptx_dir}"
        )

    click.secho(
        f"\nImporting {platform} expression for {len(samples)} sample(s) "
        f"(transform={transform}"
        f"{', +' + ','.join(include_sources) if include_sources else ''}"
        f"{', dry-run' if dry_run else ''}).\n",
        fg="green",
    )

    done, skipped, failed = [], [], []
    for child in samples:
        sid = child.sample_id
        out_path = out_dir / f"{sid}.h5ad"

        if not dry_run and out_path.exists() and not overwrite:
            click.secho(f"  [skip] {sid}: exists (use --overwrite)", fg="yellow")
            skipped.append(sid)
            continue

        model_csv = model_output_dir / f"{sid}.csv"
        if not model_csv.exists():
            click.secho(f"  [skip] {sid}: no model-outputs-csv/{sid}.csv", fg="yellow")
            skipped.append(sid)
            continue

        wsi_path = wsi_by_id.get(sid)

        try:
            if platform == "xenium-h5ad":
                h5ad_path = Path(os.fspath(child))
                if not h5ad_path.is_file():
                    click.secho(
                        f"  [skip] {sid}: h5ad path is not a file: {h5ad_path}",
                        fg="yellow",
                    )
                    skipped.append(sid)
                    continue
                # Per-sample transform dir comes from the sptx-list 3rd column.
                sample_tdir = getattr(child, "transform_dir", None)
                reg_dir = Path(sample_tdir) if sample_tdir else None
                info = _process_sample_h5ad(
                    sid,
                    h5ad_path,
                    reg_dir,
                    wsi_path,
                    model_csv,
                    out_path,
                    transform,
                    want_genes,
                    match_max_dist,
                    dry_run,
                    spatial_key=spatial_key,
                    include=include_sources,
                    results_dir=results_dir,
                )
            else:
                xdir = Path(os.fspath(child))
                if not xdir.is_dir():
                    click.secho(
                        f"  [skip] {sid}: Xenium path is not a directory: {xdir}",
                        fg="yellow",
                    )
                    skipped.append(sid)
                    continue
                info = _process_sample(
                    sid,
                    xdir,
                    wsi_path,
                    model_csv,
                    out_path,
                    transform,
                    want_genes,
                    match_max_dist,
                    dry_run,
                    include=include_sources,
                    results_dir=results_dir,
                )
            _tag = "dry" if dry_run else "ok"
            _genes = "" if info["n_genes"] is None else f"× {info['n_genes']} genes "
            click.secho(
                f"  [{_tag}]   {sid}: {info['n_cells']:,} cells {_genes}| "
                f"hit-rate {info['hit_rate_pct']:.1f}% | median match {info['median_dist_px']} px",
                fg="green",
            )
            done.append(info)
        except Exception as exc:  # noqa: BLE001
            click.secho(f"  [FAIL] {sid}: {type(exc).__name__}: {exc}", fg="red")
            failed.append(sid)

    click.secho(
        f"\nimport done: {len(done)} ok, {len(skipped)} skipped, {len(failed)} failed.\n",
        fg="green" if not failed else "yellow",
    )
    if done:
        lo = min(i["hit_rate_pct"] for i in done)
        if lo < 50:
            click.secho(
                f"  ⚠ lowest hit-rate {lo:.1f}% — a low hit-rate usually means a "
                "mis-paired sample_id or a scale/registration mismatch.",
                fg="yellow",
            )

    write_runtime_metadata(
        results_dir,
        "import",
        params=click.get_current_context().params,
        extra={
            "results": {
                "n_ok": len(done),
                "n_skipped": len(skipped),
                "n_failed": len(failed),
                "samples": done,
            }
        },
    )
