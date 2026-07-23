"""Import spatial-transcriptomics (Xenium) gene expression onto WSInsight cells.

``wsinsight import`` maps each transcriptomics cell onto the registered H&E
image using the ST2WSI (SIFT affine + bUnwarpJ B-spline) transform, matches it
to the nearest per-cell model-output detection, and writes one AnnData
``.h5ad`` per slide under ``<results-dir>/xenium-import/``. The existing
``model-outputs-csv/`` is never modified.

Inputs
------
* ``--wsi-dir``   H&E slides (used for exact target dimensions and matched to a
  sample by stem == manifest ``sample_id``).
* ``--sptx-dir``  a ``sptx-list://`` manifest (``path``<TAB>``sample_id``) whose
  first column points at each Xenium sample directory (``cells.parquet`` +
  ``cell_feature_matrix.h5`` + ``registration_params.json`` +
  ``direct_transf.txt``) and whose second column is the stable ``sample_id``
  (must equal the H&E / model-output stem).
* ``--results-dir``  holds ``model-outputs-csv/`` and receives ``xenium-import/``.

Platform is ``xenium`` by default (the only value today); it is exposed as an
option so additional platforms can be added without a breaking change.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import click

from ..bunwarp import map_cells
from ..uri_path import URIPath, URIPathType
from ._meta import write_runtime_metadata
from ._paths import default_storage_kwargs, ensure_input_directory

_STORAGE_KWARGS = default_storage_kwargs()

_WSI_EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".vms", ".vmu",
             ".bif", ".dcm", ".qptiff")

# Optional per-cell add-on sources for ``--include``. Each maps to a sidecar CSV
# that is row-aligned 1:1 with ``model-outputs-csv/<sid>.csv`` (all are derived
# from the same model detections, in the same order), so it is joined onto each
# cell by the SAME matched row index used for the model output — no extra
# spatial join. Value = (obs prefix, relative directory parts under results-dir).
# ``model`` is always imported (mandatory, canonical owner of the geometry and
# ``prob_*`` columns) and is intentionally NOT listed here.
_ADDON_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "cme":   ("cme_",   ("cme-outputs-csv", "cells")),
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
            raise ValueError(f"Unrecognized 10x matrix layout (no 'matrix' group): {h5_path}")
        g = f["matrix"]
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(int(x) for x in g["shape"][:])          # (n_genes, n_cells)
        barcodes = [b.decode() if isinstance(b, (bytes, bytearray)) else str(b)
                    for b in g["barcodes"][:]]
        feats = g["features"]
        names = [b.decode() if isinstance(b, (bytes, bytearray)) else str(b)
                 for b in feats["name"][:]]
        ftype = None
        if "feature_type" in feats:
            ftype = [b.decode() if isinstance(b, (bytes, bytearray)) else str(b)
                     for b in feats["feature_type"][:]]

    M = csc_matrix((data, indices, indptr), shape=shape)      # genes × cells
    X = M.T.tocsr()                                           # cells × genes
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


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def _process_sample(sample_id: str, xdir: Path, wsi_path: Optional[URIPath],
                    model_csv: URIPath, out_path: URIPath, transform: str,
                    want_genes: Optional[set[str]], match_max_dist: float,
                    dry_run: bool = False, *,
                    include: tuple[str, ...] = (),
                    results_dir: Optional[URIPath] = None) -> dict:
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from scipy.spatial import cKDTree

    # ---- Xenium centroids (µm) + cell ids ----
    cells = pd.read_parquet(xdir / "cells.parquet")
    cid = cells["cell_id"].map(
        lambda b: b.decode() if isinstance(b, (bytes, bytearray)) else str(b)
    ).to_numpy()
    xy_um = cells[["x_centroid", "y_centroid"]].to_numpy(float)

    # ---- transform µm -> full-res H&E px ----
    params = xdir / "registration_params.json"
    elastic = xdir / "direct_transf.txt"
    target_wh = None
    if transform == "affine+bspline":
        if wsi_path is None:
            raise click.ClickException(
                f"[{sample_id}] affine+bspline needs the H&E image (target dims); "
                "no matching --wsi-dir slide for this sample_id."
            )
        target_wh = _wsi_dims(wsi_path)
    xy_px = map_cells(
        xy_um, params, elastic if elastic.exists() else None, transform,
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
        return {"sample_id": sample_id, "n_cells": int(len(cid)),
                "n_genes": int(X.shape[1]), "hit_rate_pct": round(hit_rate, 2),
                "median_dist_px": round(float(np.nanmedian(dist)), 2) if len(dist) else None}

    # ---- AnnData: obs=geometry/link, X=sparse expression, var=genes ----
    obs = pd.DataFrame({
        "cell_id": cid.astype(str),
        "x_px": xy_px[:, 0], "y_px": xy_px[:, 1],
        "x_um": xy_um[:, 0], "y_um": xy_um[:, 1],
        "matched_box": matched,
        "match_dist_px": dist,
    })
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
    # optional add-on sources that echo the model geometry (cme/hplot re-list
    # ``minx``, ``prob_*`` …) don't duplicate them; model is the canonical owner.
    claimed: set[str] = set(md.columns)
    # Explicit link id == WSInsight's own export-h5ad obs index (<slide>-<row>);
    # None for cells with no matched detection.
    obs["model_cell_id"] = [f"{sample_id}-{int(b)}" if b >= 0 else None for b in matched]

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
    return {"sample_id": sample_id, "n_cells": int(len(obs)),
            "n_genes": int(adata.n_vars), "hit_rate_pct": round(hit_rate, 2),
            "median_dist_px": round(float(np.nanmedian(dist)), 2) if len(dist) else None}


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
    "-i", "--wsi-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="Directory (or image-list://) of H&E slides. Matched to a sample by "
         "stem == manifest sample_id; provides the exact target dimensions used "
         "by the elastic (B-spline) transform.",
)
@click.option(
    "-s", "--sptx-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="A sptx-list:// manifest (path<TAB>sample_id per line). The path points "
         "at each Xenium sample directory; sample_id is the stable id used to "
         "match the H&E / model-output (Xenium filenames collide across runs).",
)
@click.option(
    "-o", "--results-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="Directory holding WSInsight inference outputs (must contain "
         "model-outputs-csv/). xenium-import/ is written here.",
)
@click.option(
    "--platform",
    type=click.Choice(["xenium"]),
    default="xenium",
    show_default=True,
    help="Spatial-transcriptomics platform to import (xenium by default).",
)
@click.option(
    "--transform",
    type=click.Choice(["affine", "affine+bspline"]),
    default="affine+bspline",
    show_default=True,
    help="ST2WSI coordinate transform: SIFT affine only, or affine + bUnwarpJ "
         "elastic B-spline (default; requires the H&E target dimensions).",
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
         "under its own prefix: cme (cme_), hplot (hplot_), ncomp (ncomp_). "
         "The mandatory 'model' source (model_*) is always imported and need "
         "not be listed. Empty = model only. Example: --include cme,hplot",
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
      • map Xenium centroids (µm) onto the H&E via the ST2WSI transform,
      • match each cell to the nearest model-output detection,
      • write one AnnData whose obs carries EVERY model-output-csv column of the
        matched detection (prefixed ``model_``, NaN when a cell has no match) plus
        ``model_cell_id`` (== WSInsight's export-h5ad obs index), so the h5ad is
        self-contained and needs no join back to the CSV. Optional per-cell
        sources requested via ``--include`` are merged the same way under their
        own prefixes (``cme_`` / ``hplot_`` / ``ncomp_``).

    \b
    Output written to <results-dir>/:
      xenium-import/<sample_id>.h5ad
    """
    import numpy as np  # noqa: F401  (kept for parity / potential summaries)

    ensure_input_directory(wsi_dir, "--wsi-dir")
    ensure_input_directory(results_dir, "--results-dir")

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
        s for s in include_sources
        if s != "model" and not (s in seen or seen.add(s))
    )

    # ---- H&E slides indexed by stem ----
    wsi_dir = wsi_dir.coerce_image_list()
    slide_paths = [p for p in wsi_dir.iterdir()
                   if getattr(wsi_dir, "scheme", "") in ("image-list", "sptx-list") or p.is_file()]
    wsi_by_id = {p.stem: p for p in slide_paths}
    if not wsi_by_id:
        raise click.ClickException(f"No H&E slides found in: {wsi_dir}")

    model_output_dir = results_dir / "model-outputs-csv"
    if not model_output_dir.exists():
        raise click.ClickException(
            "model-outputs-csv/ not found under --results-dir. "
            "Run 'wsinsight infer' or 'wsinsight run' first."
        )

    out_dir = results_dir / "xenium-import"
    out_dir.mkdir(parents=True, exist_ok=True)

    want_genes = None
    if genes and genes.strip().lower() != "all":
        want_genes = {g.strip() for g in genes.replace(",", " ").split() if g.strip()}

    # ---- iterate the sptx-list manifest ----
    samples = list(sptx_dir.iterdir())
    if not samples:
        raise click.ClickException(f"No samples found in --sptx-dir manifest: {sptx_dir}")

    click.secho(f"\nImporting {platform} expression for {len(samples)} sample(s) "
                f"(transform={transform}"
                f"{', +' + ','.join(include_sources) if include_sources else ''}"
                f"{', dry-run' if dry_run else ''}).\n", fg="green")

    done, skipped, failed = [], [], []
    for child in samples:
        sid = child.sample_id
        xdir = Path(os.fspath(child))
        out_path = out_dir / f"{sid}.h5ad"

        if not dry_run and out_path.exists() and not overwrite:
            click.secho(f"  [skip] {sid}: exists (use --overwrite)", fg="yellow")
            skipped.append(sid)
            continue
        if not xdir.is_dir():
            click.secho(f"  [skip] {sid}: Xenium path is not a directory: {xdir}", fg="yellow")
            skipped.append(sid)
            continue
        model_csv = model_output_dir / f"{sid}.csv"
        if not model_csv.exists():
            click.secho(f"  [skip] {sid}: no model-outputs-csv/{sid}.csv", fg="yellow")
            skipped.append(sid)
            continue

        wsi_path = wsi_by_id.get(sid)
        try:
            info = _process_sample(sid, xdir, wsi_path, model_csv, out_path,
                                   transform, want_genes, match_max_dist, dry_run,
                                   include=include_sources, results_dir=results_dir)
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
        {
            "platform": platform, "transform": transform, "genes": genes,
            "include": list(include_sources),
            "match_max_dist": match_max_dist, "dry_run": dry_run,
            "n_ok": len(done), "n_skipped": len(skipped), "n_failed": len(failed),
            "samples": done,
        },
    )
