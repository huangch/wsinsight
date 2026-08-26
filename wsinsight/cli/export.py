"""Standalone export command: merge all per-cell analytics and write GeoJSON / OME-CSV."""

from __future__ import annotations

from pathlib import Path

import click

from ..export_helpers import build_export_csvs
from ..export_helpers import parse_include_sources
from ..uri_path import URIPath
from ..uri_path import URIPathType
from ._meta import write_runtime_metadata
from ._paths import default_storage_kwargs

_STORAGE_KWARGS = default_storage_kwargs()


def _to_local_path(p: URIPath | Path) -> Path:
    """Return a local filesystem Path, materialising a remote URIPath if needed."""
    if isinstance(p, URIPath):
        return Path(p.materialize())
    return Path(p)


@click.command()
@click.option(
    "-o",
    "--results-dir",
    required=True,
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    help=(
        "Results directory produced by a prior wsinsight run / infer / hplot / ncomp "
        "invocation.  Must contain a model-outputs-csv/ subfolder."
    ),
)
@click.option(
    "--geojson",
    is_flag=True,
    default=False,
    show_default=True,
    help="Export per-cell data to GeoJSON files (export-geojson/).",
)
@click.option(
    "--omecsv",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Export per-cell data to compressed OME-CSV files "
        "(export-omecsv/).  Compatible with QuPath and OMERO+."
    ),
)
@click.option(
    "--h5ad",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Export per-cell data to AnnData .h5ad files (export-h5ad/).  "
        "Compatible with scanpy / squidpy; cell centroids are stored in "
        "obsm['spatial']."
    ),
)
@click.option(
    "--patch-overlap-ratio",
    default=0.0,
    show_default=True,
    type=click.FloatRange(min=None, max=1, max_open=True),
    help=(
        "Overlap ratio used during inference (must match the original run). "
        "Controls the tile-box shrinkage written into each exported feature."
    ),
)
@click.option(
    "--object-type",
    default="detection",
    show_default=True,
    type=click.Choice(["tile", "detection", "annotation"]),
    help="QuPath object-type label embedded in each exported GeoJSON / OME-CSV feature.",
)
@click.option(
    "--workers",
    "export_workers",
    default=4,
    show_default=True,
    type=click.IntRange(min=1),
    help="Worker processes used for parallel GeoJSON / OME-CSV serialisation.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Re-build export CSVs even when export-csv/ already contains "
        "up-to-date files.  Useful after re-running hplot or ncomp."
    ),
)
@click.option(
    "--shape",
    type=click.Choice(["bbox", "polygon"]),
    default="bbox",
    show_default=True,
    help=(
        "Geometry written for export-geojson / export-omecsv / export-h5ad. "
        "'bbox' uses each cell's minx/miny/width/height rectangle (default). "
        "'polygon' emits the real segmentation contour read from patches.h5 "
        "/polygons/coords+offsets; OME-CSV / h5ad silently fall back to bbox "
        "in this commit (TODO: implement polygon emission in those writers)."
    ),
)
@click.option(
    "--include",
    "include_sources",
    multiple=True,
    help=(
        "Analysis sources to include in the export (repeatable, or comma-separated). "
        "Per-cell sources (merged into export-csv/): hplot, ncomp, niche, xenium. "
        "Simplex sources (separate directories): ecomp, tcomp, agg:<name>. "
        "Special values: 'all' (everything), 'all-cells' (per-cell only). "
        "If omitted, only per-cell sources are included (backward compatible). "
        "Example: --include hplot,ncomp,ecomp --include agg:tls"
    ),
)
def export(
    results_dir: URIPath,
    geojson: bool,
    omecsv: bool,
    h5ad: bool,
    patch_overlap_ratio: float,
    object_type: str,
    export_workers: int,
    overwrite: bool,
    include_sources: tuple[str, ...],
    shape: str,
) -> None:
    """Merge per-cell analytics and export to GeoJSON / OME-CSV / H5AD.

    Reads whatever analysis outputs are present under RESULTS_DIR:

    \b
    Per-cell sources (merged into one CSV per slide):
      model-outputs-csv/       — base inference probabilities (always included)
      hplot-outputs-csv/cells/ — H-Plot per-cell layer features
      ncomp-outputs-csv/       — node-level (cell) composition features
      niche-outputs-csv/cells/   — cell morphology embeddings
      imported-xenium/         — Xenium per-cell summaries

    \b
    Simplex sources (exported to separate directories):
      ecomp-outputs-csv/       — edge-level composition (per Delaunay edge)
      tcomp-outputs-csv/       — triad-level composition (per Delaunay triangle)
      agg-<name>-outputs-csv/  — aggregate-level features (e.g., TLS)

    \b
    When 'niche' is included and --geojson is set, the merged per-niche contour
    polygons in niche-outputs-csv/niches/ are also written to
    export-niche-regions-geojson/ (annotation polygons), mirroring
    'wsinsight niche --export-geojson'.

    At least one of --geojson, --omecsv, or --h5ad must be supplied.

    This command can be run at any time after inference — and optionally after
    hplot / ncomp / ecomp / tcomp / agg — without re-running the full pipeline.
    """
    if not geojson and not omecsv and not h5ad:
        raise click.UsageError(
            "At least one of --geojson, --omecsv, or --h5ad must be specified."
        )

    if not (results_dir / "model-outputs-csv").exists():
        raise click.ClickException(
            f"No model-outputs-csv/ found under {results_dir}. "
            "Run 'wsinsight run' or 'wsinsight infer' first."
        )

    # --- Parse and validate include sources ----------------------------------
    try:
        cell_sources, simplex_sources = parse_include_sources(
            include_sources, results_dir
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    if cell_sources or simplex_sources:
        all_sources = sorted(cell_sources) + sorted(simplex_sources)
        click.echo(f"\nIncluding sources: {', '.join(all_sources)}")
    else:
        click.echo("\nIncluding all per-cell sources (default)")

    # --- Merge all per-cell sources into export-csv/ -------------------------
    click.echo("\nMerging per-cell analytics into export CSVs...\n")
    include_set = frozenset(cell_sources) if cell_sources else None
    build_export_csvs(results_dir, overwrite=overwrite, include=include_set)

    export_dir = results_dir / "export-csv"
    export_candidates = list(export_dir.iterdir(files_only=True))
    export_csvs = [_to_local_path(p) for p in export_candidates if p.suffix == ".csv"]

    if not export_csvs:
        raise click.ClickException("No export CSVs were produced — nothing to export.")

    click.echo(f"  {len(export_csvs)} slide(s) ready for export.")

    # --- GeoJSON export (per-cell) --------------------------------------------
    if geojson:
        from ..write_geojson import write_geojsons

        click.echo("\nWriting per-cell results to GeoJSON files...\n")
        write_geojsons(
            csvs=export_csvs,
            overlap=patch_overlap_ratio,
            results_dir=results_dir,
            output_dir="export-geojson",
            prefix="prob",
            num_workers=export_workers,
            object_type=object_type,
            set_classification=True,
            overwrite=overwrite,
            annotation_shape=shape,
        )

    # --- OME-CSV export (per-cell) --------------------------------------------
    if omecsv:
        from ..write_omecsv import write_omecsvs

        click.echo("\nWriting per-cell results to OME-CSV files...\n")
        h5s: list[Path] = []
        patches_dir = results_dir / "patches"
        if patches_dir.exists():
            h5s = [
                _to_local_path(p)
                for p in patches_dir.iterdir(files_only=True)
                if p.suffix == ".h5"
            ]
        write_omecsvs(
            csvs=export_csvs,
            h5s=h5s,
            overlap=patch_overlap_ratio,
            results_dir=results_dir,
            output_dir=URIPath("export-omecsv"),
            prefix="prob",
            num_workers=export_workers,
            overwrite=overwrite,
            shape=shape,
        )

    # --- AnnData (.h5ad) export (per-cell) ------------------------------------
    if h5ad:
        from ..write_h5ad import write_h5ads

        click.echo("\nWriting per-cell results to AnnData .h5ad files...\n")
        write_h5ads(
            csvs=export_csvs,
            results_dir=results_dir,
            output_dir="export-h5ad",
            prefix="prob",
            object_type=object_type,
            overwrite=overwrite,
            shape=shape,
        )

    # --- Niche region contours (annotation-level polygons) --------------------
    # The per-cell merge above carries the niche_* one-hot columns on each cell,
    # but the niche command also writes merged per-niche contour polygons under
    # niche-outputs-csv/niches/.  Mirror `wsinsight niche --export-geojson` so
    # `export --include niche --geojson` emits those region contours too.
    niche_included = (not cell_sources) or ("niche" in cell_sources)
    if geojson and niche_included:
        niche_regions_dir = results_dir / "niche-outputs-csv" / "niches"
        if niche_regions_dir.exists():
            region_csvs = [
                _to_local_path(p)
                for p in niche_regions_dir.iterdir(files_only=True)
                if p.suffix == ".csv"
            ]
            if region_csvs:
                click.echo("\nWriting niche region contours to GeoJSON files...\n")
                write_geojsons(
                    csvs=region_csvs,
                    overlap=0,
                    results_dir=results_dir,
                    output_dir="export-niche-regions-geojson",
                    prefix="niche",
                    label_col="niche_id",
                    object_type="annotation",
                    set_classification=True,
                    annotation_shape="polygon",
                    overwrite=overwrite,
                )

    # --- Simplex exports (ecomp, tcomp, agg) ----------------------------------
    for source in simplex_sources:
        if source == "ecomp":
            _export_simplex(
                results_dir=results_dir,
                source_dir="ecomp-outputs-csv",
                output_prefix="export-ecomp",
                label="edge",
                geojson=geojson,
                omecsv=omecsv,
                export_workers=export_workers,
                overwrite=overwrite,
            )
        elif source == "tcomp":
            _export_simplex(
                results_dir=results_dir,
                source_dir="tcomp-outputs-csv",
                output_prefix="export-tcomp",
                label="triad",
                geojson=geojson,
                omecsv=omecsv,
                export_workers=export_workers,
                overwrite=overwrite,
            )
        elif source.startswith("agg:"):
            agg_name = source.split(":", 1)[1]
            _export_simplex(
                results_dir=results_dir,
                source_dir=f"agg-{agg_name}-outputs-csv",
                output_prefix=f"export-agg-{agg_name}",
                label=f"aggregate ({agg_name})",
                geojson=geojson,
                omecsv=omecsv,
                export_workers=export_workers,
                overwrite=overwrite,
            )

    write_runtime_metadata(
        results_dir,
        "export",
        params=click.get_current_context().params,
    )

    click.secho("\nExport complete.\n", fg="green")


def _export_simplex(
    results_dir: URIPath,
    source_dir: str,
    output_prefix: str,
    label: str,
    geojson: bool,
    omecsv: bool,
    export_workers: int,
    overwrite: bool,
) -> None:
    """Export a simplex-level (edge/triad/agg) source to GeoJSON / OME-CSV.

    Note: GeoJSON/OME-CSV export for edges and triads is not yet implemented
    because they have different geometry (lines/triangles) than cells (boxes).
    For now, the raw CSVs in the source directory can be consumed directly.
    """
    source_path = results_dir / source_dir
    if not source_path.exists():
        click.secho(
            f"  ⚠ {source_dir}/ not found — skipping {label} export.", fg="yellow"
        )
        return

    csvs = [
        _to_local_path(p)
        for p in source_path.iterdir(files_only=True)
        if p.suffix == ".csv"
    ]
    if not csvs:
        click.secho(
            f"  ⚠ No CSVs in {source_dir}/ — skipping {label} export.", fg="yellow"
        )
        return

    click.echo(f"\n  Found {len(csvs)} {label} CSV(s) in {source_dir}/")

    # Simplex CSVs have different geometry than cell CSVs (lines for edges,
    # triangles for triads), so standard GeoJSON/OME-CSV export is not yet
    # implemented. The raw CSVs can be consumed directly.
    if geojson or omecsv:
        click.secho(
            f"  ⚠ GeoJSON/OME-CSV export for {label}s not yet implemented — "
            f"use {source_dir}/ CSVs directly.",
            fg="yellow",
        )
