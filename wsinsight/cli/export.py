"""Standalone export command: merge all per-cell analytics and write GeoJSON / OME-CSV."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from platformdirs import user_cache_dir

from ..export_helpers import build_export_csvs
from ..uri_path import URIPath, URIPathType
from ..write_geojson import write_geojsons
from ..write_omecsv import write_omecsvs


def _storage_kwargs() -> dict[str, object]:
    cache_dir = os.getenv("WSINSIGHT_REMOTE_CACHE_DIR")
    if cache_dir is None:
        cache_dir = Path(user_cache_dir(appname="wsinsight", appauthor=False))
    storage: dict[str, object] = {"cache_dir": cache_dir}
    s3_options = os.getenv("S3_STORAGE_OPTIONS")
    if s3_options:
        try:
            parsed = json.loads(s3_options)
        except json.JSONDecodeError as exc:
            raise RuntimeError("S3_STORAGE_OPTIONS must contain valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("S3_STORAGE_OPTIONS must be a JSON object.")
        storage.update(parsed)
    return storage


_STORAGE_KWARGS = _storage_kwargs()


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
    "--export-workers",
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
def export(
    results_dir: URIPath,
    geojson: bool,
    omecsv: bool,
    patch_overlap_ratio: float,
    object_type: str,
    export_workers: int,
    overwrite: bool,
) -> None:
    """Merge all available per-cell analytics and export to GeoJSON / OME-CSV.

    Reads whatever analysis outputs are present under RESULTS_DIR:

    \b
      model-outputs-csv/       — base inference probabilities (+ reg columns)
      hplot-outputs-csv/cells/ — H-Plot per-cell layer features
      ncomp-outputs-csv/       — node-level (cell) composition features

    All per-cell sources above are left-joined into export-csv/, then
    written to export-geojson/ and/or export-omecsv/ depending on the
    flags provided.

    Edge-level (``ecomp-outputs-csv/``) and triad-level
    (``tcomp-outputs-csv/``) composition outputs are standalone
    simplicial deliverables and are NOT merged into the per-cell
    export (they have different primary keys).  Consume them
    directly from their respective subdirectories.

    At least one of --geojson or --omecsv must be supplied.

    This command can be run at any time after inference — and optionally after
    hplot / ncomp — without re-running the full inference pipeline.
    """
    if not geojson and not omecsv:
        raise click.UsageError(
            "At least one of --geojson or --omecsv must be specified."
        )

    if not (results_dir / "model-outputs-csv").exists():
        raise click.ClickException(
            f"No model-outputs-csv/ found under {results_dir}. "
            "Run 'wsinsight run' or 'wsinsight infer' first."
        )

    # --- Merge all per-cell sources into export-csv/ -------------------------
    click.echo("\nMerging per-cell analytics into export CSVs...\n")
    build_export_csvs(results_dir, overwrite=overwrite)

    export_dir = results_dir / "export-csv"
    export_candidates = list(export_dir.iterdir(files_only=True))
    export_csvs = [
        _to_local_path(p) for p in export_candidates if p.suffix == ".csv"
    ]

    if not export_csvs:
        raise click.ClickException(
            "No export CSVs were produced — nothing to export."
        )

    click.echo(f"  {len(export_csvs)} slide(s) ready for export.")

    # --- GeoJSON export -------------------------------------------------------
    if geojson:
        click.echo("\nWriting results to GeoJSON files...\n")
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
        )

    # --- OME-CSV export -------------------------------------------------------
    if omecsv:
        click.echo("\nWriting results to OME-CSV files...\n")
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
        )

    click.secho("\nExport complete.\n", fg="green")
