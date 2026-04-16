"""Standalone CLI command for post-hoc object-to-region registration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
import pandas as pd
import tqdm
from platformdirs import user_cache_dir

from ..insightlib.region_registration import register_objects_to_regions
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


def _assert_directory(path: URIPath, option_name: str) -> None:
    """Ensure the provided `URIPath` exists and points to a directory."""
    if not path.exists():
        raise click.ClickException(f"{option_name} directory not found: {path}")
    if not path.is_dir():
        raise click.ClickException(f"{option_name} must be a directory")


@click.command()
@click.option(
    "-i",
    "--wsi-dir",
    default=None,
    required=False,
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    help=(
        "If provided, only slides whose filename stem matches a file in this directory "
        "are processed.  Image files are not opened; the directory is used for name "
        "enumeration only.  Mirrors --wsi-dir in run/infer/hplot for consistent sharding."
    ),
)
@click.option(
    "-o",
    "--results-dir",
    required=True,
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    help=(
        "Results directory from a prior object-based wsinsight run containing "
        "a model-outputs-csv/ folder."
    ),
)
@click.option(
    "-r",
    "--region-inference-dir",
    required=True,
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    help=(
        "Results directory from a prior region-level (patch-based) wsinsight run "
        "containing a model-outputs-csv/ folder.  Each detected object is matched "
        "to its enclosing region and the region's class probabilities are added as "
        "region_prob_* columns in the object CSVs."
    ),
)
@click.option(
    "--geojson",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Export enriched object CSVs to GeoJSON files.  Object-level probabilities "
        "(prob_*) are written to model-outputs-geojson/."
    ),
)
@click.option(
    "--omecsv",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Export enriched object CSVs to OME-CSV files.  Object-level probabilities "
        "(prob_*) are written to model-outputs-omecsv/."
    ),
)
@click.option(
    "--export-workers",
    default=4,
    show_default=True,
    type=click.IntRange(min=1),
    help="Worker processes for GeoJSON/OME-CSV export.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Overwrite existing region_* columns in object CSVs.  Without this flag, "
        "any slide whose object CSV already contains region_* columns derived from "
        "the same region model is skipped with a warning."
    ),
)
def reg(
    wsi_dir: URIPath | None,
    results_dir: URIPath,
    region_inference_dir: URIPath,
    geojson: bool = False,
    omecsv: bool = False,
    export_workers: int = 4,
    overwrite: bool = False,
) -> None:
    """Register object-prediction CSVs to region-prediction results.

    Reads CSVs in RESULTS_DIR/model-outputs-csv/, looks up the matching region
    CSV in REGION_INFERENCE_DIR/model-outputs-csv/, spatially assigns each
    object to its enclosing region, and writes the enriched CSV back in-place
    with added region_prob_* columns.

    When --wsi-dir is supplied the slide list is derived from the image
    filenames in that directory (image files are not opened), mirroring the
    behaviour of run/infer/hplot and enabling consistent sharding.  Without
    --wsi-dir every CSV found in RESULTS_DIR/model-outputs-csv/ is processed.

    Slides missing from REGION_INFERENCE_DIR are skipped with a warning.
    If a region_prob_* column with the same name already exists it is
    overwritten; columns from a different region model are preserved.

    With --geojson or --omecsv, the enriched object CSVs are exported using
    only the object-level prob_* columns.
    """
    obj_csv_dir = results_dir / "model-outputs-csv"
    reg_csv_dir = region_inference_dir / "model-outputs-csv"

    if not obj_csv_dir.exists():
        raise click.ClickException(
            f"--results-dir does not contain a model-outputs-csv/ subfolder: {results_dir}"
        )
    if not reg_csv_dir.exists():
        raise click.ClickException(
            f"--region-inference-dir does not contain a model-outputs-csv/ subfolder: "
            f"{region_inference_dir}"
        )

    if wsi_dir is not None:
        _assert_directory(wsi_dir, "--wsi-dir")
        obj_csvs = sorted(
            obj_csv_dir / p.with_suffix(".csv").name
            for p in wsi_dir.iterdir()
            if p.is_file()
        )
    else:
        obj_csvs = sorted(p for p in obj_csv_dir.iterdir() if p.suffix == ".csv")
    if not obj_csvs:
        raise click.ClickException(f"No CSV files found in {obj_csv_dir}")

    click.secho(f"\nRegistering {len(obj_csvs)} slide(s).\n", fg="green")

    skipped = 0
    processed = 0
    with tqdm.tqdm(obj_csvs, desc="Slides", position=0) as slide_bar:
        chunk_bar = tqdm.tqdm(desc="Registering", position=1, leave=False,
                              unit="chunk")
        for obj_csv in slide_bar:
            reg_csv = reg_csv_dir / obj_csv.name
            if not reg_csv.exists():
                click.echo(f"WARNING: no region CSV for {obj_csv.name}, skipping.")
                skipped += 1
                continue

            slide_df = pd.read_csv(
                obj_csv,
                engine="c",
                memory_map=True,
                low_memory=False,
            )
            annot_df = pd.read_csv(
                reg_csv,
                engine="c",
                memory_map=True,
                low_memory=False,
            )

            if not overwrite:
                would_add = {"region_" + c for c in annot_df.columns}
                already_present = would_add & set(slide_df.columns)
                if already_present:
                    click.echo(
                        f"WARNING: skipping {obj_csv.name} — region_* columns already "
                        f"present ({', '.join(sorted(already_present)[:3])}{'...' if len(already_present) > 3 else ''}). "
                        f"Use --overwrite to replace."
                    )
                    skipped += 1
                    continue

            slide_df = register_objects_to_regions(slide_df, annot_df, pbar=chunk_bar)

            with obj_csv.open("wb") as fh:
                slide_df.to_csv(fh, index=False)

            processed += 1
        chunk_bar.close()

    click.secho(
        f"\nDone. Processed: {processed}, skipped: {skipped}.\n", fg="green"
    )

    if not (geojson or omecsv):
        return

    # Re-enumerate after registration so all (including previously-skipped) CSVs
    # are picked up for export.  Honour --wsi-dir shard when provided.
    if wsi_dir is not None:
        all_obj_csvs = sorted(
            obj_csv_dir / p.with_suffix(".csv").name
            for p in wsi_dir.iterdir()
            if p.is_file()
        )
    else:
        all_obj_csvs = sorted(p for p in obj_csv_dir.iterdir() if p.suffix == ".csv")
    local_csvs = [Path(p.__fspath__()) for p in all_obj_csvs]

    if geojson:
        click.echo("\nWriting object probabilities (prob_*) to GeoJSON files\n")
        write_geojsons(
            csvs=local_csvs,
            overlap=0.0,
            results_dir=results_dir,
            output_dir="model-outputs-geojson",
            prefix="prob",
            num_workers=export_workers,
            object_type="detection",
            set_classification=True,
        )

    if omecsv:
        click.echo("\nWriting object probabilities (prob_*) to OME-CSV files\n")
        write_omecsvs(
            csvs=local_csvs,
            h5s=[],
            overlap=0.0,
            results_dir=results_dir,
            output_dir="model-outputs-omecsv",
            prefix="prob",
            num_workers=export_workers,
        )
