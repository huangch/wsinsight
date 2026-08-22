"""Standalone CLI command for post-hoc object-to-region or object-to-object registration."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import tqdm

from ..insightlib.region_registration import register_objects_to_objects
from ..insightlib.region_registration import register_objects_to_regions
from ..io.schema import discover_prob_prefixes
from ..io.schema import make_object_prefix
from ..io.schema import make_region_prefix
from ..io.schema import resolve_no_tag_prefix
from ..uri_path import URIPath
from ..uri_path import URIPathType
from ..write_geojson import write_geojsons
from ..write_omecsv import write_omecsvs
from ..wsi import CannotReadSpacing
from ..wsi import get_avg_mpp
from ._meta import write_runtime_metadata
from ._paths import default_storage_kwargs
from ._paths import ensure_input_directory

_STORAGE_KWARGS = default_storage_kwargs()


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
    required=False,
    default=None,
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    help=(
        "Results directory from a prior region-level (patch-based) wsinsight run "
        "containing a model-outputs-csv/ folder.  Each detected object is matched "
        "to its enclosing region and the region's class probabilities are added as "
        "region_<tag>_prob_* columns in the object CSVs.  Mutually exclusive with -c."
    ),
)
@click.option(
    "-c",
    "--object-inference-dir",
    required=False,
    default=None,
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    help=(
        "Results directory from a prior object-level (cell-based) wsinsight run "
        "containing a model-outputs-csv/ folder.  Each primary object is matched "
        "to its nearest secondary object within --radius-um and that object's "
        "class probabilities are added as object_<tag>_prob_* columns in the "
        "primary CSVs.  Mutually exclusive with -r."
    ),
)
@click.option(
    "--tag",
    default="",
    show_default=False,
    type=str,
    help=(
        "Optional namespace inserted between the kind and 'prob' in the output "
        "column names: -r --tag X writes region_X_prob_*; -c --tag X writes "
        "object_X_prob_*.  Must match [a-z0-9_]+.  When omitted, the bare "
        "default (region_prob_* / object_prob_*) is used; if the bare default "
        "already exists, the smallest free integer suffix is auto-picked "
        "(region_1_prob_*, object_2_prob_*, ...) unless --overwrite is given."
    ),
)
@click.option(
    "--radius-um",
    default=5.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
    help=(
        "Maximum object-to-object match radius in micrometres.  Only consulted "
        "when -c is set."
    ),
)
@click.option(
    "--spacing-um-px",
    default=0.25,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
    help=(
        "Slide pixel size in micrometres-per-pixel, used to convert --radius-um "
        "into pixel space.  Only consulted when -c is set.  When -i/--wsi-dir "
        "is supplied, this value is treated as a fallback: each slide's MPP is "
        "read from the WSI metadata (OpenSlide / TiffSlide / tifffile / Aperio "
        "AppMag table) and used in preference; --spacing-um-px is only used for "
        "slides whose MPP cannot be read."
    ),
)
@click.option(
    "--export-geojson",
    "geojson",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Export registered object CSVs to GeoJSON files.  One subfolder is "
        "written per discovered probability prefix (prob, region_*_prob, "
        "object_*_prob) under model-outputs-geojson/."
    ),
)
@click.option(
    "--omecsv",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Export registered object CSVs to OME-CSV files.  One subfolder is "
        "written per discovered probability prefix under model-outputs-omecsv/."
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
        "Overwrite existing columns under the resolved prefix.  Without this "
        "flag, slides whose CSV already contains the resolved prefix are "
        "skipped with a warning (for explicit --tag) or auto-bumped to the "
        "next free integer suffix (for empty --tag)."
    ),
)
def reg(
    wsi_dir: URIPath | None,
    results_dir: URIPath,
    region_inference_dir: URIPath | None,
    object_inference_dir: URIPath | None,
    tag: str = "",
    radius_um: float = 5.0,
    spacing_um_px: float = 0.25,
    geojson: bool = False,
    omecsv: bool = False,
    export_workers: int = 4,
    overwrite: bool = False,
) -> None:
    """Register a primary object-CSV against a secondary inference.

    The secondary inference may be a region-level run (``-r``) or another
    object-level run (``-c``); exactly one must be supplied.  Region matching
    uses point-in-bbox containment; object matching uses nearest-neighbour
    KD-tree lookup within ``--radius-um``.  Matched secondary class
    probabilities are appended in-place to the primary CSVs under
    ``region_<tag>_prob_*`` or ``object_<tag>_prob_*`` columns.

    With ``--export-geojson`` or ``--omecsv``, all probability prefix groups
    present in the registered CSVs are exported, one subfolder per group.
    """
    if (region_inference_dir is None) == (object_inference_dir is None):
        raise click.ClickException(
            "Exactly one of -r/--region-inference-dir or "
            "-c/--object-inference-dir must be provided."
        )

    kind = "region" if region_inference_dir is not None else "object"
    secondary_dir = region_inference_dir if kind == "region" else object_inference_dir
    make_prefix = make_region_prefix if kind == "region" else make_object_prefix
    try:
        bare_prefix = make_prefix(tag)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    obj_csv_dir = results_dir / "model-outputs-csv"
    sec_csv_dir = secondary_dir / "model-outputs-csv"

    if not obj_csv_dir.exists():
        raise click.ClickException(
            f"--results-dir does not contain a model-outputs-csv/ subfolder: {results_dir}"
        )
    if not sec_csv_dir.exists():
        sec_flag = (
            "--region-inference-dir" if kind == "region" else "--object-inference-dir"
        )
        raise click.ClickException(
            f"{sec_flag} does not contain a model-outputs-csv/ subfolder: {secondary_dir}"
        )

    if wsi_dir is not None:
        wsi_dir = wsi_dir.coerce_image_list()
        ensure_input_directory(wsi_dir, "--wsi-dir")
        from ..wsi import list_slide_paths

        wsi_paths = list_slide_paths(wsi_dir)
        wsi_by_stem: dict[str, URIPath] | None = {p.stem: p for p in wsi_paths}
        obj_csvs = sorted(obj_csv_dir / p.with_suffix(".csv").name for p in wsi_paths)
    else:
        wsi_by_stem = None
        obj_csvs = sorted(p for p in obj_csv_dir.iterdir() if p.suffix == ".csv")
    if not obj_csvs:
        raise click.ClickException(f"No CSV files found in {obj_csv_dir}")

    # Resolve prefix once per run.  When --tag is empty and --overwrite is
    # not set, auto-bump to the smallest free integer suffix by scanning the
    # union of headers across all primary CSVs.
    if tag == "" and not overwrite:
        existing_cols: set[str] = set()
        for p in obj_csvs:
            try:
                existing_cols.update(pd.read_csv(p, nrows=0).columns)
            except Exception:
                continue
        prefix = resolve_no_tag_prefix(kind, existing_cols)
    else:
        prefix = bare_prefix

    click.secho(
        f"\nRegistering {len(obj_csvs)} slide(s) [{kind}] -> prefix '{prefix}'.\n",
        fg="green",
    )

    skipped = 0
    processed = 0
    low_match_slides: list[str] = []
    with tqdm.tqdm(obj_csvs, desc="Slides", position=0) as slide_bar:
        chunk_bar = tqdm.tqdm(desc="Registering", position=1, leave=False, unit="chunk")
        for obj_csv in slide_bar:
            sec_csv = sec_csv_dir / obj_csv.name
            if not sec_csv.exists():
                click.echo(f"WARNING: no secondary CSV for {obj_csv.name}, skipping.")
                skipped += 1
                continue

            slide_df = pd.read_csv(
                obj_csv,
                engine="c",
                memory_map=True,
                low_memory=False,
            )
            annot_df = pd.read_csv(
                sec_csv,
                engine="c",
                memory_map=True,
                low_memory=False,
            )

            if not overwrite:
                would_add = {prefix + c for c in annot_df.columns}
                already_present = would_add & set(slide_df.columns)
                if already_present:
                    click.echo(
                        f"WARNING: skipping {obj_csv.name} — {prefix}* columns already "
                        f"present ({', '.join(sorted(already_present)[:3])}{'...' if len(already_present) > 3 else ''}). "
                        f"Use --overwrite to replace."
                    )
                    skipped += 1
                    continue

            if kind == "region":
                slide_df, match_rate = register_objects_to_regions(
                    slide_df,
                    annot_df,
                    pbar=chunk_bar,
                    out_prefix=prefix,
                )
            else:
                # Resolve per-slide pixel spacing.  Prefer the WSI metadata
                # when --wsi-dir is supplied; fall back to --spacing-um-px.
                slide_spacing = spacing_um_px
                spacing_source = "--spacing-um-px"
                if wsi_by_stem is not None:
                    wsi_path = wsi_by_stem.get(obj_csv.stem)
                    if wsi_path is not None:
                        try:
                            slide_spacing = float(get_avg_mpp(wsi_path.__fspath__()))
                            spacing_source = "WSI metadata"
                        except (CannotReadSpacing, OSError, ValueError) as exc:
                            click.secho(
                                f"WARNING: {obj_csv.stem}: could not read MPP "
                                f"from WSI ({type(exc).__name__}); falling back "
                                f"to --spacing-um-px={spacing_um_px}.",
                                fg="yellow",
                            )
                slide_df, match_rate = register_objects_to_objects(
                    slide_df,
                    annot_df,
                    radius_um=radius_um,
                    spacing_um_px=slide_spacing,
                    out_prefix=prefix,
                    pbar=chunk_bar,
                )

            n_total = len(slide_df)
            n_matched = int(round(match_rate * n_total))
            if kind == "object":
                click.echo(
                    f"{obj_csv.stem}: matched {n_matched}/{n_total} "
                    f"({match_rate:.1%})  [spacing={slide_spacing:.4f} um/px "
                    f"via {spacing_source}]"
                )
            else:
                click.echo(
                    f"{obj_csv.stem}: matched {n_matched}/{n_total} ({match_rate:.1%})"
                )
            if kind == "object" and n_total > 0 and match_rate < 0.5:
                low_match_slides.append(obj_csv.stem)

            with obj_csv.open("wb") as fh:
                slide_df.to_csv(fh, index=False)

            processed += 1
        chunk_bar.close()

    click.secho(f"\nDone. Processed: {processed}, skipped: {skipped}.\n", fg="green")
    if low_match_slides:
        click.secho(
            f"WARNING: {len(low_match_slides)} slide(s) had object match-rate "
            f"< 50% (radius={radius_um} um). First few: "
            f"{', '.join(low_match_slides[:5])}.",
            fg="yellow",
        )

    if not (geojson or omecsv):
        return

    # Re-enumerate after registration so all (including previously-skipped) CSVs
    # are picked up for export.  Honour --wsi-dir shard when provided.
    if wsi_dir is not None:
        from ..wsi import list_slide_paths

        all_obj_csvs = sorted(
            obj_csv_dir / p.with_suffix(".csv").name for p in list_slide_paths(wsi_dir)
        )
    else:
        all_obj_csvs = sorted(p for p in obj_csv_dir.iterdir() if p.suffix == ".csv")
    local_csvs = [Path(p.__fspath__()) for p in all_obj_csvs]

    discovered = discover_prob_prefixes(local_csvs)
    if not discovered:
        click.secho(
            "WARNING: no prob_* / region_*_prob_* / object_*_prob_* columns "
            "found in the primary CSVs; nothing to export.",
            fg="yellow",
        )
        return

    click.echo(
        f"\nExporting {len(discovered)} probability group(s): "
        f"{', '.join(discovered)}\n"
    )

    for grp in discovered:
        if geojson:
            click.echo(f"  GeoJSON: {grp}_*  ->  model-outputs-geojson/{grp}/")
            write_geojsons(
                csvs=local_csvs,
                overlap=0.0,
                results_dir=results_dir,
                output_dir=Path("model-outputs-geojson") / grp,
                prefix=grp,
                num_workers=export_workers,
                object_type="detection",
                set_classification=True,
                overwrite=True,  # Always regenerate after reg modifies CSVs
            )
        if omecsv:
            click.echo(f"  OME-CSV: {grp}_*  ->  model-outputs-omecsv/{grp}/")
            write_omecsvs(
                csvs=local_csvs,
                h5s=[],
                overlap=0.0,
                results_dir=results_dir,
                output_dir=Path("model-outputs-omecsv") / grp,
                prefix=grp,
                num_workers=export_workers,
                overwrite=True,  # Always regenerate after reg modifies CSVs
            )

    write_runtime_metadata(
        results_dir,
        "reg",
        params=click.get_current_context().params,
    )
