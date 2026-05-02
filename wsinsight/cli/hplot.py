"""Standalone CLI for H-Plot generation over WSInsight inference outputs."""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import click
import pandas as pd
from platformdirs import user_cache_dir
from tqdm import tqdm

from ..insightlib.hplot_generation import hplot_generation, hplot_finalize
from ..uri_path import URIPath, URIPathType
from ._meta import write_runtime_metadata


def _coerce_number(token: str) -> int | float | str:
    """Convert CLI token to int/float when possible, otherwise lowercase text."""
    t = token.strip()
    if re.fullmatch(r"[+-]?\d+", t):
        try:
            return int(t)
        except ValueError:
            pass
    try:
        x = float(t)
        if math.isfinite(x):
            return x
    except ValueError:
        pass
    return t.lower()


def _csv_to_list(_: click.Context, __: click.Parameter, value: str | list[str] | None) -> list[int | float | str]:
    if value is None:
        return []
    tokens = value if isinstance(value, list) else [x for x in re.split(r"[,\s]+", str(value).strip()) if x]
    return [_coerce_number(str(x)) for x in tokens]


def _normalize_types(values: Iterable[object]) -> list[str]:
    """Normalize type labels to lowercase snake-case tokens."""
    return [str(value).strip().replace(" ", "_").lower() for value in values]


def _read_prob_columns(csv_path: URIPath) -> set[str]:
    """Read only the header of a model-output CSV and return its prob_* columns (lowercased, prefix stripped)."""
    try:
        with csv_path.open("r", encoding="utf-8") as fp:
            header_df = pd.read_csv(fp, nrows=0)
    except Exception:
        return set()
    return {
        c.lower().removeprefix("prob_")
        for c in header_df.columns
        if c.lower().startswith("prob_")
    }


def _collect_available_types(model_output_dir: URIPath, num_workers: int) -> set[str]:
    """Scan all model-output CSVs in parallel and return the union of available cell-type names."""
    csv_paths = [p for p in model_output_dir.iterdir() if p.name.endswith(".csv")]
    if not csv_paths:
        return set()
    available: set[str] = set()
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = [executor.submit(_read_prob_columns, p) for p in csv_paths]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Scanning available cell types",
            unit="csv",
            dynamic_ncols=True,
        ):
            available.update(fut.result())
    return available


def _validate_types(
    requested: list[str], available: set[str], option_name: str
) -> None:
    """Raise ClickException if any requested type is missing from available; warn on partial mismatches handled per-slide."""
    unknown = [t for t in requested if t not in available]
    if unknown:
        raise click.ClickException(
            f"{option_name}: unknown cell type(s) {unknown}. "
            f"Available cell types across model-outputs-csv: {sorted(available)}"
        )


def _assert_directory(path: URIPath, option_name: str) -> None:
    """Ensure the provided `URIPath` exists and points to a directory."""
    if not path.exists():
        raise click.ClickException(f"{option_name} directory not found: {path}")
    if not path.is_dir():
        raise click.ClickException(f"{option_name} must be a directory")


def _storage_kwargs() -> dict[str, object]:
    cache_dir = os.getenv("WSINSIGHT_REMOTE_CACHE_DIR")
    if cache_dir is None:
        cache_dir = Path(user_cache_dir(appname="wsinsight", appauthor=False))
    storage: dict[str, object] = {"cache_dir": cache_dir}
    s3_options = os.getenv("S3_STORAGE_OPTIONS")
    if s3_options:
        try:
            parsed = json.loads(s3_options)
        except json.JSONDecodeError as exc:  # pragma: no cover - env misconfiguration
            raise RuntimeError("S3_STORAGE_OPTIONS must contain valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("S3_STORAGE_OPTIONS must be a JSON object.")
        storage.update(parsed)
    return storage


_STORAGE_KWARGS = _storage_kwargs()


@click.command()
@click.option(
    "-i",
    "--wsi-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="Directory containing whole slide images. This directory can *only* contain whole slide images.",
)
@click.option(
    "-o",
    "--results-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="Directory to store results. If directory exists, will skip whole slides for which outputs exist.",
)
@click.option(
    "--hplot-max-neighbor-distance",
    default=25.0,
    type=click.FloatRange(min=0),
    help="A parameter of float value determining maximal distance (um) to a neighboring cell.",
)
@click.option(
    "--hplot-base-types",
    callback=_csv_to_list,
    default=None,
    help="Base cell type or cell type list that form(s) the cell cluster(s), e.g., tumor cells.",
)
@click.option(
    "--hplot-target-types",
    callback=_csv_to_list,
    default=None,
    help="Target cell type or cell type list whose layer-wise proportion is computed, e.g., lymphocytes.",
)
@click.option(
    "--hplot-k",
    default=2,
    type=click.IntRange(min=0),
    help="The maximal edge distance for defining the neighborhood of a cell.",
)
@click.option(
    "--hplot-n",
    default=8,
    type=click.IntRange(min=0),
    help="The minimal neighborhood size for a cell to be computed for determining tumor regions.",
)
@click.option(
    "--hplot-r",
    default=0.5,
    type=click.FloatRange(min=0, max=1),
    help="The minimal ratio of tumor cells in the neighborhood of a cell, determining is this cell included in a tumor region.",
)
@click.option(
    "--hplot-range-max",
    default=None,
    type=click.IntRange(min=1),
    help="The maximal layer index toward OUTSIDE of tumors for the range window of H-Plot.",
)
@click.option(
    "--hplot-range-min",
    default=None,
    type=click.IntRange(max=0),
    help="The minimal layer index toward INSIDE of tumors for the range window of H-Plot.",
)
@click.option(
    "--hplot-samples-with-valid-range-only",
    is_flag=True,
    default=False,
    show_default=True,
    help="H-Plot computing uses only samples with valid range of cellular-wise layers.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite existing H-Plot results instead of skipping slides that already have outputs.",
)
@click.option(
    "--num-workers",
    default=8,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of slides to process concurrently.",
)
def hplot(
    *,
    wsi_dir: URIPath,
    results_dir: URIPath,
    hplot_max_neighbor_distance: float = 25.0,
    hplot_base_types: List | None = None,
    hplot_target_types: List | None = None,
    hplot_k: int = 2,
    hplot_n: int = 8,
    hplot_r: float = 0.5,
    hplot_range_max: int | None = None,
    hplot_range_min: int | None = None,
    hplot_samples_with_valid_range_only: bool = False,
    overwrite: bool = False,
    num_workers: int = 8,
) -> None:
    """Run H-Plot analysis on inference outputs held inside ``results_dir``."""

    wsi_dir = wsi_dir.coerce_image_list()
    _assert_directory(wsi_dir, "--wsi-dir")
    _assert_directory(results_dir, "--results-dir")

    slide_paths = sorted([p for p in wsi_dir.iterdir() if wsi_dir.scheme == "image-list" or p.is_file()])
    if not slide_paths:
        raise click.ClickException(f"no files exist in the slide directory: {wsi_dir}")

    model_output_dir = results_dir / "model-outputs-csv"
    if not model_output_dir.exists():
        raise click.ClickException(
            "The 'model-outputs-csv' directory was not found in results directory."
        )

    if not hplot_base_types or not hplot_target_types:
        raise click.ClickException("H-Plot requires both --hplot-base-types and --hplot-target-types.")

    base_type_list = _normalize_types(hplot_base_types)
    target_type_list = _normalize_types(hplot_target_types)

    available_types = _collect_available_types(model_output_dir, num_workers)
    if not available_types:
        raise click.ClickException(
            f"No 'prob_*' columns found in any CSV under {model_output_dir}."
        )
    _validate_types(base_type_list, available_types, "--hplot-base-types")
    _validate_types(target_type_list, available_types, "--hplot-target-types")

    click.secho("\nRunning H-Plot generation.\n", fg="green")
    failed_hplot_generation = hplot_generation(
        wsi_dir=wsi_dir,
        slide_paths=slide_paths,
        results_dir=results_dir,
        base_type_list=base_type_list,
        target_type_list=target_type_list,
        max_neighbor_distance_um=hplot_max_neighbor_distance,
        hplot_k=hplot_k,
        hplot_N=hplot_n,
        hplot_R=hplot_r,
        hplot_range_max=hplot_range_max,
        hplot_range_min=hplot_range_min,
        hplot_samples_with_valid_range_only=hplot_samples_with_valid_range_only,
        num_workers=num_workers,
        overwrite=overwrite,
    )

    if failed_hplot_generation:
        click.secho(
            f"\nH-Plot generation failed for {len(failed_hplot_generation)} slides",
            fg="yellow",
        )
        click.secho("\n".join(failed_hplot_generation), fg="yellow")

    write_runtime_metadata(
        results_dir,
        "hplot",
        params=click.get_current_context().params,
    )

    click.secho("\nWSInsight tasks are all finished.\n", fg="green")


@click.command("hplot-finalize")
@click.option(
    "-o",
    "--results-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="Results directory containing hplot per-slide outputs. The aggregated "
         "hplot-outputs.csv and hmetrics-outputs.csv will be written here.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite existing hplot-outputs.csv and hmetrics-outputs.csv if they already exist.",
)
def hplot_finalize_cmd(
    *,
    results_dir: URIPath,
    overwrite: bool = False,
) -> None:
    """Rebuild hplot-outputs.csv and hmetrics-outputs.csv from per-slide intermediates.

    Use this after running parallel ``hplot`` jobs that share the same
    ``--results-dir``. Each worker writes its per-slide files; this command
    assembles the final aggregated CSVs from all of them.
    """

    _assert_directory(results_dir, "--results-dir")

    click.secho("\nFinalizing H-Plot outputs.\n", fg="green")
    hplot_finalize(output_dir=results_dir, overwrite=overwrite)

    write_runtime_metadata(
        results_dir,
        "hplot-finalize",
        params=click.get_current_context().params,
    )

    click.secho("\nH-Plot finalization complete.\n", fg="green")
