# """Detect cancerous regions in a whole slide image."""

# from __future__ import annotations

# import os
# import re
# from typing import List
# import json
# from pathlib import Path
# from platformdirs import user_cache_dir
# import click
# import math

# from ..insightlib.hplot_generation import hplot_generation
# from ..uri_path import URIPathType, URIPath


# def _coerce_number(token: str):
#     t = token.strip()
#     # int?
#     if re.fullmatch(r'[+-]?\d+', t):
#         try:
#             return int(t)
#         except ValueError:
#             pass
#     # float / scientific?
#     try:
#         x = float(t)
#         # keep only finite floats; otherwise leave as string
#         if math.isfinite(x):
#             return x
#     except ValueError:
#         pass
#     # not a number → return lower-cased string
#     return t.lower()

# def _csv_to_list(ctx, param, value):
#     if value is None:
#         return []
#     # Accept list or string input (keep usage unchanged)
#     if isinstance(value, list):
#         tokens = value
#     else:
#         # split by commas or whitespace; drop empties
#         tokens = [x for x in re.split(r'[,\s]+', str(value).strip()) if x]

#     return [_coerce_number(str(x)) for x in tokens]


# @click.command()
# @click.pass_context
# @click.option(
#     "-i",
#     "--wsi-dir",
#     type=URIPathType(exists=True, **json.loads(os.getenv("S3_STORAGE_OPTIONS"))) if os.getenv("S3_STORAGE_OPTIONS") else URIPathType(exists=True),
#     required=True,
#     help="Directory containing whole slide images. This directory can *only* contain"
#     " whole slide images.",
# )
# @click.option(
#     "-o",
#     "--results-dir",
#     type=click.Path(file_okay=False, path_type=Path),
#     required=True,
#     help="Directory to store results. If directory exists, will skip"
#     " whole slides for which outputs exist.",
# )
# @click.option(
#     "--hplot-max-neighbor-distance",
#     default=25.0,
#     type=click.FloatRange(min=0),
#     help="A parameter of float value determining maximal distance (um) to a neighboring cell.",
# )
# @click.option(
#     "--hplot-base-types",
#     callback=_csv_to_list,
#     default=None,
#     help="Base cell type or cell type list that form(s) the cell cluster(s), e.g., tumor cells.",
# )
# @click.option(
#     "--hplot-target-types",
#     callback=_csv_to_list,
#     default=None,
#     help="Target cell type cell type list for computing layer-wise proportion, e.g., lymphocytes.",
# )
# @click.option(
#     "--hplot-k",
#     default=2,
#     type=click.IntRange(min=0),
#     help="The maximal edge distance for defining the neighborhood of a cell.",
# )
# @click.option(
#     "--hplot-n",
#     default=8,
#     type=click.IntRange(min=0),
#     help="The minimal neighborhood size for a cell to be computed for determining tumor regions.",
# )
# @click.option(
#     "--hplot-r",
#     default=0.5,
#     type=click.FloatRange(min=0, max=1),
#     help="The minimal ratio of tumor cells in the neighborhood of a cell, determining "
#         "is this cell included in a tumor region.",
# )
# @click.option(
#     "--hplot-range-max",
#     default=None,
#     type=click.IntRange(min=1),
#     help="The maximal layer index toward OUTSIDE of tumors for the range window of H-Plot.",
# )
# @click.option(
#     "--hplot-range-min",
#     default=None,
#     type=click.IntRange(max=0),
#     help="The minimal layer index toward INSIDE of tumors for the range window of H-Plot.",
# )
# @click.option(
#     "--hplot-samples-with-valid-range-only",
#     is_flag=True,
#     default=False,
#     show_default=True,
#     help="H-Plot computing uses only samples with valid range of cellular-wise layers.",
# )
# @click.option(
#     "--hplot-abundance-at-peak-proximity",
#     default=0.5,
#     type=click.FloatRange(min=0, max=1),
#     help="The abundance score that maps to highest weight for proximity score.",
# )
# @click.option(
#     "--remote-cache-dir",
#     default=os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
#     type=click.Path(file_okay=False, path_type=Path),
#     required=False,
#     help="Directory for remote cache.",
# )

# def hplot(
#     wsi_dir: Path,
#     results_dir: Path,

#     hplot_max_neighbor_distance: float = 25.0,
#     hplot_base_types: List | None = None,
#     hplot_target_types: List | None = None,
#     hplot_k: int = 2,
#     hplot_n: int = 8,
#     hplot_r: float = 0.5,
#     hplot_range_max: int = None,
#     hplot_range_min: int = None,
#     hplot_samples_with_valid_range_only: bool = False,
#     hplot_abundance_at_peak_proximity: float = 0.5,
#     remote_cache_dir: Path = Path(user_cache_dir(appname="wsinsight", appauthor=False)),
# ) -> None:
#     """Perform H-Plot analysis on whole slide images.

#     This command will create a tissue mask of each WSI. Then patch coordinates will be
#     computed. The chosen model will be applied to each patch, and the results will be
#     saved to a CSV in `RESULTS_DIR/model-output`.

#     Example:

#     CUDA_VISIBLE_DEVICES=0 wsinsight run --wsi-dir slides/ --results-dir results
#     --model breast-tumor-resnet34.tcga-brca --batch-size 32 --num-workers 4

#     To list all available models and weights, use `wsinfer-zoo ls`.
#     """
        
#     remote_cache_dir = remote_cache_dir / ("s3" if str(wsi_dir).startswith("s3") else "gdc-manifest" if str(wsi_dir).startswith("gdc-manifest") else "")
#     wsi_dir = URIPath(wsi_dir, cache_dir=remote_cache_dir)
    
#     if not (wsi_dir.exists()):
#         raise FileNotFoundError(f"Whole slide image directory not found: {wsi_dir}")

#     # Test that wsi dir actually includes files. This is here for an interesting edge
#     # case. When using a Linux container and if the data directory is symlinked from a
#     # different directory, both directories need to be bind mounted onto the container.
#     # If only the symlinked directory is included, then the patching script will fail,
#     # even though it looks like there are files in the wsi_dir directory.
#     files_in_wsi_dir = [p for p in wsi_dir.iterdir() if p.is_file()]
#     if not files_in_wsi_dir:
#         raise FileNotFoundError(f"no files exist in the slide directory: {wsi_dir}")



#     # H-Plot actions
    
#     if len(hplot_base_types) == 0 or len(hplot_target_types) == 0:
#         raise click.ClickException(f"\nH-Plot requires both --hplot-base-types and hplot-target-types.")
        
#     elif len(hplot_base_types) != 0 and len(hplot_target_types) != 0:
#         click.secho("\nRunning H-Plot generation.\n", fg="green")
        
#         failed_hplot_generation = hplot_generation(
#             wsi_dir=wsi_dir,
#             results_dir=results_dir,
#             base_type_list=[c.strip().replace(' ', '_').lower() for c in hplot_base_types],
#             target_type_list=[c.strip().replace(' ', '_').lower() for c in hplot_target_types],
#             max_neighbor_distance_um=hplot_max_neighbor_distance,
#             hplot_k=hplot_k,
#             hplot_N=hplot_n,
#             hplot_R=hplot_r, 
#             hplot_range_max=hplot_range_max,
#             hplot_range_min=hplot_range_min,
#             hplot_samples_with_valid_range_only=hplot_samples_with_valid_range_only,
#             hplot_abundance_at_peak_proximity=hplot_abundance_at_peak_proximity,
#         )
        
#         if failed_hplot_generation:
#             click.secho(
#                 f"\nH-Plot generation failed for {len(failed_hplot_generation)}"
#                 " slides", fg="yellow"
#             )
            
#             click.secho("\n".join(failed_hplot_generation), fg="yellow")
            
            

        
#     click.secho("\nWSInsight tasks are all finished.\n", fg="green")
