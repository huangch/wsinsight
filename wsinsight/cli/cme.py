# """Detect cancerous regions in a whole slide image."""

# from __future__ import annotations

# import os
# import re
# import json
# from pathlib import Path
# from platformdirs import user_cache_dir
# import click
# import math

# from ..write_geojson import write_geojsons
# from ..insightlib.cme_generation import cme_generation # , DummyPatchDataset
# from ..uri_path import URIPathType, URIPath

# def _num_cpus() -> int:
#     """Get number of CPUs on the system."""
#     try:
#         return len(os.sched_getaffinity(0))
#     # os.sched_getaffinity seems to be linux only.
#     except AttributeError:
#         count = os.cpu_count()  # potentially None
#         return count or 0


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
#     "-n",
#     "--num-workers",
#     default=min(_num_cpus(), 8),  # Use at most 8 workers by default.
#     show_default=True,
#     type=click.IntRange(min=0),
#     help="Number of workers to use for data loading during model inference (n=0 for"
#     " single thread). Set this to the number of cores on your machine or lower.",
# )
# # @click.option(
# #     "--patch-overlap-median-filter-size",
# #     default=3,
# #     type=click.IntRange(min=3),
# #     help="The kernel size for median filtering when patch overlapping. Must be greater than 1 and odd.",
# # )
# # @click.option(
# #     "--red-threshold",
# #     default=0,
# #     type=click.IntRange(min=0, max=255),
# #     help="The threshold for the red channel of the image. If the red channel is greater"
# #     " than this value, the pixel is considered to be tissue. (Default: 0 no filter) (Range: 0-255)",
# # )
# @click.option(
#     "--cme-clustering-k",
#     default=None,
#     type=click.IntRange(min=0),
#     help="the n-neighbors parameter using in clustering for cmes.",
# )
# @click.option(
#     "--cme-clustering-resolutions",
#     callback=_csv_to_list,
#     default="0.5,1.0,2.0",
#     help="Resolution parameter using in clustering for cmes.",
# )
# @click.option(
#     "--remote-cache-dir",
#     default=os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
#     type=click.Path(file_okay=False, path_type=Path),
#     required=False,
#     help="Directory for remote cache.",
# )

# def cme(
#     wsi_dir: Path,
#     results_dir: Path,
#     num_workers: int = 0,
#     cme_clustering_k: int | None = None,
#     cme_clustering_resolutions = [0.5,1.0,2.0],
#     remote_cache_dir: Path = Path(user_cache_dir(appname="wsinsight", appauthor=False)),
# ) -> None:
#     """Perform cellular micro-environment analysis on whole slide images.

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

             
#     click.secho("\nRunning cme generation.\n", fg="green")
#     # Option 1: WITHOUT H-Optimus (k-hop only)
#     cme_generation(
#         wsi_dir=wsi_dir,
#         results_dir=results_dir,
#         max_edge_len_um=25,
#         max_cell_radius_um=15,
#         k_hops=2, alpha=1.0,
#         use_hoptimus=False,                 # ← off
#         hidden=64, out_dim=32, epochs=300,
#         cme_clustering_k=cme_clustering_k, 
#         cme_clustering_resolutions=cme_clustering_resolutions,
#         # seed=0,
#     )
     
#     # # Option 2: WITH H-Optimus (using a dummy dataset now; replace with your own Dataset later)
#     # patch_datasets = [DummyPatchDataset(num_cells=len(slides_inputs[0][0])),
#     #                   DummyPatchDataset(num_cells=len(slides_inputs[1][0]))]
#     #
#     # res_h0 = cme_generation(
#     #     slides_inputs,
#     #     max_edge_len_um=70.0,
#     #     k_hops=2, alpha=1.0,
#     #     use_hoptimus=True,                  # ← ON
#     #     patch_datasets=patch_datasets,      # replace with your real datasets later
#     #     sample_frac=0.2, pca_dim=128, knn_k=3, knn_sigma_um=60.0,
#     #     hidden=64, out_dim=32, epochs=300,
#     #     clusters_k=5, seed=0
#     # )
#     #
#     # # Each result contains per-slide embeddings and labels:
#     # Z_slide0 = res_h0["embeddings"][0]        # [N0, 32]
#     # y_slide0 = res_h0["labels"][0]            # [N0]
#     # kept_idx0 = res_h0["kept_idx"][0]         # map to original rows in slideA_cells.csv
 
#     click.echo("\nWriting cme detection cellular results to GeoJSON files\n")
    
#     cme_cell_csvs = list((results_dir / "cme-outputs-csv" / "cells" ).glob("*.csv"))
#     write_geojsons(csvs=cme_cell_csvs, 
#                    overlay=0, 
#                    results_dir=results_dir, 
#                    # input_dir=Path("cme-outputs-csv") / "cells", 
#                    output_dir=Path("cme-outputs-geojson") / "cells", 
#                    data_prefix="cme", 
#                    display_prefix="cme",
#                    num_workers=num_workers, 
#                    object_type="detection",
#                    set_classification=True,
#                    annotation_shape="box")
 
#     click.echo("\nWriting cme detection cme results to GeoJSON files")
   
#     cme_cme_csvs = list((results_dir / "cme-outputs-csv" / "cmes").glob("*.csv"))
#     write_geojsons(csvs=cme_cme_csvs, 
#                    overlay=0, 
#                    results_dir=results_dir, 
#                    # input_dir=Path("cme-outputs-csv") / "cmes", 
#                    output_dir=Path("cme-outputs-geojson") / "cmes", 
#                    data_prefix="cme",
#                    display_prefix="cme", 
#                    num_workers=num_workers, 
#                    object_type="annotation",
#                    set_classification=True,
#                    annotation_shape="polygon")
    
   
#     click.secho("\nWSInsight tasks are all finished.\n", fg="green")
