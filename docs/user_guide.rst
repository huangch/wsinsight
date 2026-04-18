.. _User Guide:

User Guide
==========

This guide assumes that you have installed WSInsight (see :ref:`installing`) and that you
have at least one whole slide image (WSI) ready. Example slides are available from
https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/.

For the examples below we assume your slides sit in :code:`slides/`.

.. admonition:: Citation

   If you use WSInsight in research, please cite both the WSInsight and WSInfer papers
   (bioRxiv, 2025; https://doi.org/10.1101/2025.692260),
   (npj Precision Oncology, 2024; https://doi.org/10.1038/s41698-024-00499-9).


Overview
--------

**WSInsight** is a modernized fork of WSInfer that keeps compatibility with the original
model zoo while layering in cell-centric ViT/HoverNet models, GeoJSON/OME-CSV exporters,
and reproducible CLI workflows. Key features include:

* 🔬 Cell-aware inference through WSInsight-native CellViT and HoverNet checkpoints
* ⚙️ Compatibility with WSInfer configuration/schema for legacy models
* 🧭 Deterministic output layouts (CSV + GeoJSON + OME-CSV)
* ☁️ Unified URI handling for reading WSIs from local disks, ``s3://`` buckets, or ``gdc-manifest://`` manifests and writing outputs back to local paths or S3, plus resumable runs via cached patches


Getting help
------------

Report bugs or request features via GitHub issues:
https://github.com/huangch/wsinsight/issues/new


Command line basics
-------------------

WSInsight provides a CLI. Use :code:`--help` to explore available options:

::

   wsinsight --help
   wsinsight run --help

Eight commands are available:

============================  ================================================================
Command                       Purpose
============================  ================================================================
``wsinsight run``             Convenience wrapper that extracts patches then runs inference/exports.
                              Pass ``--export-geojson`` and/or ``--export-omecsv`` to merge
                              all per-cell analytics and write GeoJSON / OME-CSV at the end.
``wsinsight patch``           Generate tissue masks + patch caches inside ``--results-dir``.
``wsinsight infer``           Reuse cached patches to run models and emit GeoJSON/OME exports.
                              Supports inline region registration via
                              ``--region-inference-dir`` and ``--overwrite``.
``wsinsight reg``             Post-hoc object-to-region registration on already-completed runs.
                              Enriches object CSVs with ``region_prob_*`` columns.
``wsinsight hplot``           Standalone H-plot analysis on existing object-based inference
                              outputs.
``wsinsight hplot-finalize``  Aggregate per-slide H-plot intermediates into a cohort-level
                              summary.
``wsinsight ncomp``           Standalone neighborhood composition analysis on existing
                              inference outputs.  For each target cell, builds a Delaunay
                              graph, collects k-hop neighbors, and records per-cell type
                              counts and proportions.
``wsinsight export``          Merge all per-cell analytics (inference, H-plot, ncomp) into
                              ``export-csv/`` and write GeoJSON and/or OME-CSV files.
                              Can be run after inference without repeating the pipeline.
============================  ================================================================

Pick ``run`` for one-shot processing. Switch to the explicit ``patch`` → ``infer`` flow
for large cohorts, resumable jobs, or when you want to reuse the same patches across
multiple model configurations. Run ``hplot`` on completed cell-detection outputs to
compute spatial tumour-microenvironment metrics, then ``hplot-finalize`` to assemble the
cohort-level summary. Use ``ncomp`` to compute per-cell neighborhood composition on
existing inference outputs. Use ``reg`` to enrich an earlier run with region-level
probabilities without re-running inference. All commands share the same URI-aware options
and support local folders, ``s3://`` buckets, ``gdc-manifest://`` manifests, and
``image-list://`` file lists.


Model catalogs
--------------

WSInsight recognizes two families of model identifiers:

* **WSInfer-compatible IDs** (e.g., ``breast-tumor-resnet34.tcga-brca``). List them with ::

      wsinfer-zoo ls

* **WSInsight-native IDs** (e.g., ``CellViT-SAM-H-x40``). These are documented in
  :ref:`available-models <available-models>` and show up in the same registry.


Running inference
-----------------

A minimal run requires a WSI directory, an output directory, and a model name.

WSInfer-compatible model example:

::

   wsinsight run \
      --wsi-dir slides/ \
      --results-dir results/ \
      --model breast-tumor-singlecell.tcga-brca

WSInsight-native CellViT example:

::

   wsinsight run \
      --wsi-dir slides/ \
      --results-dir results-cellvit/ \
      --model CellViT-SAM-H-x40 \
      --batch-size 16 \
      --num-workers 8

Both flows handle patch extraction, batched inference, and exporter steps automatically.
Pass ``--export-geojson`` and/or ``--export-omecsv`` to ``wsinsight run`` to additionally
merge all per-cell analytics and write GeoJSON / OME-CSV files at the end of the run,
equivalent to running ``wsinsight export`` as a separate step.

Two-stage workflows
-------------------

For large cohorts or multi-model studies, separate patch extraction from inference:

::

    wsinsight patch \
         --wsi-dir slides/ \
         --results-dir results/ \
         --model breast-tumor-singlecell.tcga-brca

    wsinsight infer \
         --wsi-dir slides/ \
         --results-dir results/ \
         --model breast-tumor-singlecell.tcga-brca \
         --batch-size 64 \
         --num-workers 16

``wsinsight patch`` is idempotent: re-running it skips slides whose patches already exist,
making it safe to resume interrupted jobs or share ``--results-dir`` across machines.
``wsinsight infer`` consumes the cached patches, so you can run multiple models against
the same slides without repeating tissue segmentation.

QuPath inputs and exports
-------------------------

WSInsight can both consume and generate QuPath-compatible assets:

* Use ``--qupath-detection-dir`` and ``--qupath-geojson-*-dir`` to ingest detections or
   annotations created in QuPath. Override the pseudo-model settings via
   ``--qupath-detection-patch-size``, ``--qupath-annotation-patch-size``, and
   ``--qupath-spacing-um-px``. Pass ``--qupath-name-as-class`` if you prefer QuPath's
   annotation names over its Classification column.
* Add ``--qupath`` to ``wsinsight run`` (or ``infer``) to build a QuPath project that
   links the generated layers to the original WSIs.
* ``--geojson`` and ``--omecsv`` control whether spatial outputs are emitted in those
   formats; both default to ``False`` so you only generate the artifacts you need.

Segmentation and patch controls
-------------------------------

Tissue detection can be tailored per cohort:

* ``--histoqc-dir`` points to precomputed HistoQC results that help skip low-quality
   slides.
* ``--seg-thumbsize``, ``--seg-median-filter-size``, ``--seg-binary-threshold``,
   ``--seg-closing-kernel-size``, ``--seg-min-object-size-um2``, and
   ``--seg-min-hole-size-um2`` tune the morphological pipeline.
* ``--patch-overlap-ratio`` plus ``--patch-size-um`` / ``--patch-size-px`` define how the
   patch grid is created relative to the model defaults.
* ``--cache-image-patches`` extracts HDF5 patch files during the ``patch`` stage so future
   ``infer`` runs can re-use them without touching the WSIs again.

Remote data sources and caching
-------------------------------

All CLI commands accept the same URI-aware options:

* ``--wsi-dir`` may point to local folders, ``s3://bucket/prefix`` paths,
  ``gdc-manifest://`` manifests, or ``image-list:///path/to/filelist.txt`` URIs.
  An ``image-list://`` URI references a plain text file listing one slide path per
  line (blank lines and ``#`` comments are ignored).  When ``--wsi-dir`` is a plain
  local text file it is automatically coerced to ``image-list://``.
  GDC manifests stream WSIs through the built-in cache.
* ``--results-dir`` (and the derived GeoJSON/OME outputs) may target local disks or S3
   buckets. Remote destinations do **not** need to exist beforehand; they are created as
   needed.

Environment variables tune remote behavior:

* ``S3_STORAGE_OPTIONS`` — JSON blob passed to ``fsspec`` (e.g., ``{"profile": "research"}``).
* ``WSINSIGHT_REMOTE_CACHE_DIR`` — directory where remote assets are materialized. Default
   is ``~/.cache/wsinsight``; point it at a fast SSD for large batches.
* ``WSINFER_ZOO_REGISTRY_PATH`` — optional override for the model registry JSON if you
   mirror the zoo to local/S3 storage.

With these options, it is common to read WSIs from S3, spill temporary files to a local
scratch volume, and write final GeoJSON/OME-CSV artifacts back to another S3 bucket
without any code changes.


Multi-GPU parallel runs with tmux
----------------------------------

For large cohorts on multi-GPU nodes, split the slide list into per-GPU shards and
launch one ``wsinsight run`` per GPU inside a tmux session.

1. **Split slides into shards** (one file per GPU)::

       split -n l/8 --numeric-suffixes=0 --additional-suffix=.txt \
         slides_all.txt datasets/slides_part_

2. **Launch a tmux session** with one pane per GPU.  The repository includes a ready-made
   script for 8-GPU nodes::

       bash tmux-multi-gpu.sh

   Each pane pins a single GPU via ``CUDA_VISIBLE_DEVICES`` and processes its own shard.
   All panes write to the same ``--results-dir``, so outputs merge automatically.

3. **Finalize analytics** once all panes have finished::

       wsinsight hplot-finalize --results-dir results/

.. tip::

   Adapt the number of panes and GPU assignments to your hardware.  For 4 GPUs, use a
   2×2 grid; for 2 GPUs, a simple horizontal split suffices.  See
   ``tmux-multi-gpu.sh`` in the repository root for the full 8-GPU layout.


Region registration
-------------------

``wsinsight reg`` enriches existing object-level CSV outputs with region-level class
probabilities derived from a separate region-based inference run. This is equivalent to
running ``wsinsight infer`` with ``--region-inference-dir``, but operates on
already-completed runs without repeating inference.

Required options:

* ``-o / --results-dir`` — directory of the prior object-based run.  Must contain a
  ``model-outputs-csv/`` subfolder.
* ``-r / --region-inference-dir`` — directory of the prior region-level run.  Must
  contain its own ``model-outputs-csv/`` subfolder.

Spatial matching assigns each detected object to its enclosing region polygon.  The
output CSVs gain ``region_prob_*`` columns for every class in the region model.

Optional options:

* ``-i / --wsi-dir`` — when supplied, the slide list is derived from filenames in this
  directory (images are not opened).  This mirrors the shard-enumeration behaviour of
  ``run`` / ``infer`` / ``hplot``.
* ``--overwrite`` — by default, any slide whose object CSV already contains
  ``region_*`` columns is skipped with a warning.  Pass ``--overwrite`` to
  unconditionally overwrite those columns.
* ``--geojson`` / ``--omecsv`` — export the object CSVs to GeoJSON or OME-CSV after
  registration.
* ``--export-workers`` (default 4) — worker processes for the export step.

Example::

    wsinsight reg \
        --results-dir results/ \
        --region-inference-dir results-region/ \
        --overwrite \
        --geojson


H-plot analysis
---------------

``wsinsight hplot`` computes H-plot spatial metrics from existing cell-detection
inference outputs inside ``--results-dir``.  It builds a proximity graph over detected
cells, identifies tumour-core regions, and calculates layer-wise abundance profiles for
the requested cell types.  The Delaunay triangulation is cached in
``graphs/<slide>.h5`` and shared with ``ncomp`` (see
:ref:`Neighborhood composition <Neighborhood composition>` for details).

Required options:

* ``-i / --wsi-dir`` — slide directory (used for slide enumeration; images are not
  opened).
* ``-o / --results-dir`` — directory containing a ``model-outputs-csv/`` subfolder from
  a prior object-based inference run.
* ``--hplot-base-types`` — comma-separated base cell type(s) that define tumour clusters
  (e.g. ``tumor``).
* ``--hplot-target-types`` — comma-separated target cell type(s) for the layer-wise
  proportion computation (e.g. ``lymphocyte``).

Tuning options:

* ``--hplot-max-neighbor-distance`` (default 25.0 µm) — maximum distance to a
  neighbouring cell when constructing the proximity graph.
* ``--hplot-k`` (default 2) — maximum edge distance (graph hops) defining a cell's
  neighbourhood.
* ``--hplot-n`` (default 8) — minimum neighbourhood size required for a cell to be
  included in tumour-region determination.
* ``--hplot-r`` (default 0.5) — minimum fraction of base-type cells in a cell's
  neighbourhood for that cell to be counted as part of a tumour region.
* ``--hplot-range-max`` — maximum layer index outward from the tumour boundary to
  include in the range window.
* ``--hplot-range-min`` — minimum layer index inward into the tumour to include.
* ``--hplot-samples-with-valid-range-only`` — restrict H-plot computation to slides
  that have cells at every layer within the range window.
* ``--overwrite`` — overwrite existing per-slide H-plot outputs instead of
  skipping slides that already have results.
* ``--num-workers`` (default 8) — number of slides to process concurrently.

Example::

    wsinsight hplot \
        --wsi-dir slides/ \
        --results-dir results/ \
        --hplot-base-types tumor \
        --hplot-target-types lymphocyte \
        --hplot-k 2 \
        --hplot-n 8 \
        --hplot-r 0.5 \
        --num-workers 16


H-plot finalization
-------------------

``wsinsight hplot-finalize`` aggregates per-slide H-plot intermediates that were written
by one or more ``hplot`` jobs into a single cohort-level summary.  Run this command once
after all parallel ``hplot`` workers have finished:

* ``-o / --results-dir`` (required) — the shared ``--results-dir`` used by the ``hplot``
  jobs.  The command reads per-slide H-plot files and writes two files into this
  directory:

  * ``hplot-outputs.csv`` — cohort-level H-plot profiles
  * ``hmetrics-outputs.csv`` — cohort-level H-metric summary statistics

* ``--overwrite`` — overwrite the aggregated CSVs if they already exist.

Example::

    wsinsight hplot-finalize \
        --results-dir results/ \
        --overwrite


Neighborhood composition
------------------------

``wsinsight ncomp`` computes per-cell neighborhood composition from existing
cell-detection inference outputs.  For each target cell (or every cell when no
target types are given), it builds a Delaunay proximity graph, computes k-hop
neighbors, and records the cell-type composition of each cell's local neighborhood.

Both ``hplot`` and ``ncomp`` cache the Delaunay triangulation in
``graphs/<slide>.h5`` under the results directory.  The first command to run
(whichever executes first) writes the cache; subsequent commands reuse it, skipping
the expensive ``scipy.spatial.Delaunay`` computation.  The cache stores unpruned
edges so that different ``--max-neighbor-distance`` values are served from the same
file.  If the underlying inference outputs change, the cache is automatically
invalidated via a SHA-256 hash of the cell centres.

The same analysis can be run inline via ``wsinsight run --ncomp``.

Required options:

* ``-i / --wsi-dir`` — slide directory (used for slide enumeration and µm-per-pixel
  spacing; images are not fully read).
* ``-o / --results-dir`` — directory containing a ``model-outputs-csv/`` subfolder from
  a prior inference run.

Tuning options:

* ``--ncomp-max-neighbor-distance`` (default 25.0 µm) — maximum Delaunay edge length.
* ``--ncomp-k`` (default 2) — k-hop neighborhood radius.
* ``--overwrite`` — overwrite existing per-slide ncomp outputs.
* ``--num-workers`` (default 8) — number of slides to process concurrently.

Example::

    wsinsight ncomp \
        --wsi-dir slides/ \
        --results-dir results/ \
        --ncomp-k 2 \
        --num-workers 16


Export outputs
--------------

The ``export-csv/`` directory contains merged per-cell CSVs that left-join
all available analysis outputs into a single file per slide.  It combines columns from
``model-outputs-csv/`` (base inference + region registration), ``hplot-outputs-csv/cells/``
(H-plot per-cell features), and ``ncomp-outputs-csv/`` (neighborhood composition) on
shared geometry keys (``minx``/``miny`` and ``center_x``/``center_y``).

This can be produced programmatically via ``wsinsight.export_helpers.build_export_csvs()``.


Inference performance tuning
----------------------------

``wsinsight infer`` (and ``wsinsight run``, which calls ``infer`` internally) expose
several knobs for throughput:

* ``-b / --batch-size`` (default 32) — batch size for model inference.  Increase for
  multi-GPU setups.
* ``-n / --num-workers`` (default: auto) — number of dataloader workers feeding patches
  to PyTorch.  The default heuristic is ``min(2 × GPU count, CPU count)``.
* ``--export-workers`` (default: auto) — worker processes for GeoJSON / OME-CSV export.
  The default reserves headroom for the OS and inference.
* ``--stitch-workers`` (default: auto) — thread pool size for TileFuse object-based
  detection stitching.  Default: ``min(8, CPU // 2)``.


Output structure
----------------

::

   results/
   ├── masks/                  # tissue masks with contours
   ├── model-outputs-csv/      # per-patch and per-cell classification tables
   ├── model-outputs-geojson/  # GeoJSON from wsinsight reg --geojson
   ├── model-outputs-omecsv/   # OME-CSV from wsinsight reg --omecsv
   ├── patches/                # HDF5 with patch coordinates
   ├── hplot-outputs-csv/      # per-slide H-plot intermediates
   ├── hplot-outputs.csv       # cohort-level H-plot summary (after hplot-finalize)
   ├── hmetrics-outputs.csv    # cohort-level H-metrics summary (after hplot-finalize)
   ├── ncomp-outputs-csv/      # per-cell neighborhood composition
   ├── graphs/                 # cached Delaunay triangulations (HDF5, shared by hplot/ncomp)
   ├── export-csv/             # merged per-cell CSV (inference + hplot + ncomp)
   ├── export-csv/             # merged per-cell CSV (wsinsight export)
   ├── export-geojson/         # GeoJSON export (wsinsight export --geojson)
   ├── export-omecsv/          # OME-CSV export (wsinsight export --omecsv)
   └── run_metadata_*.json     # configuration and runtime info

GeoJSON/OME outputs can be loaded into QuPath, napari, or GIS tools for spatial analysis.


Containers
----------

WSInsight can be run inside Docker or Apptainer/Singularity for reproducibility.
Prebuilt images: https://hub.docker.com/r/huangch/wsinsight/tags

Example with Docker (GPU): ::

   docker run --rm -it \
      --user $(id -u):$(id -g) \
      --mount type=bind,source=$(pwd),target=/work/ \
      --gpus all \
      huangch/wsinsight:latest run \
         --wsi-dir /work/slides/ \
         --results-dir /work/results/ \
         --model breast-tumor-singlecell.tcga-brca


Using your own model
--------------------

Custom TorchScript models are supported via JSON configuration files that follow
``wsinsight/schemas/model-config.schema.json``. Validate the JSON with any schema-aware
editor and run inference with:

::

   wsinsight run \
      --wsi-dir slides/ \
      --results-dir results/ \
      --model-path path/to/model.ts \
      --config my-config.json

Alternatively, point ``-z / --zoo-model-dir`` at a folder that already contains
``config.json`` and ``torchscript_model.pt``. This shorthand replaces the
``--config`` + ``--model-path`` pair:

::

   wsinsight run \
      --wsi-dir slides/ \
      --results-dir results/ \
      --zoo-model-dir path/to/zoo-model/


Exporting predictions
---------------------

The :code:`model-outputs-geojson/` and :code:`model-outputs-omecsv/` folders are produced
automatically when :code:`wsinsight run` completes. They can be copied directly into
QuPath projects or ingested into downstream analytics pipelines without additional CLI
steps.

For more control, use the standalone ``wsinsight export`` command.  It merges all
available per-cell analytics — base inference CSVs, H-plot cell features, and ncomp
neighborhood composition — into ``export-csv/``, then writes the merged data to
``export-geojson/`` and/or ``export-omecsv/`` depending on the flags provided.  This
command can be run at any time after inference, and optionally after ``hplot`` or
``ncomp``, without re-running the full pipeline.

At least one of ``--geojson`` or ``--omecsv`` must be supplied.

Required options:

* ``-o / --results-dir`` — results directory produced by a prior ``run`` / ``infer`` /
  ``hplot`` / ``ncomp`` invocation.  Must contain a ``model-outputs-csv/`` subfolder.

Optional options:

* ``--geojson`` — export per-cell data to GeoJSON files (``export-geojson/``).
* ``--omecsv`` — export per-cell data to compressed OME-CSV files (``export-omecsv/``).
  Compatible with QuPath and OMERO+.
* ``--patch-overlap-ratio`` (default 0.0) — overlap ratio used during inference (must
  match the original run).  Controls tile-box shrinkage in exported features.
* ``--object-type`` (default ``detection``) — QuPath object-type label embedded in each
  exported feature.  Choices: ``tile``, ``detection``, ``annotation``.
* ``--export-workers`` (default 4) — worker processes for parallel serialisation.
* ``--overwrite`` — re-build export CSVs even when ``export-csv/`` already contains
  up-to-date files.  Useful after re-running ``hplot`` or ``ncomp``.

Example::

    wsinsight export \
        --results-dir results/ \
        --geojson \
        --omecsv \
        --export-workers 8
