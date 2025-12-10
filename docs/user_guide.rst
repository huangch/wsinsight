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

Three commands are available today:

=================  ================================================================
Command            Purpose
=================  ================================================================
``wsinsight run``  Convenience wrapper that extracts patches then runs inference/exports.
``wsinsight patch``  Generate tissue masks + patch caches inside ``--results-dir``.
``wsinsight infer``  Reuse cached patches to run models and emit GeoJSON/OME exports.
=================  ================================================================

Pick ``run`` for one-shot processing. Switch to the explicit ``patch`` → ``infer`` flow
for large cohorts, resumable jobs, or when you want to reuse the same patches across
multiple model configurations. All commands share the same URI-aware options and support
local folders, ``s3://`` buckets, and ``gdc-manifest://`` manifests.


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

* ``--wsi-dir`` may point to local folders, ``s3://bucket/prefix`` paths, or
   ``gdc-manifest://`` manifests. GDC manifests stream WSIs through the built-in cache.
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


Output structure
----------------

Each inference run produces deterministic directories inside :code:`--results-dir`:

::

   results/
   ├── masks/                  # tissue masks with contours
   ├── model-outputs-csv/      # per-patch and per-cell classification tables
   ├── model-outputs-geojson/  # spatial annotations for QuPath/Geo viewers
   ├── model-outputs-omecsv/   # OME-compatible CSV exports (gzip)
   ├── patches/                # HDF5 with patch coordinates
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


Exporting predictions
---------------------

The :code:`model-outputs-geojson/` and :code:`model-outputs-omecsv/` folders are produced
automatically when :code:`wsinsight run` completes. They can be copied directly into
QuPath projects or ingested into downstream analytics pipelines without additional CLI
steps.
