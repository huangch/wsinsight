.. _cli:

Command reference
=================

.. note::

   **Experimental commands.** ``hplot``, ``hplot-finalize``, ``cme``,
   ``cme-profile``, ``ecomp``, ``tcomp``, and ``import`` (together with their
   ``hplot-outputs.csv`` / ``ecomp-outputs-csv/`` / ``tcomp-outputs-csv/``
   outputs) are research features under active development.  Their CLI flags,
   output directory layouts, and column schemas may change without notice in
   future releases.  They are hidden from ``wsinsight --help`` and refuse to
   run unless the environment variable ``WSINSIGHT_EXPERIMENTAL=1`` is set.
   ``wsinsight describe`` always emits the full schema so downstream tools
   (the QuPath extension) can discover every command; only invocation is
   gated.

Stable commands
---------------

Six CLI entry points are available by default:

============================  ================================================================
Command                       Purpose
============================  ================================================================
``wsinsight run``             One-shot workflow: tissue segmentation, patch extraction, model
                              inference, and optional ncomp analytics and exports.
                              Orchestrates ``patch`` → ``infer`` → ``ncomp`` → ``export``.
                              Pass ``--ncomp`` to enable neighborhood composition and
                              ``--export-geojson`` / ``--export-omecsv`` to write
                              GeoJSON / OME-CSV files at the end of the run.  Experimental
                              ``--hplot`` / ``--cme`` flags require
                              ``WSINSIGHT_EXPERIMENTAL=1``.
``wsinsight patch``           Segment tissue and cache patches to HDF5. By default,
                              slides with existing patch outputs are skipped; pass
                              ``--overwrite`` to regenerate.
``wsinsight infer``           Reuse cached patches to run models and produce per-cell CSV
                              outputs. By default, slides with existing CSV outputs are
                              skipped; pass ``--overwrite`` to regenerate. Supports
                              region registration via ``--region-inference-dir``.
                              Use standalone ``ncomp``/``export`` commands (or ``run``) for
                              downstream analytics.
``wsinsight reg``             Post-hoc object-to-region registration on already-completed
                              runs.  Enriches object CSVs with ``region_prob_*`` columns from
                              a prior region-level run.  Use ``--overwrite`` to replace
                              existing region columns.
``wsinsight ncomp``           Neighborhood composition analysis on existing inference outputs.
                              For each target cell, builds a Delaunay graph, collects k-hop
                              neighbors, and records per-cell type counts and proportions.
                              Can also run inline via ``wsinsight run --ncomp``.
``wsinsight export``          Merge all per-cell analytics (inference, ncomp, and — when
                              enabled — H-plot / CME) into ``export-csv/`` and write
                              GeoJSON and/or OME-CSV files.  Can be run after inference
                              without repeating the full pipeline.
============================  ================================================================

Use ``run`` for simple single-machine jobs, and switch to the explicit
``patch`` → ``infer`` flow when you need to resume work, share caches across
models, or process slides on multiple nodes.  ``run`` is the only command
that orchestrates all stages — ``infer`` focuses solely on model inference.
Use the standalone ``ncomp`` command to re-run neighborhood analytics on
existing inference outputs without repeating inference.  Use ``reg`` to enrich
earlier runs with region-level probabilities without re-running inference.
All commands share the same URI-aware options for local folders, ``s3://``
buckets, ``gs://`` buckets, ``gdc-manifest://`` manifests, and ``image-list://``
file lists
(a text file with one slide path per line; blank lines and ``#`` comments
are ignored).  A plain local text file passed directly as ``--wsi-dir`` is
rejected — prefix it with ``image-list://`` to pass a slide list.
``sptx-list://`` is a two-column variant (``path``<TAB>``sample_id`` per line)
used by ``wsinsight import`` to carry a stable ``sample_id`` alongside each
spatial-transcriptomics sample, because transcriptomics exports (e.g. Xenium)
frequently reuse the same filename across runs.

Experimental commands
---------------------

Set ``WSINSIGHT_EXPERIMENTAL=1`` to unhide and enable these research commands.

============================  ================================================================
Command                       Purpose
============================  ================================================================
``wsinsight hplot``           Standalone H-plot analysis on existing object-based inference
                              outputs.  Requires both ``--hplot-base-types`` and
                              ``--hplot-target-types``.  Computes layer-wise cell-type
                              proportions from tumour boundary outward.  Can also run inline
                              via ``wsinsight run --hplot``.  Use ``--base-by`` /
                              ``--target-by`` (``celltype`` | ``cme``) to plot the
                              fraction of cells in a discovered niche across layers
                              instead of a cell type; CME ids may be given as ``7`` or
                              ``cme_7``.
``wsinsight hplot-finalize``  Aggregate per-slide H-plot intermediates into cohort-level
                              ``hplot-outputs.csv``.  Run this
                              after parallel ``hplot`` jobs that share an output directory.
``wsinsight ecomp``           Edge-level composition analysis.  For each Delaunay edge,
                              builds the line graph, collects k-hop edge neighbors, and
                              records per-edge type counts and proportions.  Standalone
                              command — not inlined by ``run``.
``wsinsight tcomp``           Triad-level composition analysis.  For each Delaunay triangle,
                              builds the dual graph (triads sharing ≥1 vertex), collects
                              k-hop triad neighbors, and records per-triad type counts,
                              proportions, and geometry (area, perimeter, regularity).
                              Standalone command — not inlined by ``run``.
``wsinsight cme``             Cellular microenvironment (CME) analysis across a cohort of
                              slides.  Builds per-slide Delaunay cell graphs, trains a
                              global DGI encoder, clusters the embeddings, and writes
                              per-cell CME labels plus annotation-level region merges.
                              Pass ``--export-geojson`` to also write GeoJSON files.
                              Can also run inline via ``wsinsight run --cme``.  CME is a
                              cross-slide analysis (global DGI training + global clustering)
                              and cannot be parallelized across GPU shards — run it after
                              merging all per-shard inference outputs.
``wsinsight cme-profile``     Summarise each discovered CME (niche) by its dominant cell
                              types, writing ``cme-profile-composition.csv`` under the
                              results directory.  Reads the per-cell labels from ``cme``.
                              Whole-slide H&E cohorts carry no transcriptome, so no
                              marker-gene table is produced.
``wsinsight import``          Import spatial-transcriptomics (Xenium) gene expression onto
                              WSInsight cells.  Maps each transcriptomics cell onto the
                              registered H&E via the ST2WSI (SIFT affine + bUnwarpJ B-spline)
                              transform, matches it to the nearest ``model-outputs-csv``
                              detection, and writes one AnnData ``.h5ad`` per slide under
                              ``xenium-import/`` (the ``model-outputs-csv/`` is never
                              modified).  The matched detection's columns are carried into
                              ``obs`` under a ``model_`` prefix (plus ``model_cell_id``);
                              optional per-cell sidecars added with
                              ``--include cme,hplot,ncomp`` are merged under their own
                              ``cme_`` / ``hplot_`` / ``ncomp_`` prefixes (``model`` is
                              always imported).  Reads a ``sptx-list://`` manifest via
                              ``-s`` / ``--sptx-dir``; supports
                              ``--transform affine|affine+bspline`` (default), ``--genes``,
                              ``--include``, ``--match-max-dist``, and ``--dry-run`` (report
                              the cell↔detection hit-rate only, writing nothing).
============================  ================================================================


Output file formats
-------------------

``patches/<slide>.h5``
~~~~~~~~~~~~~~~~~~~~~~

Cached patch coordinates (and optionally images) produced by ``patch`` or ``run``.
One HDF5 file per slide.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Dataset / Attribute
     - Description
   * - ``/coords`` (N, 2) int32
     - Top-left patch coordinates (x, y) at level 0
   * - ``/coords`` → ``patch_size`` (attr, int32)
     - Side length of each patch in pixels
   * - ``/coords`` → ``patch_level`` (attr, int32)
     - WSI magnification level (always 0)
   * - ``/coords`` → ``patch_spacing_um_px`` (attr, float64)
     - Microns-per-pixel used for coordinate calculation
   * - ``/coords`` → ``tile_dim`` (attr, optional, int32[2])
     - Tiling dimensions ``[width, height]`` for end-to-end models
   * - ``/images`` (optional, N×H×W×3 uint8)
     - RGB patch images (when ``--save-images`` is used)
   * - ``/polygons/coords`` (optional, K×2 float32)
     - Tissue polygon vertices (ragged array)
   * - ``/polygons/offsets`` (optional, M+1 int64)
     - Ragged array offsets: polygon *i* = ``coords[offsets[i]:offsets[i+1]]``
   * - ``/slide`` → ``slide_path`` (attr, optional)
     - Original WSI file path
   * - ``/slide`` → ``slide_mpp`` (attr, optional, float64)
     - Microns-per-pixel of the WSI
   * - ``/slide`` → ``slide_width``, ``slide_height`` (attr, optional, float64)
     - WSI dimensions in pixels


``model-outputs-csv/<slide>.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Produced by ``infer``, ``run``, and ``reg``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Description
   * - ``minx``, ``miny``
     - Top-left corner of the patch/detection bounding box (pixels)
   * - ``width``, ``height``
     - Bounding box size (pixels)
   * - ``prob_<class>``
     - Model probability for each class (e.g. ``prob_tumor``)
   * - ``qupath_detection_parent``
     - Parent annotation name — only with ``--qupath-measurement-detection-dir``
   * - ``region_minx``, ``region_miny``, ``region_width``, ``region_height``
     - Matched region bounding box — only with ``--region-inference-dir``
   * - ``region_prob_<class>``
     - Region-level class probabilities — only with ``--region-inference-dir``


``hplot-outputs-csv/hplots/<slide>.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-layer H-plot curve.  One row per integer layer index.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Description
   * - ``layer``
     - Integer layer index; 0 = border, negative = inside base region, positive = outside
   * - ``target_type_prop``, ``target_type_count``
     - Proportion and count of target cells at this layer
   * - ``base_type_prop``, ``base_type_count``
     - Proportion and count of base cells at this layer
   * - ``all_type_count``
     - Total cell count at this layer
   * - ``distance``
     - Cumulative µm distance from the base-region border


``hplot-outputs-csv/cells/<slide>.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-cell file: the original ``model-outputs-csv/<slide>.csv`` extended with spatial columns.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Description
   * - ``minx``, ``miny``, ``width``, ``height``
     - Inherited from inference output
   * - ``prob_<class>``
     - Inherited from inference output
   * - ``center_x``, ``center_y``
     - Cell centre in pixels
   * - ``is_base_type``
     - ``True`` if the cell's predicted class is a base type
   * - ``is_target_type``
     - ``True`` if the cell's predicted class is a target type
   * - ``signed_distance_to_border``
     - Hop distance to the base-region boundary; negative = inside, 0 = border,
       positive = outside, NaN = unreachable


``hplot-outputs.csv``
~~~~~~~~~~~~~~~~~~~~~~

Cohort-level H-plot curve aggregated across all slides, produced by ``hplot``,
``run --hplot``, or ``hplot-finalize``.

Columns: ``id``, ``layer``, ``target_prop``, ``target_count``, ``base_prop``,
``base_count``, ``all_count``, ``distance``


``ncomp-outputs-csv/<slide>.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-cell neighborhood composition produced by ``ncomp`` or
``run --ncomp``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Description
   * - ``center_x``, ``center_y``
     - Cell centre in pixels
   * - ``cell_type``
     - Predicted cell type (argmax of ``prob_*`` columns)
   * - ``neighborhood_size``
     - Number of k-hop graph neighbors (excluding self)
   * - ``neighborhood_<class>_count``
     - Count of neighbors of each class; one column per model class
   * - ``neighborhood_<class>_prop``
     - Proportion of neighbors of each class; one column per model class


``cme-outputs-csv/cells/<slide>.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-cell CME labels and features produced by ``cme`` or ``run --cme``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Description
   * - All columns from ``model-outputs-csv``
     - Inherited inference + region columns
   * - ``cme_cluster``
     - Integer cluster label assigned by KMeans (or Leiden-derived k)
   * - ``feature_normalized_*``
     - Normalized DGI embedding features (one column per dimension)
   * - ``feature_raw_*``
     - Raw DGI embedding features (one column per dimension)


``cme-outputs-csv/cmes/<slide>.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Annotation-level merged CME regions produced by ``cme`` or ``run --cme``.
Adjacent cells sharing the same ``cme_cluster`` are dissolved into contiguous
polygonal regions.


``graphs/<slide>.h5``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cached Delaunay triangulation shared by ``hplot``, ``ncomp``, and ``cme``.  Created on
the first run and reused on subsequent runs to skip the expensive
``scipy.spatial.Delaunay`` computation.  The cache stores **unpruned** edges;
each command applies its own distance threshold at load time.

The file is automatically invalidated and rebuilt when the underlying
``model-outputs-csv/<slide>.csv`` changes (detected via cell count and a
SHA-256 hash of cell centres).

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Dataset / Attribute
     - Description
   * - ``num_cells`` (attr)
     - Row count — fast staleness check
   * - ``mpp`` (attr)
     - Microns-per-pixel used for cell centre computation
   * - ``centers_hash`` (attr)
     - SHA-256 of ``cell_centers`` bytes — bulletproof staleness check
   * - ``cell_centers`` (N, 2) int32
     - Cell centres (``center_x``, ``center_y``), row-aligned with the CSV
   * - ``simplices`` (M, 3) int32
     - Raw Delaunay triangles (3 vertex indices each)
   * - ``edges_source`` (E,) int32
     - Unique undirected edges — source vertex
   * - ``edges_target`` (E,) int32
     - Unique undirected edges — target vertex
   * - ``edges_length`` (E,) float64
     - Euclidean edge length in pixels


``export-csv/<slide>.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Merged per-cell CSV produced by ``wsinsight export``.  Left-joins the base
inference CSV, H-plot cell features, ncomp neighborhood data, and CME
labels/features on shared geometry keys.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Description
   * - All columns from ``model-outputs-csv/<slide>.csv``
     - Inherited inference + region columns
   * - ``center_x``, ``center_y``
     - Cell centre (added if absent)
   * - ``is_base_type``, ``is_target_type``
     - From H-plot cells output (when available)
   * - ``signed_distance_to_border``
     - From H-plot cells output (when available)
   * - ``cell_type``, ``neighborhood_size``
     - From ncomp output (when available)
   * - ``neighborhood_<class>_count``, ``neighborhood_<class>_prop``
     - From ncomp output (when available)
   * - ``cme_*``
     - From CME cell output (when available)
   * - ``feature_normalized_*``, ``feature_raw_*``
     - From CME cell output (when available)


Key parameters
--------------

H-plot (``--hplot-*`` in ``run`` and ``wsinsight hplot``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Description
   * - ``--hplot-base-types``
     - required
     - Comma-separated base cell types defining the tumour cluster (e.g. ``tumor``)
   * - ``--hplot-target-types``
     - required
     - Comma-separated target cell types to track across layers (e.g. ``lymphocyte``)
   * - ``--hplot-max-neighbor-distance``
     - ``25.0``
     - Maximum Delaunay edge length in µm
   * - ``--hplot-k``
     - ``2``
     - k-hop neighborhood radius for base-region detection
   * - ``--hplot-n``
     - ``8``
     - Minimum neighborhood size for base-region membership
   * - ``--hplot-r``
     - ``0.5``
     - Minimum base-type fraction for base-region membership
   * - ``--hplot-range-min``
     - ``None``
     - Innermost layer index (≤ 0) included in metrics
   * - ``--hplot-range-max``
     - ``None``
     - Outermost layer index (≥ 1) included in metrics
   * - ``--hplot-samples-with-valid-range-only``
     - off
     - Exclude slides that do not cover the full ``[range-min, range-max]`` window
   * - ``--overwrite``
     - off
     - Recompute and overwrite existing per-slide H-plot outputs

Neighborhood composition (``--ncomp-*`` in ``run`` and ``wsinsight ncomp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Description
   * - ``--ncomp-max-neighbor-distance``
     - ``25.0``
     - Maximum Delaunay edge length in µm
   * - ``--ncomp-k``
     - ``2``
     - k-hop neighborhood radius
   * - ``--overwrite``
     - off
     - Recompute and overwrite existing per-slide ncomp outputs


Simplicial composition hierarchy — ``ncomp`` / ``ecomp`` / ``tcomp``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WSInsight's three composition commands form a symmetric simplicial hierarchy
on the Delaunay triangulation:

.. list-table::
   :header-rows: 1
   :widths: 10 15 20 20 15 20

   * - Command
     - Simplex
     - Unit
     - Adjacency
     - Graph
     - Output dir
   * - ``ncomp``
     - 0-simplex (n)
     - cell
     - Delaunay edge
     - primal
     - ``ncomp-outputs-csv/``
   * - ``ecomp``
     - 1-simplex (e)
     - Delaunay edge
     - shared vertex
     - line graph
     - ``ecomp-outputs-csv/``
   * - ``tcomp``
     - 2-simplex (t)
     - Delaunay triad (triangle)
     - shared vertex
     - dual graph
     - ``tcomp-outputs-csv/``

All three commands share the Delaunay cache (``graphs/<slide>.h5``), a 25 µm
default edge filter, and a 2-hop default neighborhood radius.  Only ``tcomp``
emits per-triad geometry (area µm², perimeter µm, regularity ∈ [0, 1] where
1.0 is equilateral).  Edge- and triad-level outputs are standalone and are
**not** merged into ``export-csv/`` (they have different primary keys).

Edge composition (``--ecomp-*`` in ``run`` and ``wsinsight ecomp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Description
   * - ``--ecomp-max-edge``
     - ``25.0``
     - Maximum Delaunay edge length in µm
   * - ``--ecomp-k``
     - ``2``
     - k-hop neighborhood radius on the line graph
   * - ``--overwrite``
     - off
     - Recompute and overwrite existing per-slide ecomp outputs

Triad composition (``--tcomp-*`` in ``run`` and ``wsinsight tcomp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Description
   * - ``--tcomp-max-edge``
     - ``25.0``
     - Longest-edge threshold (µm); triads with any edge above this are pruned
   * - ``--tcomp-k``
     - ``2``
     - k-hop neighborhood radius on the dual graph
   * - ``--overwrite``
     - off
     - Recompute and overwrite existing per-slide tcomp outputs


Model selection
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Option
     - Applies to
     - Description
   * - ``-m / --model``
     - ``run``, ``patch``, ``infer``
     - Registered model from the WSInsight / WSInfer Model Zoo. Mutually exclusive
       with ``--config``, ``--model-path``, and ``--zoo-model-dir``.
   * - ``-c / --config``
     - ``run``, ``patch``, ``infer``
     - Path to a custom JSON model configuration (schema:
       ``wsinsight/schemas/model-config.schema.json``). Must be paired with
       ``--model-path``. Mutually exclusive with ``--model`` and ``--zoo-model-dir``.
   * - ``-p / --model-path``
     - ``run``, ``patch``, ``infer``
     - TorchScript weights file. Required when ``--config`` is used.
       Mutually exclusive with ``--model`` and ``--zoo-model-dir``.
   * - ``-z / --zoo-model-dir``
     - ``run``, ``patch``, ``infer``
     - Folder containing ``config.json`` and ``torchscript_model.pt``.
       Shorthand for ``--config`` + ``--model-path``.  Mutually exclusive with
       ``--model``, ``--config``, and ``--model-path``.


Inference performance
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Option
     - Default
     - Applies to
     - Description
   * - ``-b / --batch-size``
     - ``32``
     - ``run``, ``infer``
     - Batch size for model inference.  Increase for multi-GPU setups.
   * - ``-n / --num-workers``
     - auto
     - ``run``, ``infer``
     - Dataloader workers feeding patches to PyTorch.  Default heuristic:
       ``min(2 × GPU count, CPU count)``.
   * - ``--export-workers``
     - auto
     - ``infer``
     - Worker processes for GeoJSON / OME-CSV export.  Default reserves headroom
       for inference.
   * - ``--stitch-workers``
     - auto
     - ``infer``
     - Thread pool size for TileFuse object-based detection stitching.  Default:
       ``min(8, CPU // 2)``.


Example workflows
-----------------

Run inference + H-plot + ncomp + CME + export in a single command::

    wsinsight run \
      --wsi-dir slides/ \
      --results-dir results/ \
      --model pancancer-lymphocytes-inceptionv4.tcga \
      --batch-size 32 \
      --hplot \
      --hplot-base-types tumor \
      --hplot-target-types lymphocyte \
      --hplot-range-min -5 \
      --hplot-range-max 5 \
      --ncomp \
      --cme \
      --export-geojson \
      --export-omecsv

Run H-plot on existing inference outputs::

    wsinsight hplot \
      --wsi-dir slides/ \
      --results-dir results/ \
      --hplot-base-types tumor \
      --hplot-target-types lymphocyte \
      --hplot-range-min -5 \
      --hplot-range-max 5

Aggregate H-plot results after parallel per-slide runs::

    wsinsight hplot-finalize --results-dir results/

Run neighborhood composition on existing inference outputs::

    wsinsight ncomp \
      --wsi-dir slides/ \
      --results-dir results/ \
      --ncomp-k 2

Enrich object CSVs with region probabilities post-hoc::

    wsinsight reg \
      --results-dir results-cellvit/ \
      --region-inference-dir results-region/ \
      --overwrite

Export merged analytics to GeoJSON / OME-CSV::

    wsinsight export \
      --results-dir results/ \
      --geojson \
      --omecsv \
      --export-workers 8


.. click:: wsinsight.cli.cli:cli
   :prog: wsinsight
   :nested: full
   :commands: run, patch, infer, export, hplot, hplot_finalize_cmd, reg, ncomp, ecomp, tcomp, cme, sptx_import
