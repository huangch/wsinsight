Command reference
=================

Eight CLI entry points are available:

============================  ================================================================
Command                       Purpose
============================  ================================================================
``wsinsight run``             One-shot workflow: tissue segmentation, patch extraction, model
                              inference, and optional spatial analytics and exports.
                              Orchestrates ``patch`` → ``infer`` → ``hplot`` → ``ncomp``
                              → ``export``.  Pass ``--hplot`` / ``--ncomp`` to enable
                              spatial analytics and ``--export-geojson`` / ``--export-omecsv``
                              to write GeoJSON / OME-CSV files at the end of the run.
``wsinsight patch``           Segment tissue and cache patches to HDF5; safe to rerun to
                              resume interrupted jobs.
``wsinsight infer``           Reuse cached patches to run models and produce per-cell CSV
                              outputs.  Supports region registration via
                              ``--region-inference-dir`` and ``--overwrite``.
                              Use standalone ``hplot``/``ncomp``/``export`` commands
                              (or ``run``) for downstream analytics.
``wsinsight reg``             Post-hoc object-to-region registration on already-completed
                              runs.  Enriches object CSVs with ``region_prob_*`` columns from
                              a prior region-level run.  Use ``--overwrite`` to replace
                              existing region columns.
``wsinsight hplot``           Standalone H-plot analysis on existing object-based inference
                              outputs.  Requires both ``--hplot-base-types`` and
                              ``--hplot-target-types``.
``wsinsight hplot-finalize``  Aggregate per-slide H-plot intermediates into cohort-level
                              ``hplot-outputs.csv`` and ``hmetrics-outputs.csv``.  Run this
                              after parallel ``hplot`` jobs that share an output directory.
``wsinsight ncomp``           Neighborhood composition analysis on existing inference outputs.
                              For each target cell, builds a Delaunay graph, collects k-hop
                              neighbors, and records per-cell type counts and proportions.
                              Can also run inline via ``wsinsight run --ncomp``.
``wsinsight export``          Merge all per-cell analytics (inference, H-plot, ncomp) into
                              ``export-csv/`` and write GeoJSON and/or OME-CSV files.  Can
                              be run after inference — and optionally after ``hplot`` /
                              ``ncomp`` — without repeating the full pipeline.
============================  ================================================================

Use ``run`` for simple single-machine jobs, and switch to the explicit ``patch`` → ``infer``
flow when you need to resume work, share caches across models, or process slides on
multiple nodes.  ``run`` is the only command that orchestrates all stages — ``infer``
focuses solely on model inference.  Use the standalone ``hplot`` and ``ncomp`` commands
to re-run analytics on existing inference outputs without repeating inference.  Run
``hplot-finalize`` to assemble the cohort-level summary after parallel ``hplot`` jobs.  Use
``reg`` to enrich earlier runs with region-level probabilities without re-running inference.
All commands share the same URI-aware options for local folders, ``s3://`` buckets,
``gdc-manifest://`` manifests, and ``image-list://`` file lists (a text file with one
slide path per line; blank lines and ``#`` comments are ignored).  When ``--wsi-dir``
points to a plain local text file it is automatically coerced to ``image-list://``.


Output file formats
-------------------

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
     - Parent annotation name — only with ``--qupath-detection-dir``
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


``hmetrics-outputs.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~

Per-slide spatial interaction metrics.  One row per slide.

Columns: ``id``, ``valid``,
``convergence_distance (intra)``, ``abundance_score (intra)``, ``penetration_score (intra)``,
``layerwise_enrichment_index (intra)``, ``global_enrichment_index (intra)``,
``weighted_global_enrichment_index (intra)``,
``convergence_distance (peri)``, ``abundance_score (peri)``, ``proximity_score (peri)``,
``layerwise_enrichment_index (peri)``, ``global_enrichment_index (peri)``,
``weighted_global_enrichment_index (peri)``,
``exclusion_index``, ``desert_index``, ``inflammation_index``,
``layerwise_enrichment_index``, ``global_enrichment_index``,
``weighted_global_enrichment_index``


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


``export-csv/<slide>.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Merged per-cell CSV produced by ``wsinsight export``.  Left-joins the base
inference CSV, H-plot cell features, and ncomp neighborhood data on shared
geometry keys.

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

Run inference + H-plot + ncomp in a single command::

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
      --ncomp

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
   :commands: run, patch, infer, export, hplot, hplot_finalize_cmd, reg, ncomp
