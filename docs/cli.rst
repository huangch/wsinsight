Command reference
=================

Six CLI entry points are available:

=========================  ================================================================
Command                    Purpose
=========================  ================================================================
``wsinsight run``          One-shot workflow that patches slides then runs inference/exports.
``wsinsight patch``        Extract tissue masks and patches, caching them in ``--results-dir``.
``wsinsight infer``        Reuse cached patches to run models and exporters. Supports inline
                           region registration via ``--region-inference-dir`` and
                           ``--reg-overwrite``.
``wsinsight reg``          Post-hoc object-to-region registration on already-completed runs.
                           Enriches object CSVs with ``region_prob_*`` columns from a prior
                           region-level run. Use ``--reg-overwrite`` to replace existing
                           region columns.
``wsinsight hplot``        Standalone H-Plot analysis on existing object-based inference
                           outputs. Requires both ``--hplot-base-types`` and
                           ``--hplot-target-types``.
``wsinsight hplot-finalize``  Aggregate per-slide H-Plot intermediates into cohort-level
                           ``hplot-outputs.csv`` and ``hmetrics-outputs.csv``. Run this after
                           parallel ``hplot`` jobs that share an output directory.
=========================  ================================================================

Use ``run`` for simple single-machine jobs, and switch to the explicit ``patch`` → ``infer``
flow when you need to resume work, share caches across models, or process slides on
multiple nodes. Run ``hplot`` on completed cell-detection outputs to compute spatial
tumour-microenvironment metrics and ``hplot-finalize`` to assemble the cohort-level summary.
Use ``reg`` to enrich earlier runs with region-level probabilities without re-running
inference. All commands share the same URI-aware options for local folders,
``s3://`` buckets, and ``gdc-manifest://`` manifests.

.. click:: wsinsight.cli.cli:cli
   :prog: wsinsight
   :nested: full
   :commands: run, patch, infer, hplot, hplot_finalize_cmd, reg
