Command reference
=================

Three CLI entry points are available today:

=================  ================================================================
Command            Purpose
=================  ================================================================
``wsinsight run``  One-shot workflow that patches slides then runs inference/exports.
``wsinsight patch``  Extract tissue masks and patches, caching them in ``--results-dir``.
``wsinsight infer``  Reuse cached patches to run models and exporters.
=================  ================================================================

Use ``run`` for simple single-machine jobs, and switch to the explicit ``patch`` → ``infer``
flow when you need to resume work, share caches across models, or process slides on
multiple nodes. All commands share the same URI-aware options for local folders,
``s3://`` buckets, and ``gdc://`` manifests.

.. click:: wsinsight.cli.cli:cli
   :prog: wsinsight
   :nested: full
   :commands: run, patch, infer
