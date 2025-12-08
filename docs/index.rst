:html_theme.sidebar_secondary.remove:

.. image:: _static/logo.svg
  :alt: WSInsight logo
  :align: left
  :height: 48px

WSInsight: Cloud-Native Single-Cell Pathology Inference on Whole Slide Images
=============================================================================

.. image:: https://github.com/huangch/wsinsight/actions/workflows/ci.yml/badge.svg
  :alt: GitHub workflow status
  :target: https://github.com/huangch/wsinsight/actions/workflows/ci.yml
.. image:: https://readthedocs.org/projects/wsinsight/badge/?version=latest
  :alt: Documentation build status
  :target: https://wsinsight.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/pypi/v/wsinsight.svg
  :alt: PyPI version
  :target: https://pypi.org/project/wsinsight/
.. image:: https://img.shields.io/pypi/pyversions/wsinsight
  :alt: Supported Python versions
  :target: https://pypi.org/project/wsinsight/

|

WSInsight is a fork of `WSInfer <https://github.com/SBU-BMI/wsinfer>`_ that delivers
end-to-end pathology inference for giga-pixel whole slide images. It scales from laptops
to cloud clusters, orchestrates patch extraction/classification, cell detection/
classification, model inference, and downstream analytics, and produces artifacts that
can be explored in QuPath, GeoJSON-aware viewers, OMERO+, or bespoke notebooks.

.. caution::

  WSInsight is a research tool. It is not cleared for clinical workflows or
  patient-facing decisions.

.. image:: _static/diagram.drawio.png
  :alt: WSInsight workflow overview
  :align: center
  :class: workflow-diagram

Highlights
----------

- GPU-accelerated inference for registered models from the WSInfer Model Zoo or
  custom TorchScript weights
- Automated tissue segmentation, patch extraction, and batched inference with
  resumable runs
- First-class support for QuPath projects, GeoJSON/OME-CSV exports, and remote
  slides (S3, GDC manifests)
- Transparent URI handling lets you read WSIs from local disks, S3 buckets, or
  GDC manifests and write inference outputs back to either local paths or S3
  using the same CLI options
- Built for reproducibility: metadata capture, deterministic configuration, and
  container-friendly execution

Visual overview
---------------

.. |img-tissue| image:: _static/brca-tissue.png
  :alt: TCGA BRCA sample slide
.. |img-heatmap-tumor| image:: _static/brca-heatmap-neoplastic.png
  :alt: Heatmap of tumor probability
.. |img-heatmap-dead| image:: _static/brca-heatmap-dead.png
  :alt: Heatmap of dead cell probability
.. |img-heatmap-connective| image:: _static/brca-heatmap-connective.png
  :alt: Heatmap of connective cell probability

+------------------------+------------------------------+
| Original H&E           | Heatmap of Tumor Probability |
+========================+==============================+
| |img-tissue|           | |img-heatmap-tumor|          |
+------------------------+------------------------------+
| Heatmap of Dead Cells  | Heatmap of Connective Cells  |
+------------------------+------------------------------+
| |img-heatmap-dead|     | |img-heatmap-connective|     |
+------------------------+------------------------------+

Integrative patch-level and single-cell inference
-------------------------------------------------

Original H&E ROI

.. image:: _static/roi-hne.png
  :alt: Original H&E ROI
  :align: center
  :scale: 50 %

.. |img-roi-lympho-immune| image:: _static/roi-lympho-immune.png
  :alt: Immune cells in lymphoid regions
.. |img-roi-lympho-neoplastic| image:: _static/roi-lympho-neoplastic.png
  :alt: Neoplastic cells in lymphoid regions
.. |img-roi-tumor-immune| image:: _static/roi-tumor-immune.png
  :alt: Immune cells in tumor regions
.. |img-roi-tumor-neoplastic| image:: _static/roi-tumor-neoplastic.png
  :alt: Neoplastic cells in tumor regions

+----------------------------------------------+-----------------------------------------------+
| Immune Cells / Lympho Regions                | Neoplastic Cells / Lympho Regions             |
+==============================================+===============================================+
| |img-roi-lympho-immune|                      | |img-roi-lympho-neoplastic|                   |
+----------------------------------------------+-----------------------------------------------+
| Immune Cells / Tumor Regions                 | Neoplastic Cells / Tumor Regions              |
+----------------------------------------------+-----------------------------------------------+
| |img-roi-tumor-immune|                       | |img-roi-tumor-neoplastic|                    |
+----------------------------------------------+-----------------------------------------------+

Quick start
-----------

Running inference on whole slide images is done with a single command-line
entry point:

::

  wsinsight run \
    --wsi-dir slides/ \
    --results-dir results/ \
    --model breast-tumor-resnet34.tcga-brca \
    --batch-size 32 \
    --num-workers 4

Prefer a staged workflow for large cohorts? Use ``wsinsight patch`` to extract
patches and cache intermediate HDF5 files, then ``wsinsight infer`` to reuse
that cache for one or multiple models. The ``wsinsight run`` command simply
orchestrates those two steps back to back for convenience.

WSInsight accepts both WSInfer-compatible model IDs and WSInsight-native
models. List the registered WSInfer identifiers with ::

  wsinfer-zoo ls

To discover the WSInsight-native CellViT/HoverNet variants, see
:ref:`available-models`.

WSInsight-native workflow (CellViT models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WSInsight adds cell-centric Vision Transformer and HoverNet variants that are
not part of upstream WSInfer. To run them:

1. Stage your WSIs as before and ensure the conda environment includes the
  CellViT dependencies (installed automatically via the standard
  installation instructions).
2. Pick one of the WSInsight-native model identifiers (see list in
  :ref:`available-models`) from the registry.
3. Launch inference, for example with ``CellViT-SAM-H-x40``:

  ::

    wsinsight run \
     --wsi-dir slides/ \
     --results-dir results-cellvit/ \
     --model CellViT-SAM-H-x40 \
     --batch-size 16 \
     --num-workers 8

4. Review the outputs in ``results-cellvit/model-outputs-*`` and downstream
  GeoJSON artifacts just like the compatible workflow.

All commands understand local filesystem paths, ``s3://`` URIs, and
``gdc://`` manifests for ``--wsi-dir``. Outputs such as ``--results-dir``,
GeoJSON, and OME-CSV artifacts can be written to local disks or S3 using the
same URI syntax, with caching controlled via ``WSINSIGHT_REMOTE_CACHE_DIR``.

To get started, please :ref:`install WSInsight<installing>` and check out the
:ref:`User Guide`. To get help, report issues or request features, please
`submit a new issue <https://github.com/huangch/wsinsight/issues/new>`_ on our
GitHub repository. If you would like to make your patch classification model
available in WSInsight, please get in touch with us! You can `submit a new
GitHub issue <https://github.com/SBU-BMI/wsinsight/issues/new>`_.

.. admonition:: Citation

  If you find our work useful, please cite our work in npj Precision Oncology
  and the original WSInfer paper.

  Huang, C.H., Awosika, O.E., Fernandez, D. WSInsight as a cloud-native
  pipeline for single-cell pathology inference on whole-slide images.

  Kaczmarzyk, J.R., O'Callaghan, A., Inglis, F. et al. Open and reusable deep
  learning for pathology with WSInfer and QuPath. *npj Precis. Onc.* **8**, 9
  (2024). https://doi.org/10.1038/s41698-024-00499-9

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  Installing <installing>
  User Guide <user_guide>
  CLI <cli>

.. _available-models:

Available models
----------------

After installing :code:`wsinsight`, use the following command to list the most up-to-date
WSInfer-compatible models: ::

  wsinfer-zoo ls

WSInsight also publishes native CellViT/HoverNet identifiers that appear only in our
registry. A non-exhaustive list is provided below.

.. list-table::
   :header-rows: 1

   * - Classification task
     - Output classes
     - Architecture
     - Dataset
     - Resolution (px @ um/px)
     - Reference
   * - Breast adenocarcinoma detection
     - no-tumor, tumor
     - ResNet34
     - TCGA BRCA
     - 350 @ 0.25
     - `Ref <https://doi.org/10.1016%2Fj.ajpath.2020.03.012>`_
   * - Colorectal tissue classification
     - background, normal_colon_mucosa, debris, colorectal_adenocarcinoma_epithelium, adipose, mucus, smooth_muscle, cancer_associated_stroma, lymphocytes
     - ResNet50 (trained by TIAToolbox dev team)
     - NCT-CRC-HE-100K
     - 224 @ 0.5
     - `Ref <https://doi.org/10.1038/s43856-022-00186-5>`_
   * - Lung adenocarcinoma detection
     - lepidic, benign, acinar, micropapillary, mucinous, solid
     - ResNet34
     - TCGA LUAD
     - 350 @ 0.5
     - `Ref <https://github.com/SBU-BMI/quip_lung_cancer_detection>`_
   * - Lymph node metastasis detection in breast cancer
     - nomets, mets
     - ResNet50 (trained via TIAToolbox dev team)
     - PatchCamelyon
     - 96 @ 1.0
     - `Ref <https://doi.org/10.1038/s43856-022-00186-5>`_
   * - Lymphocyte detection
     - til-negative, til-positive
     - InceptionV4 (without batchnorm)
     - 23 TCGA studies
     - 100 @ 0.5
     - `Ref <https://doi.org/10.3389/fonc.2021.806603>`_
   * - Pancreatic adenocarcinoma detection
     - tumor-positive
     - Preactivation ResNet34
     - TCGA PAAD
     - 350 @ 1.5
     - `Ref <https://doi.org/10.1007/978-3-030-32239-7_60>`_
   * - Prostate adenocarcinoma detection
     - grade3, grade4or5, benign
     - ResNet34
     - TCGA PRAD
     - 175 @ 0.5
     - `Ref <https://github.com/SBU-BMI/quip_prad_cancer_detection>`_

WSInsight-native models
~~~~~~~~~~~~~~~~~~~~~~~

Cell-level and hybrid models shipped exclusively with WSInsight:

* ``CellViT-256-x20``
* ``CellViT-256-x40``
* ``CellViT-256-x40-AMP``
* ``CellViT-SAM-H-x20``
* ``CellViT-SAM-H-x40``
* ``CellViT-SAM-H-x40-AMP``
* ``CellViT-Virchow-x40-AMP``
* ``hovernet_fast_pannuke``



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
