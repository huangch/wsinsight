.. _installing:

Installing and getting started
==============================

WSInsight supports Python 3.10+ and has been tested on Linux, macOS, and Windows. Two
installation paths are maintained:

* a **reproducible conda workflow** that mirrors the automation used by the core team; and
* a **manual installation** for users who already manage their own environments.

Both methods deliver the :code:`wsinsight` CLI and Python package.

Reproducible conda workflow
---------------------------

This option recreates the reference environment used for development and publications.
Run the commands below from the repository root. Adjust the path to :code:`conda.sh` and
the environment name if needed.

.. code-block:: bash

   # reset any previous environment (optional but recommended)
   source /opt/anaconda3/etc/profile.d/conda.sh
   conda deactivate || true
   conda env remove -n wsinsight -y || true

   # create a clean env with Python 3.11 and GDAL 3.11.3
   conda create -n wsinsight python=3.11 gdal=3.11.3 -c conda-forge -y
   conda activate wsinsight
   python -m pip install --upgrade pip

   # pin numpy across every install step via constraints.txt
   python -m pip install -c ./wsinsight/constraints.txt "numpy<2"

   # install heavy ML stacks first so CUDA dependencies settle early
   python -m pip install -c ./wsinsight/constraints.txt \
     torch torchvision torch-geometric tensorflow keras stardist nvidia-ml-py
   pip uninstall -y pynvml

   # HistomicsTK wheels live on girder.github.io; keep numpy pinned for ABI safety
   python -m pip install --no-cache-dir --trusted-host github.com \
     --trusted-host raw.githubusercontent.com --trusted-host girder.github.io \
     --find-links https://girder.github.io/large_image_wheels --upgrade \
     "numpy<2" histomicstk

   # finally, install WSInsight itself in editable mode with the same constraints
   python -m pip install -c ./wsinsight/constraints.txt -e ./wsinsight

Smoke-test the CLI (optional) with representative environment variables:

.. code-block:: bash

   S3_STORAGE_OPTIONS='{"profile":"saml"}' \
   WSINFER_ZOO_REGISTRY_PATH='/workspace/wsinsight/wsinsight/zoo/wsinfer-zoo-registry.json' \
   WSINSIGHT_REMOTE_CACHE_DIR='/tmp' \
   KERAS_HOME='/workspace/wsinsight/wsinsight/keras' \
   wsinsight --help

Manual installation
-------------------

If you already maintain project environments, follow these steps.

1. **Install GPU-enabled PyTorch (and optionally TensorFlow/Keras).**

   Follow `PyTorch's install guide <https://pytorch.org/get-started/locally/>`_ for your OS
   and CUDA driver version. Verify GPU visibility with ::

       python -c "import torch; print(torch.cuda.is_available())"

   For CUDA driver compatibility, consult the
   `official matrix <https://docs.nvidia.com/deploy/cuda-compatibility/>`_.

2. **Install WSInsight from PyPI, Git, or conda-forge.**

   .. code-block:: bash

      # latest stable
      python -m pip install wsinsight

      # or latest main branch
      python -m pip install git+https://github.com/huangch/wsinsight.git

      # or via conda/mamba
      conda install -c conda-forge wsinsight

   Validate the install with ::

       wsinsight --help

3. **Developers:** install editable mode + tooling ::

       git clone https://github.com/huangch/wsinsight.git
       cd wsinsight
       python -m pip install --editable .[dev]
       pre-commit install

Supported slide backends
------------------------

WSInsight supports `OpenSlide <https://openslide.org/>`_ and
`TiffSlide <https://github.com/Bayer-Group/tiffslide>`_. TiffSlide ships automatically;
install OpenSlide (library + :code:`openslide-python`) if you need it. Select the backend via
:code:`wsinsight --backend=tiffslide` or :code:`wsinsight --backend=openslide`, or from Python
with :func:`wsinsight.wsi.set_backend`.

Containers
----------

Prebuilt images live at https://hub.docker.com/u/huangch/wsinsight/ and work with Docker,
Apptainer, or Singularity.

.. code-block:: bash

   docker pull huangch/wsinsight
   apptainer pull docker://huangch/wsinsight
   singularity pull docker://huangch/wsinsight

Getting started
---------------

The :code:`wsinsight` CLI is the primary interface. Common commands:

.. code-block:: bash

   wsinsight --help
   wsinfer-zoo ls                     # WSInfer-compatible models
   wsinsight run --wsi-dir slides/ \
       --results-dir results/ \
       --model CellViT-SAM-H-x40

Remote data configuration
-------------------------

After installation, set the following environment variables (optionally via your shell
profile or job scheduler) to enable seamless access to cloud storage and manifests:

* ``S3_STORAGE_OPTIONS`` — JSON passed directly to ``fsspec`` (examples: ``{"profile": "research"}``, ``{"key": "…", "secret": "…"}``). This unlocks reading WSIs from
   ``s3://`` URIs and writing ``--results-dir`` outputs back to S3.
* ``WSINSIGHT_REMOTE_CACHE_DIR`` — Directory used to cache remote assets locally. Defaults
   to ``~/.cache/wsinsight``; point it at a fast SSD for tera-scale slides.
* ``WSINFER_ZOO_REGISTRY_PATH`` — Optional path/URI for a custom copy of the WSInfer model
   registry (local file, ``s3://…``, etc.).

With these variables in place, all CLI commands accept local paths, ``s3://`` URIs, or
``gdc://`` manifests for ``--wsi-dir`` and can write outputs to either local disks or S3
without code changes.
