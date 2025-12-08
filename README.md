# ![WSInsight logo](docs/_static/logo48.png) WSInsight: Cloud-Native Single-Cell Pathology Inference on Whole Slide Images

WSInsight is a fork of [WSInfer](https://github.com/SBU-BMI/wsinfer) that delivers end-to-end pathology inference for giga-pixel whole slide images. It scales from laptops to cloud clusters, orchestrates patch extraction/classification, cell detection/classification, model inference, and downstream analytics, and produces artifacts that can be explored in QuPath, GeoJSON-aware viewers, OMERO+, or bespoke notebooks.

> [!IMPORTANT]
> WSInsight is a research tool. It is not cleared for clinical workflows or patient-facing decisions.

![Workflow overview](docs/_static/diagram.drawio.png)

## Highlights

- GPU-accelerated inference for registered models from the WSInfer Model Zoo or custom TorchScript weights
- Automated tissue segmentation, patch extraction, and batched inference with resumable runs
- First-class support for QuPath projects, GeoJSON/OME-CSV exports, and remote slides (S3, GDC manifests)
- Transparent URI handling lets you read WSIs from local disks, S3 buckets, or GDC manifests and write inference outputs back to either local paths or S3 using the same CLI options
- Built for reproducibility: metadata capture, deterministic configuration, and container-friendly execution

## Visual Overview

| Original H&E                                           | Heatmap of Tumor Probability                                           |
|:------------------------------------------------------:|:----------------------------------------------------------------------:|
| ![H&E example](docs/_static/brca-tissue.png)           | ![Tumor probability heatmap](docs/_static/brca-heatmap-neoplastic.png) |
| Heatmap of Dead Cell Probability                       | Heatmap of Connective Cell Probability                                 |
| ![Necrotic region](docs/_static/brca-heatmap-dead.png) | ![Connectivity heatmap](docs/_static/brca-heatmap-connective.png)      |

## Integrative Patch-Level and Single-Cell Inference

The models used in this experiment include: `CellViT-SAM-H-x40`, `breast-tumor-resnet34.tcga-brca`, and `pancancer-lymphocytes-inceptionv4.tcga`.

| Original H&E ROI                                                   |                                                                            |
|:------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
| ![original H&amp;E](docs/_static/roi-hne.png)                      |                                                                            |
| Immune Cells (green) / Lympho Regions (blue)                       | Neoplastic Cells (yellow) / Lympho Regions (blue)                          |
| ![immune cells/lympho regions](docs/_static/roi-lympho-immune.png) | ![neoplastic cells/lympho regions](docs/_static/roi-lympho-neoplastic.png) |
| Immune Cells (green) / Tumor Regions (red)                         | Neoplastic Cells (yellow) / Tumor Regions (red)                            |
| ![immune cells/tumor regions](docs/_static/roi-tumor-immune.png)   | ![neoplastic cells/tumor regions](docs/_static/roi-tumor-neoplastic.png)   |

## Documentation

- [Latest user and API guides](https://wsinsight.readthedocs.io)
- [Change history and issue reporting](https://github.com/huangch/wsinsight)

## Quick Start

### WSInfer-compatible workflow

1. Prepare a directory of whole slide images, for example the sample data under `tests/reference`.
2. Choose a registered model name from `wsinfer-zoo ls` or provide a custom configuration.
3. Run inference (one-shot workflow that performs patch extraction + inference):

   ```bash
   wsinsight run \
     --wsi-dir slides/ \
     --results-dir results/ \
     --model breast-tumor-resnet34.tcga-brca \
     --batch-size 32 \
     --num-workers 4
   ```

4. Inspect outputs in `results/model-outputs-*`, open the GeoJSON artifacts in QuPath or your preferred viewer, and review `run_metadata_*.json` for the captured environment details.

Prefer an explicit two-step flow? Run `wsinsight patch` to generate cached patches/HDF5 metadata (idempotent and resumable), then invoke `wsinsight infer` against the same `--results-dir` to produce CSV/GeoJSON/OME-CSV outputs. Both commands expose the identical URI, segmentation, and QuPath options as `wsinsight run`.

### WSInsight-native workflow (CellViT models)

WSInsight adds cell-centric Vision Transformer and HoverNet variants that are not part of upstream WSInfer. To run them:

1. Stage your WSIs as before and ensure the conda environment includes the CellViT dependencies (installed automatically via the instructions above).
2. Pick one of the WSInsight-native model identifiers (see list below) from the registry.
3. Launch inference, for example with `CellViT-SAM-H-x40`:

   ```bash
   wsinsight run \
     --wsi-dir slides/ \
     --results-dir results-cellvit/ \
     --model CellViT-SAM-H-x40 \
     --batch-size 16 \
     --num-workers 8
   ```

4. Review the outputs in `results-cellvit/model-outputs-*` and downstream GeoJSON artifacts just like the compatible workflow.

| **Method**            | **Architecture & Key Features**                                                                                                                                                   | **mPQ** | **bPQ** | **Reference**                                                                                                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CellViT**           | Vision Transformer encoder with U-Net style decoder;<br>trained on multi-tissue datasets (e.g., PanNuke);<br>supports multi-class nuclear instance segmentation & classification. | 0.4980  | 0.6793  | [Ref](https://doi.org/10.1016/j.media.2024.103143)                                                                                                                                |
| **HoVer-Net**         | ResNet50 CNN backbone with dual-branch decoder;<br>predicts nuclear masks + horizontal/vertical (HoVer) distance maps;<br>improves instance separation.                           | 0.4629  | 0.6596  | [Ref](https://doi.org/10.1016/j.media.2019.101563)                                                                                                                                |
| **StarDist–ResNet50** | ResNet50 backbone + star-convex polygon representation;<br>predicts radial distances for nuclei delineation along fixed rays.                                                     | 0.4796  | 0.6692  | [Ref](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_30), [Ref](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) |

Available WSInsight model names:

- [`CellViT-256-x20`](https://huggingface.co/huangch/CellViT-256-x20)
- [`CellViT-256-x40`](https://huggingface.co/huangch/CellViT-256-x40)
- [`CellViT-256-x40-AMP`](https://huggingface.co/huangch/CellViT-256-x40-AMP)
- [`CellViT-SAM-H-x20`](https://huggingface.co/huangch/CellViT-SAM-H-x20)
- [`CellViT-SAM-H-x40`](https://huggingface.co/huangch/CellViT-SAM-H-x40)
- [`CellViT-SAM-H-x40-AMP`](https://huggingface.co/huangch/CellViT-SAM-H-x40-AMP)
- [`CellViT-Virchow-x40-AMP`](https://huggingface.co/huangch/CellViT-Virchow-x40-AMP)
- [`hovernet_fast_pannuke`](https://huggingface.co/huangch/hovernet_fast_pannuke)

> [!TIP]
> Use `CUDA_VISIBLE_DEVICES=… wsinsight run …` to pin execution to specific GPUs. The command prints an environment summary before inference begins.

## Installation

WSInsight supports both a fully reproducible conda workflow and lighter manual installs if you already manage your own environment.

### Option A: Reproducible conda setup (recommended)

Run the following commands from the repository root to recreate the tested environment. Adjust the environment name if you need to keep multiple copies side-by-side.

```bash
# reset any previous environment
source /opt/anaconda3/etc/profile.d/conda.sh  # adapt if conda lives elsewhere
conda deactivate || true
conda env remove -n wsinsight -y || true

# create a clean env with Python 3.11 + GDAL 3.11.3
conda create -n wsinsight python=3.11 gdal=3.11.3 -c conda-forge -y
conda activate wsinsight
python -m pip install --upgrade pip

# shared constraints keep numpy < 2 across every install step
python -m pip install -c ./wsinsight/constraints.txt "numpy<2"

# install heavy ML stacks first so CUDA dependencies settle early
python -m pip install -c ./wsinsight/constraints.txt \
  torch torchvision torch-geometric tensorflow keras stardist

# HistomicsTK wheels are hosted externally; keep numpy pinned for ABI safety
python -m pip install --no-cache-dir --trusted-host github.com \
  --trusted-host raw.githubusercontent.com --trusted-host girder.github.io \
  --find-links https://girder.github.io/large_image_wheels --upgrade \
  "numpy<2" histomicstk

# finally, install WSInsight itself in editable mode with the same constraints
python -m pip install -c ./wsinsight/constraints.txt -e ./wsinsight
```

Optionally, run a smoke test to ensure the CLI starts with representative environment variables:

```bash
S3_STORAGE_OPTIONS='{"profile":"saml"}' \
WSINFER_ZOO_REGISTRY_PATH='/workspace/wsinsight/wsinsight/zoo/wsinfer-zoo-registry.json' \
WSINSIGHT_REMOTE_CACHE_DIR='/tmp' \
KERAS_HOME='/workspace/wsinsight/wsinsight/keras' \
wsinsight --help
```

> [!TIP]
> Every `python -m pip install …` line honors `constraints.txt`, keeping the dependency graph deterministic even as upstream wheels evolve.

### Option B: Manual installation

1. **Install deep learning backends**

- Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your OS / CUDA stack.
- (Optional) Bring in TensorFlow/Keras if you plan to convert models or run StarDist.
- Verify CUDA visibility with `python -c 'import torch; print(torch.cuda.is_available())'` and confirm your driver matches the [CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/).

1. **Install WSInsight**

- Stable PyPI: `python -m pip install wsinsight`
- Latest main: `python -m pip install git+https://github.com/huangch/wsinsight.git`
- Conda-Forge: `conda install -c conda-forge wsinsight` (use `mamba install` for faster solving)

1. **Install from source (development)**

```bash
git clone https://github.com/huangch/wsinsight.git
cd wsinsight
python -m pip install --editable .
pre-commit install
```

The editable install enables rapid iteration on CLI commands, model definitions, and docs. `pre-commit` keeps formatting/lint guards active during `git commit`.

## CLI Overview

Command           | Purpose
----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
`wsinsight run`   | Segment tissue, extract patches, execute model inference, and emit CSV/GeoJSON/OME-CSV outputs (one-shot orchestration of the two commands below).
`wsinsight patch` | Perform tissue segmentation, cache/crop patches to HDF5, and prepare metadata for later inference runs; safe to rerun to resume interrupted jobs.
`wsinsight infer` | Load cached patches, run the selected model, and export QuPath/GeoJSON/OME-CSV artifacts.

Pick `run` when you want a one-liner for single slides or small batches; switch to the explicit `patch` → `infer` flow to resume large jobs, share patch caches across model variants, or parallelize stages on separate machines. All commands share global options such as `--backend` (`openslide` or `tiffslide`) and `--log-level`. Use `wsinsight <command> --help` for the full option list, including QuPath integration flags and segmentation controls.

## Results Layout

- `patches/`: HDF5 patch metadata and thumbnails used during inference
- `model-outputs-csv/`: Per-slide predictions with probabilities per class
- `model-outputs-geojson/`: Spatial outputs for downstream visualization
- `run_metadata_*.json`: Captured configuration, environment, and git state for reproducibility

## Models and Configurations

- Models registered in the WSInfer Zoo can be listed with `wsinfer-zoo ls`.
- Bring-your-own models by supplying `--config` (JSON schema documented in `wsinsight/schemas/model-config.schema.json`) together with `--model-path` (TorchScript weights).
- QuPath-generated detections and annotations can be used to create pseudo-model runs via the `--qupath-*` options in `wsinsight run`.

## Remote and Large-Scale Data

- S3 URIs are supported out of the box; configure credentials via `S3_STORAGE_OPTIONS`.
- `--wsi-dir` can point to local folders, `s3://bucket/prefix` locations, or `gdc://path/to/manifest.tsv`; `--results-dir`, GeoJSON, and OME-CSV outputs can be written to local disks or S3 buckets with the same URI syntax.
- Every CLI that accepts `--wsi-dir`, `--results-dir`, `--references-dir`, or QuPath directories uses the same URI resolver as `wsinsight patch`/`infer`. Local paths require `exists=True`, while remote paths honor the `S3_STORAGE_OPTIONS` profile without checking for pre-existence—making it safe to point `--results-dir` at a brand-new bucket/key.
- `WSINSIGHT_REMOTE_CACHE_DIR` determines where remote assets are materialized locally (default: `~/.cache/wsinsight`). Set it to a fast SSD mount when you process tera-scale cohorts.
- GDC manifests can be referenced directly, and the downloaded tiles are cached via the same mechanism.
- For throughput, adjust `--num-workers` to match CPU availability and tune `--batch-size` per GPU memory.

## Development and Testing

- Ensure `ruff`, `black`, and other lint tools remain clean by running `pre-commit run --all-files`.
- Execute the test suite with `pytest` from the project root.
- Documentation lives in `docs/`; build locally with `make -C docs html`.

## Support and Feedback

- File bugs or feature requests via [GitHub issues](https://github.com/huangch/wsinsight/issues).
- For general usage questions, start a [GitHub discussion](https://github.com/huangch/wsinsight/discussions) or consult the FAQ in the documentation portal.

## License

WSInsight is released under the terms of the `LICENSE` file included with this repository.

<!--
[![Continuous Integration](https://github.com/huangch/wsinsight/actions/workflows/ci.yml/badge.svg)](https://github.com/huangch/wsinsight/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/wsinsight/badge/?version=latest)](https://wsinsight.readthedocs.io/en/latest/?badge=latest)
[![Version on PyPI](https://img.shields.io/pypi/v/wsinsight.svg)](https://pypi.org/project/wsinsight/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/wsinsight)](https://pypi.org/project/wsinsight/)
[![Published in npj Precision Oncology](https://img.shields.io/badge/Published-npj_Precision_Oncology-blue)](https://doi.org/10.1038/s41698-024-00499-9)
-->

## Citation

If you find our work useful, please cite [WSInsight](https://doi.org/10.1101/2025.692260ß) and [WSInfer](https://doi.org/10.1038/s41698-024-00499-9)!

> Huang, C.-H., Awosika, O. E., & Fernandez, D. (2025).
WSInsight as a cloud-native pipeline for single-cell pathology inference on whole-slide images. bioRxiv. https://doi.org/10.1101/2025.692260

```bibtex
@article{huang2025wsinsight,
 title        = {WSInsight as a cloud-native pipeline for single-cell pathology inference on whole-slide images},
 author       = {Huang, Chao-Hui and Awosika, Oluwamayowa E. and Fernandez, Diane},
 year         = {2025},
 journal      = {bioRxiv},
 publisher    = {Cold Spring Harbor Laboratory},
 doi          = {10.1101/2025.692260},
 url          = {https://doi.org/10.1101/2025.692260}
}
```

> Kaczmarzyk, J.R., O’Callaghan, A., Inglis, F. et al. Open and reusable deep learning for pathology with WSInfer and QuPath. *npj Precis. Onc.* **8**, 9 (2024). https://doi.org/10.1038/s41698-024-00499-9

```bibtex
@article{kaczmarzyk2024open,
  title={Open and reusable deep learning for pathology with WSInfer and QuPath},
  author={Kaczmarzyk, Jakub R. and O'Callaghan, Alan and Inglis, Fiona and Gat, Swarad and Kurc, Tahsin and Gupta, Rajarsi and Bremer, Erich and Bankhead, Peter and Saltz, Joel H.},
  journal={npj Precision Oncology},
  volume={8},
  number={1},
  pages={9},
  year={2024},
  month={Jan},
  day=10,
  doi={10.1038/s41698-024-00499-9},
  issn={2397-768X},
  url={https://doi.org/10.1038/s41698-024-00499-9}
}
```
