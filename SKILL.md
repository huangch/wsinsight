# WSInsight — Agentic AI Skill File

> **Purpose**: Enable an agentic AI (Claude, OpenClaw, Hermes, or any
> tool-using LLM agent) to autonomously fetch, install, and operate WSInsight
> for whole-slide-image pathology inference, analytics, and export.

---

## 1. What Is WSInsight?

WSInsight is a Python CLI tool that delivers end-to-end pathology inference on
giga-pixel whole slide images (WSIs).  It orchestrates tissue segmentation,
patch extraction, GPU-accelerated model inference (patch-based and cell-based),
spatial analytics (H-plot, neighborhood composition, cellular
microenvironments), and export to GeoJSON / OME-CSV.

- **Repository**: <https://github.com/huangch/wsinsight>
- **License**: Apache 2.0
- **Python**: ≥ 3.11
- **Entry point**: `wsinsight` (installed via `pip install -e .`)

---

## 2. Fetch & Install

### 2.1 Prerequisites

| Requirement       | Why                                                      |
| ----------------- | -------------------------------------------------------- |
| conda (or mamba)  | GDAL must be installed via conda — pip wheels do not ship the C library |
| CUDA toolkit      | GPU inference (PyTorch, TensorFlow, CellViT models)      |
| Git               | Clone the repository                                     |
| Python 3.11       | Minimum supported version                                |

### 2.2 Full Reproducible Install (Recommended)

Run the following commands verbatim.  Every `pip install` line honours
`constraints.txt` so dependency versions are locked and reproducible.

```bash
# 1. Clone
git clone https://github.com/huangch/wsinsight.git
cd wsinsight

# 2. Create conda environment with GDAL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda env remove -n wsinsight -y 2>/dev/null || true
conda create -n wsinsight python=3.11 gdal=3.11.3 "setuptools<67" -c conda-forge -y
conda activate wsinsight
python -m pip install --upgrade pip

# 3. Pin numpy < 2 (stardist requirement)
python -m pip install -c constraints.txt "numpy<2"

# 4. Heavy ML stacks
python -m pip install -c constraints.txt \
  torch torchvision torch-geometric tensorflow keras stardist nvidia-ml-py

# 5. HistomicsTK (external wheel host)
python -m pip install \
  --trusted-host github.com \
  --trusted-host raw.githubusercontent.com \
  --trusted-host girder.github.io \
  --find-links https://girder.github.io/large_image_wheels \
  -c constraints.txt "numpy<2" pyvips histomicstk

# 6. Remaining dependencies
python -m pip install -c constraints.txt "numpy<2" \
  scikit-learn shapely geopandas pyproj rasterio pyogrio \
  openslide-python wsidicom paquo "wsinfer-zoo>=0.6.2" \
  igraph leidenalg s3fs boto3 platformdirs timm \
  tiffslide imagecodecs opencv-python-headless orjson click

# 7. Install WSInsight (editable)
python -m pip install -c constraints.txt --no-build-isolation -e .

# 8. Verify numpy stayed below 2.0
python -c "import numpy; v=numpy.__version__; assert int(v.split('.')[0]) < 2, f'numpy {v} >= 2'"
```

### 2.3 Quick Install (Existing Environment)

If GDAL, CUDA, and compatible numpy are already available:

```bash
git clone https://github.com/huangch/wsinsight.git
cd wsinsight
pip install -e .
```

### 2.4 Docker

A GPU-enabled Docker image is available based on `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`:

```bash
# Build
bash docker-build-push.sh

# Run (mount slides and results)
docker run --gpus all \
  -v /path/to/slides:/slides \
  -v /path/to/results:/results \
  wsinsight:latest \
  wsinsight run --wsi-dir /slides --results-dir /results --model breast-tumor-resnet34.tcga-brca
```

### 2.5 Smoke Test

```bash
wsinsight --help
```

Expected: the top-level Click help listing all sub-commands without errors.

---

## 3. Environment Variables

Set these **before** running any `wsinsight` command.  In restricted or
air-gapped networks the first variable is mandatory.

| Variable                       | Required | Purpose                                                                                  |
| ------------------------------ | -------- | ---------------------------------------------------------------------------------------- |
| `WSINFER_ZOO_REGISTRY_PATH`   | Yes*     | Path to a local `wsinfer-zoo-registry.json`. Prevents network calls to HuggingFace.      |
| `S3_STORAGE_OPTIONS`           | If S3    | JSON passed to `s3fs` / `fsspec` for AWS credentials (e.g. `'{"profile":"saml"}'`).      |
| `WSINSIGHT_REMOTE_CACHE_DIR`   | No       | Local cache dir for remote assets. Default: `~/.cache/wsinsight`.                        |
| `KERAS_HOME`                   | No       | Override Keras config/weights directory.                                                  |
| `CUDA_VISIBLE_DEVICES`         | No       | Pin to specific GPU(s) (e.g. `0` or `0,1`).                                             |

\* Required when HuggingFace Hub is unreachable (SSL errors, air-gapped).

---

## 4. CLI Reference

### 4.1 Command Map

```text
wsinsight
├── run               One-shot: patch → infer → hplot → ncomp → cme → export
├── patch             Tissue segmentation + patch extraction → HDF5
├── infer             Model inference on cached patches → CSV
├── reg               Post-hoc region registration
├── hplot             H-plot spatial analytics
├── hplot-finalize    Aggregate parallel H-plot runs
├── ncomp             Neighborhood composition
├── cme               Cellular microenvironment (cross-slide)
└── export            Merge analytics → GeoJSON / OME-CSV
```

### 4.2 Global Options (All Commands)

| Flag           | Description                         |
| -------------- | ----------------------------------- |
| `--backend`    | Slide reading backend               |
| `--log-level`  | Logging verbosity                   |
| `--version`    | Print version and exit              |

### 4.3 `wsinsight run` — Full Pipeline

The one-shot orchestrator.  Delegates to `patch`, `infer`, `hplot`, `ncomp`,
`cme`, and `export` in sequence.

```bash
wsinsight run \
  --wsi-dir <WSI_DIR> \
  --results-dir <RESULTS_DIR> \
  --model <MODEL_NAME> \
  [--batch-size 32] \
  [--num-workers 4] \
  [--hplot] [--hplot-base-types tumor] [--hplot-target-types lymphocyte] \
  [--ncomp] \
  [--cme] \
  [--export-geojson] [--export-omecsv]
```

**Key options:**

| Option                   | Type      | Description                                              |
| ------------------------ | --------- | -------------------------------------------------------- |
| `--wsi-dir / -i`         | path/URI  | Directory of WSI files (local, S3, GDC manifest)         |
| `--results-dir / -o`     | path/URI  | Output directory                                         |
| `--model / -m`           | string    | Registered model name (e.g. `breast-tumor-resnet34.tcga-brca`) |
| `--config / -c`          | path      | Custom model config JSON                                 |
| `--model-path / -p`      | path      | Custom TorchScript weights                               |
| `--zoo-model-dir / -z`   | path      | Folder with `config.json` + `torchscript_model.pt`       |
| `--batch-size / -b`      | int       | Inference batch size (default 32)                        |
| `--num-workers / -n`     | int       | Dataloader workers (auto)                                |
| `--hplot`                | flag      | Enable H-plot analytics                                  |
| `--ncomp`                | flag      | Enable neighborhood composition                          |
| `--cme`                  | flag      | Enable cellular microenvironment                         |
| `--export-geojson`       | flag      | Write GeoJSON export files                               |
| `--export-omecsv`        | flag      | Write OME-CSV export files                               |
| `--overwrite`            | flag      | Recompute existing outputs                               |

### 4.4 `wsinsight patch` — Tissue Segmentation & Patch Extraction

```bash
wsinsight patch \
  --wsi-dir <WSI_DIR> \
  --results-dir <RESULTS_DIR> \
  --model <MODEL_NAME>
```

Creates `masks/` and `patches/` under `--results-dir`.

### 4.5 `wsinsight infer` — Model Inference

```bash
wsinsight infer \
  --results-dir <RESULTS_DIR> \
  --model <MODEL_NAME> \
  [--batch-size 32] [--num-workers 4] [--overwrite]
```

Reads from `patches/`, writes to `model-outputs-csv/`.

### 4.6 `wsinsight hplot` — H-Plot Analytics

```bash
wsinsight hplot \
  --wsi-dir <WSI_DIR> \
  --results-dir <RESULTS_DIR> \
  --hplot-base-types tumor \
  --hplot-target-types lymphocyte \
  [--hplot-k 2] [--hplot-n 8] [--hplot-r 0.5] \
  [--hplot-max-neighbor-distance 25.0] \
  [--hplot-range-min -5] [--hplot-range-max 5] \
  [--overwrite]
```

### 4.7 `wsinsight hplot-finalize` — Aggregate Parallel H-Plots

```bash
wsinsight hplot-finalize --results-dir <RESULTS_DIR>
```

### 4.8 `wsinsight ncomp` — Neighborhood Composition

```bash
wsinsight ncomp \
  --wsi-dir <WSI_DIR> \
  --results-dir <RESULTS_DIR> \
  [--ncomp-k 2] [--ncomp-max-neighbor-distance 25.0] [--overwrite]
```

### 4.9 `wsinsight cme` — Cellular Microenvironment

Cross-slide analysis: builds Delaunay cell graphs per slide, trains a global
DGI encoder, and clusters embeddings.  **Cannot be parallelized across GPU
shards** — run after all per-shard inference has completed.

```bash
wsinsight cme \
  --wsi-dir <WSI_DIR> \
  --results-dir <RESULTS_DIR> \
  [--cme-hoptimus] [--cme-clusters 10] [--overwrite]
```

### 4.10 `wsinsight export` — Merge & Export

```bash
wsinsight export \
  --results-dir <RESULTS_DIR> \
  --geojson --omecsv \
  [--export-workers 8] [--overwrite]
```

### 4.11 `wsinsight reg` — Region Registration

```bash
wsinsight reg \
  --results-dir <RESULTS_DIR> \
  --region-inference-dir <REGION_DIR> \
  [--geojson] [--omecsv] [--overwrite]
```

---

## 5. Model Selection

Models can be specified in four mutually exclusive ways:

| Method                 | Flag(s)                    | When to Use                                              |
| ---------------------- | -------------------------- | -------------------------------------------------------- |
| Registry name          | `--model <name>`           | Registered WSInfer Zoo / WSInsight model                 |
| Custom config + weights| `--config` + `--model-path`| Bring-your-own TorchScript model                         |
| Zoo directory          | `--zoo-model-dir`          | Folder with `config.json` + `torchscript_model.pt`       |
| List registered models | `wsinfer-zoo ls`           | Discover available model names                           |

### Available WSInsight-native Models

- `CellViT-256-x20`, `CellViT-256-x40`, `CellViT-256-x40-AMP`
- `CellViT-SAM-H-x20`, `CellViT-SAM-H-x40`, `CellViT-SAM-H-x40-AMP`
- `CellViT-Virchow-x40-AMP`
- `hovernet_fast_pannuke`

Plus all models from `wsinfer-zoo ls` (e.g. `breast-tumor-resnet34.tcga-brca`,
`pancancer-lymphocytes-inceptionv4.tcga`).

---

## 6. Results Directory Layout

After a full pipeline run, `--results-dir` contains:

```text
<results-dir>/
├── masks/
│   └── <slide>.jpg                 Tissue segmentation masks
├── patches/
│   └── <slide>.h5                  HDF5 patch files
├── model-outputs-csv/
│   └── <slide>.csv                 Per-cell inference results
├── model-outputs-geojson/
│   └── <slide>.geojson             Region-registered GeoJSON
├── model-outputs-omecsv/
│   └── <slide>.ome.csv.gz          Region-registered OME-CSV
├── hplot-outputs-csv/
│   ├── hplots/<slide>.csv          H-plot curves
│   ├── cells/<slide>.csv           Per-cell spatial annotations
│   └── hmetrics/<slide>.json       Per-slide H-plot metrics
├── hplot-outputs.csv               Aggregated H-plot (all slides)
├── hmetrics-outputs.csv            Aggregated H-plot metrics (all slides)
├── ncomp-outputs-csv/
│   └── <slide>.csv                 Neighborhood composition
├── cme-outputs-csv/
│   ├── cells/<slide>.csv           Per-cell CME labels
│   └── cmes/<slide>.csv            Merged CME regions
├── cme-outputs-geojson/
│   ├── cells/<slide>.geojson       Cell GeoJSON with CME labels
│   └── cmes/<slide>.geojson        CME region GeoJSON
├── graphs/
│   └── <slide>.h5                  Delaunay cache (shared by hplot/ncomp/cme)
├── export-csv/
│   └── <slide>.csv                 Merged per-cell CSV (all analytics)
├── export-geojson/
│   └── <slide>.geojson             GeoJSON export
├── export-omecsv/
│   └── <slide>.ome.csv.gz          OME-CSV export
└── run_metadata_*.json             Configuration & runtime info
```

---

## 7. Common Workflows

### 7.1 Basic Inference (Smallest Useful Run)

```bash
wsinsight run \
  --wsi-dir slides/ \
  --results-dir results/ \
  --model breast-tumor-resnet34.tcga-brca
```

### 7.2 Full Pipeline with All Analytics

```bash
wsinsight run \
  --wsi-dir slides/ \
  --results-dir results/ \
  --model pancancer-lymphocytes-inceptionv4.tcga \
  --batch-size 32 \
  --hplot --hplot-base-types tumor --hplot-target-types lymphocyte \
  --hplot-range-min -5 --hplot-range-max 5 \
  --ncomp \
  --cme \
  --export-geojson --export-omecsv
```

### 7.3 Step-by-Step (Resumable / Parallelizable)

```bash
# Step 1: Patch extraction (resumable)
wsinsight patch --wsi-dir slides/ --results-dir results/ --model breast-tumor-resnet34.tcga-brca

# Step 2: Inference
wsinsight infer --results-dir results/ --model breast-tumor-resnet34.tcga-brca --batch-size 32

# Step 3: Analytics (independent of each other, except CME is cross-slide)
wsinsight hplot --wsi-dir slides/ --results-dir results/ --hplot-base-types tumor --hplot-target-types lymphocyte
wsinsight ncomp --wsi-dir slides/ --results-dir results/
wsinsight cme   --wsi-dir slides/ --results-dir results/

# Step 4: Export
wsinsight export --results-dir results/ --geojson --omecsv
```

### 7.4 CellViT Cell Detection

```bash
wsinsight run \
  --wsi-dir slides/ \
  --results-dir results-cellvit/ \
  --model CellViT-SAM-H-x40 \
  --batch-size 16 --num-workers 8
```

### 7.5 CME with H-Optimus Features

```bash
wsinsight cme \
  --wsi-dir slides/ \
  --results-dir results/ \
  --cme-hoptimus --cme-clusters 10
```

### 7.6 Multi-GPU Parallel Inference

Split slide lists into per-GPU shards, run each with
`CUDA_VISIBLE_DEVICES=<N>` pinning, sharing the same `--results-dir`.
After all shards finish:

```bash
wsinsight hplot-finalize --results-dir results/
```

### 7.7 Reading Results Programmatically

```python
import pandas as pd

# Per-cell inference
df = pd.read_csv("results/model-outputs-csv/SLIDE_001.csv")
print(df[["minx", "miny", "prob_tumor"]].head())

# Merged export (all analytics joined)
df_export = pd.read_csv("results/export-csv/SLIDE_001.csv")
print(df_export.columns.tolist())

# H-plot cohort summary
hplot = pd.read_csv("results/hplot-outputs.csv")
```

---

## 8. URI & Remote Data Support

`--wsi-dir` and `--results-dir` accept:

| Scheme               | Example                                    |
| -------------------- | ------------------------------------------ |
| Local path           | `slides/` or `/data/slides`                |
| S3                   | `s3://bucket/prefix`                       |
| GDC manifest         | `gdc://path/to/manifest.tsv`              |
| Image list           | `image-list:///path/to/filelist.txt`       |

A plain local `.txt` file passed as `--wsi-dir` is auto-coerced to
`image-list://`.  S3 access requires `S3_STORAGE_OPTIONS` to be set.

---

## 9. Error Recovery & Troubleshooting

| Symptom                              | Cause                                 | Fix                                                         |
| ------------------------------------ | ------------------------------------- | ----------------------------------------------------------- |
| SSL errors / hang on startup         | HuggingFace Hub unreachable           | Set `WSINFER_ZOO_REGISTRY_PATH` to local registry JSON      |
| `numpy >= 2.0` assertion failure     | Dependency upgraded numpy             | `pip install -c constraints.txt "numpy<2"`                   |
| `ModuleNotFoundError: osgeo`         | GDAL not installed via conda          | `conda install -c conda-forge gdal=3.11.3`                  |
| CUDA out of memory                   | Batch size too large                  | Reduce `--batch-size`                                       |
| Inference produces empty CSV         | Wrong model for slide magnification   | Match model suffix (`x20`/`x40`) to slide magnification     |
| `--hplot` fails with missing types   | `--hplot-base-types` not set          | Always pass both `--hplot-base-types` and `--hplot-target-types` |
| Stale graph cache                    | CSV changed after graph was built     | Automatic: cache detects via SHA-256 hash and rebuilds       |
| CME fails across shards              | CME is cross-slide, not parallelizable| Run CME after merging all shard outputs into one results dir |

---

## 10. Agent Decision Guide

Use this flowchart when deciding which command(s) to run:

```text
Has the user provided WSIs?
├─ Yes → Do they want a one-shot run?
│        ├─ Yes → wsinsight run [--hplot] [--ncomp] [--cme] [--export-geojson]
│        └─ No  → wsinsight patch → wsinsight infer → (analytics) → wsinsight export
├─ No, but results-dir exists with model-outputs-csv/ → Skip patch+infer
│        ├─ Need H-plot?  → wsinsight hplot (requires --hplot-base-types + --hplot-target-types)
│        ├─ Need ncomp?   → wsinsight ncomp
│        ├─ Need CME?     → wsinsight cme (runs across ALL slides; not parallelizable)
│        └─ Need export?  → wsinsight export --geojson --omecsv
└─ No slides or results → Ask user for --wsi-dir
```

### Key Constraints for Agents

1. **Model is required** for `run`, `patch`, and `infer`.  Use `wsinfer-zoo ls`
   to list available models, or ask the user.
2. **`--hplot-base-types` and `--hplot-target-types`** are mandatory when
   `--hplot` is enabled.  These are comma-separated cell type names that must
   match the model's output classes (e.g. `tumor`, `lymphocyte`).
3. **CME is cross-slide** — it trains a global DGI model and clusters
   embeddings across all slides.  It cannot be run per-shard.  Run it only
   after all inference shards have completed and outputs are in one directory.
4. **`--overwrite`** is needed to recompute existing outputs.  Without it,
   completed slides are skipped (idempotent / resumable).
5. **Environment variables** must be exported before the `wsinsight` command,
   not passed as CLI flags.
6. **`constraints.txt`** should always be used with `pip install -c` to prevent
   dependency drift.

---

## 11. Verification Checklist

After installation, an agent should verify:

```bash
# 1. CLI loads without errors
wsinsight --help

# 2. All sub-commands are registered
wsinsight run --help
wsinsight patch --help
wsinsight infer --help
wsinsight hplot --help
wsinsight ncomp --help
wsinsight cme --help
wsinsight export --help

# 3. Python imports work
python -c "import wsinsight; print(wsinsight.__version__)"

# 4. CUDA is available (for GPU inference)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 5. numpy is below 2.0
python -c "import numpy; print('numpy:', numpy.__version__)"
```
