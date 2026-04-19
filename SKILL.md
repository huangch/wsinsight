---
name: wsinsight
description: Fetch, install, and operate WSInsight for whole-slide-image pathology inference, spatial analytics, and export
---

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

### 2.4 Docker (no local installation required)

A prebuilt GPU-enabled image based on `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
is published to Docker Hub.  All dependencies (conda, GDAL, PyTorch, TensorFlow,
WSInsight) are pre-installed — **users do not need to install anything locally
except Docker and the NVIDIA Container Toolkit**.

```bash
# Pull the published image
docker pull huangchtw/wsinsight:latest
```

**Option A — helper script** (`docker-run.sh`):

```bash
# Interactive shell — all GPUs
bash docker-run.sh /path/to/data

# Interactive shell — pin to GPU 2
bash docker-run.sh /path/to/data 2

# Direct command — all GPUs (no interactive shell)
bash docker-run.sh /path/to/data "" wsinsight run \
  --wsi-dir /workspace/slides --results-dir /workspace/results \
  --model breast-tumor-resnet34.tcga-brca

# Direct command — GPU 2
bash docker-run.sh /path/to/data 2 wsinsight run \
  --wsi-dir /workspace/slides --results-dir /workspace/results \
  --model breast-tumor-resnet34.tcga-brca
```

When a command is provided after the GPU argument, the container executes it
and exits.  When no command is given, you land in `/workspace` with the conda
`wsinsight` environment already activated.

**Option B — manual `docker run`**:

```bash
docker run --rm -it \
  --gpus all --shm-size=32g \
  --user $(id -u):$(id -g) \
  -v /path/to/slides:/slides \
  -v /path/to/results:/results \
  huangchtw/wsinsight:latest \
  bash -lc 'wsinsight run --wsi-dir /slides --results-dir /results --model breast-tumor-resnet34.tcga-brca'
```

`--shm-size=32g` is recommended for multi-worker dataloaders (PyTorch uses
`/dev/shm` for shared memory).  The image bakes in `WSINFER_ZOO_REGISTRY_PATH`
and `KERAS_HOME` so the CLI works without any environment setup.

**Building from source** (maintainers only):

```bash
bash docker-build-push.sh   # builds + tags + pushes huangchtw/wsinsight:latest
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

## 8. Acquiring a TCGA Slide Manifest via GDC API

The NCI Genomic Data Commons (GDC) hosts all TCGA, TARGET, and other NCI
program data.  Its REST API can return **manifest files** directly — no
`gdc-client` binary is needed.  The manifest is a TSV listing slide UUIDs and
filenames; WSInsight then downloads the actual slides on demand during
inference via the GDC data endpoint (`https://api.gdc.cancer.gov/data/{uuid}`)
with automatic retries and MD5 verification.

### 8.1 Generating a Manifest

POST to `https://api.gdc.cancer.gov/files` with `return_type=manifest`:

```bash
curl --request POST \
  --header "Content-Type: application/json" \
  --data '{
    "filters": {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": "TCGA-BRCA"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Slide Image"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "experimental_strategy",
                    "value": "Diagnostic Slide"
                }
            }
        ]
    },
    "return_type": "manifest",
    "size": "99999"
  }' \
  'https://api.gdc.cancer.gov/files' \
  > tcga-brca-dx-manifest.tsv
```

**Key parameters:**

| Parameter       | Value              | Purpose                                          |
| --------------- | ------------------ | ------------------------------------------------ |
| `return_type`   | `manifest`         | Returns TSV manifest instead of JSON metadata    |
| `size`          | `99999`            | Max results (default is 10 — always override)    |
| `filters`       | JSON object        | GDC filter syntax (see below)                    |

### 8.2 Filter Fields for Slide Images

| Field                            | Values                                              | Notes                                   |
| -------------------------------- | --------------------------------------------------- | --------------------------------------- |
| `cases.project.project_id`       | `TCGA-BRCA`, `TCGA-LUAD`, etc.                     | Required — selects the cohort           |
| `data_type`                      | `Slide Image`                                       | Required — filters to WSI files         |
| `experimental_strategy`          | `Diagnostic Slide` or `Tissue Slide`                | Diagnostic = formalin-fixed; Tissue = frozen section |
| `data_format`                    | `SVS`                                               | Optional — all TCGA slides are SVS      |
| `cases.submitter_id`             | `TCGA-A7-A0CE`, ...                                | Optional — filter to specific cases     |

Filters use the GDC query DSL with operators: `=`, `!=`, `in`, `and`, `or`.
Nested filters are combined with `"op": "and"` at the top level.

### 8.3 Common TCGA Project IDs

| Cancer Type                  | Project ID    |
| ---------------------------- | ------------- |
| Breast invasive carcinoma    | `TCGA-BRCA`   |
| Lung adenocarcinoma          | `TCGA-LUAD`   |
| Lung squamous cell carcinoma | `TCGA-LUSC`   |
| Prostate adenocarcinoma      | `TCGA-PRAD`   |
| Pancreatic adenocarcinoma    | `TCGA-PAAD`   |
| Colon adenocarcinoma         | `TCGA-COAD`   |
| Rectum adenocarcinoma        | `TCGA-READ`   |
| Glioblastoma multiforme      | `TCGA-GBM`    |
| Ovarian serous cystadenocarcinoma | `TCGA-OV` |
| Uterine corpus endometrial   | `TCGA-UCEC`   |
| Kidney renal clear cell      | `TCGA-KIRC`   |
| Head and neck squamous cell  | `TCGA-HNSC`   |
| Liver hepatocellular         | `TCGA-LIHC`   |
| Stomach adenocarcinoma       | `TCGA-STAD`   |
| Bladder urothelial           | `TCGA-BLCA`   |
| Skin cutaneous melanoma      | `TCGA-SKCM`   |

### 8.4 Manifest Format

The GDC API returns a TSV with these columns:

```text
id	filename	md5	size	state
UUID-1	TCGA-A7-A0CE-01Z-00-DX1.svs	abc123...	234567890	released
UUID-2	TCGA-A7-A13E-01Z-00-DX1.svs	def456...	345678901	released
```

WSInsight's `URIPath` reads this natively — it looks for `id`/`file_id` and
`filename`/`file_name` columns, plus optional `md5` for checksum verification.

### 8.5 Access Control

- **TCGA diagnostic and tissue slides are open-access** — no token needed.
- For controlled-access data (e.g. some TARGET projects), obtain an
  authentication token from the [GDC Data Portal](https://portal.gdc.cancer.gov/).
  There is no CLI flag for the token; pass it programmatically via the `token`
  (or `token_path`) keyword argument on `URIPath`.  For typical TCGA workflows
  this is not required.

### 8.6 Combining Filters

To select only specific cases within a project, use `"op": "in"`:

```bash
curl --request POST \
  --header "Content-Type: application/json" \
  --data '{
    "filters": {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": ["TCGA-A7-A0CE", "TCGA-A7-A13E", "TCGA-BH-A0B3"]
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Slide Image"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "experimental_strategy",
                    "value": "Diagnostic Slide"
                }
            }
        ]
    },
    "return_type": "manifest",
    "size": "99999"
  }' \
  'https://api.gdc.cancer.gov/files' \
  > tcga-brca-subset-manifest.tsv
```

### 8.7 End-to-End Example

```bash
# 1. Download manifest for all TCGA-BRCA diagnostic slides
curl --request POST \
  --header "Content-Type: application/json" \
  --data '{
    "filters": {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}}
        ]
    },
    "return_type": "manifest",
    "size": "99999"
  }' \
  'https://api.gdc.cancer.gov/files' \
  > tcga-brca-dx-manifest.tsv

# 2. Verify slide count (header + data lines)
wc -l tcga-brca-dx-manifest.tsv

# 3. Run WSInsight on the manifest
wsinsight run \
  --wsi-dir "gdc-manifest://$(pwd)/tcga-brca-dx-manifest.tsv" \
  --results-dir results-brca/ \
  --model breast-tumor-resnet34.tcga-brca \
  --batch-size 32
```

WSInsight downloads each slide on demand via `https://api.gdc.cancer.gov/data/{uuid}`,
caches it locally under the directory set by `WSINSIGHT_REMOTE_CACHE_DIR`
(defaults to `~/.cache/wsinsight` via `platformdirs.user_cache_dir`), and
processes it.

---

## 9. URI & Remote Data Support

`--wsi-dir` and `--results-dir` accept:

| Scheme               | Example                                                    |
| -------------------- | ---------------------------------------------------------- |
| Local path           | `slides/` or `/data/slides`                                |
| S3                   | `s3://bucket/prefix`                                       |
| GDC manifest         | `gdc-manifest:///absolute/path/to/manifest.tsv`            |
| Image list           | `image-list:///path/to/filelist.txt`                       |

A plain local `.txt` file passed as `--wsi-dir` is auto-coerced to
`image-list://`.  S3 access requires `S3_STORAGE_OPTIONS` to be set.

**Important:** The GDC manifest URI scheme is `gdc-manifest://` (not `gdc://`).
The path must be absolute (triple slash: `gdc-manifest:///absolute/path`).

---

## 10. Error Recovery & Troubleshooting

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

## 11. Agent Decision Guide

Use this flowchart when deciding which command(s) to run:

```text
Is WSInsight already installed / is Docker available?
├─ Docker available → Prefer Docker (Section 2.4): no install needed
│        bash docker-run.sh /path/to/data "" wsinsight run ...
│        Or interactive: bash docker-run.sh /path/to/data [GPU_ID]
├─ Not installed, no Docker → Install via conda (Section 2.1–2.3)
└─ Already installed → Continue below

Has the user provided WSIs?
├─ Yes → Do they want a one-shot run?
│        ├─ Yes → wsinsight run [--hplot] [--ncomp] [--cme] [--export-geojson]
│        └─ No  → wsinsight patch → wsinsight infer → (analytics) → wsinsight export
├─ No, but results-dir exists with model-outputs-csv/ → Skip patch+infer
│        ├─ Need H-plot?  → wsinsight hplot (requires --hplot-base-types + --hplot-target-types)
│        ├─ Need ncomp?   → wsinsight ncomp
│        ├─ Need CME?     → wsinsight cme (runs across ALL slides; not parallelizable)
│        └─ Need export?  → wsinsight export --geojson --omecsv
├─ No slides, but user mentions TCGA / GDC / cancer cohort
│        → Query GDC API for manifest (Section 8) → save .tsv
│        → wsinsight run --wsi-dir "gdc-manifest:///path/to/manifest.tsv" ...
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
7. **When the user mentions TCGA, GDC, or a cancer cohort** (e.g. "analyze
   TCGA-BRCA slides"), use the GDC API `curl` pattern from Section 8 to
   generate a manifest TSV, then pass it via
   `--wsi-dir "gdc-manifest:///absolute/path/to/manifest.tsv"`.  Do not ask
   the user to download slides manually.
8. **Prefer Docker when available** — it avoids all local dependency
   installation.  Use `bash docker-run.sh /path/to/data` (or a manual
   `docker run`) and run `wsinsight` commands inside the container.  The
   image pre-sets `WSINFER_ZOO_REGISTRY_PATH` and `KERAS_HOME`.

---

## 12. Verification Checklist

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
