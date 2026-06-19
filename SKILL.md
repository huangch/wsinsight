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
per-cell neighborhood composition on a Delaunay graph, and export to
GeoJSON / OME-CSV.

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
`/dev/shm` for shared memory).  The image bakes in `WSINSIGHT_ZOO_REGISTRY_PATH`
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
| `WSINSIGHT_ZOO_REGISTRY_PATH` | Yes*     | Path to a local `wsinsight-zoo-registry.json`. Prevents network calls to HuggingFace. Legacy `WSINFER_ZOO_REGISTRY_PATH` still honored (emits `DeprecationWarning`). |
| `S3_STORAGE_OPTIONS`           | If S3    | JSON passed to `s3fs` / `fsspec` for AWS credentials (e.g. `'{"profile":"saml"}'`).      |
| `GS_STORAGE_OPTIONS`           | If GCS   | JSON passed to `gcsfs` / `fsspec` for Google Cloud Storage (`gs://`). Optional: defaults to Application Default Credentials; override e.g. `'{"token":"/path/sa.json"}'`. |
| `WSINSIGHT_REMOTE_CACHE_DIR`   | No       | Local cache dir for remote assets. Default: `~/.cache/wsinsight`.                        |
| `KERAS_HOME`                   | No       | Override Keras config/weights directory.                                                  |
| `CUDA_VISIBLE_DEVICES`         | No       | Pin to specific GPU(s) (e.g. `0` or `0,1`).                                             |
| `WSINSIGHT_EXPERIMENTAL`       | No       | Set to `1` to unlock experimental subcommands (`hplot`, `hplot-finalize`, `ecomp`, `tcomp`, `cme`, `cme-profile`). Not needed for normal use. |

\* Required when HuggingFace Hub is unreachable (SSL errors, air-gapped).

---

## 4. CLI Reference

### 4.1 Command Map

```text
wsinsight
├── run               One-shot: patch → infer → (optional ncomp) → export
├── patch             Tissue segmentation + patch extraction → HDF5
├── infer             Model inference on cached patches → CSV
├── reg               Post-hoc region registration
├── ncomp             Node-level (cell) composition + Delaunay graph cache
├── export            Merge analytics → GeoJSON / OME-CSV
└── describe          Emit a machine-readable JSON schema of every subcommand
```

> Additional subcommands — `hplot`, `hplot-finalize`, `ecomp`, `tcomp`, `cme`,
> `cme-profile` — are gated as **experimental**. They are hidden from `--help`
> and cannot be
> executed unless `WSINSIGHT_EXPERIMENTAL=1` is exported. Their CLI flags,
> output schemas, and metric definitions may change without notice. This
> skill file documents only the stable surface.

### 4.2 Global Options (All Commands)

| Flag           | Description                         |
| -------------- | ----------------------------------- |
| `--backend`    | Slide reading backend               |
| `--log-level`  | Logging verbosity                   |
| `--version`    | Print version and exit              |

### 4.3 `wsinsight run` — Full Pipeline

The one-shot orchestrator. Delegates to `patch`, `infer`, and optionally
`ncomp` + `export` in sequence.

```bash
wsinsight run \
  --wsi-dir <WSI_DIR> \
  --results-dir <RESULTS_DIR> \
  --model <MODEL_NAME> \
  [--batch-size 32] \
  [--num-workers 8] \
  [--ncomp] \
  [--export-geojson] [--export-omecsv]
```

**Stable options:**

| Option                          | Type      | Description                                                                                  |
| ------------------------------- | --------- | -------------------------------------------------------------------------------------------- |
| `--wsi-dir / -i`                | path/URI  | Directory of WSIs (local, S3, `gdc-manifest://`, `image-list://`). Required.                 |
| `--results-dir / -o`            | path/URI  | Output directory (auto-created, including new S3 prefixes). Required.                        |
| `--model / -m`                  | string    | Registered model name (mutually exclusive with `--config`/`--model-path`/`--zoo-model-dir`). |
| `--config / -c`                 | path      | Custom model config JSON (use with `--model-path`).                                          |
| `--model-path / -p`             | path      | Custom TorchScript weights (use with `--config`).                                            |
| `--zoo-model-dir / -z`          | path      | Folder with `config.json` + `torchscript_model.pt`.                                          |
| `--batch-size / -b`             | int       | Inference batch size (default 32).                                                           |
| `--num-workers / -n`            | int       | Dataloader workers (default 8; `0` = single-threaded).                                       |
| `--cache-image-patches`         | flag      | Save extracted RGB patches into `patches/<slide>.h5` under `/images`.                        |
| `--qupath`                      | flag      | Build a QuPath project containing the inference results.                                     |
| `--region-inference-dir / -r`   | path/URI  | Prior region (patch-based) results dir; adds `region_prob_*` columns to per-cell outputs.    |
| `--qupath-measurement-detection-dir` | path | Per-slide QuPath TSV detection-measurement exports → pseudo-model.                          |
| `--qupath-geojson-detection-dir`| path      | QuPath GeoJSON detections → pseudo-model.                                                    |
| `--qupath-geojson-annotation-dir`| path     | QuPath GeoJSON annotations seed region labels.                                               |
| `--qupath-detection-patch-size` | int       | Pseudo-model patch size for detections (default 56).                                         |
| `--qupath-annotation-patch-size`| int       | Pseudo-model patch size for annotations (default 224).                                       |
| `--qupath-spacing-um-px`        | float     | Pseudo-model spacing (default 0.5).                                                          |
| `--qupath-name-as-class`        | flag      | Use the QuPath `name` field as class instead of `Classification`.                            |
| `--histoqc-dir`                 | path      | Directory of HistoQC outcomes (replaces tissue segmentation).                                |
| `--seg-thumbsize`               | str       | Thumbnail size for tissue segmentation (default `[2048,2048]`).                              |
| `--seg-median-filter-size`      | int       | Median filter kernel (odd, default 7).                                                       |
| `--seg-binary-threshold`        | int       | Binarisation threshold (default 7).                                                          |
| `--seg-closing-kernel-size`     | int       | Binary-closing kernel (default 6).                                                           |
| `--seg-min-object-size-um2`     | float     | Min retained tissue object area in µm² (default 40000).                                      |
| `--seg-min-hole-size-um2`       | float     | Min retained hole area in µm² (default 36100).                                               |
| `--patch-overlap-ratio`         | float     | Patch overlap ratio (default 0.0 = non-overlapping).                                         |
| `--patch-size-um`               | float     | Patch side length in µm (default 0 → use model default).                                     |
| `--patch-size-px`               | float     | Patch side length in px (default 0 → use model default).                                     |
| `--ncomp`                       | flag      | Run node-level cell composition after inference.                                             |
| `--ncomp-max-neighbor-distance` | float     | Max Delaunay edge length (µm) for ncomp (default 25.0).                                      |
| `--ncomp-k`                     | int       | k-hop radius for ncomp (default 2).                                                          |
| `--export-geojson`              | flag      | After analytics merge per-cell tables → `export-geojson/`.                                   |
| `--export-omecsv`               | flag      | Same, → `export-omecsv/`.                                                                    |
| `--overwrite`                   | flag      | Recompute existing outputs across every stage.                                               |

**Experimental run flags** (require `WSINSIGHT_EXPERIMENTAL=1`):
`--hplot` (+ `--hplot-max-neighbor-distance`, `--hplot-base-types`,
`--hplot-target-types`, `--hplot-k`, `--hplot-n`, `--hplot-r`,
`--hplot-range-min`, `--hplot-range-max`, `--hplot-samples-with-valid-range-only`),
`--ecomp` (+ `--ecomp-max-edge`, `--ecomp-k`),
`--tcomp` (+ `--tcomp-max-edge`, `--tcomp-k`),
`--cme`   (+ `--cme-hoptimus`, `--cme-clusters`).

These flags are accepted by `run` only when the corresponding subcommand is
enabled; they remain undocumented and unstable.

### 4.4 `wsinsight patch` — Tissue Segmentation & Patch Extraction

```bash
wsinsight patch \
  --wsi-dir <WSI_DIR> \
  --results-dir <RESULTS_DIR> \
  --model <MODEL_NAME>
```

Creates `masks/` and `patches/` under `--results-dir`. Honors all
`--seg-*`, `--patch-*`, `--qupath-*`, `--histoqc-dir`, `--region-inference-dir`,
and `--cache-image-patches` options listed in §4.3 (the `patch` and `infer`
stages share the same surface as `run`). Writes `patch_metadata_<ts>.json`.

### 4.5 `wsinsight infer` — Model Inference

```bash
wsinsight infer \
  --results-dir <RESULTS_DIR> \
  --model <MODEL_NAME> \
  [--batch-size 32] [--num-workers 4] [--stitch-workers 8] [--overwrite]
```

Reads from `patches/`, writes to `model-outputs-csv/` plus
`infer_metadata_<ts>.json`. Accepts the same `--region-inference-dir`,
`--qupath-*`, and `--patch-*` options as `run`. `--stitch-workers` controls
the TileFuse thread pool used to assemble object-based detections.

### 4.6 `wsinsight ncomp` — Node-level (Cell) Composition

Builds (or reuses) a Delaunay cell graph per slide under
`graphs/<slide>.h5` and emits per-cell k-hop neighborhood composition.

```bash
wsinsight ncomp \
  --wsi-dir <WSI_DIR> \
  --results-dir <RESULTS_DIR> \
  [--ncomp-k 2] [--ncomp-max-neighbor-distance 25.0] \
  [--num-workers 8] [--overwrite]
```

Defaults: 25 µm edge filter, 2-hop neighborhood radius, 8 concurrent slides.
Outputs go to `ncomp-outputs-csv/<slide>.csv`. The `graphs/<slide>.h5` cache
is keyed by a SHA-256 hash of the cell-center coordinates, so `ncomp` reruns
are idempotent and safe to resume.

### 4.7 `wsinsight export` — Merge & Export

```bash
wsinsight export \
  --results-dir <RESULTS_DIR> \
  --geojson --omecsv \
  [--object-type detection] [--patch-overlap-ratio 0.0] \
  [--export-workers 4] [--overwrite]
```

Left-joins per-cell analytics under `RESULTS_DIR` (`model-outputs-csv/`,
`hplot-outputs-csv/cells/`, `ncomp-outputs-csv/`) into `export-csv/`, then
serialises to `export-geojson/` and/or `export-omecsv/`. Edge-level
(`ecomp-outputs-csv/`) and triad-level (`tcomp-outputs-csv/`) products use
different primary keys and are **not** merged — consume them directly.
`--object-type` is one of `tile`, `detection` (default), or `annotation` and
is embedded into each exported feature for QuPath. `--patch-overlap-ratio`
must match the value used at inference time to recover the correct
shrunk-tile geometry.

### 4.8 `wsinsight reg` — Post-hoc Object Registration

Post-hoc registration of an existing object-level (cell) inference against a
region-level (patch) inference and/or against another object-level inference:

```bash
wsinsight reg \
  --results-dir <CELL_RESULTS> \
  [--region-inference-dir <REGION_RESULTS>] \
  [--object-inference-dir <OTHER_CELL_RESULTS>] \
  [--tag <NAMESPACE>] \
  [--radius-um 5.0] [--spacing-um-px 0.25] \
  [--geojson] [--omecsv] [--export-workers 4] [--overwrite]
```

For each cell, region matching adds `region_prob_*` columns from the patch
inference; object-to-object matching pairs cells across two object runs
within `--radius-um` (using `--spacing-um-px` to convert). `--tag` namespaces
the added columns (`<kind>_<tag>_prob_*`). `--wsi-dir / -i`, when supplied,
restricts the run to slides whose stem appears under that directory.
GeoJSON / OME-CSV exports for the registered tables land in dedicated
subfolders.

### 4.9 `wsinsight describe` — Machine-readable CLI Schema

Emits a JSON description of every subcommand, its options, types, defaults,
and flag forms. Intended for downstream tooling (e.g. the QuPath extension)
that needs to render forms without hard-coding the CLI.

```bash
wsinsight describe                         # stdout
wsinsight describe --output schema.json    # file
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

### Available Models in the Bundled WSInsight Zoo

Resolved automatically from `wsinsight/zoo/wsinsight-zoo-registry.json` via
`WSINSIGHT_ZOO_REGISTRY_PATH` (legacy `WSINFER_ZOO_REGISTRY_PATH` is still
respected with a deprecation warning).

**WSInsight-native (cell-level / object-based)**

- `CellViT-256-x20`, `CellViT-256-x40`, `CellViT-256-x40-AMP`
- `CellViT-SAM-H-x20`, `CellViT-SAM-H-x40`, `CellViT-SAM-H-x40-AMP`
- `CellViT-Virchow-x40-AMP`
- `10xGenomics-BRCA-CellViT-SAM-H-x40`,
  `10xGenomics-CRC-CellViT-SAM-H-x40`
- `hovernet_fast_pannuke`
- `hne_cell_classification`

**WSInfer Zoo (region / patch-level), pre-registered for convenience**

- `breast-tumor-resnet34.tcga-brca`
- `lung-tumor-resnet34.tcga-luad`
- `pancreas-tumor-preactresnet34.tcga-paad`
- `prostate-tumor-resnet34.tcga-prad`
- `pancancer-lymphocytes-inceptionv4.tcga`
- `lymphnodes-tiatoolbox-resnet50.patchcamelyon`
- `colorectal-tiatoolbox-resnet50.kather100k`
- `colorectal-resnet34.penn`

Any additional WSInfer Zoo model can be added by extending the registry JSON
or by passing `--config` + `--model-path` (or `--zoo-model-dir`).

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
│   └── <slide>.geojson             Inference GeoJSON (when produced by run)
├── model-outputs-omecsv/
│   └── <slide>.ome.csv.gz          Inference OME-CSV (when produced by run)
├── ncomp-outputs-csv/
│   └── <slide>.csv                 Per-cell composition (node-level)
├── graphs/
│   └── <slide>.h5                  Delaunay cache (produced by ncomp/ecomp/tcomp/cme)
├── export-csv/
│   └── <slide>.csv                 Merged per-cell CSV (model + ncomp [+ hplot])
├── export-geojson/
│   └── <slide>.geojson             GeoJSON export
├── export-omecsv/
│   └── <slide>.ome.csv.gz          OME-CSV export
├── patch_metadata_<ts>.json        Patch-stage configuration & runtime info
└── infer_metadata_<ts>.json        Inference-stage configuration & runtime info
```

> Experimental subcommands add their own subdirectories alongside the stable
> ones: `hplot-outputs-csv/{cells,...}/`, `ecomp-outputs-csv/<slide>.csv`,
> `tcomp-outputs-csv/<slide>.csv`, `cme-outputs-csv/{cells,cmes}/<slide>.csv`,
> and matching `*-outputs-geojson/` folders. Schemas for these are
> intentionally not pinned here.

### 6.1 `patches/<slide>.h5` — Patch Coordinates (HDF5)

| HDF5 path | Type | Shape / dtype | Description |
|---|---|---|---|
| `/slide` | group | — | Slide metadata group |
| `/slide.attrs["slide_path"]` | attribute | UTF-8 string | Original WSI file path |
| `/slide.attrs["slide_mpp"]` | attribute | float | Microns per pixel |
| `/slide.attrs["slide_width"]` | attribute | float | Slide width in pixels |
| `/slide.attrs["slide_height"]` | attribute | float | Slide height in pixels |
| `/coords` | dataset | `(N, 2)` int32 | Patch top-left `[x, y]` at level 0; gzip-compressed |
| `/coords.attrs["patch_size"]` | attribute | int | Patch side length in pixels |
| `/coords.attrs["patch_level"]` | attribute | int | Always `0` |
| `/coords.attrs["patch_spacing_um_px"]` | attribute | float | Spacing in µm/px |
| `/coords.attrs["tile_dim"]` | attribute | `(2,)` int (optional) | Tile dimensions when set |
| `/images` | dataset (optional) | `(N, patch_size, patch_size, 3)` uint8 | Cached RGB patches; only when `cache_image_patches=True` |
| `/polygons` | group (optional) | — | Present when polygons are supplied |
| `/polygons/coords` | dataset | `(K, 2)` float32 | All polygon vertices concatenated; attr `columns=["x","y"]` |
| `/polygons/offsets` | dataset | `(N+1,)` int64 | Ragged offsets: polygon `i` → `coords[offsets[i]:offsets[i+1]]` |
| `/polygons.attrs["layout"]` | attribute | string | Always `"ragged_offsets"` |

### 6.2 `model-outputs-csv/<slide>.csv` — Inference Results

| Column | dtype | Description |
|---|---|---|
| `minx` | int | Patch top-left x (level-0 pixels) |
| `miny` | int | Patch top-left y (level-0 pixels) |
| `width` | int | Patch width in pixels |
| `height` | int | Patch height in pixels |
| `prob_<class>` | float32 | One column per model class (e.g. `prob_tumor`, `prob_lymphocyte`). Names from `model_info.config.class_names` |

Conditional columns:

| Column | When present |
|---|---|
| `qupath_detection_parent` | QuPath pseudo-model with detection TSV |
| `region_minx`, `region_miny`, `region_width`, `region_height`, `region_prob_<class>`, … | `--region-inference-dir` set with `object_based=True`. All region CSV columns prefixed with `region_` |

**Deriving a per-cell tumor mask from `region_prob_*`**: when region inference
is enabled, each cell carries one `region_prob_<class>` column per region class.
Argmax over those columns yields a per-cell region label — select the tumor
column (e.g. `region_prob_Tumor` for BRCA, `region_prob_ColorectalAdenocarcinomaEpithelium`
for CRC) to build a `bool` tumor/non-tumor mask for region-stratified analytics.

```python
import pandas as pd
df = pd.read_csv("results/model-outputs-csv/SLIDE.csv")
region_cols = [c for c in df.columns if c.startswith("region_prob_")]
region_argmax = df[region_cols].to_numpy().argmax(axis=1)
in_tumor = region_argmax == region_cols.index("region_prob_Tumor")
```

### 6.3 `graphs/<slide>.h5` — Delaunay Graph Cache (HDF5)

| HDF5 path | Type | Shape / dtype | Description |
|---|---|---|---|
| `file.attrs["num_cells"]` | attribute | int64 | Number of cell centers N |
| `file.attrs["mpp"]` | attribute | float64 | Microns per pixel |
| `file.attrs["centers_hash"]` | attribute | bytes (np.void) | SHA-256 of `cell_centers.tobytes()` for cache invalidation |
| `cell_centers` | dataset | `(N, 2)` int32 | Cell center `[x, y]` |
| `simplices` | dataset | `(M, 3)` int32 | Delaunay triangle vertex indices |
| `edges_source` | dataset | `(E,)` int32 | Source node index per undirected edge |
| `edges_target` | dataset | `(E,)` int32 | Target node index per undirected edge |
| `edges_length` | dataset | `(E,)` float64 | Euclidean edge length in pixels |

Edges are stored **unpruned**; pruning to `max_edge_length_px` happens at read time.

### 6.4 `ncomp-outputs-csv/<slide>.csv` — Node-level (Cell) Composition

| Column | Description |
|---|---|
| `center_x` | Cell center x |
| `center_y` | Cell center y |
| `cell_type` | Argmax class label (e.g. `"tumor"`, `"lymphocyte"`) |
| `neighborhood_size` | Number of k-hop neighbors (excluding self) |
| `neighborhood_<type>_count` | Count of neighbors per class (e.g. `neighborhood_tumor_count`) |
| `neighborhood_<type>_prop` | Proportion of neighbors per class; NaN when `neighborhood_size == 0` |

### 6.5 `export-csv/<slide>.csv` — Merged Per-Cell Export

Left-join of the available analytics on a per-cell basis:

| Source | Join key | Columns added |
|---|---|---|
| `model-outputs-csv/` | *(base table)* | `minx`, `miny`, `width`, `height`, `prob_<class>`, `center_x`, `center_y` |
| `ncomp-outputs-csv/` | `center_x`, `center_y` | `cell_type`, `neighborhood_size`, `neighborhood_<type>_count`, `neighborhood_<type>_prop` |

When experimental subcommands were run, `export-csv/` may include additional
columns sourced from their outputs.

### 6.6 GeoJSON Outputs

`model-outputs-geojson/` and `export-geojson/` use the same schema — a GeoJSON FeatureCollection:

```json
{
  "type": "Feature",
  "id": "<uuid4>",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[x1,y1],[x2,y2],...]]
  },
  "properties": {
    "isLocked": true,
    "objectType": "tile",
    "classification": { "name": "prob_<winner>", "color": [R,G,B] },
    "measurements": { "<every numeric non-geometry column>": value }
  }
}
```

Geometry is the overlap-shrunk patch rectangle by default. `measurements` includes all numeric columns **except** `minx`, `miny`, `width`, `height`, `center_x`, `center_y`.

### 6.7 OME-CSV Outputs

`model-outputs-omecsv/` and `export-omecsv/` write gzip-compressed OME-CSV files:

| Column | Description |
|---|---|
| `object` | Row index (int) |
| `secondary_object` | Same as `object` |
| `polygon` | WKT polygon string for the overlap-shrunk box |
| `objectType` | Always `"tile"` |
| `classification` | Argmax class name (prob_ prefix stripped) |
| *(all numeric non-geometry columns)* | `prob_*` and `ncomp` columns when present. NaN → `"NaN"` |

---

## 7. Common Workflows

### 7.1 Basic Inference (Smallest Useful Run)

```bash
wsinsight run \
  --wsi-dir slides/ \
  --results-dir results/ \
  --model breast-tumor-resnet34.tcga-brca
```

### 7.2 Full Pipeline with ncomp + Export

```bash
wsinsight run \
  --wsi-dir slides/ \
  --results-dir results/ \
  --model pancancer-lymphocytes-inceptionv4.tcga \
  --batch-size 32 \
  --ncomp \
  --export-geojson --export-omecsv
```

### 7.3 Step-by-Step (Resumable / Parallelizable)

```bash
# Step 1: Patch extraction (resumable)
wsinsight patch --wsi-dir slides/ --results-dir results/ --model breast-tumor-resnet34.tcga-brca

# Step 2: Inference
wsinsight infer --results-dir results/ --model breast-tumor-resnet34.tcga-brca --batch-size 32

# Step 3: Composition analytics (builds graphs/<slide>.h5 on first run)
wsinsight ncomp --wsi-dir slides/ --results-dir results/

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

### 7.5 Multi-GPU Parallel Inference

Split slide lists into per-GPU shards and run each with
`CUDA_VISIBLE_DEVICES=<N>` pinning, sharing the same `--results-dir`.
`patch`, `infer`, `ncomp`, and `export` are all idempotent per-slide, so
they can run concurrently across shards.

### 7.6 Reading Results Programmatically

```python
import pandas as pd

# Per-cell inference
df = pd.read_csv("results/model-outputs-csv/SLIDE_001.csv")
print(df[["minx", "miny", "prob_tumor"]].head())

# Merged export (model + ncomp joined per cell)
df_export = pd.read_csv("results/export-csv/SLIDE_001.csv")
print(df_export.columns.tolist())
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

## 9. Acquiring TCGA Clinical & Molecular Data

WSInsight produces per-slide morphological outputs.  Linking them to clinical
endpoints (survival, treatment, subtypes) requires external clinical tables.
TCGA slide filenames encode the patient barcode:

```text
TCGA-A7-A0CE-01Z-00-DX1.svs
└─── patient ───┘
```

Extract the first 12 characters (`TCGA-A7-A0CE`) as the join key.

### 9.1 GDC `/cases` API — Demographics, Staging & Treatment

```bash
curl 'https://api.gdc.cancer.gov/cases?filters={"op":"=","content":{"field":"cases.project.project_id","value":"TCGA-BRCA"}}&expand=diagnoses,diagnoses.treatments,demographic&size=99999&format=TSV' \
  > tcga-brca-clinical.tsv
```

Fields returned via `expand=`:

| Expand path              | Key fields                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `demographic`            | `vital_status`, `days_to_death`, `days_to_birth`, `gender`, `race`, `ethnicity`                |
| `diagnoses`              | `ajcc_pathologic_stage`, `ajcc_pathologic_t/n/m`, `primary_diagnosis`, `age_at_diagnosis`, `days_to_last_follow_up`, `days_to_recurrence`, `progression_or_recurrence`, `tumor_grade` |
| `diagnoses.treatments`   | `treatment_type` (Surgery / Radiation / Pharmaceutical), `therapeutic_agents`, `treatment_intent_type` (Adjuvant / First-Line) |

Join key: `submitter_id` in the TSV (e.g. `TCGA-A7-A0CE`).

### 9.2 Curated Survival — Liu et al. 2018 (Recommended)

The GDC raw fields (`days_to_death`, `days_to_last_follow_up`) require manual
curation.  **Liu et al.** already did this for all 33 TCGA cancer types:

> Liu J et al. "An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive
> High-Quality Survival Outcome Analytics." *Cell* 173(2):400-416, 2018.
> DOI: [10.1016/j.cell.2018.02.052](https://doi.org/10.1016/j.cell.2018.02.052)

Download **Supplementary Table 1** (Excel).  Columns:

| Column      | Meaning                                |
| ----------- | -------------------------------------- |
| `bcr_patient_barcode` | Patient ID (e.g. `TCGA-A7-A0CE`) |
| `OS`, `OS.time`       | Overall Survival (event + days)  |
| `PFI`, `PFI.time`     | Progression-Free Interval        |
| `DFI`, `DFI.time`     | Disease-Free Interval            |
| `DSS`, `DSS.time`     | Disease-Specific Survival        |

This is the gold-standard source for survival analysis on TCGA data.

### 9.3 Molecular Subtypes & Biomarkers

| Data type               | Best source                                | Join key           | Notes                                                        |
| ----------------------- | ------------------------------------------ | ------------------ | ------------------------------------------------------------ |
| **PAM50** (BRCA)        | TCGA BRCA paper supplementary or cBioPortal | `PATIENT_ID`       | Luminal A/B, HER2-enriched, Basal-like, Normal-like          |
| **MSI-H / MSS**         | TCGA PanCanAtlas or MANTIS/MSISensor scores | `PATIENT_ID`       | Relevant for COAD, READ, STAD, UCEC                          |
| **Immune subtypes**     | Thorsson et al. 2018 (*Immunity*)          | `TCGA Participant Barcode` | C1–C6 pan-cancer immune subtypes                     |
| **ER / PR / HER2**      | cBioPortal clinical data tab               | `PATIENT_ID`       | Receptor status for breast cancer                            |
| **Mutations / CNA**     | cBioPortal or GDC MAF files                | `PATIENT_ID`       | TP53, PIK3CA, BRAF, KRAS, etc.                               |

### 9.4 cBioPortal — One-Stop Download

[cBioPortal](https://www.cbioportal.org/) aggregates clinical, molecular, and
genomic data into downloadable TSVs.  Example for TCGA-BRCA:

```text
https://www.cbioportal.org/study/clinicalData?id=brca_tcga_pan_can_atlas_2018
```

Download the "Clinical Data" tab as TSV — it includes PAM50, ER/PR/HER2
status, survival, and staging in a single table keyed by `PATIENT_ID`.

### 9.5 Joining Clinical Data with WSInsight Outputs

```python
import pandas as pd
from pathlib import Path

# Load WSInsight per-slide output
slide_csv = Path("results/model-outputs-csv/TCGA-A7-A0CE-01Z-00-DX1.csv")
df = pd.read_csv(slide_csv)

# Extract patient barcode from filename
patient_id = slide_csv.stem.rsplit("-", 3)[0]   # "TCGA-A7-A0CE"

# Load clinical table (e.g. Liu et al. or cBioPortal)
clinical = pd.read_csv("tcga-brca-clinical.tsv", sep="\t")

# Join
patient_row = clinical[clinical["bcr_patient_barcode"] == patient_id]
print(patient_row[["OS", "OS.time", "PFI", "PFI.time"]].iloc[0])
```

For cohort-level analysis, iterate over all CSVs in `model-outputs-csv/`,
extract each patient barcode, and merge into a single DataFrame.

### 9.6 Example: cross-cohort biomarker-landscape screen

A reference end-to-end pattern (TCGA-BRCA + TCGA-CRC) lives at
`experiments/biomarker-landscape.ipynb`. It uses only two WSInsight artefacts
per slide — `model-outputs-csv/<slide>.csv` and `graphs/<slide>.h5` (both
produced by the stable `infer` + `ncomp` pipeline) — and builds per-slide
features stratified into three region strata (`all` / `tumor` / `nontumor`)
from the cached Delaunay graph:

```python
import h5py, numpy as np
with h5py.File("results/graphs/SLIDE.h5", "r") as fh:
    simplices = fh["simplices"][()]           # (M, 3) int32
    centers   = fh["cell_centers"][()]        # (N, 2) int32

# Prune simplices by µm edge length (25 µm default, matches ncomp).
EDGE_LIMIT_UM = 25.0
SPACING_UM_PX = 0.25                          # 40x slides
edge_limit_px = EDGE_LIMIT_UM / SPACING_UM_PX
coords = centers[simplices]
max_edge = np.linalg.norm(coords[:, [0, 1, 0]] - coords[:, [1, 2, 2]], axis=2).max(axis=1)
simp_ok = simplices[max_edge <= edge_limit_px]
```

Slide features are averaged to the patient level (first-12-char barcode) and
screened univariately against every available clinical score per cohort —
binary outcomes (receptor status, stage, MSI-H, 5-year OS event) via
two-sided Mann–Whitney *U* / AUC, continuous outcomes (tumor purity, CYT,
GZMA, PRF1, Leukocyte Fraction, Thorsson immune scores, age) via Spearman *ρ*,
with Benjamini–Hochberg FDR correction per (cohort, score). Survival is
handled with `lifelines` (KM + Cox HR).

The notebook ships with a BRCA pretreatment filter
(`history_of_neoadjuvant_treatment == "No"`, 1085/1099 patients) so
morphology-derived biomarkers are not confounded by neoadjuvant therapy.
Cohort artefacts land in `biomarker_landscape_results/`
(`slide_features_<cohort>.csv`, `biomarker_landscape_screen.csv`,
`biomarker_landscape_best.csv`, `km_best_per_cohort.{png,svg}`, …).
Adding a new cohort only requires a `COHORTS[...]` entry and a clinical-score
assembly function that emits a long-form
`(patient_barcode, score_name, score_type, value)` table.

> Any features in this notebook that go beyond what `ncomp` emits (e.g.
> triad-level summaries computed from the raw `simplices` dataset) are
> *user-side* derivations. They are not part of the stable WSInsight surface
> and will not track schema changes in the experimental `tcomp` command.

---

## 10. URI & Remote Data Support

`--wsi-dir` and `--results-dir` accept:

| Scheme               | Example                                                    |
| -------------------- | ---------------------------------------------------------- |
| Local path           | `slides/` or `/data/slides`                                |
| S3                   | `s3://bucket/prefix`                                       |
| GCS                  | `gs://bucket/prefix`                                       |
| GDC manifest         | `gdc-manifest:///absolute/path/to/manifest.tsv`            |
| Image list           | `image-list:///path/to/filelist.txt`                       |

A plain local `.txt` file passed as `--wsi-dir` is rejected with a clear
error — prefix it with `image-list://` to pass a slide list.  S3 access
requires `S3_STORAGE_OPTIONS` to be set; GCS access uses Application
Default Credentials by default and can be overridden with `GS_STORAGE_OPTIONS`.

**Important:** The GDC manifest URI scheme is `gdc-manifest://` (not `gdc://`).
The path must be absolute (triple slash: `gdc-manifest:///absolute/path`).

---

## 11. Error Recovery & Troubleshooting

| Symptom                              | Cause                                 | Fix                                                         |
| ------------------------------------ | ------------------------------------- | ----------------------------------------------------------- |
| SSL errors / hang on startup         | HuggingFace Hub unreachable           | Set `WSINSIGHT_ZOO_REGISTRY_PATH` to local registry JSON    |
| `numpy >= 2.0` assertion failure     | Dependency upgraded numpy             | `pip install -c constraints.txt "numpy<2"`                   |
| `ModuleNotFoundError: osgeo`         | GDAL not installed via conda          | `conda install -c conda-forge gdal=3.11.3`                  |
| CUDA out of memory                   | Batch size too large                  | Reduce `--batch-size`                                       |
| Inference produces empty CSV         | Wrong model for slide magnification   | Match model suffix (`x20`/`x40`) to slide magnification     |
| Stale graph cache                    | CSV changed after graph was built     | Automatic: cache detects via SHA-256 hash and rebuilds       |
| Experimental subcommand refuses to run | `WSINSIGHT_EXPERIMENTAL` not set    | Export `WSINSIGHT_EXPERIMENTAL=1` before invoking it         |

---

## 12. Agent Decision Guide

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
│        ├─ Yes → wsinsight run [--ncomp] [--export-geojson] [--export-omecsv]
│        └─ No  → wsinsight patch → wsinsight infer → [ncomp] → wsinsight export
├─ No, but results-dir exists with model-outputs-csv/ → Skip patch+infer
│        ├─ Need per-cell neighborhood composition? → wsinsight ncomp
│        └─ Need GeoJSON / OME-CSV?                 → wsinsight export --geojson --omecsv
├─ No slides, but user mentions TCGA / GDC / cancer cohort
│        → Query GDC API for manifest (Section 8) → save .tsv
│        → wsinsight run --wsi-dir "gdc-manifest:///path/to/manifest.tsv" ...
├─ User needs clinical / molecular labels (survival, PAM50, MSI, treatment)
│        → See Section 9: GDC /cases API, Liu et al. 2018, or cBioPortal
│        → Join on first 12 chars of slide filename (patient barcode)
└─ No slides or results → Ask user for --wsi-dir
```

### Key Constraints for Agents

1. **Model is required** for `run`, `patch`, and `infer`. Use `wsinfer-zoo ls`
   to list available models, or ask the user.
2. **`--overwrite`** is needed to recompute existing outputs. Without it,
   completed slides are skipped (idempotent / resumable).
3. **Environment variables** must be exported before the `wsinsight` command,
   not passed as CLI flags.
4. **`constraints.txt`** should always be used with `pip install -c` to prevent
   dependency drift.
5. **When the user mentions TCGA, GDC, or a cancer cohort** (e.g. "analyze
   TCGA-BRCA slides"), use the GDC API `curl` pattern from Section 8 to
   generate a manifest TSV, then pass it via
   `--wsi-dir "gdc-manifest:///absolute/path/to/manifest.tsv"`. Do not ask
   the user to download slides manually.
6. **Prefer Docker when available** — it avoids all local dependency
   installation. Use `bash docker-run.sh /path/to/data` (or a manual
   `docker run`) and run `wsinsight` commands inside the container. The
   image pre-sets `WSINSIGHT_ZOO_REGISTRY_PATH` and `KERAS_HOME`.
7. **When the user needs clinical labels** (survival, PAM50, MSI, treatment),
   refer to Section 9. Use the GDC `/cases` API for demographics/staging,
   Liu et al. 2018 for curated survival endpoints, and cBioPortal for
   molecular subtypes. Join on the first 12 characters of the slide filename.
8. **Do not recommend the experimental subcommands** (`hplot`,
   `hplot-finalize`, `ecomp`, `tcomp`, `cme`) unless the user has explicitly
   opted in via `WSINSIGHT_EXPERIMENTAL=1`. Their CLI surfaces and output
   schemas are unstable.

---

## 13. Verification Checklist

After installation, an agent should verify:

```bash
# 1. CLI loads without errors
wsinsight --help

# 2. All stable sub-commands are registered
wsinsight run --help
wsinsight patch --help
wsinsight infer --help
wsinsight ncomp --help
wsinsight reg --help
wsinsight export --help
wsinsight describe --help

# 3. Python imports work
python -c "import wsinsight; print(wsinsight.__version__)"

# 4. CUDA is available (for GPU inference)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 5. numpy is below 2.0
python -c "import numpy; print('numpy:', numpy.__version__)"
```

---

## 14. MCP server (FastMCP)

WSInsight ships an optional [Model Context Protocol](https://modelcontextprotocol.io/)
server that exposes the same CLI surface to MCP-compatible clients
(Claude Desktop, VS Code Copilot, custom agents).

```bash
pip install 'wsinsight[mcp]'
wsinsight-mcp                       # stdio (default)
wsinsight-mcp --http 127.0.0.1:8765 # streamable HTTP, localhost-only
```

Auto-registered tools (stable surface):

- **Long-running** (return `job_id`, poll `job_status` / `job_logs`,
  stop with `cancel_job`): `run`, `patch`, `infer`, `ncomp`.
- **Short-running** (block, return exit code + log tail): `export`, `reg`.
- **Meta**: `job_status`, `job_logs`, `cancel_job`, `list_jobs`.
- **Resources**: `wsinsight://schema`, `wsinsight://models`,
  `wsinsight://results/{results_dir}/layout`.
- **Prompt**: `reproduce_tcga_crc`.

With `wsinsight-mcp --experimental` the server additionally exposes
`hplot`, `ecomp`, `tcomp`, `cme` (long-running) and `hplot-finalize`
(short-running). Child processes inherit `WSINSIGHT_EXPERIMENTAL=1`
automatically. The set of long-running commands lives in
`wsinsight/mcp/schema.py::LONG_RUNNING_COMMANDS`.

Each tool's input schema mirrors the CLI parameter names, types, and
defaults verbatim (the CLI JSON schema in `wsinsight/cli/cli_schema.json`
is the single source of truth). One `cancel_job` call sends `SIGINT`
(engaging the existing two-press cooperative-cancel handler in
`wsinsight/cancel.py`); a second call escalates to `SIGTERM`.

Concurrency defaults to the number of visible GPUs (parsed from
`CUDA_VISIBLE_DEVICES`, else `torch.cuda.device_count()`, else 1) and
each running job is pinned to one GPU via `CUDA_VISIBLE_DEVICES` in the
child process's environment. Override with `--max-concurrent N`.

Full docs: [`wsinsight/mcp/README.md`](wsinsight/mcp/README.md).

### Claude Desktop config snippet

```json
{
  "mcpServers": {
    "wsinsight": { "command": "wsinsight-mcp", "args": [] }
  }
}
```

### VS Code Copilot MCP config snippet

```json
{
  "servers": {
    "wsinsight": { "type": "stdio", "command": "wsinsight-mcp" }
  }
}
```
