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

## 2. Assumed Install

Assume one of these already exists, and verify with `wsinsight --help` before
anything else. Installing is the fallback, not the normal path: only when that
verification fails **and** Docker is unavailable do you run `conda-setup.sh`.
This is the precedence the §10 decision guide follows.

| Path                 | Command                                                          |
| -------------------- | ---------------------------------------------------------------- |
| Docker (preferred)   | `docker pull huangchtw/wsinsight:latest`                          |
| Conda                | `bash ./conda-setup.sh <ENV_NAME> [-m\|--mcp] [-d\|--dev] [-r\|--reset]` |

`conda-setup.sh` is the maintained recipe; do not hand-assemble a conda
environment from a list of `pip install` lines.

### 2.1 Docker (no local installation required)

A prebuilt GPU-enabled image based on `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
is published to Docker Hub.  All dependencies (conda, GDAL, PyTorch, TensorFlow,
WSInsight) are pre-installed — **users do not need to install anything locally
except Docker and the NVIDIA Container Toolkit**.

```bash
# Pull the published image
docker pull huangchtw/wsinsight:latest
```

### 2.2 Running via `docker run`

Invoke `docker run` directly. The canonical form is:

```bash
docker run --rm -i \
  --gpus all \
  --shm-size=32g \
  --init \
  -e HOST_UID -e HOST_GID \
  -e TMPDIR=/tmp \
  -v /path/to/data:/workspace \
  -v wsinsight-hf-cache:/app/hf-cache \
  huangchtw/wsinsight:latest \
  bash -lc 'wsinsight run --wsi-dir /workspace/slides --results-dir /workspace/results \
            --zoo-model-dir /workspace/zoo/<hf_repo_id>/<hf_revision>'
```

Every flag is load-bearing:

| Flag | Why it is there |
| ---- | ---------------- |
| `--gpus all` | All GPUs. Pin one with `--gpus device=2`. |
| `--shm-size=32g` | PyTorch DataLoader workers use `/dev/shm`; the 64 MB default causes worker crashes. |
| `--init` | Reaps zombie children so a cancelled run does not leave orphans. |
| `-e HOST_UID -e HOST_GID` | Forwarded **only when set** in your shell. Controls the uid the entrypoint drops to. |
| `-e TMPDIR=/tmp` | PyTorch creates temp dirs at import; point it at `/workspace/.tmp` when the container's `/tmp` is small. |
| `-v <data>:/workspace` | Must be `/workspace` — the entrypoint stats that exact path to decide the uid. |
| `-v wsinsight-hf-cache:/app/hf-cache` | Named volume; persists downloaded weights between runs. |
| `bash -lc '<cmd>'` | The login shell activates the conda env. Omit it entirely for an interactive shell. |

Use `-it` for an interactive session, but **`-i` alone when running
non-interactively** — `-t` fails without a TTY, which is the usual case for an
agent or CI job.

**Interactive shell, all GPUs**

```bash
docker run --rm -it --gpus all --shm-size=32g --init \
  -e HOST_UID -e HOST_GID -e TMPDIR=/tmp \
  -v /path/to/data:/workspace -v wsinsight-hf-cache:/app/hf-cache \
  huangchtw/wsinsight:latest
```

**One command on GPU 2, scratch redirected onto the mount**

```bash
docker run --rm -i --gpus device=2 --shm-size=32g --init \
  -e HOST_UID -e HOST_GID -e TMPDIR=/workspace/.tmp \
  -v /path/to/data:/workspace -v wsinsight-hf-cache:/app/hf-cache \
  huangchtw/wsinsight:latest \
  bash -lc 'wsinsight run -i /workspace/slides -o /workspace/results \
            -z /workspace/zoo/huangch/CellViT-SAM-H-x40/main'
```

**Writing files you own**

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
```

Do **not** pass `--user`. The image starts as root by design: the entrypoint
resolves a target uid from `HOST_UID`/`HOST_GID`, else from the owner of
`/workspace`, else `1000`, then drops privileges with `setpriv`. Starting
non-root makes the entrypoint bail out of that logic
(`if [ "$(id -u)" -ne 0 ]`). If `/workspace` is root-owned and the two
variables are unset, the session runs as root and new files will be root-owned.

The image bakes in `WSINSIGHT_ZOO_REGISTRY_PATH` and `KERAS_HOME`, so the CLI
works inside the container with no environment setup. Refresh it with
`docker pull huangchtw/wsinsight:latest`.

---

## 3. Environment Variables

Set these **before** running any `wsinsight` command.  In restricted or
air-gapped networks the first variable is mandatory.

| Variable                       | Required | Purpose                                                                                  |
| ------------------------------ | -------- | ---------------------------------------------------------------------------------------- |
| `WSINSIGHT_ZOO_REGISTRY_PATH` | Conditional | Path to a local `wsinsight-zoo-registry.json`. Prevents network calls to HuggingFace. Legacy `WSINFER_ZOO_REGISTRY_PATH` still honored (emits `DeprecationWarning`). |
| `S3_STORAGE_OPTIONS`           | If S3    | JSON passed to `s3fs` / `fsspec` for AWS credentials (e.g. `'{"profile":"saml"}'`).      |
| `GS_STORAGE_OPTIONS`           | If GCS   | JSON passed to `gcsfs` / `fsspec` for Google Cloud Storage (`gs://`). Optional: defaults to Application Default Credentials; override e.g. `'{"token":"/path/sa.json"}'`. |
| `WSINSIGHT_REMOTE_CACHE_DIR`   | No       | Local cache dir for remote assets. Default: `~/.cache/wsinsight`.                        |
| `KERAS_HOME`                   | No       | Override Keras config/weights directory. Point it at the folder *containing* `models/`; csbdeep appends `models/StarDist2D/<name>` itself. |
| `SSL_CERT_FILE`                | Conditional | CA bundle for StarDist's weight download. Behind a TLS-inspecting proxy, Python's bundled `certifi` lacks the corporate root and the fetch fails with `CERTIFICATE_VERIFY_FAILED`; the system bundle (e.g. `/etc/pki/tls/certs/ca-bundle.crt`) usually has it. |
| `HF_HUB_OFFLINE`               | Conditional | Set to `1` to stop `huggingface_hub` making any network call. Needed behind a TLS-inspecting proxy, where even an already-downloaded model triggers a HEAD request that fails on the injected certificate. Does **not** by itself make locally-stored weights load — see §5.1. |
| `CUDA_VISIBLE_DEVICES`         | No       | Pin to specific GPU(s) (e.g. `0` or `0,1`).                                             |
| `WSINSIGHT_EXPERIMENTAL`       | No       | Set to `1` to unlock experimental subcommands (`hplot`, `hplot-finalize`, `ecomp`, `tcomp`, `niche`, `niche-profile`, `agg`, `import`). Not needed for normal use. |

\* Required when HuggingFace Hub is unreachable (air-gapped or SSL-restricted
networks); optional otherwise.

---

## 4. CLI Reference

### 4.1 Command Map

```text
wsinsight
├── run               PRIMARY — one-shot: patch → infer → (optional ncomp) → export
├── patch             Tissue segmentation + patch extraction → HDF5
├── infer             Model inference on cached patches → CSV
├── reg               Post-hoc region registration
├── ncomp             Node-level (cell) composition + Delaunay graph cache
├── export            Merge analytics → GeoJSON / OME-CSV
└── schema            Emit a machine-readable JSON schema of every subcommand
```

`WSINSIGHT_EXPERIMENTAL=1` adds `hplot`, `hplot-finalize`, `ecomp`, `tcomp`,
`niche`, `niche-profile`, `agg` and `import`. `wsinsight schema` lists whatever
the running build actually exposes.

> Additional subcommands — `hplot`, `hplot-finalize`, `ecomp`, `tcomp`, `niche`,
> `niche-profile`, `agg`, `import` — are gated as **experimental**. They are hidden from `--help`
> and cannot be
> executed unless `WSINSIGHT_EXPERIMENTAL=1` is exported. Their CLI flags,
> output schemas, and metric definitions may change without notice. This
> skill file documents only the stable surface.
>
> `import` (experimental) maps spatial-transcriptomics expression onto WSInsight
> cells. It reads an `sptx-list://` manifest (`path`<TAB>`sample_id`<TAB>`transform_dir`,
> columns 2 and 3 optional) via `-s`/`--sptx-dir`, transforms each transcriptomics
> cell onto the registered H&E through the ST2WSI SIFT-affine + bUnwarpJ
> B-spline transform, matches it to the nearest `model-outputs-csv` detection,
> and writes one AnnData `.h5ad` per slide under `imported-xenium/`
> (`model-outputs-csv/` is never modified). Supports `--platform xenium`
> (raw Xenium directories) and `--platform xenium-h5ad` (annotated `.h5ad`
> inputs). For `xenium-h5ad`, column 3 should point to each sample transform
> folder containing `registration_params.json`. Every matched
> `model-outputs-csv` column is carried onto the cell under a `model_` prefix
> (plus `model_cell_id`); optional per-cell sidecars requested with
> `--include niche,hplot,ncomp` are merged the same way under their own
> `niche_` / `hplot_` / `ncomp_` prefixes (`model` is always imported and need
> not be listed). Supports `--transform affine|affine+bspline|none` (default
> affine+bspline), `--genes`, `--include`, `--match-max-dist`, and `--dry-run`
> (report the cell↔detection hit-rate only, writing nothing).

### 4.2 Global Options (All Commands)

| Flag           | Description                         |
| -------------- | ----------------------------------- |
| `--backend`    | Slide reading backend               |
| `--log-level`  | Logging verbosity                   |
| `--version`    | Print version and exit              |

### 4.2.1 Pixel Spacing (MPP)

`patch` and `run` accept `--spacing-um-px`:

| Value            | Behaviour                                                        |
| ---------------- | ---------------------------------------------------------------- |
| `0` (default)    | Read the MPP from the slide metadata; error if the slide has none |
| any value `> 0`  | **Override** the metadata with this value, with a warning         |

`patch` records the spacing it used in `patches/<slide>.h5`, and the analysis
commands (`ncomp`, `ecomp`, `tcomp`, `agg`, `hplot`, `niche`) read it back from
there. One `--spacing-um-px` therefore governs the whole pipeline; they only
re-open the slide when that record is missing.

> `reg` has its own `--spacing-um-px` with different, fallback-only semantics.

### 4.2.2 Which Commands Take `--wsi-dir`

Only the stages that read slide pixels: `patch`, `run`, and `import`.

`ncomp`, `ecomp`, `tcomp`, `agg`, `hplot` and `niche` work from `--results-dir`
alone — cells come from `model-outputs-csv/` and the slide list and spacing from
`patches/`. They **reject** `--wsi-dir`, as `export`, `hplot-finalize` and
`niche-profile` always have.

### 4.2.3 What Each Stage Prints

Every stage prints two blocks:

```
Command line
------------
wsinsight run --zoo-model-dir /zoo/my-model/main --spacing-um-px 0.5 ...

Parameters in effect for this stage
-----------------------------------
spacing_um_px = 0.5
...
```

The first is the invocation as typed, shell-quoted so it can be copied and
re-run. The second is what that stage resolved, which is not always the same:
`run` chains `patch` and `infer` with arguments it has already validated. Both
land in `<command>_metadata_<ts>.json` too, as `runtime.args` (a list, so a path
containing spaces stays unambiguous) and `params`.

### 4.3 `wsinsight run` — Full Pipeline

**Start here. `run` is the primary command and covers almost every task.** It
drives the whole workflow end to end; use `patch`, `infer`, `ncomp` or `export`
on their own only to re-run a single stage against results that already exist.

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

**Choosing the model flag.** Use `--model/-m` only when the weights still have
to be downloaded. When the model folder is already on disk, pass
`--zoo-model-dir/-z` with the path to that folder — `-m` always routes through
the HuggingFace Hub, even for a model the registry lists locally:

```bash
wsinsight run -i <WSI_DIR> -o <RESULTS_DIR> \
  -z /path/to/zoo/<hf_repo_id>/<hf_revision>
```

See §5.1 for how to find that path and why `-m` fails without network access.

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
| `--pin-memory / --no-pin-memory`| flag      | Pin DataLoader tensors to CUDA memory (default on). Disable with `--no-pin-memory` in memory-constrained environments where workers are killed by the OOM killer. |
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
| `--export-workers`              | int       | Worker processes for GeoJSON/OME-CSV export (default: auto).                                 |
| `--export-object-type`          | choice    | Object type written to GeoJSON/OME-CSV: `detection` (default) or `annotation`.               |
| `--stitch-workers`              | int       | Thread pool size for TileFuse object-based detection stitching (default: `min(8, CPU // 2)`).|
| `--agg`                         | flag      | Run density-gated aggregate detection after inference (requires `--agg-name` + `--agg-types`).|
| `--agg-name`                    | string    | Product label for the aggregate run (e.g. `tls`); namespaces every artifact.                 |
| `--agg-types`                   | string    | Comma-separated ingredient cell types (e.g. `t_cell,b_cell`).                                |
| `--agg-max-neighbor-distance`   | float     | Max Delaunay edge length (µm) for the aggregate gate (default 25.0).                          |
| `--agg-k`                       | int       | k-hop radius for the density gate (default 2).                                               |
| `--agg-n`                       | int       | Minimum neighborhood size for membership (default 8).                                        |
| `--agg-r`                       | float     | Minimum ingredient-type fraction for membership (default 0.5).                               |
| `--agg-min-size`                | int       | Drop aggregates with fewer than this many cells (default 10).                                |
| `--overwrite`                   | flag      | Recompute existing outputs across every stage.                                               |

Every option in the table above is stable and safe to emit.

#### 4.3.1 Experimental pass-through flags (require `WSINSIGHT_EXPERIMENTAL=1`)

The flag groups below are **not** part of the stable surface. They are accepted
by `run` only when `WSINSIGHT_EXPERIMENTAL=1` is exported and the corresponding
experimental subcommand is enabled; their names, defaults, and semantics remain
undocumented and unstable. Do not emit them unless the user explicitly asks for
an experimental stage.

`--hplot` (+ `--max-neighbor-distance`, `--base-types`,
`--target-types`, `--k`, `--n`, `--r`,
`--range-min`, `--range-max`, `--samples-with-valid-range-only`),
`--ecomp` (+ `--max-edge`, `--k`),
`--tcomp` (+ `--max-edge`, `--k`),
`--niche`   (+ `--hoptimus`, `--clusters`, `--leiden-res`, `--embed-dim`, `--k-hops`, `--max-edge-len-um`, `--max-cell-radius-um`, `--soft`, `--alpha`, `--epochs`, `--patience`, `--min-delta`, `--min-epochs`, `--amp`, `--seed`, `--export-geojson`).

> **Option naming convention.** A standalone subcommand never repeats its own
> name in its options — it is `wsinsight niche --clusters --k-hops --alpha`.
> Only `wsinsight run` prefixes them (`--niche-clusters`, `--niche-k-hops`,
> `--niche-alpha`), because `run` orchestrates many stages and has to keep their
> option namespaces apart.

### 4.4 `wsinsight patch` — Tissue Segmentation & Patch Extraction

```bash
wsinsight patch \
  --wsi-dir <WSI_DIR> \
  --results-dir <RESULTS_DIR> \
  --model <MODEL_NAME>
```

Creates `masks/` and `patches/` under `--results-dir`. Honors all
`--seg-*`, `--qupath-*`, `--histoqc-dir`, `--region-inference-dir`,
and `--cache-image-patches` options listed in §4.3 (the `patch` and `infer`
stages share the same surface as `run`). The patch-sizing flags are exactly
three; on `patch` they carry no prefix: `--overlap-ratio`, `--size-um`,
`--size-px` (on `run` and `infer` the same three are `--patch-overlap-ratio`,
`--patch-size-um`, `--patch-size-px`). Writes `patch_metadata_<ts>.json`.

A slide whose segmentation raises is reported and skipped, and the run
continues; the failures are listed at the end. When **every** slide fails the
command exits non-zero, so a chained `run` stops instead of inferring on
whatever patches an earlier run happened to leave behind:

```
Error: Segmentation failed for every slide; see the errors above. Failed: <slide>
```

### 4.5 `wsinsight infer` — Model Inference

```bash
wsinsight infer \
  --results-dir <RESULTS_DIR> \
  --model <MODEL_NAME> \
  [--batch-size 32] [--num-workers 4] [--stitch-workers 8] \
  [--pin-memory | --no-pin-memory] [--overwrite]
```

Reads from `patches/`, writes to `model-outputs-csv/` plus
`infer_metadata_<ts>.json`. Accepts the same `--region-inference-dir`,
`--qupath-*`, and patch-sizing (`--patch-overlap-ratio`, `--patch-size-um`,
`--patch-size-px`) options as `run`. `--stitch-workers` controls
the TileFuse thread pool used to assemble object-based detections.
`--no-pin-memory` disables pinned (page-locked) memory for DataLoaders,
which helps in memory-constrained environments where workers are killed
by the system OOM killer.

### 4.6 `wsinsight ncomp` — Node-level (Cell) Composition

Builds (or reuses) a Delaunay cell graph per slide under
`graphs/<slide>.h5` and emits per-cell k-hop neighborhood composition.

```bash
wsinsight ncomp \
  --results-dir <RESULTS_DIR> \
  [--k 2] [--max-neighbor-distance 25.0] \
  [--num-workers 8] [--overwrite]
```

Defaults: 25 µm edge filter, 2-hop neighborhood radius, 8 concurrent slides.
Outputs go to `ncomp-outputs-csv/<slide>.csv`. The `graphs/<slide>.h5` cache
is keyed by a SHA-256 hash of the cell-center coordinates, so `ncomp` reruns
are idempotent and safe to resume. There is no `--wsi-dir` (see §4.2.2).

### 4.7 `wsinsight export` — Merge & Export

```bash
wsinsight export \
  --results-dir <RESULTS_DIR> \
  --geojson --omecsv \
  [--object-type detection] [--overlap-ratio 0.0] \
  [--export-workers 4] [--overwrite]
```

Left-joins per-cell analytics under `RESULTS_DIR` (`model-outputs-csv/`,
`hplot-outputs-csv/cells/`, `ncomp-outputs-csv/`) into `export-csv/`, then
serialises to `export-geojson/` and/or `export-omecsv/`. Edge-level
(`ecomp-outputs-csv/`) and triad-level (`tcomp-outputs-csv/`) products use
different primary keys and are **not** merged — consume them directly.
`--object-type` is one of `tile`, `detection` (default), or `annotation` and
is embedded into each exported feature for QuPath. `--overlap-ratio`
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
  [--export-geojson] [--omecsv] [--export-workers 4] [--overwrite]
```

For each cell, region matching adds `region_prob_*` columns from the patch
inference; object-to-object matching pairs cells across two object runs
within `--radius-um` (using `--spacing-um-px` to convert). `--tag` namespaces
the added columns (`<kind>_<tag>_prob_*`). `--wsi-dir / -i`, when supplied,
restricts the run to slides whose stem appears under that directory.
GeoJSON / OME-CSV exports for the registered tables land in dedicated
subfolders.

### 4.9 `wsinsight schema` — Machine-readable CLI Schema

Emits a JSON description of every subcommand, its options, types, defaults,
and flag forms. Intended for downstream tooling (e.g. the QuPath extension)
that needs to render forms without hard-coding the CLI.

```bash
wsinsight schema                         # stdout
wsinsight schema --output schema.json    # file
wsinsight schema --models-only           # zoo only, no command surface
```

**`--models-only`** drops the `commands` block and adds the registry actually
in use — the answer to "which zoo am I resolving against?" without parsing a
~130 kB document (the trimmed form is ~6 kB):

```json
{
  "schema_version": 1,
  "wsinsight_version": "...",
  "registry_path": "/home/<user>/.wsinfer-zoo/wsinfer-zoo-registry.json",
  "registry_resolved": "/path/to/zoo/wsinsight-zoo-registry.json",
  "models": [ { "name": "...", "path": "...", ... } ]
}
```

`registry_path` is where the lookup landed; `registry_resolved` follows any
symlink, and its directory is the root the local weight paths are resolved
against. The two differ whenever the default
`~/.wsinfer-zoo/wsinfer-zoo-registry.json` is a link to a fuller registry.

---

## 5. Model Selection

Models can be specified in four mutually exclusive ways:

| Method                 | Flag(s)                    | When to Use                                              |
| ---------------------- | -------------------------- | -------------------------------------------------------- |
| Zoo directory          | `--zoo-model-dir` / `-z`   | **Preferred when the weights are already on disk** — loads directly, no Hub call |
| Registry name          | `--model` / `-m`           | Registered model whose weights still need downloading    |
| Custom config + weights| `--config` + `--model-path`| Bring-your-own TorchScript model                         |
| List registered models | `wsinsight schema`         | Discover model names *and* their local paths (`wsinfer-zoo ls` contacts the Hub) |

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

### 5.1 Loading Weights Without Network Access

`--model <name>` **always resolves through `huggingface_hub`**, even when the
registry entry points at a directory that already holds the weights.
`WSINSIGHT_ZOO_REGISTRY_PATH` only makes the name *valid*; it does not change
where the weights are loaded from. On a host that cannot reach
`huggingface.co`:

| Attempt | Result |
| ------- | ------ |
| `--model <name>` | `SSLError` / connection failure while locating the file on the Hub |
| `--model <name>` + `HF_HUB_OFFLINE=1` | `LocalEntryNotFoundError` — the Hub cache under `HF_HOME` is searched, not the registry directory |
| `--zoo-model-dir <dir>` | Loads straight from disk; the Hub is never contacted |

Use `--zoo-model-dir`. It takes the directory holding `config.json` and
`torchscript_model.pt`, which sits beside the registry as
`<registry-dir>/<hf_repo_id>/<hf_revision>/`:

```bash
wsinsight run -i ./slides -o ./results \
  -z /path/to/zoo/<hf_repo_id>/<hf_revision>
```

`wsinsight schema` reports the resolved directory of every model already present
on disk, so the right value can be looked up programmatically:

```bash
wsinsight schema --models-only | python -c "
import json, sys
for m in json.load(sys.stdin)['models']:
    if m['path']:
        print(m['name'], m['path'])
"
```

Prefer this over `wsinfer-zoo ls`, which contacts the Hub.

**Registry discovery without an environment variable.** With
`WSINSIGHT_ZOO_REGISTRY_PATH` unset, resolution falls back to
`~/.wsinfer-zoo/wsinfer-zoo-registry.json`. Pointing that path at a fuller
registry makes every caller agree on the model list — including subprocesses
that never inherit the variable. A symlink works: the path is `resolve()`d, so
the weights root follows the link target rather than the symlink's own
directory.

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
│   └── <slide>.h5                  Delaunay cache (produced by ncomp/ecomp/tcomp/niche)
├── export-csv/
│   └── <slide>.csv                 Merged per-cell CSV (model + ncomp [+ hplot])
├── export-geojson/
│   └── <slide>.geojson             GeoJSON export
├── export-omecsv/
│   └── <slide>.ome.csv.gz          OME-CSV export
└── <command>_metadata_<ts>.json    Per-command run log; every subcommand (run,
                                    patch, infer, export, ncomp, hplot, niche, …)
                                    writes one with the same {command, params,
                                    runtime, timestamp} schema (patch/infer also
                                    embed the model record)
```

> Experimental subcommands add their own subdirectories alongside the stable
> ones: `hplot-outputs-csv/{cells,...}/`, `ecomp-outputs-csv/<slide>.csv`,
> `tcomp-outputs-csv/<slide>.csv`, `niche-outputs-csv/{cells,niches}/<slide>.csv`,
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
wsinsight ncomp --results-dir results/

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

## 8. URI & Remote Data Support

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

Building a TCGA/GDC manifest, and joining results to clinical endpoints
(survival, PAM50, MSI, treatment), are covered in
[`reference/remote-data.md`](reference/remote-data.md). Read that file only when
the task actually involves a TCGA cohort.

---

## 9. Error Recovery & Troubleshooting

| Symptom                              | Cause                                 | Fix                                                         |
| ------------------------------------ | ------------------------------------- | ----------------------------------------------------------- |
| `SSLError` / `CERTIFICATE_VERIFY_FAILED` while locating a model | `-m/--model` always resolves through the Hub, even when the weights are already cached | Use `-z/--zoo-model-dir <dir>` (§5.1). Setting `WSINSIGHT_ZOO_REGISTRY_PATH` alone does **not** avoid this |
| `LocalEntryNotFoundError` (with `HF_HUB_OFFLINE=1`) | Only the Hub cache under `HF_HOME` is searched; local weights sit beside the registry | Use `-z/--zoo-model-dir <dir>` (§5.1) |
| Model name rejected as an invalid choice | The registry in use does not list it, so it is not a valid `--model` value | `wsinsight schema --models-only` shows what actually resolves |
| `numpy >= 2.0` assertion failure     | Dependency upgraded numpy             | `pip install -c constraints.txt "numpy<2"`                   |
| `ModuleNotFoundError: osgeo`         | GDAL not installed via conda          | `conda install -c conda-forge gdal=3.11.3`                  |
| CUDA out of memory                   | Batch size too large                  | Reduce `--batch-size`                                       |
| DataLoader worker killed by signal   | System OOM killer (pinned memory)     | Use `--no-pin-memory --num-workers 2`; auto-recovery retries |
| Inference produces empty CSV         | Wrong model for slide magnification   | Match model suffix (`x20`/`x40`) to slide magnification     |
| Stale graph cache                    | CSV changed after graph was built     | Automatic: cache detects via SHA-256 hash and rebuilds       |
| Experimental subcommand refuses to run | `WSINSIGHT_EXPERIMENTAL` not set    | Export `WSINSIGHT_EXPERIMENTAL=1` before invoking it         |

---

## 10. Agent Decision Guide

Use this flowchart when deciding which command(s) to run:

```text
Is WSInsight already installed / is Docker available?
├─ Docker available → Prefer Docker (Section 2.2): no install needed
│        docker run --rm -i --gpus all --shm-size=32g --init \
│          -e HOST_UID -e HOST_GID -v /path/to/data:/workspace \
│          -v wsinsight-hf-cache:/app/hf-cache \
│          huangchtw/wsinsight:latest bash -lc 'wsinsight run ...'
├─ `wsinsight --help` fails and no Docker → fallback install (§2):
│        bash ./conda-setup.sh <ENV_NAME>
└─ Already installed → Continue below

Has the user provided WSIs?
├─ Yes → Do they want a one-shot run?
│        ├─ Yes → wsinsight run [--ncomp] [--export-geojson] [--export-omecsv]
│        └─ No  → wsinsight patch → wsinsight infer → [ncomp] → wsinsight export
├─ No, but results-dir exists with model-outputs-csv/ → Skip patch+infer
│        (Before skipping patch/infer, verify both patches/ and
│         model-outputs-csv/ exist and are non-empty for the target slides;
│         if either is missing or incomplete, run the missing upstream stage
│         first rather than proceeding to ncomp/export.)
│        ├─ Need per-cell neighborhood composition? → wsinsight ncomp
│        │    (ncomp requires model-outputs-csv/ to exist. If it is absent,
│        │     run wsinsight infer first.)
│        └─ Need GeoJSON / OME-CSV?                 → wsinsight export --geojson --omecsv
├─ No slides, but user mentions TCGA / GDC / cancer cohort
│        → reference/remote-data.md §1 — GDC API manifest → save .tsv
│        → wsinsight run --wsi-dir "gdc-manifest:///path/to/manifest.tsv" ...
├─ User needs clinical / molecular labels (survival, PAM50, MSI, treatment)
│        → reference/remote-data.md §2 — GDC /cases, Liu et al. 2018, cBioPortal
│        → Join on first 12 chars of slide filename (patient barcode)
└─ No slides or results → Ask user for --wsi-dir
```

### Key Constraints for Agents

1. **A model is required** for `run`, `patch`, and `infer`. Call
   `wsinsight schema --models-only` (or the `list_models` MCP tool) to see what
   this installation can actually resolve — do not guess, and do not use
   `wsinfer-zoo ls`, which contacts the Hub and fails on restricted networks.
   Prefer `-z/--zoo-model-dir` for any model whose `path` is non-null. If a
   requested model has a null `path` (weights not on disk) and the host cannot
   reach huggingface.co, stop and tell the user the weights must first be
   downloaded on a networked host and placed under the zoo registry directory;
   do not attempt `-m`.
2. **`--overwrite`** is needed to recompute existing outputs. Without it,
   completed slides are skipped (idempotent / resumable).
3. **Environment variables** must be exported before the `wsinsight` command,
   not passed as CLI flags.
4. **`constraints.txt`** should always be used with `pip install -c` to prevent
   dependency drift.
5. **When the user mentions TCGA, GDC, or a cancer cohort** (e.g. "analyze
   TCGA-BRCA slides"), use the GDC API `curl` pattern in
   [`reference/remote-data.md`](reference/remote-data.md) §1 to generate a
   manifest TSV, then pass it via
   `--wsi-dir "gdc-manifest:///absolute/path/to/manifest.tsv"`. Do not ask
   the user to download slides manually.
6. **Prefer Docker when available** — it avoids all local dependency
   installation. Call `docker run` directly (§2.2): mount the data at
   `/workspace`, wrap the command in `bash -lc`, and use `-i` rather than
   `-it` when there is no TTY. The image pre-sets
   `WSINSIGHT_ZOO_REGISTRY_PATH` and `KERAS_HOME`.
7. **When the user needs clinical labels** (survival, PAM50, MSI, treatment),
   see [`reference/remote-data.md`](reference/remote-data.md) §2. Use the GDC
   `/cases` API for demographics/staging, Liu et al. 2018 for curated survival
   endpoints, and cBioPortal for molecular subtypes. Join on the first 12
   characters of the slide filename.
8. **Do not recommend the experimental subcommands** (`hplot`,
   `hplot-finalize`, `ecomp`, `tcomp`, `niche`, `niche-profile`, `agg`, `import`) unless the user
   has explicitly
   opted in via `WSINSIGHT_EXPERIMENTAL=1`. Their CLI surfaces and output
   schemas are unstable.

---

## 11. MCP server (FastMCP)

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
- **Meta**: `job_status`, `job_logs`, `cancel_job`, `list_jobs`, `list_models`.
- **Resources**: `wsinsight://schema`, `wsinsight://models`,
  `wsinsight://results/{results_dir}/layout`.
- **Prompt**: `reproduce_tcga_crc`.

With `wsinsight-mcp --experimental` the server additionally exposes
`hplot`, `ecomp`, `tcomp`, `niche`, `agg` (long-running) and `hplot-finalize`,
`niche-profile`, `import` (short-running). Child processes inherit
`WSINSIGHT_EXPERIMENTAL=1`
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
