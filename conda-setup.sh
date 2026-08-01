#!/usr/bin/env bash
# conda-setup.sh — create and populate the wsinsight conda environment.
#
# Usage:  bash ./conda-setup.sh [-n ENV_NAME] [-r|--reset] [--mcp]
#
#   -n | --name  ENV_NAME   Conda environment to use (default: current active env).
#   -r | --reset            Deactivate, remove, recreate, and activate the env.
#                           Without this flag the script skips env creation and
#                           only (re-)installs packages into the existing env.
#   --mcp                   Also install fastmcp (MCP server support).
#                           Not installed by default to avoid jaraco.* version scanning.
#
# Key workarounds:
#   1. PIP_CACHE_DIR=/tmp/...   — redirects pip wheel cache to /tmp to bypass NAS inode quotas
#   2. histomicstk --no-deps    — skips broken girder-client 3.2.11 on PyPI (broken 2026-06)
#   3. large-image explicit      — histomicstk imports large_image at module init; must install
#   4. pyvips SSL fallback chain — handles Pfizer proxy certificate interception
#   5. stringzilla pre-install  — isolates GCC-incompatible source build so failure
#                                 doesn't roll back click/break the smoke test

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Argument parsing ───────────────────────────────────────────────────────────
ENV_NAME="${CONDA_DEFAULT_ENV:-}"   # default = current active env
DO_RESET=0
DO_MCP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -n/--name requires an environment name." >&2
                exit 1
            fi
            ENV_NAME="$2"
            shift 2
            ;;
        -r|--reset)
            DO_RESET=1
            shift
            ;;
        --mcp)
            DO_MCP=1
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash ./conda-setup.sh [-n ENV_NAME] [-r|--reset] [--mcp]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$ENV_NAME" ]]; then
    echo "Error: no conda environment specified and no environment is currently active." >&2
    echo "       Use -n ENV_NAME to specify one." >&2
    exit 1
fi

echo "Target conda environment: ${ENV_NAME}  (reset=${DO_RESET}, mcp=${DO_MCP})"

# ── (Re-)create environment ────────────────────────────────────────────────────
source /opt/anaconda3/etc/profile.d/conda.sh

if [[ "$DO_RESET" -eq 1 ]]; then
    conda deactivate
    conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
    # conda gdal first as you already do
    conda create -n "${ENV_NAME}" python=3.11 gdal=3.11.3 "setuptools<67" -c conda-forge -y
fi

conda activate "${ENV_NAME}"
pip install --upgrade pip

# ── Pip cache fix (NAS inode quota) ───────────────────────────────────────────
# PIP_CACHE_DIR redirects ALL cache writes (including temp wheel builds) to /tmp.
# This is more reliable than PIP_NO_CACHE_DIR=1 which doesn't prevent temp writes.
pip cache purge || true
export PIP_CACHE_DIR=/tmp/pip-cache-wsinsight

# torch, tensorflow, and other ML deps are declared in pyproject.toml;
# they are installed by `pip install -e .` below together with all other deps.
# In case of SSL issues behind a corporate proxy, pre-install pyvips with cert check disabled,
# then install histomicstk normally.
pip install --trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io \
    --find-links https://girder.github.io/large_image_wheels \
    -c "${SCRIPT_DIR}/constraints.txt" "numpy<2" pyvips \
    2>/dev/null \
  || pip install --trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io \
    --find-links https://girder.github.io/large_image_wheels \
    -c "${SCRIPT_DIR}/constraints.txt" "numpy<2" pyvips \
    --cert /etc/pki/tls/certs/ca-bundle.crt \
    2>/dev/null \
  || pip install --trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io \
    --find-links https://girder.github.io/large_image_wheels \
    -c "${SCRIPT_DIR}/constraints.txt" "numpy<2" pyvips \
    --cert /etc/ssl/certs/ca-certificates.crt \
    2>/dev/null \
  || PIP_TRUSTED_HOST="github.com girder.github.io raw.githubusercontent.com" \
    CURL_CA_BUNDLE="" \
    pip install -c "${SCRIPT_DIR}/constraints.txt" "numpy<2" pyvips \
    --find-links https://girder.github.io/large_image_wheels \
    2>/dev/null \
  || echo "WARNING: pyvips install failed (all SSL fallbacks exhausted); continuing"

# ── large-image from girder wheel index ──────────────────────────────────────
# Pre-installed before `pip install -e .` to get the girder pre-built wheels.
# `pip install -e .` sees them as already satisfied and skips re-downloading.
pip install \
      --trusted-host github.com --trusted-host raw.githubusercontent.com \
      --trusted-host girder.github.io \
      --find-links https://girder.github.io/large_image_wheels \
      -c "${SCRIPT_DIR}/constraints.txt" \
      "large-image" "large-image-source-tifffile" "large-image-source-pil" \
      "large-image-source-openslide" "large-image-source-vips" \
      "large-image-converter" \
  || pip install -c "${SCRIPT_DIR}/constraints.txt" "large-image" "large-image-converter" \
  || echo "WARNING: large-image install failed; histomicstk will not import"

# ── All wsinsight dependencies declared in pyproject.toml ────────────────────
# Installs torch, tensorflow, scikit-learn, scipy, nimfa, girder-client, etc.
# pyvips and large-image above are already satisfied; pip skips them.
# histomicstk is NOT in pyproject.toml (girder-client==3.2.11 hardpin issue);
# it is installed below with --no-deps once all its transitive deps are present.
# --no-build-isolation: uses the current env's setuptools so the wsinsight entry
#   point is created correctly without a separate isolated build environment.
pip install --no-build-isolation -c "${SCRIPT_DIR}/constraints.txt" -e "${SCRIPT_DIR}"

# This script does not use `set -e`, so a failed editable install would otherwise
# scroll past and only surface later as "wsinsight: command not found".
command -v wsinsight >/dev/null || {
    echo "ERROR: 'pip install -e' did not create the wsinsight console script." >&2
    echo "       Scroll up for the pip resolver error; do not ignore it." >&2
    exit 1
}

# ── histomicstk --no-deps (girder-client==3.2.11 hardpin bypass) ─────────────
# histomicstk hardpins girder-client==3.2.11 which is broken on PyPI (2026-06).
# All histomicstk runtime deps are already installed above via pyproject.toml.
pip install --no-deps \
    --trusted-host github.com --trusted-host raw.githubusercontent.com \
    --trusted-host girder.github.io \
    --find-links https://girder.github.io/large_image_wheels \
    -c "${SCRIPT_DIR}/constraints.txt" histomicstk

# ── MCP server support (optional, --mcp flag) ─────────────────────────────────
# Installed separately to avoid entangling fastmcp's jaraco.* dep chain with the
# main wsinsight resolution. Pin versions are in constraints.txt.
if [[ "${DO_MCP}" -eq 1 ]]; then
    echo "Installing fastmcp (MCP server support)..."
    pip install -c "${SCRIPT_DIR}/constraints.txt" fastmcp
fi

# ── CellViT training dependencies ─────────────────────────────────────────────
# albumentations<2.0: versions >=2.0 require albucore which requires stringzilla,
# a C extension that fails to compile on GCC < 11 (system has GCC 8.5).
# CellViT training only uses basic augmentations available in albumentations 1.x.
pip install -c "${SCRIPT_DIR}/constraints.txt" stringzilla 2>/dev/null \
  || pip install stringzilla --no-build-isolation 2>/dev/null \
  || echo "WARNING: stringzilla build failed (GCC < 11 incompatible with AVX-512 byte intrinsics); albumentations advanced features may be limited"

pip install -c "${SCRIPT_DIR}/constraints.txt" "numpy<2" \
    "cupy-cuda12x<14" \
    wandb "albumentations<1.4.0" colorama einops schema torchstain natsort \
    geojson ujson ray torchmetrics "evalutils==0.5.0" torchinfo

# ── Safety check ──────────────────────────────────────────────────────────────
python -c 'import numpy; v=numpy.__version__; assert int(v.split(".")[0])<2, "ERROR: numpy "+v+" >= 2.0; stardist will break"; print("numpy "+v+"  OK")'

# ── Smoke test ────────────────────────────────────────────────────────────────
S3_STORAGE_OPTIONS='{"profile":"saml"}' \
WSINSIGHT_ZOO_REGISTRY_PATH='/workspace/wsinsight/devel/zoo/wsinsight-zoo-registry.json' \
WSINSIGHT_REMOTE_CACHE_DIR='/tmp' \
KERAS_HOME='/workspace/wsinsight/devel/keras' \
wsinsight

