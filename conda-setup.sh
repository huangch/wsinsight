#!/usr/bin/env bash
# conda-setup.sh — create and populate the wsinsight conda environment.
#
# <<<USAGE_START>>>
# Usage:  bash ./conda-setup.sh ENV_NAME [-r|--reset] [-m|--mcp] [-d|--dev] [-h|--help]
#
#   ENV_NAME                (positional, REQUIRED) Conda environment to use/create.
#                           There is NO fallback to the currently-activated conda env:
#                           the name is mandatory so `-r` can never accidentally
#                           destroy a different active environment.
#   -r | --reset            Deactivate, remove, recreate, and activate ENV_NAME.
#                           Without this flag the script skips env creation and
#                           only (re-)installs packages into the existing env.
#   -m | --mcp              Also install fastmcp (MCP server support).
#                           Not installed by default to avoid jaraco.* version scanning.
#   -d | --dev              Also install the [dev] extra (pytest, pytest-cov, ruff,
#                           pre_commit) so the post-install smoke test can run the
#                           real test suite. Without -d the suite is SKIPped if
#                           pytest is missing; with -d it FAILS (you asked for it).
#                           The package itself is always installed editable (-e).
#   -h | --help             Print this help message and exit.
# <<<USAGE_END>>>
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
# ENV_NAME is the FIRST POSITIONAL argument and is REQUIRED. We deliberately do
# NOT fall back to $CONDA_DEFAULT_ENV: with `-r` in play, a hidden dependency on
# whatever env happens to be active is a footgun (it would silently destroy an
# unrelated env). Make the caller name the env explicitly, every time.
DO_RESET=0
DO_MCP=0
DO_DEV=0

print_usage() {
    awk '
        /<<<USAGE_START>>>/ {capture=1; next}
        /<<<USAGE_END>>>/   {capture=0}
        capture            {sub(/^# ?/, ""); print}
    ' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        -r|--reset)
            DO_RESET=1
            shift
            ;;
        -m|--mcp)
            DO_MCP=1
            shift
            ;;
        -d|--dev)
            DO_DEV=1
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Run '${0##*/} --help' for usage." >&2
            exit 1
            ;;
        *)
            # First non-flag token is ENV_NAME. Reject a second positional
            # argument (we only have one positional slot).
            if [[ -n "${ENV_NAME:-}" ]]; then
                echo "Error: only one positional argument (ENV_NAME) is accepted; got '$ENV_NAME' and '$1'." >&2
                echo "Run '${0##*/} --help' for usage." >&2
                exit 1
            fi
            ENV_NAME="$1"
            shift
            ;;
    esac
done

if [[ -z "${ENV_NAME:-}" ]]; then
    echo "Error: ENV_NAME is required." >&2
    echo "       Got: $0 $*" >&2
    echo "       Run '${0##*/} --help' for usage." >&2
    exit 1
fi

echo "Target conda environment: ${ENV_NAME}  (reset=${DO_RESET}, mcp=${DO_MCP}, dev=${DO_DEV})"

# ── (Re-)create environment ────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "${CONDA_BASE}" ]]; then
    for _base in /opt/conda /opt/anaconda3; do
        if [[ -f "${_base}/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="${_base}"
            break
        fi
    done
fi
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Error: cannot locate conda.sh. Activate conda first or set CONDA_BASE." >&2
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

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
# Exported BEFORE any purge: `pip cache purge` obeys this variable, so running it
# first wiped the user's global ~/.cache/pip (4.3 GB observed). The redirect on
# its own solves the quota problem, so nothing is purged here. One shared dir
# lets the sibling repos reuse the multi-hundred-MB torch/TF/cuDNN wheels.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-wsinsight-stack}"

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
# With -d/--dev, also install the [dev] extra (pytest, pytest-cov, ruff,
# pre_commit) so the smoke test can run the suite; without -d, the suite is
# SKIPped if pytest is missing and only WARN-ed if it fails.
if [[ "${DO_DEV}" -eq 1 ]]; then
    pip install --no-build-isolation -c "${SCRIPT_DIR}/constraints.txt" -e "${SCRIPT_DIR}[dev]"
else
    pip install --no-build-isolation -c "${SCRIPT_DIR}/constraints.txt" -e "${SCRIPT_DIR}"
fi

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

# ── Smoke test ────────────────────────────────────────────────────────────────
# Hard checks are fatal: a half-installed env must not look like a success.
# The test suite is reported but does not fail the setup.
echo "---- smoke test ----"
SMOKE_FAIL=0
smoke() {                       # smoke <label> <command...>
    label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf '  PASS  %s\n' "$label"
    else
        printf '  FAIL  %s\n' "$label"
        SMOKE_FAIL=$((SMOKE_FAIL + 1))
    fi
}

python -c 'import numpy, sys; print("  numpy", numpy.__version__, "| python", sys.version.split()[0])' || true

smoke "wsinsight on PATH"    command -v wsinsight
smoke "wsinsight --help"     wsinsight --help
smoke "import wsinsight"     python -c 'import wsinsight'
# numpy 2.x breaks stardist and the zarr<3 / anndata generation.
smoke "numpy < 2"            python -c 'import numpy, sys; sys.exit(int(numpy.__version__.split(".")[0]) >= 2)'
if [[ "${DO_MCP}" -eq 1 ]]; then
    smoke "wsinsight-mcp on PATH" command -v wsinsight-mcp
    smoke "wsinsight-mcp --help"  wsinsight-mcp --help
fi

if [[ -d "${SCRIPT_DIR}/tests" ]]; then
    if python -c "import pytest" >/dev/null 2>&1; then
        python -m pytest "${SCRIPT_DIR}/tests" -q \
            && echo "  PASS  test suite" \
            || echo "  WARN  test suite did not pass (non-fatal)"
    elif [[ "${DO_DEV}" -eq 1 ]]; then
        # User asked for the [dev] extra: pytest should be present. FAIL loudly
        # instead of silently SKIPping, or the install is misconfigured.
        echo "  FAIL  test suite: pytest missing but -d/--dev was requested" >&2
        smoke "pytest importable (dev)" python -c "import pytest"
    else
        echo "  SKIP  test suite (pytest not installed; rerun with -d/--dev)"
    fi
fi

if [[ "${SMOKE_FAIL}" -ne 0 ]]; then
    echo "smoke test: ${SMOKE_FAIL} check(s) FAILED" >&2
    exit 1
fi
echo "smoke test: all checks passed"
