# ====================================
# CUDA 12.8 + cuDNN + Ubuntu 22.04
# ====================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 

# ------------------------------------
# Basic system dependencies + OpenJDK 17
# ------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget unzip vim ca-certificates build-essential pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    openjdk-17-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# ------------------------------------
# Install AWS CLI v2
# ------------------------------------
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip
    
# ------------------------------------
# Install Miniconda (Python 3.11 base)
# ------------------------------------
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p "$CONDA_DIR" \
 && rm /tmp/mc.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# ------------------------------------
# Accept Anaconda Terms of Service
# (Required since Sept 2024 for pkgs/main and pkgs/r)
# ------------------------------------
RUN conda --version && \
    (conda tacs accept --override-channels --channel https://repo.anaconda.com/pkgs/main -y || true) && \
    (conda tacs accept --override-channels --channel https://repo.anaconda.com/pkgs/r -y || true)

# ------------------------------------
# Create environment (conda-forge to avoid TOS re-prompts)
# ------------------------------------
RUN conda update -n base --yes --override-channels -c conda-forge conda && \
    conda create -y --override-channels -n wsinsight -c conda-forge python=3.11 gdal=3.11.3 pip "setuptools<67" && \
    conda clean -afy
RUN python -m pip install --upgrade pip 

# ------------------------------------
# Global Conda initialization
# Fix: Docker bash doesn’t read /etc/profile.d/*
# Solution: write hook into both /etc/bash.bashrc and user skeleton (~/.bashrc)
# ------------------------------------
RUN echo '. /opt/conda/etc/profile.d/conda.sh' >> /etc/bash.bashrc && \
    echo 'conda activate wsinsight' >> /etc/bash.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /etc/skel/.bashrc && \
    echo 'conda activate wsinsight' >> /etc/skel/.bashrc

# ------------------------------------
# Preload Conda env path
# ------------------------------------
ENV CONDA_DEFAULT_ENV=wsinsight
ENV PATH="$CONDA_DIR/envs/wsinsight/bin:$PATH"

# ------------------------------------
# Set working directory
# ------------------------------------
WORKDIR /app/wsinsight
COPY . .
RUN mv /app/wsinsight/keras /app/
RUN mv /app/wsinsight/zoo /app/

# ------------------------------------
# Pre-install heavy ML stack explicitly
# Torch + TF are in pyproject.toml but pre-installing ensures they land even
# if the full dep walk of pip install -e . hits a constraint conflict.
# ------------------------------------
RUN pip install --retries 10 -c /app/wsinsight/constraints.txt "numpy<2" \
    torch torchvision torch-geometric tensorflow keras stardist nvidia-ml-py
# pynvml conflicts with nvidia-ml-py; removal is best-effort only.
RUN pip uninstall -y pynvml || true

# ------------------------------------
# Pre-install pyvips (SSL fallback chain for corporate proxy)
# ------------------------------------
RUN set -eu; \
    CONSTR=/app/wsinsight/constraints.txt; \
    GIRDER="--trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io --find-links https://girder.github.io/large_image_wheels"; \
    ( pip install --retries 5 $GIRDER -c "$CONSTR" pyvips \
      || pip install --retries 5 $GIRDER -c "$CONSTR" pyvips --cert /etc/pki/tls/certs/ca-bundle.crt \
      || pip install --retries 5 $GIRDER -c "$CONSTR" pyvips --cert /etc/ssl/certs/ca-certificates.crt \
      || CURL_CA_BUNDLE="" pip install --retries 5 $GIRDER -c "$CONSTR" pyvips \
      || echo "WARNING: pyvips install failed (all SSL fallbacks exhausted); continuing" )

# ------------------------------------
# Pre-install large-image from girder wheel index
# MUST honour constraints.txt: without it large-image pulls zarr>=3 / dask 2026.x,
# which then makes the constrained `pip install -e .` below unsatisfiable
# (the pinned scanpy/squidpy/spatialdata line requires zarr<3).
# ------------------------------------
RUN set -eu; \
    CONSTR=/app/wsinsight/constraints.txt; \
    GIRDER="--trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io --find-links https://girder.github.io/large_image_wheels"; \
    ( pip install --retries 5 $GIRDER -c "$CONSTR" "large-image" "large-image-source-tifffile" "large-image-source-pil" \
        "large-image-source-openslide" "large-image-source-vips" "large-image-converter" \
      || pip install --retries 5 -c "$CONSTR" "large-image" "large-image-converter" \
      || echo "WARNING: large-image install failed; histomicstk may not import" )

# ------------------------------------
# Install remaining wsinsight dependencies (via pyproject.toml)
# torch/tf/pyvips/large-image above are already satisfied — pip skips them.
# histomicstk is NOT in pyproject.toml (girder-client hardpin); installed below.
# ------------------------------------
RUN pip install --retries 10 -c /app/wsinsight/constraints.txt -e "/app/wsinsight"
# pynvml conflicts with nvidia-ml-py; removal is best-effort only.
RUN pip uninstall -y pynvml || true
RUN pip install --retries 10 -c /app/wsinsight/constraints.txt fastmcp

# ------------------------------------
# histomicstk --no-deps (girder-client==3.2.11 hardpin bypass)
# ------------------------------------
RUN pip install --no-deps \
    --trusted-host github.com --trusted-host raw.githubusercontent.com \
    --trusted-host girder.github.io \
    --find-links https://girder.github.io/large_image_wheels \
    -c /app/wsinsight/constraints.txt histomicstk

# ------------------------------------
# Sanity check (runs at build time)
# ------------------------------------
RUN python - <<'PY'
import os, subprocess, torch, tensorflow as tf
print("JAVA_HOME:", os.environ.get("JAVA_HOME"))
subprocess.run(["java","-version"], check=False)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU?", torch.cuda.is_available())
print("TF:", tf.__version__, "GPUs:", tf.config.list_physical_devices("GPU"))
PY

# ------------------------------------
# Environment variables
# ------------------------------------
ENV WSINSIGHT_ZOO_REGISTRY_PATH=/app/zoo/wsinsight-zoo-registry.json
ENV KERAS_HOME=/app/keras
# Persistent Hugging Face cache for first-run model auto-download.
# Mount a named volume on /app/hf-cache (see docker-run.sh) to keep weights
# across container restarts.
ENV HF_HOME=/app/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Fail the build if the console scripts were not installed. Without this the
# image can ship with a working Python package but no `wsinsight` on PATH.
RUN python -c "import wsinsight; print('wsinsight package OK:', wsinsight.__file__)" && \
    command -v wsinsight && \
    command -v wsinsight-mcp && \
    wsinsight --help > /dev/null

# ------------------------------------
# Non-root user
# ------------------------------------
# The container starts as root; the runtime entrypoint (docker-entrypoint.sh)
# remaps the pre-created ``user`` account to the owner of the mounted /workspace
# (or to $HOST_UID/$HOST_GID) and then drops privileges via setpriv. The uid/gid
# baked here is only a throwaway placeholder, immediately overwritten at run
# time, so it is hard-coded (1000) rather than exposed as a build ARG.
RUN groupadd -g 1000 user && \
    useradd -m -u 1000 -g 1000 -s /bin/bash user && \
    # ensure new user's ~/.bashrc inherits conda setup
    bash -lc 'echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/user/.bashrc' && \
    bash -lc 'echo "conda activate wsinsight" >> /home/user/.bashrc' && \
    mkdir -p /app/hf-cache && \
    chown -R 1000:1000 /home/user /app/hf-cache /app/zoo /app/keras

# Install the runtime uid/gid-remapping entrypoint.
RUN install -m 0755 /app/wsinsight/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace
RUN chown -R 1000:1000 /workspace
# NOTE: no ``USER`` here on purpose — the container starts as root so the
# entrypoint can remap ``user`` to the mount owner, then drops privileges via
# setpriv. Passing ``docker run --user ...`` still works: the entrypoint detects
# a non-root start and execs the command unchanged.

# ------------------------------------
# Default interactive shell
# ------------------------------------
SHELL ["/bin/bash","-lc"]
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
