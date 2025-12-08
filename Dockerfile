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
    git curl wget ca-certificates build-essential pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    openjdk-17-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

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
    conda create -y --override-channels -n wsinsight -c conda-forge python=3.11 gdal=3.11.3 pip && \
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

# ------------------------------------
# Install ML libraries (Torch + TensorFlow + keras)
# ------------------------------------
RUN pip install -c constraints.txt "numpy<2" torch torchvision torch-geometric tensorflow keras nvidia-ml-py
RUN pip uninstall -y pynvml || true

# ------------------------------------
# Install stardist for cell prediction (based on TensorFlow + Keras)
# ------------------------------------
RUN pip install -c constraints.txt "numpy<2" stardist

# ------------------------------------
# Install HistomicsTK for staining normalization
# ------------------------------------
RUN pip install -c constraints.txt --trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io --find-links https://girder.github.io/large_image_wheels "numpy<2" histomicstk

# ------------------------------------
# Install H-Plot and WSInsight packages
# ------------------------------------
# RUN pip install --upgrade "numpy<2" -e ./hplot
RUN pip install -c constraints.txt "numpy<2" -e .

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
# Non-root user
# ------------------------------------
ARG USERNAME=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} && \
    # ensure new user's ~/.bashrc inherits conda setup
    bash -lc 'echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/'"${USERNAME}"'/.bashrc' && \
    bash -lc 'echo "conda activate wsinsight" >> /home/'"${USERNAME}"'/.bashrc' && \
    chown -R ${UID}:${GID} /home/${USERNAME}
WORKDIR /workspace
RUN chown -R ${UID}:${GID} /workspace
USER ${USERNAME}

# ------------------------------------
# Environment variables
# ------------------------------------
ENV WSINFER_ZOO_REGISTRY_PATH=/app/wsinsight/zoo/wsinfer-zoo-registry.json
ENV KERAS_HOME=/app/wsinsight/keras

# ------------------------------------
# Default interactive shell
# ------------------------------------
SHELL ["/bin/bash","-lc"]
CMD ["bash"]
