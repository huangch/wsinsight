#!/bin/sh

IMAGE_ID=huangchtw/wsinsight:latest
docker pull ${IMAGE_ID}

DATA_DIR="${1}"
GPU_ID="${2}"

if [ -z "${DATA_DIR}" ]; then
    echo "Usage: docker-run.sh /path/to/data [GPU_ID] [COMMAND ...]"
    echo ""
    echo "Examples:"
    echo "  docker-run.sh /data                          # interactive shell, all GPUs"
    echo "  docker-run.sh /data 2                        # interactive shell, GPU 2"
    echo "  docker-run.sh /data \"\" wsinsight run ...     # run command directly, all GPUs"
    echo "  docker-run.sh /data 2  wsinsight run ...     # run command directly, GPU 2"
    exit 1
fi

# Shift past DATA_DIR and GPU_ID to collect the remaining args as the command
shift
if [ ! -z "${GPU_ID}" ]; then
    shift
    GPU_FLAG="device=${GPU_ID}"
else
    GPU_FLAG="all"
fi

if [ $# -gt 0 ]; then
    # Direct command mode: run the provided command and exit
    echo docker run --rm -it --gpus "${GPU_FLAG}" --shm-size=32g --init -v "${DATA_DIR}":/workspace ${IMAGE_ID} bash -lc "$*"
    docker run --rm -it --gpus "${GPU_FLAG}" --shm-size=32g --init -v "${DATA_DIR}":/workspace ${IMAGE_ID} bash -lc "$*"
else
    # Interactive mode: drop into a shell
    echo docker run --rm -it --gpus "${GPU_FLAG}" --shm-size=32g --init -v "${DATA_DIR}":/workspace ${IMAGE_ID}
    docker run --rm -it --gpus "${GPU_FLAG}" --shm-size=32g --init -v "${DATA_DIR}":/workspace ${IMAGE_ID}
fi
