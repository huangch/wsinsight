#!/bin/sh

# IMAGE_ID=lj-docker-reg.pfizer.com/huangc78/wsinsight:latest
IMAGE_ID=wsinsight:latest
docker pull ${IMAGE_ID}

# The container's uid/gid is set at run time by the image entrypoint: by default
# it becomes the owner of the mounted /workspace (so you can always write to your
# data). Export HOST_UID / HOST_GID before running to force a specific id; the
# ``-e HOST_UID -e HOST_GID`` on the docker run lines forward them only when set.

# Named volume that persists the Hugging Face model cache between runs.
# First invocation triggers an auto-download of any model referenced from the
# WSInsight zoo registry; subsequent runs reuse the cached weights.
HF_CACHE_VOLUME=wsinsight-hf-cache

# Parse arguments: --gpu <id> is optional and may appear anywhere before the command.
DATA_DIR=""
GPU_FLAG="all"

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)
            GPU_FLAG="device=$2"
            shift 2
            ;;
        --gpu=*)
            GPU_FLAG="device=${1#--gpu=}"
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            # First non-option arg is the data dir; remaining are the command.
            if [ -z "${DATA_DIR}" ]; then
                DATA_DIR="$1"
                shift
            else
                break
            fi
            ;;
    esac
done

if [ -z "${DATA_DIR}" ]; then
    echo "Usage: wsinsight-docker-run.sh [--gpu <ID>] /path/to/data [COMMAND ...]"
    echo ""
    echo "Options:"
    echo "  --gpu <ID>   Use a specific GPU (default: all GPUs)"
    echo ""
    echo "Examples:"
    echo "  wsinsight-docker-run.sh /data                         # interactive shell, all GPUs"
    echo "  wsinsight-docker-run.sh --gpu 2 /data                 # interactive shell, GPU 2"
    echo "  wsinsight-docker-run.sh /data wsinsight run ...       # run command, all GPUs"
    echo "  wsinsight-docker-run.sh --gpu 2 /data wsinsight run . # run command, GPU 2"
    exit 1
fi

if [ $# -gt 0 ]; then
    # Direct command mode: run the provided command and exit
    echo docker run --rm -it --gpus "${GPU_FLAG}" --shm-size=32g --init -e HOST_UID -e HOST_GID -v "${DATA_DIR}":/workspace -v "${HF_CACHE_VOLUME}":/app/hf-cache ${IMAGE_ID} bash -lc "$*"
    docker run --rm -it --gpus "${GPU_FLAG}" --shm-size=32g --init -e HOST_UID -e HOST_GID -v "${DATA_DIR}":/workspace -v "${HF_CACHE_VOLUME}":/app/hf-cache ${IMAGE_ID} bash -lc "$*"
else
    # Interactive mode: drop into a shell
    echo docker run --rm -it --gpus "${GPU_FLAG}" --shm-size=32g --init -e HOST_UID -e HOST_GID -v "${DATA_DIR}":/workspace -v "${HF_CACHE_VOLUME}":/app/hf-cache ${IMAGE_ID}
    docker run --rm -it --gpus "${GPU_FLAG}" --shm-size=32g --init -e HOST_UID -e HOST_GID -v "${DATA_DIR}":/workspace -v "${HF_CACHE_VOLUME}":/app/hf-cache ${IMAGE_ID}
fi
