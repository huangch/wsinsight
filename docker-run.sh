#!/bin/sh

IMAGE_ID=huangchtw/wsinsight:latest
docker pull ${IMAGE_ID}

if [ ! "${1}" == "" ] && [ ! "${2}" == "" ]; then
echo docker run --rm -it --gpus "device=${2}" --shm-size=32g --init -v "${1}":/workspace ${IMAGE_ID}
docker run --rm -it --gpus "device=${2}" --shm-size=32g --init -v "${1}":/workspace ${IMAGE_ID}
elif [ ! "${1}" == "" ]; then
# docker run --rm -it --gpus all --shm-size=16g -v "${1}":/workspace ${IMAGE_ID}
echo docker run --rm -it --gpus all --shm-size=32g --init -v "${1}":/workspace ${IMAGE_ID}
docker run --rm -it --gpus all --shm-size=32g --init -v "${1}":/workspace ${IMAGE_ID}
else
echo "Usage: docker-run.sh /path/to/data/folder GPU_ID [ENTER]"
fi
