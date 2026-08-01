#!/bin/sh
# The container uid/gid is chosen at RUN time by the image entrypoint (it remaps
# the in-image "user" to the owner of the mounted /workspace, or to
# $HOST_UID/$HOST_GID), so the build never bakes the caller's id.
docker build -f ./Dockerfile -t wsinsight:latest .
docker tag wsinsight:latest huangchtw/wsinsight:latest
docker push huangchtw/wsinsight:latest

# IMAGE_ID=lj-docker-reg.pfizer.com/huangc78/wsinsight:latest
# IMAGE_ID=wsinsight:latest
# docker pull ${IMAGE_ID}
