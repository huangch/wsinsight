#!/bin/sh
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -f ./Dockerfile -t wsinsight:latest .
docker tag wsinsight:latest huangchtw/wsinsight:latest
docker push huangchtw/wsinsight:latest
