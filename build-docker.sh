docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -f ./Dockerfile -t wsinsight:latest .
docker tag wsinsight:latest https://hub.docker.com/r/huangchtw/wsinsight:latest
docker push https://hub.docker.com/r/huangchtw/wsinsight:latest
