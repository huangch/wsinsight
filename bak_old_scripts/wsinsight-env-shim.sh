#! /bin/sh
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PATH=/usr/bin:/bin WSINSIGHT_EXPERIMENTAL=1 WSINSIGHT_ZOO_REGISTRY_PATH=/workspace/wsinsight/devel/zoo/wsinsight-zoo-registry.json KERAS_HOME=/workspace/wsinsight/devel/keras HF_HOME=/workspace/wsinsight/devel/zoo/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt /opt/anaconda3/envs/wsi/bin/wsinsight $*
