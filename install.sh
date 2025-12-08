# Reset environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda env remove -n wsinsight -y

# conda gdal first as you already do
conda create -n wsinsight python=3.11 gdal=3.11.3 -c conda-forge -y
conda activate wsinsight
pip install --upgrade pip
pip uninstall -y pynvml || true
pip install -c ./wsinsight/constraints.txt "numpy<2"

# heavy stacks first (optional but speeds up):
pip install -c ./wsinsight/constraints.txt torch torchvision torch-geometric tensorflow keras stardist nvidia-ml-py

# histomicstk wheel source (same as before), still honoring constraints:
# pip install -c ./wsinsight/constraints.txt "numpy<2" histomicstk --find-links https://girder.github.io/large_image_wheels
# In case of SSL issues, use below installing histomicstk
pip install --no-cache-dir --trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io --find-links https://girder.github.io/large_image_wheels --upgrade "numpy<2" histomicstk

# the rest + your package
pip install -c ./wsinsight/constraints.txt -e ./wsinsight

# Test the main entry
S3_STORAGE_OPTIONS='{"profile":"saml"}' \
WSINFER_ZOO_REGISTRY_PATH='/workspace/wsinsight/wsinsight/zoo/wsinfer-zoo-registry.json' \
WSINSIGHT_REMOTE_CACHE_DIR='/tmp' \
KERAS_HOME='/workspace/wsinsight/wsinsight/keras' \
wsinsight



