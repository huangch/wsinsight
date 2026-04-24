# Reset environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda env remove -n wsinsight -y

# conda gdal first as you already do
conda create -n wsinsight python=3.11 gdal=3.11.3 "setuptools<67" -c conda-forge -y
conda activate wsinsight
pip install --upgrade pip
pip install -c constraints.txt "numpy<2"

# heavy stacks first (optional but speeds up):
pip install -c constraints.txt torch torchvision torch-geometric tensorflow keras stardist nvidia-ml-py
# pip uninstall -y pynvml

# histomicstk wheel source (same as before), still honoring constraints:
# pip install -c constraints.txt "numpy<2" histomicstk --find-links https://girder.github.io/large_image_wheels
# In case of SSL issues behind a corporate proxy, pre-install pyvips with cert check disabled,
# then install histomicstk normally.
pip install --trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io \
    --find-links https://girder.github.io/large_image_wheels \
    -c constraints.txt "numpy<2" pyvips \
    2>/dev/null \
  || pip install --trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io \
    --find-links https://girder.github.io/large_image_wheels \
    -c constraints.txt "numpy<2" pyvips \
    --cert /etc/ssl/certs/ca-certificates.crt \
    2>/dev/null \
  || PIP_TRUSTED_HOST="github.com girder.github.io raw.githubusercontent.com" \
    CURL_CA_BUNDLE="" \
    pip install -c constraints.txt "numpy<2" pyvips \
    --find-links https://girder.github.io/large_image_wheels

pip install --trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io --find-links https://girder.github.io/large_image_wheels -c constraints.txt "numpy<2" histomicstk

# Pre-install remaining heavy deps that cause resolver backtracking
pip install -c constraints.txt "numpy<2" \
    scikit-learn shapely geopandas pyproj rasterio pyogrio \
    openslide-python wsidicom paquo "wsinfer-zoo>=0.6.2" \
    igraph leidenalg s3fs boto3 platformdirs timm \
    tiffslide imagecodecs opencv-python-headless orjson click

# the rest + your package (use --no-build-isolation to speed up resolve)
pip install -c constraints.txt --no-build-isolation -e .

# install CellViT training dependencies (required for pan-tissue model training):
#   - cupy-cuda12x<14 : pre-built binary wheel; pinned <14 because cupy 14.x
#                       requires numpy>=2 which conflicts with our numpy<2 pin.
#                       Replace with cupy-cuda11x if running on CUDA 11.
pip install -c constraints.txt "numpy<2" "cupy-cuda12x<14" \
    wandb albumentations colorama einops schema torchstain natsort \
    geojson ujson ray torchmetrics "evalutils==0.5.0" torchinfo

# Safety check: ensure numpy stayed below 2.0
python -c "import numpy; v=numpy.__version__; assert int(v.split('.')[0]) < 2, f'ERROR: numpy {v} >= 2.0 detected; stardist will break. Re-run: pip install -c constraints.txt \"numpy<2\"'"

# Test the main entry
S3_STORAGE_OPTIONS='{"profile":"saml"}' \
WSINSIGHT_ZOO_REGISTRY_PATH='/workspace/wsinsight/devel/zoo/wsinsight-zoo-registry.json' \
WSINSIGHT_REMOTE_CACHE_DIR='/tmp' \
KERAS_HOME='/workspace/wsinsight/devel/keras' \
wsinsight



