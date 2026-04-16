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
# In case of SSL issues, use below installing histomicstk
pip install --trusted-host github.com --trusted-host raw.githubusercontent.com --trusted-host girder.github.io --find-links https://girder.github.io/large_image_wheels -c constraints.txt "numpy<2" histomicstk

# the rest + your package
pip install -c constraints.txt -e .

# install CellViT training dependencies (optional):
# pip install -c constraints.txt "numpy<2" cupy wandb albumentations colorama einops schema torchstain natsort geojson ujson ray torchmetrics "evalutils==0.5.0" torchinfo

# Safety check: ensure numpy stayed below 2.0
python -c "import numpy; v=numpy.__version__; assert int(v.split('.')[0]) < 2, f'ERROR: numpy {v} >= 2.0 detected; stardist will break. Re-run: pip install -c constraints.txt \"numpy<2\"'"

# Test the main entry
S3_STORAGE_OPTIONS='{"profile":"saml"}' \
WSINFER_ZOO_REGISTRY_PATH='/workspace/wsinsight/devel/zoo/wsinfer-zoo-registry.json' \
WSINSIGHT_REMOTE_CACHE_DIR='/tmp' \
KERAS_HOME='/workspace/wsinsight/wsinsight/keras' \
wsinsight



