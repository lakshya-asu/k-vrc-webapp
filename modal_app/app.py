import modal

# Persistent volume for model weights + data
volume = modal.Volume.from_name("kvrc-data", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App("kvrc-animation", image=image)

VOLUME_PATH = "/data"
MODEL_PATH = "/data/models"
DATA_PATH = "/data/train"
