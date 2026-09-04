from pathlib import Path

# Absolute paths (handles different working directories)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "styles.csv"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Model config
MODEL_NAME = "ResNet50"
EMBEDDING_DIM = 2048

# UI config
ITEMS_PER_PAGE = 6
