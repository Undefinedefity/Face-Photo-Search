import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = DATA_DIR / "photos"
TMP_DIR = DATA_DIR / "tmp"
DB_PATH = DATA_DIR / "app.db"

# Thresholds for clustering
INSIGHTFACE_COSINE_THRESHOLD = float(os.getenv("INSIGHTFACE_THRESHOLD", "0.5"))
FACEREC_EUCLIDEAN_THRESHOLD = float(os.getenv("FACEREC_THRESHOLD", "0.6"))

# Thumbnail defaults
DEFAULT_THUMB_WIDTH = 256

# Status polling interval hint for frontend
POLL_INTERVAL_SECONDS = 1.0


def ensure_dirs() -> None:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

