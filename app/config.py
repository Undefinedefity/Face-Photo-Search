import json
import os
from pathlib import Path
from typing import Dict, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = DATA_DIR / "photos"
TMP_DIR = DATA_DIR / "tmp"
DB_PATH = DATA_DIR / "app.db"
CONFIG_FILE = DATA_DIR / "config.json"

# Default thresholds
_DEFAULT_INSIGHT = float(os.getenv("INSIGHTFACE_THRESHOLD", "0.6"))
_DEFAULT_FACEREC = float(os.getenv("FACEREC_THRESHOLD", "0.6"))

_settings: Dict[str, float] = {
    "insightface_threshold": _DEFAULT_INSIGHT,
    "facerec_threshold": _DEFAULT_FACEREC,
}

# Thumbnail defaults
DEFAULT_THUMB_WIDTH = 256

# Status polling interval hint for frontend
POLL_INTERVAL_SECONDS = 1.0


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    _load_settings()


def _load_settings() -> None:
    if CONFIG_FILE.exists():
        try:
            loaded = json.loads(CONFIG_FILE.read_text())
            if "insightface_threshold" in loaded:
                _settings["insightface_threshold"] = float(loaded["insightface_threshold"])
            if "facerec_threshold" in loaded:
                _settings["facerec_threshold"] = float(loaded["facerec_threshold"])
        except Exception:
            # ignore parse errors and keep defaults
            pass


def save_settings() -> None:
    CONFIG_FILE.write_text(json.dumps(_settings, indent=2))


def get_thresholds() -> Tuple[float, float]:
    return _settings["insightface_threshold"], _settings["facerec_threshold"]


def set_threshold(engine: str, value: float) -> None:
    if engine == "insightface":
        _settings["insightface_threshold"] = value
    elif engine == "face_recognition":
        _settings["facerec_threshold"] = value
    else:
        raise ValueError("Unsupported engine")
    save_settings()
