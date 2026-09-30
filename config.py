"""
Configuration file for ANPR (Automatic Number Plate Recognition) System.
Uses pathlib for cross-platform compatibility (Windows, Linux, macOS).
"""

from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Model Configuration
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best.pt"

# Data Directories
DATA_DIR = BASE_DIR / "data"
DATA_YAML_PATH = DATA_DIR / "data.yaml"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

# Test & Output Directories
TEST_IMAGES_DIR = BASE_DIR / "test_images"
OUTPUTS_DIR = BASE_DIR / "outputs"

# History Configuration
HISTORY_DIR = BASE_DIR / "history"
HISTORY_CSV_PATH = HISTORY_DIR / "detections.csv"

# Inference Settings
DEFAULT_CONFIDENCE = 0.40
YOLO_IMAGE_SIZE = 640
OCR_LANGUAGES = ['en']

# Image Upload Settings
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]

def ensure_directories():
    """Ensure all required directories exist on startup."""
    for folder in [MODELS_DIR, DATA_DIR, IMAGES_DIR, LABELS_DIR, TEST_IMAGES_DIR, OUTPUTS_DIR, HISTORY_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Subdirectories for data splitting
    for split in ["train", "valid", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

# Run directory check on import
ensure_directories()
