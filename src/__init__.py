"""
ANPR Source Package Initialization.
Exposes core detector, preprocessing, ocr, and history functionality.
"""

from .detector import NumberPlateDetector
from .preprocessing import preprocess_plate_for_ocr, get_preprocessing_stages
from .ocr import recognize_plate_text
from .history import save_detection, load_history, clear_history, get_dashboard_stats

__all__ = [
    "NumberPlateDetector",
    "preprocess_plate_for_ocr",
    "get_preprocessing_stages",
    "recognize_plate_text",
    "save_detection",
    "load_history",
    "clear_history",
    "get_dashboard_stats"
]
