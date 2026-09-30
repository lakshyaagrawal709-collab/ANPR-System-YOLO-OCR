"""
Image Preprocessing Pipeline for Number Plate OCR.
Provides modular image enhancement functions using OpenCV.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional

def validate_crop(crop: np.ndarray, min_width: int = 10, min_height: int = 10) -> bool:
    """Validate if the cropped image is non-empty and has reasonable dimensions."""
    if crop is None or not isinstance(crop, np.ndarray):
        return False
    if crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return h >= min_height and w >= min_width

def resize_plate(crop: np.ndarray, target_height: int = 100) -> np.ndarray:
    """Resize the license plate crop while maintaining aspect ratio for optimal OCR."""
    if not validate_crop(crop):
        return crop
    
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop
        
    aspect_ratio = w / float(h)
    new_height = target_height
    new_width = int(new_height * aspect_ratio)
    
    # Avoid resizing if too large or invalid
    if new_width <= 0 or new_height <= 0:
        return crop
        
    # Use INTER_CUBIC for upscaling, INTER_AREA for downscaling
    interpolation = cv2.INTER_CUBIC if new_height > h else cv2.INTER_AREA
    return cv2.resize(crop, (new_width, new_height), interpolation=interpolation)

def preprocess_plate_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Main preprocessing pipeline for OCR readiness:
    1. Resize for character clarity
    2. Grayscale conversion
    3. Noise reduction (Bilateral filter preserves sharp character edges)
    4. Contrast enhancement (CLAHE)
    5. Adaptive/Otsu Thresholding
    """
    if not validate_crop(crop):
        raise ValueError("Invalid image crop passed to preprocessing pipeline.")

    # 1. Resize crop
    resized = resize_plate(crop, target_height=120)

    # 2. Grayscale conversion
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    elif len(resized.shape) == 3 and resized.shape[2] == 4:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
    else:
        gray = resized.copy()

    # 3. Bilateral Filter (Noise reduction preserving text edges)
    denoised = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)

    # 4. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Otsu Thresholding to convert to binary image
    _, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded

def get_preprocessing_stages(crop: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Returns intermediate preprocessing stage outputs for visual inspection in Streamlit.
    Useful for demonstration and viva explanation.
    """
    if not validate_crop(crop):
        return {}

    resized = resize_plate(crop, target_height=120)
    
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized.copy()

    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "Resized": resized,
        "Grayscale": gray,
        "Denoised": denoised,
        "CLAHE Enhanced": enhanced,
        "Otsu Binary": thresholded
    }
