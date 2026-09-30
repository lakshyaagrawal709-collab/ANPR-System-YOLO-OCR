"""
OCR Module using EasyOCR.
Handles character recognition, text cleaning, confidence calculation,
and Indian number plate format validation.
"""

import re
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Global cache for EasyOCR reader instance
_READER_INSTANCE = None

def get_ocr_reader(lang_list: list = ['en'], gpu: bool = False):
    """
    Get or initialize singleton EasyOCR Reader.
    Prevents expensive re-initialization on every frame/call.
    """
    global _READER_INSTANCE
    if _READER_INSTANCE is None:
        try:
            import easyocr
            # Initialize EasyOCR reader
            _READER_INSTANCE = easyocr.Reader(lang_list, gpu=gpu)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR reader: {str(e)}")
    return _READER_INSTANCE

def clean_ocr_text(raw_text: str) -> str:
    """
    Clean OCR text:
    - Convert to uppercase
    - Remove spaces and newline characters
    - Remove special characters (keep only A-Z and 0-9)
    """
    if not raw_text:
        return ""
    
    # Uppercase and strip whitespace
    text = raw_text.upper().strip()
    # Remove everything except uppercase letters A-Z and digits 0-9
    cleaned = re.sub(r'[^A-Z0-9]', '', text)
    return cleaned

def validate_indian_plate_format(plate_number: str) -> Dict[str, Any]:
    """
    Optional helper to check standard Indian vehicle registration formats:
    - Standard: e.g. MH12AB1234, DL01C1234, KA05M9999
    - BH-Series: e.g. 22BH1234A
    Note: Returns informational status without rejecting valid non-standard plates.
    """
    if not plate_number:
        return {"is_valid_format": False, "format_type": "Unknown"}
    
    # Standard format: 2 letters (State) + 2 digits (District) + 1-3 letters + 4 digits
    std_pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$'
    # BH Series pattern: 2 digits (Year) + BH + 4 digits + 1-2 letters
    bh_pattern = r'^\d{2}BH\d{4}[A-Z]{1,2}$'

    if re.match(std_pattern, plate_number):
        return {"is_valid_format": True, "format_type": "Standard Indian State"}
    elif re.match(bh_pattern, plate_number):
        return {"is_valid_format": True, "format_type": "Bharat Series (BH)"}
    else:
        return {"is_valid_format": False, "format_type": "Custom / Non-Standard"}

def recognize_plate_text(processed_img: np.ndarray, reader=None) -> Tuple[str, float, Dict[str, Any]]:
    """
    Run EasyOCR on the processed plate image.
    
    Returns:
        tuple: (cleaned_plate_number, average_confidence, format_info)
    """
    if processed_img is None or processed_img.size == 0:
        return "", 0.0, {"is_valid_format": False, "format_type": "Empty Image"}

    if reader is None:
        try:
            reader = get_ocr_reader()
        except Exception as e:
            print(f"[OCR Warning] EasyOCR not available: {e}")
            return "", 0.0, {"is_valid_format": False, "format_type": "OCR Reader Error"}

    try:
        # EasyOCR readtext returns list of (bbox, text, prob)
        results = reader.readtext(processed_img)
        
        if not results:
            return "", 0.0, {"is_valid_format": False, "format_type": "No Text Detected"}

        raw_texts = []
        confidences = []

        for bbox, text, prob in results:
            cleaned_sub = clean_ocr_text(text)
            if cleaned_sub:
                raw_texts.append(cleaned_sub)
                confidences.append(prob)

        if not raw_texts:
            return "", 0.0, {"is_valid_format": False, "format_type": "No Valid Characters"}

        full_cleaned_text = "".join(raw_texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        format_info = validate_indian_plate_format(full_cleaned_text)

        return full_cleaned_text, avg_confidence, format_info

    except Exception as e:
        print(f"[OCR Error] Exception during OCR execution: {e}")
        return "", 0.0, {"is_valid_format": False, "format_type": f"Error: {str(e)}"}
