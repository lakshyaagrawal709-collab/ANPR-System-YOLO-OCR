"""
Utility Helper Functions for ANPR Project.
Handles image conversions, sample test image generation, and UI helpers.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Union, Tuple, Optional
import config

def load_image_as_bgr(input_source: Union[str, Path, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
    """Convert various image input types (path, PIL Image, numpy array) into OpenCV BGR numpy array."""
    if input_source is None:
        return None

    if isinstance(input_source, (str, Path)):
        img_path = Path(input_source)
        if not img_path.exists():
            return None
        return cv2.imread(str(img_path))

    elif isinstance(input_source, Image.Image):
        # Convert PIL Image to RGB numpy array then to BGR
        rgb_array = np.array(input_source.convert("RGB"))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    elif isinstance(input_source, np.ndarray):
        # Check if 2D or 3D
        if len(input_source.shape) == 2:
            return cv2.cvtColor(input_source, cv2.COLOR_GRAY2BGR)
        elif len(input_source.shape) == 3 and input_source.shape[2] == 3:
            return input_source
        elif len(input_source.shape) == 3 and input_source.shape[2] == 4:
            return cv2.cvtColor(input_source, cv2.COLOR_BGRA2BGR)
            
    return None

def bgr_to_rgb(bgr_img: np.ndarray) -> np.ndarray:
    """Convert BGR OpenCV image to RGB for Streamlit/PIL rendering."""
    if bgr_img is None:
        return None
    if len(bgr_img.shape) == 2:
        return cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

def generate_sample_test_images():
    """
    Generate synthetic sample vehicle images with standard Indian number plates
    in test_images/ directory for quick application demonstration.
    """
    test_dir = config.TEST_IMAGES_DIR
    test_dir.mkdir(parents=True, exist_ok=True)

    sample_plates = [
        ("MH12AB1234", "sample_car_mh12.jpg", (30, 60, 90)),
        ("DL01C5678", "sample_car_dl01.jpg", (90, 40, 40)),
        ("KA05M9999", "sample_car_ka05.jpg", (40, 80, 50))
    ]

    for plate_text, filename, car_color in sample_plates:
        file_path = test_dir / filename
        if file_path.exists():
            continue

        # Create synthetic car front view (width 800, height 500)
        img = Image.new("RGB", (800, 500), color=(220, 225, 230))
        draw = ImageDraw.Draw(img)

        # Draw road ground
        draw.rectangle([0, 350, 800, 500], fill=(70, 75, 80))

        # Draw car body
        draw.rectangle([150, 200, 650, 420], fill=car_color)
        # Car windshield
        draw.polygon([(220, 200), (300, 100), (500, 100), (580, 200)], fill=(180, 220, 240))
        # Car headlights
        draw.ellipse([170, 260, 240, 310], fill=(255, 255, 200), outline=(50, 50, 50))
        draw.ellipse([560, 260, 630, 310], fill=(255, 255, 200), outline=(50, 50, 50))
        # Car grille
        draw.rectangle([280, 270, 520, 330], fill=(40, 40, 40))

        # Number Plate background (White plate with black border)
        plate_box = [300, 340, 500, 395]
        draw.rectangle(plate_box, fill=(255, 255, 255), outline=(0, 0, 0), width=3)
        # IND strip
        draw.rectangle([305, 345, 320, 390], fill=(0, 51, 160))

        # Draw text on plate
        try:
            # Attempt basic font or default
            font = ImageFont.load_default()
        except Exception:
            font = None

        draw.text((335, 355), plate_text, fill=(0, 0, 0), font=font)

        img.save(file_path)
