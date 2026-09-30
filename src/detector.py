"""
YOLO Object Detector Module for Number Plate Detection.
Handles model loading, fallback demo mode, inference, bounding box extraction,
and plate cropping.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import config

class NumberPlateDetector:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else config.MODEL_PATH
        self.model = None
        self.is_demo_mode = False
        self.error_message = ""
        
        self._load_model()

    def _load_model(self):
        """Attempt to load YOLO model from best.pt; activate fallback if missing."""
        if not self.model_path.exists():
            self.is_demo_mode = True
            self.error_message = (
                f"YOLO model weights file not found at '{self.model_path}'. "
                "Operating in Demo/Fallback Mode."
            )
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.is_demo_mode = False
            self.error_message = ""
        except Exception as e:
            self.is_demo_mode = True
            self.model = None
            self.error_message = f"Failed to load YOLO model: {str(e)}"

    def detect(self, image: np.ndarray, conf_threshold: float = config.DEFAULT_CONFIDENCE, force_full_image: bool = False) -> Dict[str, Any]:
        """
        Run detection on an input OpenCV BGR image.
        
        Returns:
            dict containing:
                - 'success': bool
                - 'is_demo': bool
                - 'annotated_image': np.ndarray
                - 'detections': List[Dict] with keys:
                    'bbox': (xmin, ymin, xmax, ymax)
                    'confidence': float
                    'crop': np.ndarray
                - 'message': str
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "success": False,
                "is_demo": self.is_demo_mode,
                "annotated_image": None,
                "detections": [],
                "message": "Invalid input image provided."
            }

        # Handle Demo / Fallback Mode when best.pt is not available
        if self.is_demo_mode or self.model is None:
            return self._run_demo_detection(image, force_full_image=force_full_image)

        # Real YOLO Model Inference
        try:
            results = self.model.predict(image, conf=conf_threshold, verbose=False)
            if not results or len(results) == 0:
                return {
                    "success": True,
                    "is_demo": False,
                    "annotated_image": image.copy(),
                    "detections": [],
                    "message": "No number plate detected."
                }

            result = results[0]
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                return {
                    "success": True,
                    "is_demo": False,
                    "annotated_image": image.copy(),
                    "detections": [],
                    "message": "No number plate detected."
                }

            detections = []
            annotated_img = image.copy()
            h, w = image.shape[:2]

            for box in boxes:
                # Bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                
                # Clip coordinates to image boundaries
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)

                conf = float(box.conf[0].cpu().numpy())
                
                # Crop plate region
                crop = image[ymin:ymax, xmin:xmax]

                if crop.size > 0:
                    detections.append({
                        "bbox": (xmin, ymin, xmax, ymax),
                        "confidence": conf,
                        "crop": crop
                    })

                    # Draw bounding box and label on annotated image
                    cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                    label = f"Plate: {conf:.2f}"
                    cv2.putText(annotated_img, label, (xmin, max(ymin - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Sort detections by confidence descending
            detections.sort(key=lambda x: x["confidence"], reverse=True)

            return {
                "success": True,
                "is_demo": False,
                "annotated_image": annotated_img,
                "detections": detections,
                "message": f"Successfully detected {len(detections)} number plate(s)."
            }

        except Exception as e:
            return {
                "success": False,
                "is_demo": False,
                "annotated_image": image.copy(),
                "detections": [],
                "message": f"Error during YOLO inference: {str(e)}"
            }

    def _run_demo_detection(self, image: np.ndarray, force_full_image: bool = False) -> Dict[str, Any]:
        """
        Fallback demo detection mode when trained YOLO model (best.pt) is not present.
        Intelligently detects whether the input photo is already a cropped license plate
        or a full vehicle image.
        """
        h, w = image.shape[:2]
        aspect_ratio = w / float(h) if h > 0 else 1.0

        # If image is wide (aspect ratio >= 1.8) or force_full_image is set,
        # the photo is ALREADY a cropped license plate! Use full image area.
        if force_full_image or aspect_ratio >= 1.8:
            xmin, ymin, xmax, ymax = 0, 0, w, h
            mode_label = "DEMO MODE (Full License Plate Detected)"
        else:
            # Full vehicle image: focus on central vehicle plate region
            ymin = int(h * 0.45)
            ymax = int(h * 0.85)
            xmin = int(w * 0.15)
            xmax = int(w * 0.85)
            mode_label = "DEMO MODE (Simulated Vehicle ROI)"

        crop = image[ymin:ymax, xmin:xmax]
        annotated_img = image.copy()
        
        # Draw demo bounding box in Amber/Orange to signal Demo Mode
        cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 165, 255), 3)
        cv2.putText(annotated_img, mode_label, (max(xmin, 10), max(ymin - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        demo_detection = {
            "bbox": (xmin, ymin, xmax, ymax),
            "confidence": 0.85, # Simulated confidence
            "crop": crop if crop.size > 0 else image
        }

        return {
            "success": True,
            "is_demo": True,
            "annotated_image": annotated_img,
            "detections": [demo_detection],
            "message": "Demo Mode active (Trained model missing). License plate region extracted."
        }
