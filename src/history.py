"""
Detection History Tracker using CSV.
Manages persistent detection records, data retrieval, statistics, and clearing.
"""

import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import config

# Standard CSV Headers
CSV_HEADERS = ["timestamp", "plate_number", "detection_confidence", "ocr_confidence", "source"]

def init_history_file() -> Path:
    """Initialize CSV history file with headers if it doesn't exist."""
    csv_path = config.HISTORY_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            
    return csv_path

def save_detection(plate_number: str, det_conf: float, ocr_conf: float, source: str = "Image") -> bool:
    """
    Save a new detection entry to the CSV history.
    
    Args:
        plate_number: Cleaned recognized plate text
        det_conf: YOLO detection confidence score (0.0 to 1.0)
        ocr_conf: EasyOCR confidence score (0.0 to 1.0)
        source: Source of image ('Image', 'Webcam', 'Demo')
    """
    if not plate_number:
        return False

    csv_path = init_history_file()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prevent saving rapid duplicate webcam entries (within last 5 seconds with same plate)
    try:
        if source == "Webcam":
            df = load_history()
            if not df.empty:
                last_row = df.iloc[-1]
                last_plate = str(last_row.get("plate_number", ""))
                last_source = str(last_row.get("source", ""))
                if last_plate == plate_number and last_source == "Webcam":
                    # Check time difference
                    try:
                        last_time = datetime.strptime(str(last_row["timestamp"]), "%Y-%m-%d %H:%M:%S")
                        if (datetime.now() - last_time).total_seconds() < 5.0:
                            return False  # Skip duplicate log
                    except Exception:
                        pass
    except Exception as e:
        print(f"[History Warning] Error checking duplicate entry: {e}")

    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                plate_number,
                f"{det_conf:.4f}",
                f"{ocr_conf:.4f}",
                source
            ])
        return True
    except Exception as e:
        print(f"[History Error] Failed to save detection: {e}")
        return False

def load_history() -> pd.DataFrame:
    """Load detection history into a Pandas DataFrame."""
    csv_path = init_history_file()
    try:
        df = pd.read_csv(csv_path)
        # Ensure correct column names
        if list(df.columns) != CSV_HEADERS:
            df = pd.DataFrame(columns=CSV_HEADERS)
        return df
    except Exception as e:
        print(f"[History Error] Could not read CSV file: {e}")
        return pd.DataFrame(columns=CSV_HEADERS)

def clear_history() -> bool:
    """Clear all entries from the detection history CSV file."""
    csv_path = config.HISTORY_CSV_PATH
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        return True
    except Exception as e:
        print(f"[History Error] Could not clear CSV file: {e}")
        return False

def get_dashboard_stats() -> Dict[str, Any]:
    """Calculate summary statistics for the Streamlit Dashboard."""
    df = load_history()
    if df.empty:
        return {
            "total_detections": 0,
            "unique_plates": 0,
            "latest_plate": "N/A",
            "avg_det_confidence": 0.0,
            "avg_ocr_confidence": 0.0
        }

    total = len(df)
    unique_plates = df["plate_number"].nunique()
    latest_plate = str(df.iloc[-1]["plate_number"])
    
    # Calculate average confidences handling possible string conversions
    try:
        avg_det_conf = float(pd.to_numeric(df["detection_confidence"], errors='coerce').mean())
    except Exception:
        avg_det_conf = 0.0
        
    try:
        avg_ocr_conf = float(pd.to_numeric(df["ocr_confidence"], errors='coerce').mean())
    except Exception:
        avg_ocr_conf = 0.0

    return {
        "total_detections": total,
        "unique_plates": unique_plates,
        "latest_plate": latest_plate,
        "avg_det_confidence": round(avg_det_conf * 100, 1),
        "avg_ocr_confidence": round(avg_ocr_conf * 100, 1)
    }
