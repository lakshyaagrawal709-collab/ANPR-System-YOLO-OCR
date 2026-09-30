"""
YOLO Training Script for Number Plate Detector.
Trains YOLOv8 model on custom dataset defined in data/data.yaml.
"""

import sys
import argparse
from pathlib import Path
import config

def train_yolo(data_yaml: Path, epochs: int = 30, imgsz: int = 640, batch: int = 8, output_dir: Path = config.MODELS_DIR):
    """
    Train YOLOv8 model on custom number plate dataset.
    
    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        imgsz: Image resolution size
        batch: Batch size
        output_dir: Directory to save trained model (models/)
    """
    print("=" * 60)
    print("  ANPR SYSTEM - YOLO DETECTOR TRAINING SCRIPT")
    print("=" * 60)
    
    if not data_yaml.exists():
        print(f"[Error] data.yaml not found at: {data_yaml}")
        print("Please prepare your dataset inside data/ folder following data/data.yaml structure.")
        sys.exit(1)

    # Check if dataset image folders exist and contain images
    train_images_dir = config.IMAGES_DIR / "train"
    if not train_images_dir.exists() or len(list(train_images_dir.glob("*.*"))) == 0:
        print(f"[Warning] Training images directory '{train_images_dir}' is empty or missing.")
        print("Please place your annotated dataset images and labels into data/images/ and data/labels/ before training.")
        print("Required subfolders: train, valid, test")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Error] ultralytics package is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    print(f"\n[Info] Loading base pre-trained YOLOv8 nano model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    print(f"[Info] Starting training with settings:")
    print(f"       - Data config: {data_yaml}")
    print(f"       - Epochs: {epochs}")
    print(f"       - Image size: {imgsz}")
    print(f"       - Batch size: {batch}")
    print(f"       - Target output: {config.MODEL_PATH}")

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name="anpr_yolo_run",
            exist_ok=True
        )

        # Retrieve best weights file path
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if best_weights.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            target_best = config.MODEL_PATH
            import shutil
            shutil.copy(best_weights, target_best)
            print("\n" + "=" * 60)
            print(f"[SUCCESS] Training complete!")
            print(f"[SUCCESS] Best weights saved to: {target_best}")
            print("=" * 60)
        else:
            print("[Warning] Training completed but best.pt could not be copied automatically.")
            
    except Exception as e:
        print(f"[Error] Training failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for Number Plate Detection.")
    parser.add_argument("--data", type=str, default=str(config.DATA_YAML_PATH), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=Path(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
