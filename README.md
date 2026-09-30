# Automatic Number Plate Recognition (ANPR) System

> **College IVP (Image & Video Processing) Academic Mini-Project**  
> *Built with Python, Ultralytics YOLOv8, OpenCV, EasyOCR, and Streamlit*

---

## 📌 Project Overview

This project is a complete, modular, end-to-end **Automatic Number Plate Recognition (ANPR) System**. It processes vehicle images or live webcam streams to automatically detect the license plate region, apply image preprocessing, extract registration numbers using Optical Character Recognition (OCR), and log detection history to a CSV database.

The implementation is kept **understandable, modular, and easy to explain in a viva examination** without unnecessary over-engineering.

---

## 🚀 Features

- **Object Detection with YOLO**: Detects number plate bounding boxes on vehicle images.
- **OpenCV Image Preprocessing Pipeline**: 5-stage enhancement (Resize, Grayscale, Bilateral Denoising, CLAHE Contrast, Otsu Binary Thresholding).
- **Deep Learning OCR with EasyOCR**: Extracts alphanumeric text and calculates recognition confidence.
- **Indian Plate Format Verification**: Regex-based format helper for standard state registration and BH-series plates.
- **Interactive Streamlit Web Dashboard**: 5 navigation views (Dashboard, Image Detection, Webcam Detection, History, Viva Guide).
- **Persistent CSV History**: Log timestamp, plate text, detection confidence, OCR confidence, and source with search & download capabilities.
- **Graceful Fallback / Demo Mode**: UI functions seamlessly with a simulated detection ROI even if trained YOLO model (`best.pt`) is absent.
- **YOLO Training Pipeline**: Built-in `train.py` script for training custom datasets on laptop/GPU.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Computer Vision & Image Processing**: OpenCV (`opencv-python`), Pillow (`PIL`), NumPy
- **Object Detection**: Ultralytics YOLOv8 (`ultralytics`)
- **Character Recognition**: EasyOCR (`easyocr`)
- **Web UI Framework**: Streamlit (`streamlit`)
- **Data Handling**: Pandas (`pandas`), PyYAML (`pyyaml`), Python CSV

---

## 📂 Project Structure

```text
ANPR_Project/
│
├── app.py                  # Main Streamlit Web Application
├── config.py               # Configuration & Pathlib relative paths
├── train.py                # Script to train YOLO on custom dataset
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation & viva guide
├── .gitignore              # Git ignore rules
│
├── models/                 # Model directory
│   └── best.pt             # Trained YOLO weights (User places trained file here)
│
├── src/                    # Modular source code
│   ├── __init__.py         # Package initialization
│   ├── detector.py         # YOLO detector class & Fallback Demo mode
│   ├── preprocessing.py    # 5-Stage OpenCV image preprocessing pipeline
│   ├── ocr.py              # EasyOCR engine & Indian plate format check
│   ├── history.py          # CSV detection logger & dashboard statistics
│   └── utils.py            # Image conversion & synthetic sample generator
│
├── data/                   # Dataset directory for training
│   ├── images/             # Subfolders: train/, valid/, test/
│   ├── labels/             # Subfolders: train/, valid/, test/
│   └── data.yaml           # Dataset configuration file
│
├── test_images/            # Pre-generated sample test images
├── outputs/                # Directory for saved predictions
└── history/                # History storage
    └── detections.csv      # CSV detection database
```

---

## 💻 Installation & Setup Guide

### Step 1: Clone or Navigate to Project Folder
```powershell
cd ANPR_Project
```

### Step 2: Create a Virtual Environment (Recommended)
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(On Linux/macOS: `source venv/bin/activate`)*

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🏃 How to Run the Application

Execute the Streamlit web application:

```powershell
streamlit run app.py
```

After running the command, open your browser at `http://localhost:8501`.

---

## 🎯 Model Handling & Demo Mode

1. **If `models/best.pt` exists**: The application loads your trained YOLO model for real plate localization.
2. **If `models/best.pt` does NOT exist**: The app operates in **Demo Mode**. The sidebar displays a warning, and a simulated plate bounding box is extracted so that the preprocessing, EasyOCR, UI, and CSV history pipeline can be evaluated immediately without crashing.

---

## 🏋️ Training Custom YOLO Model (Optional)

If you wish to train your own YOLOv8 number plate detector:

### 1. Prepare Dataset
Place your annotated dataset into the `data/` directory:
- `data/images/train/`, `data/images/valid/`
- `data/labels/train/`, `data/labels/valid/`

Ensure `data/data.yaml` contains:
```yaml
path: ../data
train: images/train
val: images/valid

nc: 1
names:
  0: number_plate
```

### 2. Run Training Command
```powershell
python train.py --epochs 30 --imgsz 640 --batch 8
```

The script will automatically train the model and save `best.pt` into `models/best.pt`.

---

## 📊 Detection History Management

All successful detections are logged into `history/detections.csv`:

| timestamp | plate_number | detection_confidence | ocr_confidence | source |
| :--- | :--- | :--- | :--- | :--- |
| 2026-09-29 18:30:20 | MH12AB1234 | 0.9120 | 0.8750 | Image |

From the **Detection History** page in the Streamlit UI, you can:
- Search records by plate number.
- Export/Download history as a CSV file.
- Clear history data.

---

## 🎓 Viva-Voce Examination Guide

### Q1: What is the main objective of this project?
**Answer:** The objective is to design an Automatic Number Plate Recognition (ANPR) system that detects license plates from vehicle images or camera feeds, enhances the image quality using computer vision techniques, recognizes the vehicle registration number using OCR, and stores logs for security/traffic monitoring.

### Q2: What is YOLO's role vs. OpenCV vs. EasyOCR?
- **YOLO (You Only Look Once)**: Responsible for **Object Localization**. It scans the entire frame to locate the number plate bounding box coordinates.
- **OpenCV**: Responsible for **Image Preprocessing**. It cleans the cropped plate image (grayscaling, bilateral noise reduction, CLAHE contrast enhancement, Otsu thresholding) to optimize character clarity.
- **EasyOCR**: Responsible for **Character Recognition (OCR)**. It converts the preprocessed binary character pixels into machine-readable text string.

### Q3: Why are YOLO and OCR split into separate stages?
**Answer:** Running OCR on a full vehicle image is computationally expensive and prone to high false-positive rates due to background clutter (headlights, stickers, car logos, street signs). Using YOLO first to crop the exact license plate isolates the region of interest, making OCR faster and significantly more accurate.

### Q4: What image preprocessing steps are applied and why?
1. **Resizing**: Upscales small plate crops to a standard height (~120px) to prevent pixelation.
2. **Grayscale**: Reduces 3-channel color data into single-channel luminance.
3. **Bilateral Filter**: Smooths background noise while keeping character edges sharp.
4. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Equalizes illumination under harsh sunlight or dark shadows.
5. **Otsu Binarization**: Converts gray pixels into high-contrast black & white binary mask for text extraction.

---

## ⚠️ Academic Disclaimer & Limitations

- **Accuracy**: Recognition performance depends on lighting conditions, plate cleanliness, font standard, and camera angle.
- **Scope**: Developed as an academic mini-project for demonstration and educational purposes. It does not claim 100% recognition accuracy or production-level traffic enforcement capability.
