"""
Automatic Number Plate Recognition (ANPR) System
Streamlit Web Application
Tech Stack: Python, OpenCV, Ultralytics YOLO, EasyOCR, Streamlit, Pandas
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Set Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="ANPR System - YOLO + OpenCV + EasyOCR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (NMIMS Academic UI Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .plate-badge {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 3px;
        color: #0F172A;
        background-color: #FEF08A;
        padding: 10px 20px;
        border-radius: 6px;
        border: 2px solid #EAB308;
        display: inline-block;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

import config
from src.detector import NumberPlateDetector
from src.preprocessing import preprocess_plate_for_ocr, get_preprocessing_stages
from src.ocr import recognize_plate_text, get_ocr_reader
from src.history import save_detection, load_history, clear_history, get_dashboard_stats
from src.utils import load_image_as_bgr, bgr_to_rgb, generate_sample_test_images

# Ensure sample images exist in test_images/
generate_sample_test_images()

# Initialize Cached Detector and OCR Reader to avoid reload overhead
@st.cache_resource(show_spinner="Initializing YOLO Detector...")
def load_detector():
    return NumberPlateDetector(config.MODEL_PATH)

@st.cache_resource(show_spinner="Loading EasyOCR Engine...")
def load_ocr():
    try:
        return get_ocr_reader(config.OCR_LANGUAGES)
    except Exception as e:
        return None

detector = load_detector()
ocr_reader = load_ocr()

# ==========================================
# SIDEBAR NAVIGATION & MODEL STATUS
# ==========================================
st.sidebar.title("📌 ANPR Navigation")
menu = st.sidebar.radio(
    "Select Module:",
    ["Dashboard", "Image Detection", "Webcam Detection", "Detection History", "About & Viva Guide"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status & Settings")

# Model Status Alert in Sidebar
if config.MODEL_PATH.exists():
    st.sidebar.success("✅ Trained YOLO Model: Loaded (`models/best.pt`)")
else:
    st.sidebar.warning("⚠️ YOLO Model: Missing\n*(Demo Mode Active)*")

# Detection Confidence Slider
conf_threshold = st.sidebar.slider(
    "YOLO Confidence Threshold:",
    min_value=0.10,
    max_value=0.90,
    value=config.DEFAULT_CONFIDENCE,
    step=0.05
)

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-header">Automatic Number Plate Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Computer Vision Mini-Project | YOLOv8 + OpenCV + EasyOCR</div>', unsafe_allow_html=True)

if detector.is_demo_mode:
    st.info("ℹ️ **Demo Mode Active** — Trained YOLO model (`models/best.pt`) was not found. Using simulated plate detection bounding box to demonstrate the full UI pipeline.")

# ==========================================
# MODULE 1: DASHBOARD
# ==========================================
if menu == "Dashboard":
    st.header("📊 Detection System Dashboard")
    stats = get_dashboard_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Detections", stats["total_detections"])
    with col2:
        st.metric("Unique Vehicle Plates", stats["unique_plates"])
    with col3:
        st.metric("Latest Detected Plate", stats["latest_plate"])
    with col4:
        st.metric("Avg Detection Conf.", f"{stats['avg_det_confidence']}%" if stats["total_detections"] > 0 else "N/A")

    st.markdown("---")
    
    if stats["total_detections"] == 0:
        st.info("No detections recorded yet. Navigate to **Image Detection** or **Webcam Detection** to perform ANPR scans.")
    else:
        st.subheader("📋 Recent Detections Log")
        df_history = load_history()
        st.dataframe(df_history.tail(10).iloc[::-1], use_container_width=True)

# ==========================================
# MODULE 2: IMAGE DETECTION
# ==========================================
elif menu == "Image Detection":
    st.header("🖼️ Image Detection & OCR Pipeline")

    input_option = st.radio("Choose Image Input Source:", ["Upload Vehicle Image", "Use Sample Test Images"], horizontal=True)
    
    bgr_img = None
    input_filename = "Uploaded_Image"

    if input_option == "Upload Vehicle Image":
        uploaded_file = st.file_uploader("Upload a vehicle photo (JPG, JPEG, PNG):", type=config.ALLOWED_EXTENSIONS)
        if uploaded_file is not None:
            image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            bgr_img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            input_filename = uploaded_file.name
    else:
        test_images = list(config.TEST_IMAGES_DIR.glob("*.*"))
        if test_images:
            selected_test_file = st.selectbox("Select a Sample Test Image:", [f.name for f in test_images])
            selected_path = config.TEST_IMAGES_DIR / selected_test_file
            bgr_img = cv2.imread(str(selected_path))
            input_filename = selected_test_file
        else:
            st.warning("No sample test images found in test_images/ directory.")

    if bgr_img is not None:
        h_img, w_img = bgr_img.shape[:2]
        is_wide_crop = (w_img / float(h_img)) >= 1.8 if h_img > 0 else False
        
        force_full_image = st.checkbox(
            "🖼️ Photo is already a cropped license plate (Use 100% of image area)",
            value=is_wide_crop,
            help="Check this if your uploaded image is already a close-up photo of the license plate."
        )

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("📷 Original Image")
            st.image(bgr_to_rgb(bgr_img), use_container_width=True)

        # Run ANPR Pipeline Button
        if st.button("🚀 Run ANPR Recognition", type="primary", use_container_width=True):
            with st.spinner("Processing image through YOLO & EasyOCR pipeline..."):
                det_result = detector.detect(bgr_img, conf_threshold=conf_threshold, force_full_image=force_full_image)

            if not det_result["success"]:
                st.error(f"Detection Error: {det_result['message']}")
            elif not det_result["detections"]:
                st.warning("⚠️ No number plate detected in this image. Try lowering the confidence threshold slider in the sidebar.")
                with col_img2:
                    st.subheader("🔍 Annotated Image")
                    st.image(bgr_to_rgb(det_result["annotated_image"]), use_container_width=True)
            else:
                top_det = det_result["detections"][0]
                crop_bgr = top_det["crop"]
                det_conf = top_det["confidence"]

                with col_img2:
                    st.subheader("🎯 YOLO Bounding Box")
                    st.image(bgr_to_rgb(det_result["annotated_image"]), use_container_width=True)

                st.markdown("---")
                st.subheader("🔬 Plate Preprocessing & Character Recognition")
                
                col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
                with col_p1:
                    st.markdown("**Cropped Plate**")
                    st.image(bgr_to_rgb(crop_bgr), use_container_width=True)

                # OpenCV Preprocessing
                try:
                    processed_plate = preprocess_plate_for_ocr(crop_bgr)
                    with col_p2:
                        st.markdown("**Preprocessed (Binary)**")
                        st.image(processed_plate, use_container_width=True)
                except Exception as e:
                    st.error(f"Preprocessing error: {e}")
                    processed_plate = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

                # EasyOCR Recognition
                plate_text, ocr_conf, format_info = recognize_plate_text(processed_plate, reader=ocr_reader)

                with col_p3:
                    st.markdown("**Recognized Number Plate Result**")
                    if plate_text:
                        st.markdown(f'<div class="plate-badge">{plate_text}</div>', unsafe_allow_html=True)
                        st.write(f"• **YOLO Detection Confidence:** `{det_conf * 100:.1f}%`")
                        st.write(f"• **OCR Confidence:** `{ocr_conf * 100:.1f}%`")
                        
                        fmt_icon = "✅" if format_info["is_valid_format"] else "ℹ️"
                        st.write(f"• **Format Verification:** {fmt_icon} {format_info['format_type']}")

                        # Save to CSV history automatically
                        saved = save_detection(
                            plate_number=plate_text,
                            det_conf=det_conf,
                            ocr_conf=ocr_conf,
                            source="Demo Mode" if det_result["is_demo"] else "Image"
                        )
                        if saved:
                            st.toast("✅ Detection logged into history CSV!", icon="💾")
                    else:
                        st.error("Could not extract legible text from the cropped plate.")

                # Expandable Preprocessing Stages Breakdown for Viva Explanation
                with st.expander("🔍 View All 5 OpenCV Preprocessing Stages (For Viva / Demo)"):
                    stages = get_preprocessing_stages(crop_bgr)
                    stage_cols = st.columns(len(stages))
                    for idx, (stage_name, stage_img) in enumerate(stages.items()):
                        with stage_cols[idx]:
                            st.caption(f"Stage {idx+1}: {stage_name}")
                            if len(stage_img.shape) == 2:
                                st.image(stage_img, use_container_width=True)
                            else:
                                st.image(bgr_to_rgb(stage_img), use_container_width=True)

# ==========================================
# MODULE 3: WEBCAM DETECTION
# ==========================================
elif menu == "Webcam Detection":
    st.header("📹 Live Webcam / Camera Detection")
    st.write("Capture a frame from your webcam for real-time license plate extraction.")

    camera_image = st.camera_input("Take a photo of vehicle number plate")

    if camera_image is not None:
        image_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        bgr_frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if bgr_frame is not None:
            st.subheader("⚡ Processing Captured Frame")
            det_result = detector.detect(bgr_frame, conf_threshold=conf_threshold)

            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.markdown("**Annotated Detection Frame**")
                st.image(bgr_to_rgb(det_result["annotated_image"]), use_container_width=True)

            with col_w2:
                if det_result["detections"]:
                    top_det = det_result["detections"][0]
                    crop_bgr = top_det["crop"]
                    det_conf = top_det["confidence"]

                    processed_plate = preprocess_plate_for_ocr(crop_bgr)
                    plate_text, ocr_conf, format_info = recognize_plate_text(processed_plate, reader=ocr_reader)

                    st.markdown("**Extracted Plate Details**")
                    if plate_text:
                        st.markdown(f'<div class="plate-badge">{plate_text}</div>', unsafe_allow_html=True)
                        st.write(f"• **YOLO Confidence:** `{det_conf * 100:.1f}%`")
                        st.write(f"• **OCR Confidence:** `{ocr_conf * 100:.1f}%`")
                        st.write(f"• **Format:** {format_info['format_type']}")

                        save_detection(
                            plate_number=plate_text,
                            det_conf=det_conf,
                            ocr_conf=ocr_conf,
                            source="Webcam"
                        )
                        st.success("Log saved to detection history!")
                    else:
                        st.warning("No readable text detected on captured plate.")
                else:
                    st.warning("No number plate detected in captured webcam frame.")

# ==========================================
# MODULE 4: DETECTION HISTORY
# ==========================================
elif menu == "Detection History":
    st.header("📜 Persistent Detection History")
    df_history = load_history()

    if df_history.empty:
        st.info("No detection history records found in CSV.")
    else:
        # Search & Filter
        search_query = st.text_input("🔍 Search Plate Number:", "").upper().strip()
        if search_query:
            filtered_df = df_history[df_history["plate_number"].str.contains(search_query, na=False)]
        else:
            filtered_df = df_history

        st.dataframe(filtered_df.iloc[::-1], use_container_width=True)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download History CSV",
                data=csv_data,
                file_name="anpr_detection_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_h2:
            if st.button("🗑️ Clear All Detection History", type="secondary", use_container_width=True):
                if clear_history():
                    st.success("Detection history cleared successfully!")
                    st.rerun()

# ==========================================
# MODULE 5: ABOUT & VIVA GUIDE
# ==========================================
elif menu == "About & Viva Guide":
    st.header("📚 Project Architecture & Viva Q&A Guide")
    
    st.markdown("""
    ### 🏗️ System Architecture Pipeline
    ```text
    [Input Image / Frame] ──> [YOLOv8 Plate Detector] ──> [Bounding Box Crop]
                                                                  │
    [Saved History CSV] <── [Clean OCR Text] <── [EasyOCR Reader] <── [OpenCV Preprocessing]
    ```

    ---
    ### 🎓 Essential Viva Questions & Answers

    **Q1: What is the role of YOLO in this project?**
    > **Answer:** YOLO (You Only Look Once) is an object detection model used strictly for **localization**. It scans the entire vehicle image and outputs the bounding box coordinates `(xmin, ymin, xmax, ymax)` surrounding the number plate.

    **Q2: What is the role of OpenCV?**
    > **Answer:** OpenCV performs **image preprocessing** on the cropped plate image. It resizes, converts to grayscale, removes noise via Bilateral Filtering, enhances contrast using CLAHE, and applies Otsu thresholding to make character outlines crisp for OCR.

    **Q3: What does EasyOCR do?**
    > **Answer:** EasyOCR is a deep learning OCR engine built on PyTorch. It takes the preprocessed binary plate image, recognizes character glyphs, and extracts the text string along with individual character probabilities.

    **Q4: Why separate YOLO and OCR into two stages?**
    > **Answer:** Running OCR directly on a full car image is slow and inaccurate because background noise (headlights, car badges, road text) causes false positives. YOLO quickly isolates the exact number plate region first, allowing OCR to focus strictly on characters.

    **Q5: How does the system handle Indian number plate variations?**
    > **Answer:** Indian plates vary across states and formats (e.g. Standard State `MH12AB1234` vs Bharat Series `22BH1234A`). The regex helper identifies standard formats for UI feedback but does not discard custom plates.
    """)
