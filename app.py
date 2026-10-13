"""
Automatic Number Plate Recognition (ANPR) System
Enhanced Mobile-Friendly Streamlit Application
Tech: YOLOv8 · OpenCV · EasyOCR · Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import socket
import time
from io import BytesIO

# ── Must be FIRST Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="ANPR System – YOLO + OCR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Mobile-optimised CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

* { box-sizing: border-box; }

/* Set the default font on the app, without overriding Streamlit icon fonts */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp { background: #080D1A; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0C1220 0%,#111827 100%);
    border-right: 1px solid #1E3A5F;
}

/* ── Header ─── */
.main-header {
    font-size: clamp(1.6rem, 5vw, 2.6rem);
    font-weight: 900;
    background: linear-gradient(135deg,#3B82F6,#8B5CF6,#EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 2px;
}
.sub-header {
    font-size: clamp(0.75rem,2vw,0.95rem);
    color: #475569;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 18px;
}

/* ── Plate badge ─── */
.plate-badge {
    font-size: clamp(1.5rem,6vw,2.4rem);
    font-weight: 900;
    letter-spacing: 5px;
    color: #0F172A;
    background: linear-gradient(135deg,#FEF08A,#FDE047);
    padding: 14px 28px;
    border-radius: 10px;
    border: 3px solid #CA8A04;
    display: inline-block;
    font-family: 'Courier New', monospace !important;
    box-shadow: 0 4px 24px rgba(234,179,8,.45);
    word-break: break-all;
}

/* ── Metric cards ─── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#0F1629,#1A2540);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { font-size: clamp(1.1rem,3vw,1.7rem) !important; font-weight: 700 !important; color: #60A5FA !important; }
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.78rem !important; }

/* ── Touch-friendly buttons ─── */
.stButton > button {
    min-height: 52px !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#3B82F6,#8B5CF6) !important;
    border: none !important; color: white !important;
    box-shadow: 0 4px 16px rgba(59,130,246,.45) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(59,130,246,.6) !important;
}

/* ── Camera input ─── */
[data-testid="stCameraInput"] > div {
    border-radius: 16px !important;
    border: 2px dashed #3B82F6 !important;
    min-height: 240px;
    background: rgba(59,130,246,.04);
}
/* ── File uploader ─── */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 16px !important;
    border: 2px dashed #3B82F6 !important;
    background: rgba(59,130,246,.05) !important;
}

/* ── Info pills ─── */
.info-pill {
    display: inline-block;
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 4px 3px;
}
.conf-high { background:rgba(34,197,94,.15);  color:#22C55E; border:1px solid #22C55E; }
.conf-mid  { background:rgba(234,179,8,.15);  color:#EAB308; border:1px solid #EAB308; }
.conf-low  { background:rgba(239,68,68,.15);  color:#EF4444; border:1px solid #EF4444; }
.pill-blue { background:rgba(59,130,246,.15); color:#60A5FA; border:1px solid #3B82F6;  }

/* ── Cards ─── */
.result-card {
    background: linear-gradient(135deg,#0F1629,#1A2540);
    border: 1px solid #1E3A5F;
    border-radius: 20px;
    padding: 20px 24px;
    margin: 12px 0;
}
.demo-banner {
    background: linear-gradient(135deg,rgba(234,179,8,.12),rgba(249,115,22,.12));
    border: 1px solid rgba(234,179,8,.4);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 16px;
    color: #FDE68A;
}
.section-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 24px 0 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1E3A5F;
}
.live-badge {
    display: inline-block;
    background: rgba(239,68,68,.15);
    color: #EF4444;
    border: 1px solid #EF4444;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 700;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

/* ── QR section ─── */
.qr-box {
    background: white;
    border-radius: 14px;
    padding: 10px;
    text-align: center;
    margin: 8px 0;
}

/* ── Mobile overrides ─── */
@media (max-width:768px) {
    .plate-badge { letter-spacing:2px; padding:10px 14px; }
    .main-header { text-align:center; }
    .sub-header  { text-align:center; }
}

/* ── Fix Viva Q&A expander text and icon layout ─── */
[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    min-height: 44px !important;
}

[data-testid="stExpander"] summary p {
    display: block !important;
    position: static !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.6 !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
}

[data-testid="stExpander"] summary svg {
    flex-shrink: 0 !important;
}

/* ── Clean up default Streamlit chrome ─── */
#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }
[data-testid="stToolbar"] { display:none; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def conf_pill(label: str, value: float) -> str:
    css = "conf-high" if value >= 0.75 else ("conf-mid" if value >= 0.50 else "conf-low")
    return f"<span class='info-pill {css}'>{label} {value*100:.1f}%</span>"

def run_full_pipeline(bgr_img, conf_thresh, force_full, source):
    """Run detect → preprocess → OCR → save. Returns display dict."""
    det_result = detector.detect(bgr_img, conf_threshold=conf_thresh, force_full_image=force_full)
    if not det_result["success"] or not det_result["detections"]:
        return det_result, None, "", 0.0, 0.0, {}
    top = det_result["detections"][0]
    crop = top["crop"]
    det_conf = top["confidence"]
    try:
        processed = preprocess_plate_for_ocr(crop)
    except Exception:
        processed = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    plate_text, ocr_conf, fmt_info = recognize_plate_text(processed, reader=ocr_reader)
    if plate_text:
        save_detection(plate_text, det_conf, ocr_conf, source=source)
    return det_result, crop, plate_text, det_conf, ocr_conf, fmt_info


# ── Lazy imports (safe for Cloud) ─────────────────────────────────────────────
import config
from src.detector import NumberPlateDetector
from src.preprocessing import preprocess_plate_for_ocr, get_preprocessing_stages
from src.ocr import recognize_plate_text, get_ocr_reader
from src.history import save_detection, load_history, clear_history, get_dashboard_stats
from src.utils import bgr_to_rgb, generate_sample_test_images

generate_sample_test_images()

@st.cache_resource(show_spinner="Initialising YOLO Detector…")
def load_detector():
    return NumberPlateDetector(config.MODEL_PATH)

@st.cache_resource(show_spinner="Loading EasyOCR Engine…")
def load_ocr():
    try:
        return get_ocr_reader(config.OCR_LANGUAGES)
    except Exception:
        return None

detector  = load_detector()
ocr_reader = load_ocr()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 ANPR Navigator")

    menu = st.radio(
        "nav",
        ["📊 Dashboard", "📷 Image Detection", "📹 Camera Scan", "📜 History", "ℹ️ About & Viva"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Model status
    if config.MODEL_PATH.exists():
        st.success("✅ YOLO Model: Loaded")
    else:
        st.warning("⚠️ **Demo Mode** – No `models/best.pt`.\nOCR pipeline is fully functional.")

    st.markdown("---")

    conf_threshold = st.slider("🎚️ Detection Confidence", 0.10, 0.90, config.DEFAULT_CONFIDENCE, 0.05)

    st.markdown("---")

    # ── QR Code for Mobile Access ──────────────────────────────────────────
    local_ip   = get_local_ip()
    network_url = f"http://{local_ip}:8502"

    st.markdown("### 📱 Open on Mobile")
    st.caption("Same WiFi → scan QR code:")

    try:
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=7, border=2,
                           error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(network_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="#1E40AF", back_color="white")
        buf = BytesIO()
        qr_img.save(buf, format="PNG")
        st.markdown('<div class="qr-box">', unsafe_allow_html=True)
        st.image(buf.getvalue(), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception:
        st.info("Install `qrcode[pil]` for QR display")

    st.code(network_url, language=None)
    st.caption("Copy URL → open in mobile browser")


# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Automatic Number Plate Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">YOLOv8 Detection · OpenCV Preprocessing · EasyOCR Recognition</div>',
            unsafe_allow_html=True)

if detector.is_demo_mode:
    st.markdown("""
    <div class="demo-banner">
    ⚠️ <b>Demo Mode Active</b> — No trained <code>models/best.pt</code> found.
    Place your trained weights there for real YOLO plate localisation.
    The full preprocessing → EasyOCR → history pipeline works right now.
    </div>""", unsafe_allow_html=True)

page = menu.split(" ", 1)[1]

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 – DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown('<div class="section-title">Live Statistics</div>', unsafe_allow_html=True)
    stats = get_dashboard_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Detections",  stats["total_detections"])
    c2.metric("Unique Plates",     stats["unique_plates"])
    c3.metric("Latest Plate",      stats["latest_plate"])
    c4.metric("Avg YOLO Conf.",    f"{stats['avg_det_confidence']}%" if stats["total_detections"] > 0 else "N/A")

    st.markdown('<div class="section-title">Recent Detections</div>', unsafe_allow_html=True)

    if stats["total_detections"] == 0:
        st.info("No detections yet. Use **Image Detection** or **Camera Scan** to start recognising plates.")
    else:
        df = load_history()
        st.dataframe(df.iloc[::-1].head(20), width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 – IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Image Detection":
    st.markdown('<div class="section-title">Upload & Recognise</div>', unsafe_allow_html=True)

    input_option = st.radio("Source:", ["Upload Photo", "Sample Test Images"], horizontal=True)
    bgr_img = None

    if input_option == "Upload Photo":
        up = st.file_uploader("Upload vehicle or plate image", type=config.ALLOWED_EXTENSIONS)
        if up:
            raw = np.asarray(bytearray(up.read()), dtype=np.uint8)
            bgr_img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    else:
        test_imgs = list(config.TEST_IMAGES_DIR.glob("*.*"))
        if test_imgs:
            sel = st.selectbox("Sample:", [f.name for f in test_imgs])
            bgr_img = cv2.imread(str(config.TEST_IMAGES_DIR / sel))
        else:
            st.warning("No sample images in test_images/.")

    if bgr_img is not None:
        h_img, w_img = bgr_img.shape[:2]
        is_wide  = (w_img / float(h_img)) >= 1.8
        force_full = st.checkbox("🖼️ Already a cropped plate photo (use full image area)",
                                  value=is_wide,
                                  help="Tick this when your image shows only the plate, not the whole car.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Original Image")
            st.image(bgr_to_rgb(bgr_img), width="stretch")

        if st.button("🚀 Run ANPR Recognition", type="primary", width="stretch"):
            with st.spinner("YOLO → OpenCV → EasyOCR …"):
                det_result, crop_bgr, plate_text, det_conf, ocr_conf, fmt_info = \
                    run_full_pipeline(bgr_img, conf_threshold, force_full,
                                      source="Demo" if detector.is_demo_mode else "Image")

            if not det_result["success"]:
                st.error(f"Detection error: {det_result['message']}")
            elif not det_result["detections"]:
                with col_b:
                    st.caption("Annotated")
                    if det_result["annotated_image"] is not None:
                        st.image(bgr_to_rgb(det_result["annotated_image"]), width="stretch")
                st.warning("No plate detected. Lower confidence threshold or try another image.")
            else:
                with col_b:
                    st.caption("YOLO Detection")
                    st.image(bgr_to_rgb(det_result["annotated_image"]), width="stretch")

                st.markdown('<div class="section-title">OCR Pipeline</div>', unsafe_allow_html=True)

                try:
                    processed = preprocess_plate_for_ocr(crop_bgr)
                except Exception:
                    processed = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

                col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
                with col_p1:
                    st.caption("Cropped Plate")
                    st.image(bgr_to_rgb(crop_bgr), width="stretch")
                with col_p2:
                    st.caption("Preprocessed")
                    st.image(processed, width="stretch")
                with col_p3:
                    st.caption("Recognition Result")
                    if plate_text:
                        st.markdown(f'<div class="plate-badge">{plate_text}</div>', unsafe_allow_html=True)
                        st.markdown(
                            conf_pill("YOLO", det_conf) +
                            conf_pill("OCR",  ocr_conf) +
                            f"<span class='info-pill pill-blue'>{fmt_info.get('format_type','—')}</span>",
                            unsafe_allow_html=True
                        )
                        st.toast("Saved to history!", icon="💾")
                    else:
                        st.error("No readable text found on the plate.")

                with st.expander("🔬 View 5-Stage OpenCV Preprocessing"):
                    stages = get_preprocessing_stages(crop_bgr)
                    scols = st.columns(len(stages))
                    for i, (name, img) in enumerate(stages.items()):
                        with scols[i]:
                            st.caption(f"Stage {i+1}\n{name}")
                            st.image(img if len(img.shape) == 2 else bgr_to_rgb(img),
                                     width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 – CAMERA SCAN (MOBILE OPTIMISED)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Camera Scan":
    st.markdown('<div class="section-title">📹 Live Camera Scanner</div>', unsafe_allow_html=True)

    # Mobile instructions card
    st.markdown("""
    <div class="result-card">
    <b>📱 How to use on mobile:</b>
    <ol style="margin:8px 0 0; padding-left:20px; line-height:1.9; color:#CBD5E1;">
      <li>Open this page at <code style="background:#1E3A5F;padding:2px 6px;border-radius:4px;">{url}</code> on your phone</li>
      <li>Tap the <b>Camera</b> button below</li>
      <li>Allow camera permission → the <b>back camera</b> opens by default</li>
      <li>Point steadily at the number plate in good light</li>
      <li>Tap <b>Capture Photo</b> — result appears instantly</li>
    </ol>
    <br><small style="color:#64748B;">💡 Tip: Keep 30–50 cm distance, ensure plate is fully visible, avoid glare.</small>
    </div>
    """.format(url=f"http://{get_local_ip()}:8502"), unsafe_allow_html=True)

    auto_scan = st.toggle("⚡ Auto-Scan after capture", value=True,
                          help="Automatically runs ANPR as soon as a photo is taken")

    st.markdown('<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)
    camera_img = st.camera_input("📸 Capture number plate")

    if camera_img:
        raw = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
        bgr_frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        scan_now = auto_scan or st.button("🔍 Scan Plate", type="primary", width="stretch")

        if scan_now and bgr_frame is not None:
            h_f, w_f = bgr_frame.shape[:2]
            is_wide   = (w_f / float(h_f)) >= 1.8

            with st.spinner("Scanning plate…"):
                det_result, crop_bgr, plate_text, det_conf, ocr_conf, fmt_info = \
                    run_full_pipeline(bgr_frame, conf_threshold, is_wide, source="Camera")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.caption("Captured Frame")
                st.image(bgr_to_rgb(bgr_frame), width="stretch")
            with col_r2:
                st.caption("YOLO Overlay")
                if det_result.get("annotated_image") is not None:
                    st.image(bgr_to_rgb(det_result["annotated_image"]), width="stretch")

            if plate_text:
                st.markdown("---")
                st.markdown(
                    f'<div style="text-align:center;padding:16px;">'
                    f'<div class="plate-badge">{plate_text}</div>'
                    f'<br><br>'
                    + conf_pill("YOLO", det_conf)
                    + conf_pill("OCR",  ocr_conf)
                    + f"<span class='info-pill pill-blue'>{fmt_info.get('format_type','—')}</span>"
                    + '</div>',
                    unsafe_allow_html=True
                )
                st.success(f"✅ Plate **{plate_text}** recognised and logged!")
            elif det_result["detections"]:
                st.warning("Plate region found but text could not be read clearly. Move closer or improve lighting.")
            else:
                st.warning("No plate detected. Ensure plate fills most of the frame and is well-lit.")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 – HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "History":
    st.markdown('<div class="section-title">Detection History</div>', unsafe_allow_html=True)

    df = load_history()
    if df.empty:
        st.info("No detection history yet. Recognise some plates first!")
    else:
        search = st.text_input("🔍 Search plate number:", "").upper().strip()
        filtered = df[df["plate_number"].str.contains(search, na=False)] if search else df

        st.dataframe(filtered.iloc[::-1], width="stretch")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv_bytes, "anpr_history.csv",
                               "text/csv", width="stretch")
        with col_d2:
            if st.button("🗑️ Clear History", type="secondary", width="stretch"):
                if clear_history():
                    st.success("History cleared!")
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 – ABOUT & VIVA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About & Viva":
    st.markdown('<div class="section-title">Project Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
```
[Mobile Back Camera / PC Webcam / Uploaded Photo]
            │
            ▼
  [YOLOv8 Number Plate Detector]   ◄─ models/best.pt
            │  bounding box (x,y,w,h) + confidence
            ▼
  [OpenCV Preprocessing Pipeline]
   Resize → Grayscale → Bilateral Filter
   → CLAHE → Otsu Binary Threshold
            │  clean binary plate image
            ▼
  [EasyOCR Character Recognition]
            │  plate text + char confidences
            ▼
  [Text Cleaning]  mh 12-ab 1234! → MH12AB1234
            │
            ▼
  [CSV History Logger]  detections.csv
```
    """)

    st.markdown('<div class="section-title">Viva Q&A</div>', unsafe_allow_html=True)

    qa = [
        ("What does YOLO do?",
         "YOLO (You Only Look Once) is an object detection model. It scans the entire vehicle image in one forward pass and outputs the bounding box coordinates (x, y, width, height) of the number plate along with a confidence score."),
        ("What does OpenCV do?",
         "OpenCV performs 5-stage image preprocessing on the cropped plate: Resize (for character clarity), Grayscale, Bilateral Filter (edge-preserving noise removal), CLAHE (contrast under shadows/glare), and Otsu Binarisation (black/white characters)."),
        ("What does EasyOCR do?",
         "EasyOCR is a deep-learning OCR engine (ResNet + LSTM). It takes the preprocessed binary image and recognises each character, returning the text string and per-character confidence scores."),
        ("Why are YOLO and OCR two separate stages?",
         "YOLO handles spatial localisation (where is the plate?). OCR handles character recognition (what does it say?). Running OCR on a full car image would introduce background noise (headlights, brand logos, road text) causing false positives. YOLO crops only the plate region, making OCR dramatically more accurate."),
        ("What preprocessing steps are applied and why?",
         "1. Resize – upscale small crops so characters are not pixelated. 2. Grayscale – reduce 3-channel colour to 1-channel luminance. 3. Bilateral Filter – smooth noise without blurring character edges. 4. CLAHE – equalise brightness across the plate (handles shadow/glare). 5. Otsu Threshold – automatically convert grey pixels to black/white binary, maximising contrast for OCR."),
        ("How does the mobile camera work?",
         "The app uses Streamlit's st.camera_input() which calls the browser's native getUserMedia API. On Android Chrome and iOS Safari, this opens the device camera—back camera by default—allowing the user to capture a photo of the number plate directly from their phone."),
        ("What is the accuracy of this system?",
         "Accuracy depends on plate quality, lighting, angle, and training data. This is an academic prototype and does not claim production-level accuracy. Real-world accuracy requires a large annotated dataset and proper training."),
    ]

    for q, a in qa:
        with st.expander(f"❓ {q}"):
            st.markdown(f"> {a}")