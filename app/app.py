import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEVICE, EMOTIONS
from src.predict import predict_image
from app.inference import initialize_model, draw_emotion_on_frame, detect_faces
from app.webcam import get_webcam_processor

st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

EMOJI_MAP = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sad": "😢",
    "Surprise": "😮",
    "Neutral": "😐",
}


# background: linear-gradient(135deg, #0a2e1a 0%, #1a5c3a 50%, #0d4a2a 100%); // Green
# background: linear-gradient(135deg, #3d2b1f 0%, #8b5a2b 50%, #5c3a21 100%); // Brown
# background: linear-gradient(135deg, #7a3a00 0%, #d97706 50%, #b45309 100%); // Orange
def inject_custom_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=M+PLUS+Rounded+1c:wght@400;700;900&family=Poppins:wght@400;600;700&display=swap');

.stApp {
    
    background: linear-gradient(135deg, #003366 0%, #0077be 50%, #005b96 100%); // Blue
    font-family: 'Poppins', 'M PLUS Rounded 1c', sans-serif;
}

.floating-face {
    position: fixed;
    pointer-events: none;
    z-index: 0;
    animation: floatFace var(--dur) ease-in-out infinite;
    animation-delay: var(--delay);
    opacity: 0.08;
    font-size: var(--size);
    left: var(--left);
    top: var(--top);
}

@keyframes floatFace {
    0%, 100% { transform: translateY(0) rotate(0deg) scale(1); }
    25% { transform: translateY(-30px) rotate(12deg) scale(1.1); }
    50% { transform: translateY(-10px) rotate(-6deg) scale(0.95); }
    75% { transform: translateY(-40px) rotate(8deg) scale(1.05); }
}

.glass-panel {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    margin-bottom: 1.5rem;
    transition: box-shadow 0.4s ease;
}

.glass-panel:hover {
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

@keyframes pageEnter {
    0% { opacity: 0; transform: scale(0.92) translateY(24px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

.page-content {
    animation: pageEnter 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
}

h1, h2, h3 {
    font-family: 'Fredoka One', 'M PLUS Rounded 1c', sans-serif !important;
    color: #00ff88 !important;
    text-shadow: 0 0 24px rgba(0, 255, 136, 0.25);
    letter-spacing: 1px;
}

h1 { font-size: 2.6rem !important; }
h2 { font-size: 1.7rem !important; }
h3 { font-size: 1.25rem !important; }

p, li, .stMarkdown, .stText {
    color: rgba(255, 255, 255, 0.82) !important;
    font-family: 'Poppins', sans-serif !important;
}

strong { color: #00ff88 !important; }

.stButton > button {
    background: linear-gradient(135deg, #00cc66, #00ff88) !important;
    border: none !important;
    border-radius: 50px !important;
    color: #0a2e1a !important;
    font-weight: 700 !important;
    font-family: 'Poppins', sans-serif !important;
    padding: 0.6rem 2.5rem !important;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    box-shadow: 0 4px 20px rgba(0, 255, 136, 0.25) !important;
    letter-spacing: 0.5px;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: -60%;
    left: -60%;
    width: 220%;
    height: 220%;
    background: radial-gradient(circle, rgba(255,255,255,0.35) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.5s ease;
    pointer-events: none;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.06) !important;
    box-shadow: 0 12px 40px rgba(0, 255, 136, 0.5) !important;
}

.stButton > button:hover::before {
    opacity: 1;
}

.stButton > button:active {
    transform: translateY(0) scale(0.97) !important;
}

section[data-testid="stSidebar"] {
    background: rgba(8, 24, 14, 0.75) !important;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

section[data-testid="stSidebar"] .stRadio > label {
    color: rgba(255, 255, 255, 0.6) !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.95rem !important;
}

section[data-testid="stSidebar"] .stRadio > div {
    background: transparent !important;
    gap: 0.2rem !important;
}

section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 14px !important;
    padding: 0.65rem 1.2rem !important;
    margin: 0.15rem 0 !important;
    border: 1px solid transparent !important;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    cursor: pointer;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0, 255, 136, 0.08) !important;
    border-color: rgba(0, 255, 136, 0.25) !important;
    transform: translateX(6px) scale(1.02);
}

div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(0, 204, 102, 0.2), rgba(0, 255, 136, 0.08)) !important;
    border-color: rgba(0, 255, 136, 0.4) !important;
    box-shadow: 0 0 24px rgba(0, 255, 136, 0.12) !important;
}

hr {
    border: none !important;
    height: 2px !important;
    background: linear-gradient(90deg, transparent, rgba(0,255,136,0.5), transparent) !important;
    margin: 1.5rem 0 !important;
}

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

[data-testid="stMetric"] label {
    color: rgba(255, 255, 255, 0.7) !important;
    font-family: 'Poppins', sans-serif !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #00ff88 !important;
    font-family: 'Fredoka One', sans-serif !important;
    font-size: 2rem !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #00cc66, #00ff88) !important;
    border-radius: 10px !important;
}

.stProgress > div {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 2px dashed rgba(0, 255, 136, 0.25) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: all 0.35s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 255, 136, 0.6) !important;
    background: rgba(255, 255, 255, 0.08) !important;
}

.stSpinner > div {
    border-color: #00ff88 !important;
    border-top-color: transparent !important;
}

.stCodeBlock {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(0, 255, 136, 0.12) !important;
    border-radius: 12px !important;
}

.stCodeBlock code {
    color: #00ff88 !important;
}

.stImage img {
    border-radius: 16px !important;
    border: 2px solid rgba(0, 255, 136, 0.12) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

.stechart {
    background: rgba(0, 0, 0, 0.2) !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
}

::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); }
::-webkit-scrollbar-thumb { background: rgba(0,255,136,0.25); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,255,136,0.5); }

.stAlert {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(10px) !important;
}

.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: white !important;
}
</style>
<div class="floating-face" style="--left:3%;--top:8%;--size:32px;--dur:13s;--delay:0s">😊</div>
<div class="floating-face" style="--left:92%;--top:15%;--size:28px;--dur:15s;--delay:2s">😢</div>
<div class="floating-face" style="--left:15%;--top:75%;--size:30px;--dur:11s;--delay:4s">😠</div>
<div class="floating-face" style="--left:78%;--top:82%;--size:26px;--dur:14s;--delay:1s">😮</div>
<div class="floating-face" style="--left:50%;--top:5%;--size:34px;--dur:12s;--delay:3s">😨</div>
<div class="floating-face" style="--left:8%;--top:50%;--size:24px;--dur:16s;--delay:5s">🤢</div>
<div class="floating-face" style="--left:85%;--top:45%;--size:30px;--dur:13s;--delay:6s">😐</div>
<div class="floating-face" style="--left:35%;--top:88%;--size:22px;--dur:14s;--delay:2.5s">😊</div>
<div class="floating-face" style="--left:65%;--top:12%;--size:26px;--dur:11s;--delay:4.5s">😢</div>
<div class="floating-face" style="--left:22%;--top:30%;--size:20px;--dur:15s;--delay:7s">😮</div>
<div class="floating-face" style="--left:72%;--top:60%;--size:28px;--dur:12s;--delay:1.5s">😠</div>
<div class="floating-face" style="--left:45%;--top:40%;--size:18px;--dur:16s;--delay:5.5s">😨</div>
""",
        unsafe_allow_html=True,
    )


def glass_container(content_func):
    def wrapper(*args, **kwargs):
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        content_func(*args, **kwargs)
        st.markdown("</div>", unsafe_allow_html=True)

    return wrapper


@glass_container
def home_page_content():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("About the Project")
        st.markdown(
            "This system uses a deep learning Convolutional Neural Network (CNN) trained on the "
            "**FER2013** dataset to recognize **7 basic emotions** from facial expressions in real time."
        )
        st.markdown("---")
        st.subheader("Detected Emotions")
        cols = st.columns(7)
        for i, (emotion, emoji) in enumerate(EMOJI_MAP.items()):
            with cols[i]:
                st.markdown(
                    f"<div style='text-align:center'><span style='font-size:2.5rem'>{emoji}</span><br><span style='font-size:0.85rem;color:rgba(255,255,255,0.7)'>{emotion}</span></div>",
                    unsafe_allow_html=True,
                )
    with col2:
        st.subheader("Quick Instructions")
        st.markdown("""
        1. Navigate using the **sidebar**
        2. **Upload Prediction** – upload a photo
        3. **Webcam** – real-time detection via camera
        4. **About** – learn more about the project
        """)
    st.markdown("---")
    st.subheader("System Workflow")
    st.code(
        "Input Image → Face Detection → Grayscale Conversion → Resize (48×48) → "
        "Normalization → CNN Model → Softmax → Emotion Classification",
        language="text",
    )


def home_page():
    st.title("Facial Emotion Recognition")
    home_page_content()


@glass_container
def upload_page_content():
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_rgb = np.array(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Prediction Results")
            if st.button("Predict Emotion", type="primary"):
                with st.spinner("Analyzing..."):
                    model = initialize_model()
                    faces = detect_faces(image_bgr)
                    if len(faces) > 0:
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                        face_roi = gray[y : y + h, x : x + w]
                        face_roi = cv2.resize(face_roi, (48, 48))
                        prediction = predict_image(model, face_roi, DEVICE)
                        annotated = draw_emotion_on_frame(
                            image_bgr.copy(), prediction, (x, y, w, h)
                        )
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        col1.image(annotated_rgb, use_container_width=True)
                    else:
                        prediction = predict_image(model, image, DEVICE)
                    emotion = prediction["emotion"]
                    confidence = prediction["confidence"]
                    emoji = EMOJI_MAP.get(emotion, "")
                    st.markdown(
                        f"<div style='text-align:center;font-size:2rem'>{emoji} <strong>{emotion}</strong></div>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.progress(int(confidence))
                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame(
                        {
                            "Emotion": list(prediction["probabilities"].keys()),
                            "Probability (%)": list(
                                prediction["probabilities"].values()
                            ),
                        }
                    ).set_index("Emotion")
                    st.bar_chart(prob_df)


def upload_page():
    st.title("Upload & Predict")
    st.markdown("Upload an image to detect facial emotion.")
    upload_page_content()


@glass_container
def webcam_page_content():
    processor = get_webcam_processor()
    if "webcam_on" not in st.session_state:
        st.session_state.webcam_on = False
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button(
            "Start Webcam" if not st.session_state.webcam_on else "Stop Webcam"
        ):
            st.session_state.webcam_on = not st.session_state.webcam_on
            if not st.session_state.webcam_on:
                st.rerun()
    if st.session_state.webcam_on:
        stframe = st.image([])
        fps_display = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam. Please check your camera permissions.")
            st.session_state.webcam_on = False
            st.rerun()
        fps_counter = 0
        fps = 0
        last_time = time.time()
        while st.session_state.webcam_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break
            annotated = processor.process_frame(frame)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, channels="RGB", use_container_width=True)
            fps_counter += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_time = now
                fps_display.metric("FPS", fps)
        cap.release()
        st.session_state.webcam_on = False


def webcam_page():
    st.title("Real-time Webcam Emotion Detection")
    st.markdown("Point your face at the camera to see real-time emotion predictions.")
    webcam_page_content()


@glass_container
def about_page_content():
    st.subheader("Contributor")
    st.markdown("Built with ❤️ by **Nam Lav**")
    st.subheader("Technologies Used")
    techs = {
        "🐍": "Python",
        "🔥": "PyTorch",
        "👁️": "OpenCV",
        "📊": "Streamlit",
        "🔢": "NumPy",
        "🐼": "Pandas",
        "📈": "Matplotlib / Seaborn",
        "🧠": "Scikit-learn",
    }
    cols = st.columns(4)
    for i, (emoji, tech) in enumerate(techs.items()):
        with cols[i % 4]:
            st.markdown(f"{emoji} **{tech}**")
    st.subheader("Model Architecture")
    st.markdown(
        "**EmotionCNN** – a 4-block convolutional neural network:\n\n"
        "- 4 convolutional blocks with BatchNorm, ReLU, MaxPool, Dropout\n"
        "- Doubling filter count per block: 64 → 128 → 256 → 512\n"
        "- Fully connected classifier (512×3×3 → 512 → 7)\n"
        "- ~2.5M trainable parameters"
    )
    st.subheader("Dataset")
    st.markdown(
        "**FER2013** (Facial Expression Recognition 2013):\n\n"
        "- 35,887 grayscale 48×48 face images\n"
        "- 7 emotion categories\n"
        "- Training / PublicTest / PrivateTest split\n"
        "- Kaggle competition dataset"
    )
    st.subheader("GitHub")
    st.markdown("[View on GitHub](https://github.com/namlav/facee)")


def about_page():
    st.title("About")
    about_page_content()


PAGES = {
    "Home": home_page,
    "Upload Prediction": upload_page,
    "Webcam": webcam_page,
    "About": about_page,
}


def main():
    inject_custom_css()
    st.sidebar.markdown(
        "<div style='font-family:Fredoka One;font-size:1.6rem;color:#00ff88;"
        "text-shadow:0 0 20px rgba(0,255,136,0.3);padding:1rem 0;text-align:center'>✦ FaceE ✦</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    selection = st.sidebar.radio(
        "Navigation", list(PAGES.keys()), label_visibility="collapsed"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align:center;color:rgba(255,255,255,0.4);font-size:0.8rem;padding:1rem 0'>"
        "🌸 Emotional Detection</div>",
        unsafe_allow_html=True,
    )
    page_class = f"page-content-{selection.replace(' ', '-').lower()}"
    st.markdown(
        f"<div class='page-content {page_class}' key='page-{selection}'>",
        unsafe_allow_html=True,
    )
    PAGES[selection]()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
