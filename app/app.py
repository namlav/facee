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
from app.inference import initialize_model, draw_emotion_on_frame, detect_faces, put_unicode_text
from app.webcam import get_webcam_processor
from matplotlib import pyplot as plt

from app.export_utils import export_to_csv, export_to_image, export_to_pdf

st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

EMOJI_MAP = {
    "Giận dữ": "😠",
    "Ghê tởm": "🤢",
    "Sợ hãi": "😨",
    "Vui vẻ": "😊",
    "Buồn bã": "😢",
    "Kinh ngạc": "😮",
    "Bình thường": "😐",
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

.dashboard-container {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    margin-bottom: 1.5rem;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #00cc66, #00ff88) !important;
    border: none !important;
    border-radius: 50px !important;
    color: #0a2e1a !important;
    font-weight: 700 !important;
    font-family: 'Poppins', sans-serif !important;
    padding: 0.6rem 2.5rem !important;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    box-shadow: 0 4px 20px rgba(0, 255, 136, 0.25) !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-3px) scale(1.06) !important;
    box-shadow: 0 12px 40px rgba(0, 255, 136, 0.5) !important;
}

[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span,
[data-testid="stMetricValue"] p,
[data-testid="stMetricValue"] span {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word !important;
    line-height: 1.3 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    margin-bottom: 0 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    word-break: break-word !important;
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


def show_dashboard(results, total_label="Tổng số khuôn mặt"):
    emotion_counts = {}
    for r in results:
        em = r['emotion']
        emotion_counts[em] = emotion_counts.get(em, 0) + 1

    total_faces = len(results)

    if total_faces > 0:
        most_common = max(emotion_counts, key=emotion_counts.get)
        most_common_count = emotion_counts[most_common]
        most_common_pct = (most_common_count / total_faces) * 100
    else:
        most_common = "N/A"
        most_common_pct = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(total_label, total_faces)
    with col2:
        st.metric("Biểu cảm phổ biến", most_common)
    with col3:
        st.metric("Tỉ lệ phổ biến", f"{most_common_pct:.1f}%")
    with col4:
        st.metric("Loại biểu cảm", f"{len(emotion_counts)}/7")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Phân bố biểu cảm")
        df_counts = pd.DataFrame({
            'Biểu cảm': list(emotion_counts.keys()),
            'Số lượng': list(emotion_counts.values())
        }).set_index('Biểu cảm')
        st.bar_chart(df_counts, color='#00ff88')

        with col_chart2:
            st.subheader("Tỉ lệ phần trăm")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor('none')
            colors = plt.cm.Set2(np.linspace(0, 1, len(emotion_counts)))
            wedges, texts, autotexts = ax.pie(
                list(emotion_counts.values()),
                labels=None,
                autopct='%1.1f%%',
                colors=colors,
                startangle=140,
                textprops={'color': 'white', 'fontsize': 9},
                pctdistance=0.75
            )
            for t in autotexts:
                t.set_color('white')
            ax.legend(
                wedges, list(emotion_counts.keys()),
                loc='center left', bbox_to_anchor=(1, 0.5),
                fontsize=8, frameon=False,
                labelcolor='white'
            )
            st.pyplot(fig)
            plt.close(fig)


def show_export_buttons(results, annotated_image=None):
    st.subheader("📥 Tải xuống kết quả")

    if annotated_image is not None:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)

    with col1:
        csv_data, csv_name = export_to_csv(results)
        st.download_button(
            label="📥 Xuất CSV",
            data=csv_data,
            file_name=csv_name,
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        pdf_data, pdf_name = export_to_pdf(results, len(results))
        st.download_button(
            label="📥 Xuất PDF",
            data=pdf_data,
            file_name=pdf_name,
            mime="application/pdf",
            use_container_width=True,
        )

    if annotated_image is not None:
        with col3:
            img_data, img_name = export_to_image(annotated_image, results)
            st.download_button(
                label="📥 Xuất IMAGE",
                data=img_data,
                file_name=img_name,
                mime="image/png",
                use_container_width=True,
            )


@glass_container
def home_page_content():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Giới thiệu về đề tài")
        st.markdown(
            "Hệ thống này sử dụng mạng nơ-ron tích chập (CNN) học sâu được huấn luyện trên tập dữ liệu "
            "**FER2013** để nhận diện **7 biểu cảm cơ bản** từ các biểu hiện khuôn mặt trong thời gian thực."
        )
        st.markdown("---")
        st.subheader("Các biểu cảm được phát hiện")
        cols = st.columns(7)
        for i, (emotion, emoji) in enumerate(EMOJI_MAP.items()):
            with cols[i]:
                st.markdown(
                    f"<div style='text-align:center'><span style='font-size:2.5rem'>{emoji}</span><br><span style='font-size:0.85rem;color:rgba(255,255,255,0.7)'>{emotion}</span></div>",
                    unsafe_allow_html=True,
                )
    with col2:
        st.subheader("Hướng dẫn nhanh")
        st.markdown("""
        1. Điều hướng bằng cách sử dụng **sidebar**
        2. **Upload Prediction** – tải lên một ảnh
        3. **Webcam** – phát hiện thời gian thực qua camera
        4. **About** – tìm hiểu thêm về đề tài
        """)
    st.markdown("---")
    st.subheader("Quy trình làm việc của hệ thống")
    st.code(
        "Ảnh đầu vào → Nhận diện khuôn mặt → Chuyển đổi sang ảnh xám → Thay đổi kích thước (48×48) → "
        "Chuẩn hóa → Mô hình CNN → Softmax → Phân loại biểu cảm",
        language="text",
    )


def home_page():
    st.title("Nhận Diện Biểu Cảm Khuôn Mặt")
    home_page_content()


@glass_container
def upload_page_content():
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_rgb = np.array(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh đã tải lên")
            st.image(image, use_container_width=True)
        with col2:
            if st.button("Nhận diện biểu cảm", type="primary"):
                with st.spinner("Đang phân tích..."):
                    model = initialize_model()
                    faces = detect_faces(image_bgr)
                    annotated = image_bgr.copy()
                    all_results = []

                    if len(faces) > 0:
                        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                        for (x, y, w, h) in faces:
                            face_roi = gray[y : y + h, x : x + w]
                            face_roi = cv2.resize(face_roi, (48, 48))
                            prediction = predict_image(model, face_roi, DEVICE)
                            prediction['face_box'] = (x, y, w, h)
                            all_results.append(prediction)
                            annotated = draw_emotion_on_frame(
                                annotated, prediction, (x, y, w, h)
                            )
                    else:
                        put_unicode_text(annotated, "Không phát hiện khuôn mặt trong ảnh", (10, 30), font_size=14, color=(0, 0, 255))

                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.session_state.upload_results = all_results
                    st.session_state.upload_annotated = annotated_rgb

        if st.session_state.get('upload_results') is not None:
            results = st.session_state.upload_results
            annotated_rgb = st.session_state.upload_annotated

            with col1:
                st.subheader("Kết quả nhận diện")
                st.image(annotated_rgb, use_container_width=True)

            with col2:
                if len(results) > 0:
                    show_dashboard(results)
                    st.markdown("---")
                    show_export_buttons(results, annotated_rgb)

            if len(results) > 0:
                st.markdown("---")
                st.subheader("Chi tiết kết quả phát hiện")
                detail_rows = []
                for i, r in enumerate(results):
                    box = r.get('face_box')
                    detail_rows.append({
                        "STT": i + 1,
                        "Biểu cảm": r['emotion'],
                        "Độ tin cậy": f"{r.get('confidence', 0):.1f}%",
                        "Vị trí": f"({box[0]}, {box[1]})" if box else "Toàn bộ ảnh",
                    })
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)
    else:
        st.session_state.upload_results = None
        st.session_state.upload_annotated = None


def upload_page():
    st.title("Tải lên và Dự đoán biểu cảm")
    st.markdown("Tải ảnh lên để nhận diện biểu cảm trên khuôn mặt.")
    upload_page_content()


@glass_container
def webcam_page_content():
    if "webcam_on" not in st.session_state:
        st.session_state.webcam_on = False

    col1, col2, col3 = st.columns(3)
    with col2:
        btn_label = "📷 Bật Webcam" if not st.session_state.webcam_on else "⏹ Tắt Webcam"
        if st.button(btn_label):
            st.session_state.webcam_on = not st.session_state.webcam_on
            if st.session_state.webcam_on:
                st.session_state.webcam_results = []
                st.session_state.webcam_emotion_counts = {}
                st.session_state.webcam_total_detections = 0
            else:
                st.rerun()

    if st.session_state.webcam_on:
        stframe = st.image([])
        fps_display = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Không thể truy cập webcam. Vui lòng kiểm tra quyền truy cập camera.")
            st.session_state.webcam_on = False
            st.rerun()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        model = initialize_model()
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        fps_counter = 0
        fps = 0
        last_time = time.time()
        frame_idx = 0
        process_every_n = 2
        last_faces = []
        last_results = []

        while st.session_state.webcam_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể chụp khung hình.")
                break

            frame = cv2.flip(frame, 1)
            frame_results = []
            display_frame = frame.copy()

            if frame_idx % process_every_n == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                faces = face_cascade.detectMultiScale(
                    small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    last_faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]
                    last_results = []
                    for (x, y, w, h) in last_faces:
                        face_roi = gray[y:y + h, x:x + w]
                        face_resized = cv2.resize(face_roi, (48, 48))
                        prediction = predict_image(model, face_resized, DEVICE)
                        last_results.append(prediction)
                else:
                    last_faces = []
                    last_results = []

            if len(last_faces) > 0:
                for i, (x, y, w, h) in enumerate(last_faces):
                    if i < len(last_results):
                        prediction = last_results[i]
                        display_frame = draw_emotion_on_frame(display_frame, prediction, (x, y, w, h))
                        frame_results.append(prediction)
            else:
                put_unicode_text(display_frame, "Không phát hiện khuôn mặt", (10, 30), font_size=14, color=(0, 0, 255))

            if frame_idx % 5 == 0:
                for pred in frame_results:
                    st.session_state.webcam_results.append(pred)
                    em = pred['emotion']
                    st.session_state.webcam_emotion_counts[em] = \
                        st.session_state.webcam_emotion_counts.get(em, 0) + 1
                    st.session_state.webcam_total_detections += 1
            frame_idx += 1

            annotated_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
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
        st.rerun()

    if st.session_state.get('webcam_total_detections', 0) > 0:
        st.markdown("---")
        st.title("📊 Thống kê phiên nhận diện")
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)

        emotion_counts = st.session_state.webcam_emotion_counts
        total = st.session_state.webcam_total_detections

        if emotion_counts:
            most_common = max(emotion_counts, key=emotion_counts.get)
            most_common_count = emotion_counts[most_common]
        else:
            most_common = "N/A"
            most_common_count = 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng lượt phát hiện", total)
        with col2:
            st.metric("Biểu cảm phổ biến", most_common)
        with col3:
            pct = (most_common_count / total * 100) if total > 0 else 0
            st.metric("Tỉ lệ phổ biến", f"{pct:.1f}%")
        with col4:
            st.metric("Loại biểu cảm", f"{len(emotion_counts)}/7")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Phân bố biểu cảm")
            df_counts = pd.DataFrame({
                'Biểu cảm': list(emotion_counts.keys()),
                'Số lượng': list(emotion_counts.values())
            }).set_index('Biểu cảm')
            st.bar_chart(df_counts, color='#00ff88')

        with col_chart2:
            st.subheader("Tỉ lệ phần trăm")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor('none')
            colors = plt.cm.Set2(np.linspace(0, 1, len(emotion_counts)))
            wedges, texts, autotexts = ax.pie(
                list(emotion_counts.values()),
                labels=None,
                autopct='%1.1f%%',
                colors=colors,
                startangle=140,
                textprops={'color': 'white', 'fontsize': 9},
                pctdistance=0.75
            )
            for t in autotexts:
                t.set_color('white')
            ax.legend(
                wedges, list(emotion_counts.keys()),
                loc='center left', bbox_to_anchor=(1, 0.5),
                fontsize=8, frameon=False,
                labelcolor='white'
            )
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

        webcam_results = st.session_state.webcam_results
        if webcam_results:
            st.subheader("📥 Tải xuống báo cáo")
            col_csv, col_pdf = st.columns(2)
            with col_csv:
                csv_data, csv_name = export_to_csv(webcam_results)
                st.download_button(
                    label="📥 Xuất CSV",
                    data=csv_data,
                    file_name=csv_name,
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_pdf:
                pdf_data, pdf_name = export_to_pdf(webcam_results, total)
                st.download_button(
                    label="📥 Xuất PDF",
                    data=pdf_data,
                    file_name=pdf_name,
                    mime="application/pdf",
                    use_container_width=True,
                )


def webcam_page():
    st.title("Nhận diện biểu cảm từ webcam trong thời gian thực")
    st.markdown("Hướng khuôn mặt của bạn vào camera để xem dự đoán biểu cảm.")
    webcam_page_content()


@glass_container
def about_page_content():
    st.subheader("Contributor")
    st.markdown("Built with ❤️ by **NHÓM 10**")
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
        "**EmotionCNN** – một mạng nơ-ron tích chập 4 khối:\n\n"
        "- 4 khối tích chập với BatchNorm, ReLU, MaxPool, Dropout\n"
        "- Tăng gấp đôi số lượng bộ lọc trên mỗi khối: 64 → 128 → 256 → 512\n"
        "- Bộ phân loại được kết nối đầy đủ (512×3×3 → 512 → 7)\n"
        "- ~7 triệu tham số có thể huấn luyện"
    )
    st.subheader("Dataset")
    st.markdown(
        "**FER2013** (Facial Expression Recognition 2013):\n\n"
        "- 35.887 hình ảnh khuôn mặt đen trắng 48×48\n"
        "- 7 loại cảm xúc: Giận dữ, Ghê tởm, Sợ hãi, Vui vẻ, Buồn bã, Kinh ngạc, Bình thường\n"
        "- Các giai đoạn: Huấn luyện / Kiểm thử công khai / Kiểm thử riêng tư\n"
        "- Bộ dữ liệu cuộc thi Kaggle năm 2013 về nhận diện cảm xúc"
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

    if 'upload_results' not in st.session_state:
        st.session_state.upload_results = None
    if 'upload_annotated' not in st.session_state:
        st.session_state.upload_annotated = None
    if 'webcam_results' not in st.session_state:
        st.session_state.webcam_results = []
    if 'webcam_emotion_counts' not in st.session_state:
        st.session_state.webcam_emotion_counts = {}
    if 'webcam_total_detections' not in st.session_state:
        st.session_state.webcam_total_detections = 0

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
        "🌸 NHẬN DIỆN BIỂU CẢM</div>",
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
