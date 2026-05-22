import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import DEVICE, EMOTIONS
from src.predict import predict_image
from app.inference import initialize_model, draw_emotion_on_frame, detect_faces
from app.webcam import get_webcam_processor

st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

EMOJI_MAP = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😊', 'Sad': '😢', 'Surprise': '😮', 'Neutral': '😐'
}


def home_page():
    st.title("Facial Emotion Recognition")
    st.markdown("---")
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
                st.markdown(f"## {emoji}")
                st.markdown(f"**{emotion}**")

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
        language='text'
    )


def upload_page():
    st.title("Upload & Predict")
    st.markdown("Upload an image to detect facial emotion.")

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_rgb = np.array(image.convert('RGB'))
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
                        face_roi = gray[y:y + h, x:x + w]
                        face_roi = cv2.resize(face_roi, (48, 48))
                        prediction = predict_image(model, face_roi, DEVICE)
                        annotated = draw_emotion_on_frame(image_bgr.copy(), prediction, (x, y, w, h))
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        col1.image(annotated_rgb, use_container_width=True)
                    else:
                        prediction = predict_image(model, image, DEVICE)

                    emotion = prediction['emotion']
                    confidence = prediction['confidence']
                    emoji = EMOJI_MAP.get(emotion, '')

                    st.markdown(f"## {emoji} {emotion}")
                    st.metric("Confidence", f"{confidence:.1f}%")

                    st.progress(int(confidence))

                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Emotion': list(prediction['probabilities'].keys()),
                        'Probability (%)': list(prediction['probabilities'].values())
                    }).set_index('Emotion')
                    st.bar_chart(prob_df)


def webcam_page():
    st.title("Real-time Webcam Emotion Detection")
    st.markdown("Point your face at the camera to see real-time emotion predictions.")

    processor = get_webcam_processor()

    if 'webcam_on' not in st.session_state:
        st.session_state.webcam_on = False

    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("Start Webcam" if not st.session_state.webcam_on else "Stop Webcam"):
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


def about_page():
    st.title("About")

    st.subheader("Team")
    st.markdown("Built with ❤️ by the FER Team")

    st.subheader("Technologies Used")
    techs = {
        '🐍': 'Python', '🔥': 'PyTorch', '👁️': 'OpenCV',
        '📊': 'Streamlit', '🔢': 'NumPy', '🐼': 'Pandas',
        '📈': 'Matplotlib / Seaborn', '🧠': 'Scikit-learn'
    }
    for emoji, tech in techs.items():
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
    st.markdown("[View on GitHub](https://github.com)")


PAGES = {
    "Home": home_page,
    "Upload Prediction": upload_page,
    "Webcam": webcam_page,
    "About": about_page,
}


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Facial Emotion Recognition")
    st.sidebar.markdown("CNN-based real-time emotion detection")
    PAGES[selection]()


if __name__ == '__main__':
    main()
