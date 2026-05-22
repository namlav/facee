import cv2
import numpy as np
from PIL import Image

from src.config import DEVICE, IMG_SIZE
from src.model import EmotionCNN
from src.predict import load_model, predict_image

_model = None
_device = None


def initialize_model(model_path='models/best_model.pth'):
    global _model, _device
    _device = DEVICE
    _model = load_model(model_path, _device, EmotionCNN)
    return _model


def _ensure_initialized():
    global _model, _device
    if _model is None:
        initialize_model()


def predict_from_image_file(image_file):
    _ensure_initialized()
    image = Image.open(image_file).convert('L').resize((IMG_SIZE, IMG_SIZE))
    return predict_image(_model, image, _device)


def predict_from_cv2_frame(frame):
    _ensure_initialized()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return predict_image(_model, gray, _device)


def draw_emotion_on_frame(frame, prediction, face_box=None):
    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{prediction['emotion']}: {prediction['confidence']:.1f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def detect_faces(image):
    if isinstance(image, Image.Image):
        gray = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
    else:
        return np.array([])
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
