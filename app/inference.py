from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import DEVICE, IMG_SIZE
from src.model import EmotionCNN
from src.predict import load_model, predict_image


def _get_font(size=16):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def put_unicode_text(frame, text, position, font_size=16, color=(0, 255, 0), bg_color=None):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = position
    if bg_color:
        draw.rectangle([(x, y - th), (x + tw, y + 4)], fill=bg_color)
    draw.text((x, y - th), text, font=font, fill=color)
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

_label_zones = {}

_MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'

_model = None
_device = None


def initialize_model(model_path=None):
    global _model, _device
    if model_path is None:
        model_path = str(_MODEL_DIR / 'best_model.pth')
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
    global _label_zones
    if face_box is None:
        return frame

    x, y, w, h = face_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = f"{prediction['emotion']}: {prediction['confidence']:.1f}%"

    frame_id = id(frame)
    if frame_id not in _label_zones:
        _label_zones[frame_id] = []
    if len(_label_zones) > 200:
        _label_zones.clear()
        _label_zones[frame_id] = []

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_sizes = [14, 13, 12, 11, 10]
    chosen_size = 12
    for size in font_sizes:
        font = _get_font(size)
        bb = draw.textbbox((0, 0), label, font=font)
        if bb[2] - bb[0] <= w - 8 and bb[3] - bb[1] <= h * 0.5:
            chosen_size = size
            break
    else:
        chosen_size = 10

    font = _get_font(chosen_size)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    candidates = [
        y - 8 - th,
        y + h + 8,
        y + 4,
        y + h - th - 4,
    ]
    final_y = candidates[0]
    for py in candidates:
        zone = (py - 2, py + th + 2)
        if zone[0] < 0 or zone[1] > frame.shape[0]:
            continue
        if any(max(z[0], zone[0]) < min(z[1], zone[1]) for z in _label_zones[frame_id]):
            continue
        if (py == candidates[2] or py == candidates[3]) and (py < y or py + th > y + h):
            continue
        final_y = py
        break

    _label_zones[frame_id].append((final_y - 2, final_y + th + 2))
    draw.rectangle([(x, final_y - 2), (x + tw + 4, final_y + th + 2)], fill=(0, 255, 0))
    draw.text((x + 2, final_y), label, font=font, fill=(0, 0, 0))
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
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
