import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.config import EMOTIONS, EMOTION_LIST, NUM_CLASSES, DEVICE, MODELS_DIR
from src.model import EmotionCNN
from src.preprocessing import preprocess_image


def get_emotion_label(class_idx):
    return EMOTIONS[int(class_idx)]


def _extract_state_dict(checkpoint):
    """Accept either a raw model state_dict or a training checkpoint dict."""
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint


def load_model(model_path, device, model_class=EmotionCNN):
    model = model_class(num_classes=NUM_CLASSES).to(device)
    state_dict = _extract_state_dict(torch.load(model_path, map_location=device))
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model, image, device):
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Unable to load image: {image}')
    elif isinstance(image, Image.Image):
        image = np.array(image.convert('L'), dtype=np.uint8)

    processed = preprocess_image(image)
    tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0).to(device)

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        outputs = model(tensor)
        probs = softmax(outputs)

    probs_np = probs.cpu().numpy()[0]
    class_idx = int(np.argmax(probs_np))
    confidence = float(probs_np[class_idx])
    emotion = get_emotion_label(class_idx)
    probabilities = {get_emotion_label(i): float(p) * 100 for i, p in enumerate(probs_np)}

    return {
        'emotion': emotion,
        'confidence': confidence * 100,
        'probabilities': probabilities,
        'class_idx': class_idx,
    }


def predict_image_batch(model, image_paths, device):
    results = []
    for path in image_paths:
        pred = predict_image(model, path, device)
        pred['image_path'] = str(path)
        results.append(pred)
    return results


def _get_font(size=18):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_prediction(image, prediction):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        image = np.array(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    label = f"{prediction['emotion']}: {prediction['confidence']:.1f}%"
    font = _get_font(18)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([(10, 10), (10 + tw + 8, 10 + th + 12)], fill=(0, 255, 0))
    draw.text((14, 14), label, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description='Predict emotion from image')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--model_path', type=str, default=str(MODELS_DIR / 'best_model.pth'))
    parser.add_argument('--device', type=str, default=str(DEVICE))
    parser.add_argument('--save', type=str, default=None, help='Save annotated image to path')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model_path, device)

    prediction = predict_image(model, args.image_path, device)

    print(f"Emotion: {prediction['emotion']}")
    print(f"Confidence: {prediction['confidence']:.2f}%")
    print('Probabilities:')
    for emotion, conf in prediction['probabilities'].items():
        print(f'  {emotion}: {conf:.2f}%')

    annotated = draw_prediction(args.image_path, prediction)

    if args.save:
        cv2.imwrite(args.save, annotated)
        print(f'Annotated image saved to {args.save}')
    else:
        cv2.imshow('Emotion Prediction', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
