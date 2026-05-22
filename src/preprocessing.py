import cv2
import numpy as np
import pandas as pd
import torch

from .config import EMOTIONS, IMG_SIZE


def preprocess_image(image):
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 1:
            image = image[:, :, 0]
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def preprocess_image_tensor(image):
    image = preprocess_image(image)
    tensor = torch.from_numpy(image).float().unsqueeze(0)
    return tensor


def load_and_preprocess(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return preprocess_image(image)


def apply_histogram_equalization(image, method='clahe'):
    if image.dtype != np.uint8:
        image_uint8 = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        image_uint8 = image
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(image_uint8)
    elif method == 'standard':
        result = cv2.equalizeHist(image_uint8)
    else:
        raise ValueError(f"Unknown method: {method}")
    return result.astype(np.float32) / 255.0


def normalize_image(image, mean=0.5, std=0.5):
    return (image - mean) / std


def prepare_fer2013_csv(raw_csv_path, output_path):
    df = pd.read_csv(raw_csv_path)
    required_cols = {'emotion', 'pixels'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    expected_pixels = IMG_SIZE * IMG_SIZE
    valid_rows = []
    for idx, row in df.iterrows():
        try:
            pixels = str(row['pixels']).split()
            if len(pixels) == expected_pixels:
                valid_rows.append(row)
        except Exception:
            continue
    cleaned = pd.DataFrame(valid_rows)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def explore_dataset(csv_path):
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    emotion_counts = df['emotion'].value_counts().sort_index()
    class_distribution = {EMOTIONS.get(int(k), str(k)): int(v) for k, v in emotion_counts.items()}
    sample_images = []
    for emotion in sorted(df['emotion'].unique()):
        sample = df[df['emotion'] == emotion].iloc[0]
        pixels = np.array(sample['pixels'].split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
        sample_images.append(pixels)
    example_pixels = np.array(df.iloc[0]['pixels'].split(), dtype=np.uint8)
    return {
        'total_samples': total_samples,
        'class_distribution': class_distribution,
        'sample_images': sample_images,
        'example_pixels_shape': example_pixels.shape,
    }
