from pathlib import Path

import torch

EMOTIONS = {
    0: "Giận dữ",
    1: "Ghê tởm",
    2: "Sợ hãi",
    3: "Vui vẻ",
    4: "Buồn bã",
    5: "Kinh ngạc",
    6: "Bình thường",
}

EMOTION_LIST = list(EMOTIONS.values())

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
APP_DIR = BASE_DIR / "app"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE = 48

NUM_CLASSES = 7
IN_CHANNELS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 42
