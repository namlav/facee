from pathlib import Path
import torch

EMOTIONS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

EMOTION_LIST = list(EMOTIONS.values())

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
RESULTS_DIR = BASE_DIR / 'results'
REPORTS_DIR = BASE_DIR / 'reports'
APP_DIR = BASE_DIR / 'app'

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE = 48

NUM_CLASSES = 7
IN_CHANNELS = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RANDOM_SEED = 42
