import random
import time
from contextlib import contextmanager

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import BASE_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, REPORTS_DIR, APP_DIR, DEVICE, RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return DEVICE


def ensure_dirs():
    dirs = [
        DATA_DIR, RAW_DIR, PROCESSED_DIR,
        MODELS_DIR, CHECKPOINTS_DIR,
        RESULTS_DIR, REPORTS_DIR,
        APP_DIR,
        RESULTS_DIR / 'plots',
        RESULTS_DIR / 'metrics',
        RESULTS_DIR / 'confusion_matrix',
        REPORTS_DIR / 'figures',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, acc, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curves')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig


def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig


def calculate_metrics(y_true, y_pred, average='weighted'):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            print(f"{self.name} took {self.elapsed:.4f}s")
