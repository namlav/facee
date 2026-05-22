import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix as sk_confusion_matrix,
    classification_report as sk_classification_report,
)

from .config import EMOTIONS, EMOTION_LIST, NUM_CLASSES, DEVICE, MODELS_DIR, RESULTS_DIR
from .model import EmotionCNN
from .dataset import get_dataloaders
from .utils import ensure_dirs, plot_confusion_matrix as plot_cm_util


def get_predictions(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_predictions = []
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    avg_loss = test_loss / total
    acc = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def plot_confusion_matrix(model, dataloader, class_names, device, save_path=None):
    all_labels, all_predictions, _ = get_predictions(model, dataloader, device)
    cm = sk_confusion_matrix(all_labels, all_predictions)
    if save_path is not None:
        plot_cm_util(cm, class_names, save_path=save_path)
    return cm


def classification_report(model, dataloader, class_names, device):
    all_labels, all_predictions, _ = get_predictions(model, dataloader, device)
    return sk_classification_report(all_labels, all_predictions, target_names=class_names)


def per_class_accuracy(model, dataloader, class_names, device):
    all_labels, all_predictions, _ = get_predictions(model, dataloader, device)
    cm = sk_confusion_matrix(all_labels, all_predictions)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return {name: float(acc) for name, acc in zip(class_names, per_class_acc)}


def main():
    parser = argparse.ArgumentParser(description='Evaluate Emotion CNN')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to FER2013 CSV')
    parser.add_argument('--model_path', type=str, default=str(MODELS_DIR / 'best_model.pth'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=str(DEVICE))
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device(args.device)

    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    _, _, test_loader = get_dataloaders(args.csv_path, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, criterion, device)

    print('Evaluation Results:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')

    cm_dir = RESULTS_DIR / 'confusion_matrix'
    cm_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        model, test_loader, EMOTION_LIST, device,
        save_path=cm_dir / 'confusion_matrix.png',
    )

    metrics_dir = RESULTS_DIR / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / 'evaluation_results.txt', 'w') as f:
        for k, v in metrics.items():
            f.write(f'{k}: {v:.4f}\n')
        f.write('\nPer-class accuracy:\n')
        pca = per_class_accuracy(model, test_loader, EMOTION_LIST, device)
        for name, acc in pca.items():
            f.write(f'  {name}: {acc:.4f}\n')
        f.write('\nClassification Report:\n')
        f.write(classification_report(model, test_loader, EMOTION_LIST, device))


if __name__ == '__main__':
    main()
