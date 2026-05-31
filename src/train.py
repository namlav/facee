import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

from .config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES,
    DEVICE, RANDOM_SEED, CHECKPOINTS_DIR, MODELS_DIR,
    RESULTS_DIR
)
from .model import EmotionCNN
from .dataset import get_class_weights, get_dataloaders
from .utils import set_seed, ensure_dirs, plot_training_history


class FocalLoss(nn.Module):
    """Focal loss, useful when easy majority-class examples dominate training."""

    def __init__(self, weight=None, gamma=1.5):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc='Train')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Val')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    avg_loss = running_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    return avg_loss, accuracy, macro_f1


def _selection_score(val_acc, val_macro_f1, metric):
    if metric == 'val_acc':
        return val_acc
    if metric == 'macro_f1':
        return val_macro_f1
    return 0.5 * val_acc + 0.5 * val_macro_f1


def _save_training_checkpoint(path, model, optimizer, epoch, val_loss, val_acc):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, path)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=30,
    device='cpu',
    checkpoint_dir='checkpoints',
    experiment_name='emotion_cnn',
    best_model_path=None,
    selection_metric='balanced_score',
):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    if best_model_path is not None:
        best_model_path = Path(best_model_path)
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}
    best_score = -1.0
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1 = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_macro_f1)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}')

        score = _selection_score(val_acc, val_macro_f1, selection_metric)
        if score > best_score:
            best_score = score
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path / f'{experiment_name}_best.pth')
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0:
            _save_training_checkpoint(
                checkpoint_path / f'{experiment_name}_epoch_{epoch}.pth',
                model,
                optimizer,
                epoch,
                val_loss,
                val_acc,
            )

    return {
        'history': history,
        'best_score': best_score,
        'best_val_acc': best_val_acc,
        'best_val_macro_f1': best_val_macro_f1,
        'best_epoch': best_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Emotion CNN')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to FER2013 CSV')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--device', type=str, default=str(DEVICE))
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--checkpoint_dir', type=str, default=str(CHECKPOINTS_DIR))
    parser.add_argument('--best_model_path', type=str, default=str(MODELS_DIR / 'best_model.pth'))
    parser.add_argument('--experiment_name', type=str, default='emotion_cnn')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--loss', choices=['ce', 'focal'], default='ce')
    parser.add_argument('--no_class_weights', action='store_true')
    parser.add_argument('--class_weight_power', type=float, default=0.5)
    parser.add_argument('--max_class_weight', type=float, default=3.0)
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    parser.add_argument(
        '--selection_metric',
        choices=['balanced_score', 'val_acc', 'macro_f1'],
        default='balanced_score',
    )
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs()

    train_loader, val_loader, _ = get_dataloaders(
        args.csv_path,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    class_weights = None
    if not args.no_class_weights:
        class_weights = get_class_weights(
            args.csv_path,
            power=args.class_weight_power,
            max_weight=args.max_class_weight,
        ).to(device)
        print(f'Using class weights: {[round(float(w), 4) for w in class_weights.cpu()]}')
    if args.loss == 'focal':
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    result = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler=scheduler, num_epochs=args.epochs,
        device=device, checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        best_model_path=args.best_model_path,
        selection_metric=args.selection_metric,
    )

    final_path = Path(args.checkpoint_dir) / f'{args.experiment_name}_final.pth'
    torch.save(model.state_dict(), final_path)

    print(
        f'Best score: {result["best_score"]:.4f}, '
        f'val acc: {result["best_val_acc"]:.4f}, '
        f'val macro-F1: {result["best_val_macro_f1"]:.4f} '
        f'at epoch {result["best_epoch"]}'
    )
    print(f'Best model saved to: {args.best_model_path}')
    print(f'Final model saved to: {final_path}')

    plots_dir = RESULTS_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(result['history'], save_path=plots_dir / f'{args.experiment_name}_history.png')


if __name__ == '__main__':
    main()
