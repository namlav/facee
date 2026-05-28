import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES,
    DEVICE, RANDOM_SEED, CHECKPOINTS_DIR, MODELS_DIR,
    RESULTS_DIR
)
from .model import EmotionCNN
from .dataset import get_dataloaders
from .utils import set_seed, ensure_dirs, plot_training_history


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
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


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
):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    if best_model_path is not None:
        best_model_path = Path(best_model_path)
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
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

    return {'history': history, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch}


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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    result = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler=scheduler, num_epochs=args.epochs,
        device=device, checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        best_model_path=args.best_model_path,
    )

    final_path = Path(args.checkpoint_dir) / f'{args.experiment_name}_final.pth'
    torch.save(model.state_dict(), final_path)

    print(f'Best validation accuracy: {result["best_val_acc"]:.4f} at epoch {result["best_epoch"]}')
    print(f'Best model saved to: {args.best_model_path}')
    print(f'Final model saved to: {final_path}')

    plots_dir = RESULTS_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(result['history'], save_path=plots_dir / f'{args.experiment_name}_history.png')


if __name__ == '__main__':
    main()
