import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import BATCH_SIZE, IMG_SIZE, RANDOM_SEED


class FER2013Dataset(Dataset):
    """PyTorch dataset for FER2013 CSV rows."""

    def __init__(self, data, transform=None):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise TypeError('data must be a path (str) or pandas DataFrame')
        required_cols = {'emotion', 'pixels'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f'Dataset must contain columns: {required_cols}')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
        image = torch.from_numpy(pixels).float().unsqueeze(0) / 255.0
        label = int(row['emotion'])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_train_transform():
    """Return augmentation and normalization used during training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_eval_transform():
    """Return deterministic preprocessing used for validation, test, and inference."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def split_fer2013_dataframe(df, val_split=0.1, test_split=0.1, seed=RANDOM_SEED):
    """Split FER2013 data using Usage column when available, otherwise seeded random split."""
    if 'Usage' in df.columns:
        train_df = df[df['Usage'] == 'Training']
        val_df = df[df['Usage'] == 'PublicTest']
        test_df = df[df['Usage'] == 'PrivateTest']
    else:
        n = len(df)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        train_end = int(n * (1 - val_split - test_split))
        val_end = train_end + int(n * val_split)
        train_df = df.iloc[indices[:train_end]]
        val_df = df.iloc[indices[train_end:val_end]]
        test_df = df.iloc[indices[val_end:]]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_dataloaders(
    csv_path,
    batch_size=BATCH_SIZE,
    val_split=0.1,
    test_split=0.1,
    augment=True,
    num_workers=0,
):
    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = split_fer2013_dataframe(df, val_split, test_split)

    train_transform = get_train_transform() if augment else get_eval_transform()
    val_transform = get_eval_transform()
    train_dataset = FER2013Dataset(train_df, transform=train_transform)
    val_dataset = FER2013Dataset(val_df, transform=val_transform)
    test_dataset = FER2013Dataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_data_loaders_with_augmentation(csv_path, batch_size=BATCH_SIZE, aug_transform=None):
    if aug_transform is None:
        aug_transform = get_train_transform()

    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = split_fer2013_dataframe(df)
    val_transform = get_eval_transform()

    train_dataset = FER2013Dataset(train_df, transform=aug_transform)
    val_dataset = FER2013Dataset(val_df, transform=val_transform)
    test_dataset = FER2013Dataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
