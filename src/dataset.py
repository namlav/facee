import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import BATCH_SIZE, IMG_SIZE, RANDOM_SEED


class FER2013Dataset(Dataset):
    def __init__(self, data, transform=None):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise TypeError('data must be a path (str) or pandas DataFrame')
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


def _get_default_transform(normalize=True):
    transform_list = [transforms.ToPILImage(), transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(transform_list)


def get_dataloaders(csv_path, batch_size=BATCH_SIZE, val_split=0.1, test_split=0.1):
    df = pd.read_csv(csv_path)
    if 'Usage' in df.columns:
        train_df = df[df['Usage'] == 'Training']
        val_df = df[df['Usage'] == 'PublicTest']
        test_df = df[df['Usage'] == 'PrivateTest']
    else:
        n = len(df)
        indices = np.random.permutation(n)
        train_end = int(n * (1 - val_split - test_split))
        val_end = train_end + int(n * val_split)
        train_df = df.iloc[indices[:train_end]]
        val_df = df.iloc[indices[train_end:val_end]]
        test_df = df.iloc[indices[val_end:]]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = FER2013Dataset(train_df, transform=train_transform)
    val_dataset = FER2013Dataset(val_df, transform=val_transform)
    test_dataset = FER2013Dataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_data_loaders_with_augmentation(csv_path, batch_size=BATCH_SIZE, aug_transform=None):
    if aug_transform is None:
        aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    df = pd.read_csv(csv_path)
    if 'Usage' in df.columns:
        train_df = df[df['Usage'] == 'Training']
        val_df = df[df['Usage'] == 'PublicTest']
        test_df = df[df['Usage'] == 'PrivateTest']
    else:
        n = len(df)
        indices = np.random.permutation(n)
        train_end = int(n * 0.8)
        val_end = train_end + int(n * 0.1)
        train_df = df.iloc[indices[:train_end]]
        val_df = df.iloc[indices[train_end:val_end]]
        test_df = df.iloc[indices[val_end:]]

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = FER2013Dataset(train_df, transform=aug_transform)
    val_dataset = FER2013Dataset(val_df, transform=val_transform)
    test_dataset = FER2013Dataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
