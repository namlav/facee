import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RANDOM_SEED


USAGE_TO_SPLIT = {
    'Training': 'train',
    'PublicTest': 'val',
    'PrivateTest': 'test',
}


def split_fer2013_csv(
    csv_path,
    output_dir=PROCESSED_DIR,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=RANDOM_SEED,
):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required_cols = {'emotion', 'pixels'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'CSV must contain columns: {required_cols}')

    if 'Usage' in df.columns:
        splits = {
            split_name: df[df['Usage'] == usage].copy()
            for usage, split_name in USAGE_TO_SPLIT.items()
        }
    else:
        if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
            raise ValueError('val_ratio and test_ratio must be positive and sum to less than 1.')
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(df))
        train_end = int(len(df) * (1 - val_ratio - test_ratio))
        val_end = train_end + int(len(df) * val_ratio)
        splits = {
            'train': df.iloc[indices[:train_end]].copy(),
            'val': df.iloc[indices[train_end:val_end]].copy(),
            'test': df.iloc[indices[val_end:]].copy(),
        }

    output_paths = {}
    for split_name, split_df in splits.items():
        output_path = output_dir / f'{split_name}.csv'
        split_df.to_csv(output_path, index=False)
        output_paths[split_name] = output_path

    return output_paths


def main():
    parser = argparse.ArgumentParser(description='Split FER2013 CSV into train/val/test files.')
    parser.add_argument('--csv_path', required=True, help='Path to cleaned FER2013 CSV.')
    parser.add_argument('--output_dir', default=str(PROCESSED_DIR), help='Directory for split CSV files.')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    output_paths = split_fer2013_csv(
        args.csv_path,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    for split_name, output_path in output_paths.items():
        print(f'{split_name}: {output_path}')


if __name__ == '__main__':
    main()
