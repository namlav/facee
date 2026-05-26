import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import EMOTIONS, IMG_SIZE, PROCESSED_DIR, RAW_DIR, REPORTS_DIR


DEFAULT_FILES = {
    'raw': RAW_DIR / 'fer2013.csv',
    'clean': PROCESSED_DIR / 'fer2013_clean.csv',
    'train': PROCESSED_DIR / 'train.csv',
    'val': PROCESSED_DIR / 'val.csv',
    'test': PROCESSED_DIR / 'test.csv',
}


def _read_existing_csvs(files):
    datasets = {}
    for name, path in files.items():
        path = Path(path)
        if path.exists():
            datasets[name] = pd.read_csv(path)
    if not datasets:
        raise FileNotFoundError('No dataset CSV files were found.')
    return datasets


def _add_emotion_name(df):
    result = df.copy()
    result['emotion_name'] = result['emotion'].astype(int).map(EMOTIONS)
    return result


def plot_class_distribution(df, output_path):
    df = _add_emotion_name(df)
    order = [EMOTIONS[i] for i in sorted(EMOTIONS)]
    counts = df['emotion_name'].value_counts().reindex(order, fill_value=0)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, palette='viridis', hue=counts.index, legend=False)
    ax.set_title('FER2013 Class Distribution')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Number of Images')
    ax.bar_label(ax.containers[0], fmt='%d', padding=3)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_split_distribution(datasets, output_path):
    split_counts = {
        name: len(df)
        for name, df in datasets.items()
        if name in {'train', 'val', 'test'}
    }
    if not split_counts:
        return False

    labels = list(split_counts.keys())
    values = list(split_counts.values())

    plt.figure(figsize=(8, 6))
    colors = ['#4C78A8', '#F58518', '#54A24B']
    wedges, _, autotexts = plt.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors[:len(values)],
    )
    for text in autotexts:
        text.set_color('white')
        text.set_fontweight('bold')
    plt.title('Train / Validation / Test Split')
    plt.legend(wedges, [f'{label}: {value}' for label, value in zip(labels, values)], loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def plot_split_class_distribution(datasets, output_path):
    rows = []
    for split_name in ['train', 'val', 'test']:
        if split_name not in datasets:
            continue
        df = _add_emotion_name(datasets[split_name])
        counts = df['emotion_name'].value_counts()
        for emotion_name, count in counts.items():
            rows.append({'split': split_name, 'emotion': emotion_name, 'count': count})

    if not rows:
        return False

    plot_df = pd.DataFrame(rows)
    order = [EMOTIONS[i] for i in sorted(EMOTIONS)]

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=plot_df,
        x='emotion',
        y='count',
        hue='split',
        order=order,
        palette='Set2',
    )
    ax.set_title('Class Distribution by Split')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Number of Images')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def plot_pixel_intensity(df, output_path, sample_size=2000, seed=42):
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)

    values = []
    for pixel_string in df['pixels']:
        values.append(np.fromstring(pixel_string, dtype=np.uint8, sep=' '))
    pixels = np.concatenate(values)

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(pixels, bins=50, color='#4C78A8')
    ax.set_title('Pixel Intensity Distribution')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_sample_grid(df, output_path, samples_per_class=5):
    df = _add_emotion_name(df)
    emotions = [EMOTIONS[i] for i in sorted(EMOTIONS)]
    fig, axes = plt.subplots(len(emotions), samples_per_class, figsize=(samples_per_class * 1.8, 12))

    for row_idx, emotion_name in enumerate(emotions):
        class_df = df[df['emotion_name'] == emotion_name].head(samples_per_class)
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            ax.axis('off')
            if col_idx < len(class_df):
                pixels = np.fromstring(class_df.iloc[col_idx]['pixels'], dtype=np.uint8, sep=' ')
                image = pixels.reshape(IMG_SIZE, IMG_SIZE)
                ax.imshow(image, cmap='gray')
            if col_idx == 0:
                ax.set_ylabel(emotion_name, rotation=0, labelpad=35, va='center')

    fig.suptitle('FER2013 Sample Images by Emotion', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_dataset_visualizations(output_dir=REPORTS_DIR / 'figures', sample_size=2000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _read_existing_csvs(DEFAULT_FILES)
    base_df = datasets['clean'] if 'clean' in datasets else datasets['raw']

    outputs = []
    class_path = output_dir / 'dataset_class_distribution.png'
    plot_class_distribution(base_df, class_path)
    outputs.append(class_path)

    split_path = output_dir / 'dataset_split_distribution.png'
    if plot_split_distribution(datasets, split_path):
        outputs.append(split_path)

    split_class_path = output_dir / 'dataset_split_class_distribution.png'
    if plot_split_class_distribution(datasets, split_class_path):
        outputs.append(split_class_path)

    pixel_path = output_dir / 'dataset_pixel_intensity_distribution.png'
    plot_pixel_intensity(base_df, pixel_path, sample_size=sample_size)
    outputs.append(pixel_path)

    samples_path = output_dir / 'dataset_sample_grid.png'
    plot_sample_grid(base_df, samples_path)
    outputs.append(samples_path)

    return outputs


def main():
    parser = argparse.ArgumentParser(description='Create visual charts for FER2013 dataset analysis.')
    parser.add_argument('--output_dir', default=str(REPORTS_DIR / 'figures'))
    parser.add_argument('--sample_size', type=int, default=2000)
    args = parser.parse_args()

    outputs = create_dataset_visualizations(args.output_dir, sample_size=args.sample_size)
    print('Created dataset visualizations:')
    for output in outputs:
        print(f'- {output}')


if __name__ == '__main__':
    main()
