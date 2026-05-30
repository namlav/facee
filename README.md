# Facial Emotion Recognition System

## Overview

AI-powered facial emotion recognition system using deep learning (Convolutional Neural Networks with PyTorch). The model classifies human facial expressions into seven emotion categories and is deployed as an interactive Streamlit web application with real-time webcam support.

## Features

- **7 emotion classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Image upload prediction** – Upload any facial image and get instant emotion classification
- **Real-time webcam recognition** – Live emotion detection via webcam feed
- **Face detection** – Automatic face detection with bounding boxes using OpenCV
- **Confidence scores** – Probability distribution across all emotion classes
- **Training history** – Loss and accuracy curves, confusion matrix, and classification metrics

## Project Structure

```
facial-emotion-recognition/
│
├── app/
│   ├── app.py                 # Streamlit web application entry point
│   ├── inference.py           # Model inference helpers
│   ├── webcam.py              # Real-time webcam detection
│   └── assets/                # Static assets (icons, styles)
│
├── data/
│   ├── raw/                   # Original FER2013 CSV files
│   ├── processed/             # Cleaned and preprocessed data
│   └── external/              # External resources (e.g., haarcascade)
│
├── models/
│   ├── checkpoints/           # Intermediate model checkpoints
│   └── best_model.pth         # Best performing model weights
│
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis notebook
│
├── reports/
│   ├── figures/               # Report figures and charts
│   └── final_report.docx      # Project final report
│
├── results/
│   ├── confusion_matrix/       # Confusion matrix visualizations
│   ├── metrics/                # Evaluation metrics (precision, recall, F1)
│   └── plots/                  # Training curves and other plots
│
├── src/
│   ├── dataset.py             # PyTorch Dataset and DataLoader
│   ├── preprocessing.py       # Data cleaning and normalization
│   ├── augmentation.py        # Data augmentation transforms
│   ├── model.py               # CNN model architecture definition
│   ├── train.py               # Training loop and logging
│   ├── evaluate.py            # Evaluation and metrics computation
│   ├── predict.py             # Single image prediction script
│   ├── utils.py               # Common utility functions
│   └── config.py              # Hyperparameters and file paths
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

```bash
git clone <repo-url>
cd facial-emotion-recognition
pip install -r requirements.txt
```

### Download Dataset

1. Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)
2. Place `fer2013.csv` in `data/raw/`
3. Run the preprocessing pipeline:

```bash
python -m src.preprocessing
```

### Dataset Preprocessing and Visualization

The dataset preparation pipeline uses the following files:

```text
data/raw/fer2013.csv                  # Original FER2013 CSV
data/processed/fer2013_clean.csv      # Cleaned/preprocessed CSV
data/processed/train.csv              # Training split
data/processed/val.csv                # Validation split
data/processed/test.csv               # Test split
```

Recommended commands:

```bash
python -m src.download_dataset
python -m src.preprocessing --raw_csv data/raw/fer2013.csv --output data/processed/fer2013_clean.csv --equalize
python -m src.split_dataset --csv_path data/processed/fer2013_clean.csv --output_dir data/processed
python -m src.visualize_dataset --output_dir reports/figures
```

#### Raw vs Preprocessed Dataset

| Aspect | Raw dataset | Preprocessed dataset |
|---|---|---|
| File | `data/raw/fer2013.csv` | `data/processed/fer2013_clean.csv` |
| Number of samples | 35,887 | 35,887 |
| Main columns | `emotion`, `pixels`, `Usage` | `emotion`, `pixels`, `Usage` |
| Pixel validation | Not explicitly checked | Rows are checked to ensure each image has exactly `48 x 48 = 2,304` pixel values |
| Image contrast | Original grayscale pixel values | CLAHE contrast equalization can be applied with `--equalize` |
| Split files | Uses FER2013 `Usage` column if available | Additional `train.csv`, `val.csv`, and `test.csv` are generated |
| Purpose | Preserve the original dataset | Use for training, validation, evaluation, and reporting |

In the current processed dataset, no invalid rows were removed, so the number of samples remains unchanged. The main difference is that the processed file has passed validation and can optionally contain CLAHE-enhanced pixel values, which may improve visibility of facial features under poor lighting or low contrast.

Current split sizes:

| Split | Samples | Approximate ratio |
|---|---:|---:|
| Train | 28,709 | 80% |
| Validation | 3,589 | 10% |
| Test | 3,589 | 10% |

#### Dataset Visualization Outputs

The visualization script creates the following figures in `reports/figures/`:

| Figure | Meaning |
|---|---|
| `dataset_class_distribution.png` | Shows the number of images for each emotion class. This chart helps identify class imbalance. FER2013 is usually imbalanced: `Happy` has many samples, while `Disgust` has very few. |
| `dataset_split_distribution.png` | Shows the proportion of train, validation, and test samples. This confirms whether the dataset split follows the expected 80/10/10 structure. |
| `dataset_split_class_distribution.png` | Compares emotion-class counts across train, validation, and test sets. This helps verify that each split contains all emotion classes and that imbalance is consistent across splits. |
| `dataset_pixel_intensity_distribution.png` | Shows the distribution of grayscale pixel values from sample images. This helps inspect brightness, contrast, and whether images are mostly dark, bright, or evenly distributed. |
| `dataset_sample_grid.png` | Displays example face images grouped by emotion label. This is useful for visually checking image quality, label meaning, and preprocessing effects. |

These figures are intended for the EDA and dataset-analysis section of the project report.

## Usage

### Training

Train the CNN model from scratch:

```bash
python -m src.train --csv_path data/raw/fer2013.csv --epochs 30 --batch_size 32
```

Optional arguments: `--lr`, `--model_save_path`, `--device`.

### Evaluation

Evaluate the trained model on the test set:

```bash
python -m src.evaluate
```

Outputs confusion matrix, classification report, and metric plots to `results/`.

### Run Web App

Launch the Streamlit web interface:

```bash
streamlit run app/app.py
```

Available pages:
- **Home** – Project overview and instructions
- **Upload** – Image upload with emotion prediction
- **Webcam** – Real-time webcam emotion detection
- **About** – Team information and technology stack

### Run Webcam (standalone)

```bash
python -m app.webcam
```

Press `q` to quit the webcam feed.

### Single Image Prediction

```bash
python -m src.predict --image path/to/image.jpg
```

## Model Architecture

- Custom CNN with 4 convolutional blocks
- Each block: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
- Fully connected layers with Dropout for regularization
- Approximately 7 million trainable parameters
- Input: 48×48 grayscale images
- Output: Softmax over 7 emotion classes

```
Input (48×48 grayscale)
        ↓
Conv2D (3×3, 64) → BatchNorm → ReLU → MaxPool (2×2)
        ↓
Conv2D (3×3, 128) → BatchNorm → ReLU → MaxPool (2×2)
        ↓
Conv2D (3×3, 256) → BatchNorm → ReLU → MaxPool (2×2)
        ↓
Conv2D (3×3, 512) → BatchNorm → ReLU → MaxPool (2×2)
        ↓
Flatten → FC (512→256) → Dropout → ReLU
        ↓
FC (256→128) → Dropout → ReLU
        ↓
FC (128→7) → Softmax
```

## Results

| Metric              | Target      |
|---------------------|-------------|
| Validation Accuracy | 65% – 75%   |
| Real-time FPS       | 15+ FPS     |
| Inference Time      | < 100 ms    |

Confusion matrices, precision/recall/F1 reports, and training curves are saved in `results/`.

## Technologies

- **Deep Learning**: PyTorch, TorchVision
- **Computer Vision**: OpenCV, Pillow
- **Web Application**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: scikit-learn

## Team

| Role                  | Member          |
|-----------------------|-----------------|
| Data Engineer & Documentation | Nguyễn Quốc Duy - [@QuocDuyNguyen](https://github.com/QuocDuyNguyen) |
| AI Engineer                    | Nguyễn Thanh Phong - [@Phong2607-H](https://github.com/Phong2607-H)|
| Application Engineer           | La Văn Nam (author) - [@namlav](https://github.com/namlav) |

## License

MIT
