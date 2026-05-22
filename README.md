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
| Data Engineer & Documentation | Team A |
| AI Engineer                    | Team B |
| Application Engineer           | Team C |

## License

MIT
