# ECG Signal Classification for Myocardial Infarction Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning and Computer Vision pipeline designed to classify Electrocardiogram (ECG) time-series signals into normal cardiac rhythms and abnormal beats indicating **Myocardial Infarction (MI)**. 

This repository implements data preprocessing, class imbalance handling via **SMOTE**, and 2D spatial transformations—specifically **Continuous Wavelet Transform (CWT)** and **Gramian Angular Field (GAF)**—to extract spatial and time-frequency representations from 1D heartbeat signals for enhanced classification accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

---

## Overview

Cardiovascular disease diagnosis relies heavily on Electrocardiogram (ECG) recordings. Detecting subtle morphological changes in heartbeat signals like Myocardial Infarction (heart attack) manually can be time-consuming. 

This project benchmarks traditional machine learning models trained on raw 1D time series against modern 2D transformation strategies (CWT scalograms and GAF image matrices) evaluated with machine learning and Convolutional Neural Network (CNN) architectures.

---

## Key Features

- **Standardized Preprocessing**: Cleans raw 96-timestep ECG signals, scales feature ranges, and standardizes binary target labels (`0`: Abnormal / MI, `1`: Normal).
- **Imbalance Management (SMOTE)**: Applies Synthetic Minority Over-sampling Technique to balance minority cardiac event classes without synthetic data leakage.
- **Time-Series to Image Transformations**:
  - **Continuous Wavelet Transform (CWT)**: Decomposes signals into 2D time-frequency scalograms to isolate localized spectral changes.
  - **Gramian Angular Field (GAF)**: Maps 1D temporal series into 2D polar coordinate matrices preserving temporal correlations.
- **Multi-Model Benchmarking**: Trains and evaluates Logistic Regression, Support Vector Machines (SVM), Random Forests, and 2D Convolutional Neural Networks (CNN).

---

## Pipeline Architecture

```text
               ┌────────────────────────┐
               │    Raw ECG Signals     │
               │  (96 Timesteps / Beat) │
               └───────────┬────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │ Preprocessing & Label Alignment  │
          │     ({-1, 1} ➔ {0: MI, 1: Normal})│
          └────────────────┬─────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │  SMOTE Class Imbalance Resampling│
          └────────────────┬─────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 1D Raw Time │     │ 2D CWT      │     │ 2D GAF      │
│   Series    │     │ Scalograms  │     │ Matrices    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Classifiers │     │ ML / 2D CNN │     │ ML / 2D CNN │
│ (LR/SVM/RF) │     │ Classifier  │     │ Classifier  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │ Evaluation: Accuracy, Precision, │
          │   Recall, F1-Score & ROC-AUC     │
          └──────────────────────────────────┘
```

---

## Dataset

This project utilizes the benchmark **ECG200** dataset:
- **Time steps per sample**: 96 readings corresponding to a single heartbeat cycle.
- **Class Labels**:
  - `1`: Normal Heartbeat
  - `0` (`-1` raw): Abnormal Heartbeat (Myocardial Infarction)

---

## Project Structure

```text
.
├── data/
│   ├── ECG200_TRAIN.txt          # Raw training set
│   └── ECG200_TEST.txt           # Raw testing set
├── notebooks/
│   └── ecg_classification.ipynb  # Exploratory notebook & step-by-step pipeline
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Data loading, label conversion, SMOTE scaling
│   ├── feature_extraction.py     # CWT and GAF transformation modules
│   ├── models.py                 # ML baselines & PyTorch/Keras CNN architectures
│   └── evaluate.py               # Performance metrics & confusion matrix plotters
├── requirements.txt              # Dependency file
├── LICENSE                       # License information
└── README.md                     # Project documentation
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ecg-mi-classification-cwt-gaf.git
   cd ecg-mi-classification-cwt-gaf
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running via Jupyter Notebook
Launch the end-to-end interactive notebook:
```bash
jupyter notebook notebooks/ecg_classification.ipynb
```

### Python API Example
```python
from src.preprocessing import load_and_preprocess
from src.feature_extraction import transform_cwt, transform_gaf
from src.models import train_classifier

# 1. Load and Preprocess
X_train, y_train, X_test, y_test = load_and_preprocess('data/ECG200_TRAIN.txt', 'data/ECG200_TEST.txt')

# 2. Extract CWT / GAF 2D Representations
X_train_cwt, X_test_cwt = transform_cwt(X_train), transform_cwt(X_test)
X_train_gaf, X_test_gaf = transform_gaf(X_train), transform_gaf(X_test)

# 3. Train & Evaluate
results = train_classifier(X_train_gaf, y_train, X_test_gaf, y_test, model_type='cnn')
print(results)
```

---

## Methodology

1. **Data Preprocessing & Standardisation**: Features are standardized using `StandardScaler`. Labels are cast to binary format to adhere to standard clinical evaluation norms.
2. **Resampling**: `SMOTE` (Synthetic Minority Over-sampling Technique) is applied strictly on the training set to resolve class imbalance while preventing data leakage.
3. **Continuous Wavelet Transform (CWT)**: Utilizes Morlet wavelets to extract dynamic frequency components over time, outputting a 2D matrix per sample.
4. **Gramian Angular Field (GAF)**: Rescales time-series values between $[-1, 1]$ and converts them into polar coordinates to calculate Gramian Summation/Difference angular fields.

---

## Results

Models were evaluated across raw signals and 2D spatial transformations:

| Feature Input | Model Architecture | Accuracy | Precision | Recall (MI) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Raw Time Series | Logistic Regression | 0.81 | 0.83 | 0.80 | 0.81 |
| Raw Time Series | Random Forest | 0.85 | 0.86 | 0.84 | 0.85 |
| CWT Scalograms | Support Vector Machine (RBF) | 0.87 | 0.88 | 0.86 | 0.87 |
| **GAF Matrices** | **2D CNN** | **0.89** | **0.90** | **0.88** | **0.89** |

*Note: High Recall for class 0 (Myocardial Infarction) is prioritized in health diagnostics to reduce false negative risk.*

---

## Dependencies

- Python 3.8+
- `numpy`, `pandas`, `scikit-learn`
- `imbalanced-learn`
- `PyWavelets`
- `pyts`
- `PyTorch` / `TensorFlow`
- `matplotlib`, `seaborn`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
