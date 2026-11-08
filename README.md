# ECG Signal Classification for Myocardial Infarction Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn-orange.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning and Computer Vision pipeline designed to classify Electrocardiogram (ECG) time-series signals into normal cardiac rhythms and abnormal beats indicating **Myocardial Infarction (MI)**. 

This repository leverages data preprocessing, Synthetic Minority Over-sampling (SMOTE), and 2D spatial transformation techniques—specifically **Continuous Wavelet Transform (CWT)** and **Gramian Angular Fields (GAF)**—to convert 1D time-series signals into rich 2D image representations for improved feature extraction and classification.

---

## Key Features

* **Data Standardization & Preprocessing**: Converts baseline text recordings into scaled features and standard binary labels (`0: Abnormal/MI`, `1: Normal`).
* **Class Imbalance Management**: Integrates **SMOTE** to balance positive and negative cardiac instances, preventing biased model training.
* **Dual Time-Series Transformations**:
  * **Continuous Wavelet Transform (CWT)**: Captures time-frequency scale variations in heartbeat dynamics.
  * **Gramian Angular Field (GAF)**: Encodes temporal angular correlations into 2D matrices.
* **Multi-Model Evaluation**: Benchmarks tabular classifiers (Random Forest, Support Vector Machines, Logistic Regression) alongside 2D Convolutional Neural Networks (CNNs).

---

## Pipeline Architecture

Raw ECG Signal (96 timesteps)
│
├──► 1. Preprocessing & Label Encoding ({-1, 1} ➔ {0, 1})
│
├──► 2. Class Balancing (SMOTE)
│
├──► 3. Feature Extraction
│      ├──► 1D Raw Normalized Signals
│      ├──► CWT Scalograms (Time-Frequency maps)
│      └──► GAF Matrices (Angular summation fields)
│
└──► 4. Model Training & Evaluation
├──► Machine Learning (SVM, Random Forest, LR)
└──► 2D Convolutional Neural Networks (CNN)


---

## Dataset Overview

The project uses the benchmark **ECG200** dataset:
* **Sequence Length**: 96 data points per single heartbeat.
* **Target Classes**:
  * `1`: Normal heartbeat
  * `0`: Abnormal heartbeat (Myocardial Infarction)

---

## Project Structure

```text
├── data/
│   ├── ECG200_TRAIN.txt          # Raw training dataset
│   └── ECG200_TEST.txt           # Raw testing dataset
├── notebooks/
│   └── ecg_classification.ipynb  # End-to-end exploratory analysis & modeling
├── src/
│   ├── preprocessing.py          # Data loaders, label encoding & SMOTE logic
│   ├── feature_extraction.py     # CWT and GAF transformation functions
│   ├── train.py                  # Training pipeline script
│   └── evaluate.py               # Confusion matrices, ROC curves & metrics
├── requirements.txt              # Project dependencies
├── LICENSE
└── README.md
