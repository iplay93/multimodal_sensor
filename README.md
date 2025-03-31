# Transformer Model for Multimodal Sensor-based Classification

## Overview
This project implements a Transformer-based classifier for multimodal time-series data. The model processes EEG, IMU, and PPG sensor data to classify mental states into three categories: Relaxed, Neutral, and Focused.

## Features
- Transformer-based classifier with attention extraction
- Multimodal data processing (EEG, IMU, PPG)
- Five-fold cross-validation setup
- Training and evaluation pipeline
- Attention map visualization

## Requirements
Ensure the following libraries are installed:
```bash
pip install numpy torch scikit-learn matplotlib seaborn shap
```

## Directory Structure
```
multimodal_sensor/
├── data/cross_val_splits/      # Folder containing cross-validation data splits
├── data/label_dict/            # Folder containing original data
├── feature_extraction/         # Extracts important feature for each signals(not use in this)
├── model_design/trainmodel.py  # Main script for training
├── model_design/testmodel.py   # Main script for testing
├── model_design/analysis/      # Stores analysis results (e.g. confusion matrix, etc.)
├── model_design/checkpoints/   # Stores trained model checkpoints
├── preprocessing/dataLoader.py # Data Acquisition & Data Preprocessing  
├── README.md                   # Project documentation

## Model Architecture
The Transformer-based classifier consists of:
- An embedding layer to project input features into `d_model` dimensions.
- Transformer encoder layers with multi-head self-attention.
- A fully connected layer for classification.
- Optional attention extraction for interpretability.

## Data Preparation
Data is stored in `.npy` format inside `data/cross_val_splits/`. Each fold directory contains:
- `train.npy`: Training data
- `val.npy`: Test data

Each sample consists of:
```python
(eeg_array, imu_array, ppg_array, label)
```