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