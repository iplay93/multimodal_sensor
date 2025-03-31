# Transformer Model for Multimodal Sensor-based Classification

## Overview
This project implements a Transformer-based classifier for multimodal time-series data. The model processes EEG, IMU, and PPG sensor data to classify mental states into three categories: Relaxed, Neutral, and Focused.

## Features
- Transformer-based classifier with attention extraction
- Multimodal sensor data processing (EEG, IMU, PPG)
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
├── preprocessing/dataLoader.py # Data acquisition & Data preprocessing  
├── feature_extraction/         # Extracts important feature for each signals(not use in this)
├── model_design/model.py       # Designs a Transformer-based model
├── analysis/                   # Stores analysis results (e.g. confusion matrix, etc.)
├── checkpoints/                # Stores trained model checkpoints
├── trainModel.py               # Main script for training
├── testModel.py                # Main script for testing
├── README.md                   # Project documentation
```

## Model Architecture
The Transformer-based classifier consists of:
- An embedding layer to project input features into `d_model` dimensions.
- Transformer encoder layers with multi-head self-attention.
- A fully connected layer for classification.
- Optional attention extraction for interpretability.

## Data Preparation
Data is stored in `.npy` format inside `data/cross_val_splits/`. Each fold directory contains:
- `train.npy`: Training data
- `test.npy`: Test data

Each sample consists of:
```python
(eeg_array, imu_array, ppg_array, label)
```

## Training
Run the training script with:
```bash
python trainmodel.py
```
This will:
- Train the model on each fold
- Save trained checkpoints
- Generate attention maps for interpretability

## Attention Map Visualization
The script automatically saves attention maps in `analysis/`. These maps help understand which input features contribute most to the classification.

## Model Evaluation
The script computes:
- **Classification report** using precision, recall, and F1-score.
- **Confusion matrix** to analyze misclassifications.
- **ROC AUC Score** to assess model performance.

## Checkpoints
Trained models are saved in `checkpoints/` as:
```
checkpoints/model_fold_{fold_number}.pth
```
These can be used for inference or fine-tuning.

## Notes
- Adjust `input_dim` based on the number of features in the input data.
- Modify `num_classes` to match the number of classification categories.
- The model runs on GPU if available; otherwise, it falls back to CPU.

## Future Improvements
- Experiment with different Transformer hyperparameters.
- Implement early stopping for better generalization.
- Extend support for additional sensor modalities.