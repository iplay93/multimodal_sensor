import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Filtering function with valid frequency range
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    if lowcut >= nyquist or highcut >= nyquist:
        raise ValueError("Critical frequencies must be between 0 and Nyquist frequency.")
    low = max(lowcut / nyquist, 1e-5)  # Ensure valid range
    high = min(highcut / nyquist, 0.99)  # Ensure valid range
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)