#!/usr/bin/env python3
"""
Deep Learning Models Training with Grid Search for 1000 Hz EEG Data

This script trains multiple deep learning models on the original 1000 Hz EEG dataset
with comprehensive hyperparameter grid search to find optimal parameters for each model.
All models are trained with full reproducibility using random seeds.

FIXED VERSION: Addresses critical bugs identified in code analysis
- Fixed: Final model now trains on full training set (not subset)
- Fixed: Memory cleanup in grid search to prevent GPU OOM
- Fixed: Window label handling inconsistencies
- Fixed: Data loading column name standardization

Author: DILANJAN DK
Email: DDIYABAL@UWO.CA

Models Included:
    - CNN2D: 2D Convolutional Neural Network
    - LSTM: Bidirectional Long Short-Term Memory
    - Transformer: Transformer Encoder
    - CNN-Transformer: Hybrid CNN-Transformer architecture
    - DeepCNN-LSTM: Deep CNN followed by LSTM
    - Window-Aware Models: Models using window labels as input
    - Multi-Task Models: Models predicting window labels as auxiliary task

Usage Examples:
    # Train all models with grid search on pain_threshold task
    python scripts/dl_train_1000hz_gridsearch_fixed.py --task pain_threshold
    
    # Train specific models with grid search
    python scripts/dl_train_1000hz_gridsearch_fixed.py --task pain_threshold --models cnn lstm transformer
    
    # Train without grid search (use best known parameters)
    python scripts/dl_train_1000hz_gridsearch_fixed.py --task pain_threshold --no-grid-search
    
    # Custom seed for reproducibility
    python scripts/dl_train_1000hz_gridsearch_fixed.py --task pain_threshold --seed 123
    
    # Quick test run with fewer samples
    python scripts/dl_train_1000hz_gridsearch_fixed.py --task pain_threshold --quick --quick_n_per_subj 50

Output:
    - Results printed to console with detailed metrics
    - Results saved to JSON file (default: results_1000hz_gridsearch.json)
    - Includes accuracy, balanced accuracy, F1, precision, recall, confusion matrices
    - Best hyperparameters for each model are saved
"""

import os
import sys
import argparse
import json
import random
import time
import hashlib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Any, List
from collections import Counter
from itertools import product
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ============================================================================
# AUTHOR INFORMATION
# ============================================================================
__author__ = "DILANJAN DK"
__email__ = "DDIYABAL@UWO.CA"
__version__ = "1.1.1-fixed"

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.
    """
    # Python random module
    random.seed(seed)
    
    # NumPy random number generator
    np.random.seed(seed)
    
    # PyTorch random number generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Random seed set to {seed} for reproducibility")
    print(f"  - Python random: {seed}")
    print(f"  - NumPy random: {seed}")
    print(f"  - PyTorch random: {seed}")
    print(f"  - CUDA deterministic: True")
    print(f"  - CuDNN deterministic: True")

# ============================================================================
# DATA LOADING
# ============================================================================

def find_data_root(proj_root: str, dataset_name: str = 'Data') -> str:
    candidate = os.path.join(proj_root, dataset_name)
    return candidate if os.path.isdir(candidate) else proj_root

def load_index(root: str) -> pd.DataFrame:
    index_path = os.path.join(root, 'index.csv')
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.csv not found at {index_path}")
    
    df = pd.read_csv(index_path)
    
    # Filter out rejected epochs if reject_flag column exists
    if 'reject_flag' in df.columns:
        df = df[df['reject_flag'] == False].copy()
        print(f"  Filtered out rejected epochs: {len(pd.read_csv(index_path)) - len(df)} samples removed")
    
    return df

def get_most_common_channels(df: pd.DataFrame) -> int:
    if 'n_channels' in df.columns:
        cnt = Counter(df['n_channels'].dropna().astype(int))
        return cnt.most_common(1)[0][0] if cnt else 64
    return 64

class EEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, most_ch: int, return_window: bool = False, window_le: LabelEncoder = None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.most_ch = most_ch
        self.return_window = return_window
        self.window_le = window_le
        
        if self.return_window and self.window_le and 'window' in self.df.columns:
            self.window_indices = self.window_le.transform(self.df['window'])
        else:
            self.window_indices = np.zeros(len(self.df), dtype=int)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        # FIXED: Standardized column handling
        npz_file = row.get('npz_file', row.get('path', ''))
        if not npz_file:
            raise ValueError(f"Row {idx} missing both 'npz_file' and 'path' columns")
        npz_path = os.path.join(self.root, 'npz', npz_file)
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        with np.load(npz_path, allow_pickle=True) as data:
            X = data['X']  # Shape: (channels, time)
            
        ch, T = X.shape
        if ch < self.most_ch:
            pad = np.zeros((self.most_ch - ch, T), dtype=X.dtype)
            X = np.vstack([X, pad])
        elif ch > self.most_ch:
            X = X[:self.most_ch, :]
        
        # Z-score normalization per channel
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mean) / std
        
        window_idx = self.window_indices[idx] if self.return_window else 0
        return torch.FloatTensor(X), window_idx

def load_task_data(df: pd.DataFrame, task: str) -> Tuple[np.ndarray, LabelEncoder, pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Prepare labels for a given classification task.
    
    Args:
        df: DataFrame with rating_bin column
        task: Task name ('pain_5class', 'none_vs_pain', 'pain_only', 'pain_threshold')
    
    Returns:
        Tuple of (encoded labels, label encoder, filtered dataframe, window labels, window encoder)
    """
    le = LabelEncoder()
    window_le = LabelEncoder()
    y_window = np.zeros(len(df), dtype=int)
    
    # FIXED: Window handling - fit window_le first before any filtering
    if 'window' in df.columns:
        window_le.fit(df['window'])
        y_window = window_le.transform(df['window'])
    else:
        print("⚠ 'window' column not found in dataframe. Window-aware models will not work correctly.")
        df = df.copy()
        df['window'] = 'unknown'
        window_le.fit(['unknown'])
        y_window = np.zeros(len(df), dtype=int)
    
    if task == 'pain_5class':
        y = le.fit_transform(df['rating_bin'])
    elif task == 'none_vs_pain':
        # Strictly None vs Any Pain
        df = df.copy()
        df['binary'] = df['rating_bin'].apply(lambda x: 'none' if str(x).lower() in ['none', 'no_pain', 'no'] else 'pain')
        y = le.fit_transform(df['binary'])
    elif task == 'pain_only':
        pain_df = df[~df['rating_bin'].isin(['none', 'no_pain', 'no'])].copy()
        if len(pain_df) == 0:
            raise ValueError("No pain samples found")
        y = le.fit_transform(pain_df['rating_bin'])
        df = pain_df
        # FIXED: Re-transform window labels after filtering
        y_window = window_le.transform(df['window'])
    elif task == 'pain_threshold':
        # User definition: No Pain = {none, low}, Pain = {mid, high, extreme}
        print("  Grouping: No Pain = {none, low}, Pain = {mid, high, extreme}")
        df = df.copy()
        def threshold_label(rating_bin):
            rb = str(rating_bin).strip().lower()
            if rb in ['none', 'low', 'no_pain', 'no']:
                return 'no_significant_pain'
            else:
                return 'significant_pain'
        df['threshold_label'] = df['rating_bin'].apply(threshold_label)
        y = le.fit_transform(df['threshold_label'])
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return y, le, df, y_window, window_le

def load_all_segments(df: pd.DataFrame, root: str, most_ch: int, return_window: bool = False, window_le: LabelEncoder = None) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"\n{'='*70}")
    print(" DATA LOADING")
    print(f"{'='*70}")
    print(f"Total samples to load: {len(df):,}")
    print(f"Most common channels: {most_ch}")
    
    # FIXED: Standardized column handling
    if 'npz_file' not in df.columns:
        if 'path' in df.columns:
            df = df.copy()
            df['npz_file'] = df['path']
            print("Using 'path' column as 'npz_file'")
        else:
            raise ValueError("DataFrame must have either 'npz_file' or 'path' column")
    else:
        print("Using 'npz_file' column")
    
    if 'n_channels' in df.columns:
        ch_counts = Counter(df['n_channels'].dropna().astype(int))
        print(f"\nChannel distribution:")
        for ch, count in sorted(ch_counts.items())[:10]:
            print(f"  {ch} channels: {count:,} samples")
    
    dataset = EEGDataset(df, root, most_ch, return_window=return_window, window_le=window_le)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    all_X = []
    all_W = []
    print(f"\nLoading segments in batches of 32...")
    for batch_idx, (xb, wb) in enumerate(tqdm(loader, desc="Loading segments", unit="batch")):
        all_X.append(xb)
        if return_window:
            all_W.append(wb)
        if batch_idx == 0:
            print(f"  First batch shape: {xb.shape}")
    
    X_tensor = torch.cat(all_X, dim=0)
    W_tensor = torch.cat(all_W, dim=0) if return_window else torch.zeros(len(X_tensor), dtype=torch.long)
    
    print(f"\n✓ Successfully loaded {X_tensor.shape[0]:,} segments")
    print(f"  Shape: (samples={X_tensor.shape[0]}, channels={X_tensor.shape[1]}, time_points={X_tensor.shape[2]})")
    print(f"  Memory usage: ~{X_tensor.numel() * 4 / 1024**2:.2f} MB (float32)")
    
    return X_tensor, W_tensor

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class WindowAware_DeepCNN_LSTM(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, n_windows: int = 3,
                 cnn_filters: List[int] = [32, 64, 128], 
                 lstm_hidden: int = 192, 
                 lstm_layers: int = 2,
                 window_embed_dim: int = 16,
                 dropout: float = 0.4):
        super().__init__()
        self.window_embedding = nn.Embedding(n_windows, window_embed_dim)
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels//8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))
        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))
        
        time_reduced = n_time // 8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))
        lstm_input_dim = cnn_filters[2] + window_embed_dim
        
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden, batch_first=True, 
                           num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0, 
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2 + window_embed_dim, n_classes)
    
    def forward(self, x, window_idx):
        batch_size = x.size(0)
        window_emb = self.window_embedding(window_idx)
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(2).transpose(1, 2)
        
        seq_len = x.size(1)
        window_emb_expanded = window_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, window_emb_expanded], dim=2)
        
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = torch.cat([h, window_emb], dim=1)
        h = self.dropout(h)
        return self.fc(h)

class WindowAware_CNN_Transformer(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, n_windows: int = 3,
                 cnn_filters: List[int] = [32, 64, 128],
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 window_embed_dim: int = 16,
                 dropout: float = 0.3,
                 time_downsample: int = 4):
        super().__init__()
        self.window_embedding = nn.Embedding(n_windows, window_embed_dim)
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels//8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))
        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))
        
        time_reduced = n_time // 8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))
        self.time_downsample = time_downsample
        time_for_transformer = time_reduced // time_downsample
        self.cnn_to_transformer = nn.Linear(cnn_filters[2] + window_embed_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, time_for_transformer + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model + window_embed_dim, n_classes)
    
    def forward(self, x, window_idx):
        batch_size = x.size(0)
        window_emb = self.window_embedding(window_idx)
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(2).transpose(1, 2)
        
        T = x.size(1)
        T_down = (T // self.time_downsample) * self.time_downsample
        x = x[:, :T_down, :]
        x = x.reshape(batch_size, T_down // self.time_downsample, -1, self.time_downsample)
        x = x.mean(dim=3)
        
        window_emb_expanded = window_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, window_emb_expanded], dim=2)
        x = self.cnn_to_transformer(x)
        
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = torch.cat([x, window_emb], dim=1)
        x = self.dropout(x)
        return self.fc(x)

class CNN2D(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(n_channels//8, 7), padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((1, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 4))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden: int = 128, 
                 dropout: float = 0.5, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True, num_layers=2, 
                           dropout=dropout if dropout > 0 else 0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        fc_in = hidden * 2 if bidirectional else hidden
        self.fc = nn.Linear(fc_in, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h = h[-1]
        h = self.dropout(h)
        return self.fc(h)

class TransformerModel(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_classes: int, 
                 d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.downsample = 4
        self.proj = nn.Linear(n_channels * self.downsample, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_time // self.downsample + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, 
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, T = x.shape
        T_down = (T // self.downsample) * self.downsample
        x = x[:, :, :T_down]
        x = x.reshape(batch_size, C, -1, self.downsample)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, T_down // self.downsample, -1)
        x = self.proj(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)

class CNN_Transformer(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_classes: int,
                 cnn_filters: int = 64, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(n_channels//8, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))
        self.conv2 = nn.Conv2d(32, cnn_filters, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters)
        self.pool2 = nn.AdaptiveAvgPool2d((1, n_time // 4))
        self.cnn_to_transformer = nn.Linear(cnn_filters, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_time // 4 + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.squeeze(2).transpose(1, 2)
        x = self.cnn_to_transformer(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)

class DeepCNN_LSTM(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2,
                 cnn_filters: List[int] = [32, 64, 128], 
                 lstm_hidden: int = 192, 
                 lstm_layers: int = 2,
                 dropout: float = 0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels//8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))
        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))
        time_reduced = n_time // 8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))
        self.lstm = nn.LSTM(cnn_filters[2], lstm_hidden, batch_first=True, 
                           num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0, 
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(2).transpose(1, 2)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h)

class MultiTask_DeepCNN_LSTM(nn.Module):
    """
    Multi-Task Deep CNN-LSTM for pain and window prediction.
    
    Heads:
        1. Pain Classification (Main Task)
        2. Window Classification (Auxiliary Task: Baseline/ERP/Post)
    
    This architecture forces the shared encoder to learn features relevant 
    to temporal phases (context), improving robustness without needing 
    window labels at inference time.
    """
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, n_windows: int = 3,
                 cnn_filters: List[int] = [32, 64, 128], 
                 lstm_hidden: int = 192, 
                 lstm_layers: int = 2,
                 dropout: float = 0.4):
        super().__init__()
        
        # Shared Encoder (Deep CNN)
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels//8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))
        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))
        time_reduced = n_time // 8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))
        
        # Shared Temporal (LSTM)
        self.lstm = nn.LSTM(cnn_filters[2], lstm_hidden, batch_first=True, 
                           num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0, 
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
        # Task Heads
        self.fc_pain = nn.Linear(lstm_hidden * 2, n_classes)
        self.fc_window = nn.Linear(lstm_hidden * 2, n_windows)
    
    def forward(self, x):
        # Shared Encoder
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(2).transpose(1, 2)
        
        # Shared Temporal
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = self.dropout(h)
        
        # Heads
        pain_out = self.fc_pain(h)
        window_out = self.fc_window(h)
        return pain_out, window_out

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def class_weights_from_y(y: np.ndarray, class_names: list = None) -> Tuple[torch.FloatTensor, Dict]:
    unique, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(unique) * counts)
    weight_info = {}
    for i, (cls_idx, count) in enumerate(zip(unique, counts)):
        cls_name = class_names[cls_idx] if class_names else f"Class_{cls_idx}"
        weight_info[cls_name] = {
            'count': int(count),
            'weight': float(weights[i]),
            'percentage': float(count / len(y) * 100)
        }
    return torch.FloatTensor(weights), weight_info

def train_one_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    y_train: np.ndarray, device: str, epochs: int = 50, patience: int = 15,
                    lr: float = 1e-3, weight_decay: float = 1e-5, class_names: list = None,
                    model_name: str = "Model", verbose: bool = True) -> Tuple[nn.Module, float, Dict]:
    
    if verbose:
        print(f"\n{'='*70}")
        print(f" TRAINING SETUP: {model_name}")
        print(f"{'='*70}")
    
    model.to(device)
    if verbose:
        print(f"Model moved to device: {device}")
    
    weights, weight_info = class_weights_from_y(y_train, class_names)
    weights = weights.to(device)
    if verbose:
        print(f"\nClass weights for loss function:")
        for cls_name, info in weight_info.items():
            print(f"  {cls_name:20s}: count={info['count']:6d} ({info['percentage']:5.2f}%), weight={info['weight']:.4f}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Determine model type
    is_window_aware = isinstance(model, (WindowAware_DeepCNN_LSTM, WindowAware_CNN_Transformer))
    is_multitask = isinstance(model, MultiTask_DeepCNN_LSTM)
    
    if is_multitask:
        criterion_window = nn.CrossEntropyLoss().to(device)
        if verbose:
            print(f"  Note: Multi-task training enabled (Pain + Window prediction)")
    elif is_window_aware and verbose:
        print(f"  Note: Window-aware training enabled")
    
    best_state, best_val_loss = None, float('inf')
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    current_lr = lr
    
    if verbose:
        print(f"\n{'='*70}")
        print(f" TRAINING PROGRESS")
        print(f"{'='*70}")
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10} {'Val F1':<10} {'LR':<12} {'Time':<8} {'Status':<15}")
        print("-" * 70)
    
    for epoch in range(epochs):
        # Timing setup
        if device == 'cuda' and torch.cuda.is_available():
            epoch_start_time = torch.cuda.Event(enable_timing=True)
            epoch_end_time = torch.cuda.Event(enable_timing=True)
            epoch_start_time.record()
            use_cuda_timing = True
        else:
            epoch_start = time.time()
            use_cuda_timing = False
        
        # Training phase
        model.train()
        train_losses = []
        
        if verbose:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")
        else:
            train_pbar = train_loader
        
        for batch in train_pbar:
            optimizer.zero_grad()
            
            if is_window_aware:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                out = model(xb, wb)
                loss = criterion(out, yb)
            elif is_multitask:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                out_pain, out_window = model(xb)
                loss_p = criterion(out_pain, yb)
                loss_w = criterion_window(out_window, wb)
                loss = loss_p + loss_w
            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                out = model(xb)
                loss = criterion(out, yb)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_loss = loss.item()
            train_losses.append(batch_loss)
            
            if verbose:
                train_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg_loss': f'{np.mean(train_losses):.4f}'})
        
        tr_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        
        if verbose:
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, unit="batch")
        else:
            val_pbar = val_loader
        
        with torch.no_grad():
            for batch in val_pbar:
                if is_window_aware:
                    xb, wb, yb = batch
                    xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                    out = model(xb, wb)
                    loss = criterion(out, yb)
                elif is_multitask:
                    xb, wb, yb = batch
                    xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                    out_pain, out_window = model(xb)
                    loss_p = criterion(out_pain, yb)
                    loss_w = criterion_window(out_window, wb)
                    loss = loss_p + loss_w
                    out = out_pain  # For metrics
                else:
                    xb, yb = batch[0].to(device), batch[-1].to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    
                val_losses.append(loss.item())
                val_preds.append(out.argmax(dim=1).cpu().numpy())
                val_targets.append(yb.cpu().numpy())
                
                if verbose:
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch time
        if use_cuda_timing:
            epoch_end_time.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0
        else:
            epoch_time = time.time() - epoch_start
        
        vloss = np.mean(val_losses)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        val_acc = (val_preds == val_targets).mean()
        
        old_lr = current_lr
        scheduler.step(vloss)
        current_lr = optimizer.param_groups[0]['lr']
        lr_changed = abs(current_lr - old_lr) > 1e-8
        
        if vloss < best_val_loss - 1e-4:
            status = "✓ New best!"
            best_val_loss = vloss
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            status = "⏹ Early stop" if no_improve >= patience else f"No improve ({no_improve}/{patience})"
        
        if verbose:
            lr_str = f"{current_lr:.2e}" + (" ↓" if lr_changed else "")
            time_str = f"{epoch_time:.1f}s"
            print(f"{epoch+1:<8} {tr_loss:<12.6f} {vloss:<12.6f} {val_acc:<10.4f} {val_f1:<10.4f} {lr_str:<12} {time_str:<8} {status:<15}")
        elif epoch % 5 == 0 or epoch == epochs - 1:
            lr_str = f"{current_lr:.2e}" + (" ↓" if lr_changed else "")
            time_str = f"{epoch_time:.1f}s"
            print(f"  [{epoch+1}/{epochs}] Train Loss: {tr_loss:.4f} | Val Loss: {vloss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {time_str} | {status}")
        
        if no_improve >= patience:
            if verbose:
                print(f"\n⏹ Early stopping triggered at epoch {epoch+1}")
            break
    
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if verbose:
            print(f"\n✓ Loaded best model from epoch {best_epoch}")
            print(f"  Best validation loss: {best_val_loss:.6f}")
            print(f"  Best validation accuracy: {best_val_acc:.4f}")
            print(f"  Best validation F1: {best_val_f1:.4f}")
    else:
        if verbose: print(f"\n⚠ No best model state saved (using final model)")
    
    training_info = {
        'best_epoch': best_epoch, 'best_val_loss': float(best_val_loss),
        'best_val_acc': float(best_val_acc), 'best_val_f1': float(best_val_f1),
        'total_epochs': epoch + 1, 'early_stopped': no_improve >= patience
    }
    
    return model, best_val_f1, training_info

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str, le: LabelEncoder, 
                   model_name: str = "Model", window_le: LabelEncoder = None) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f" EVALUATION: {model_name}")
    print(f"{'='*70}")
    print(f"Evaluating on {len(test_loader)} test batches...")
    
    model.eval()
    all_preds, all_targets, all_probs, all_windows = [], [], [], []
    all_window_preds = []
    
    is_window_aware = isinstance(model, (WindowAware_DeepCNN_LSTM, WindowAware_CNN_Transformer))
    is_multitask = isinstance(model, MultiTask_DeepCNN_LSTM)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            if is_window_aware:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                out = model(xb, wb)
                all_windows.append(wb.cpu().numpy())
            elif is_multitask:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                out_pain, out_window = model(xb)
                out = out_pain
                all_windows.append(wb.cpu().numpy())
                all_window_preds.append(out_window.argmax(dim=1).cpu().numpy())
            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                out = model(xb)
                if len(batch) > 2:
                    all_windows.append(batch[1].numpy())
            
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_probs = np.concatenate(all_probs)
    
    acc = (y_pred == y_true).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1_per_class, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    report = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    max_probs = np.max(y_probs, axis=1)
    mean_confidence = np.mean(max_probs)
    std_confidence = np.std(max_probs)
    
    print(f"\nOverall Metrics (Pain task: no_significant_pain vs significant_pain; none+low vs mid/high/extreme):")
    print(f"  Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy:  {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"  Macro F1-Score:     {f1_macro:.4f}")
    print(f"\nPain Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    print(f"Confusion Matrix (Pain):")
    print(cm)
    
    metrics = {
        'accuracy': acc, 'balanced_accuracy': bal_acc, 'f1_macro': f1_macro,
        'f1_weighted': f1_weighted, 'report': report, 'confusion_matrix': cm.tolist(),
        'class_names': le.classes_.tolist(), 'mean_confidence': float(mean_confidence),
        'std_confidence': float(std_confidence)
    }
    
    if all_windows and window_le:
        windows = np.concatenate(all_windows)
        per_window_metrics = {}
        print(f"\nPer-Window Performance:")
        for w_idx, w_name in enumerate(window_le.classes_):
            mask = (windows == w_idx)
            if np.any(mask):
                w_acc = (y_pred[mask] == y_true[mask]).mean()
                w_f1 = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
                per_window_metrics[w_name] = {'accuracy': float(w_acc), 'f1_macro': float(w_f1), 'n_samples': int(mask.sum())}
                print(f"  {w_name:<15}: Acc={w_acc:.4f}, F1={w_f1:.4f} ({mask.sum()} samples)")
        metrics['per_window_metrics'] = per_window_metrics
    
    # Multi-task auxiliary window head metrics (if available)
    if all_window_preds and window_le is not None:
        y_window_true = np.concatenate(all_windows)
        y_window_pred = np.concatenate(all_window_preds)
        window_acc = (y_window_pred == y_window_true).mean()
        window_bal_acc = balanced_accuracy_score(y_window_true, y_window_pred)
        window_f1 = f1_score(y_window_true, y_window_pred, average='macro', zero_division=0)
        window_report = classification_report(y_window_true, y_window_pred, target_names=window_le.classes_, zero_division=0)
        window_cm = confusion_matrix(y_window_true, y_window_pred)
        
        print(f"\nAuxiliary Window Task Metrics (Baseline/ERP/Post):")
        print(f"  Accuracy:           {window_acc:.4f} ({window_acc*100:.2f}%)")
        print(f"  Balanced Accuracy:  {window_bal_acc:.4f} ({window_bal_acc*100:.2f}%)")
        print(f"  Macro F1-Score:     {window_f1:.4f}")
        print(f"\nWindow Classification Report:")
        print(window_report)
        print(f"Confusion Matrix (Window):")
        print(window_cm)
        
        metrics['window_task'] = {
            'accuracy': float(window_acc),
            'balanced_accuracy': float(window_bal_acc),
            'f1_macro': float(window_f1),
            'report': window_report,
            'confusion_matrix': window_cm.tolist(),
            'class_names': window_le.classes_.tolist()
        }
    
    return metrics

# ============================================================================
# GRID SEARCH
# ============================================================================

def grid_search_model(model_name: str, X_all: torch.Tensor, y_all: np.ndarray, groups: np.ndarray,
                     n_classes: int, n_channels: int, n_time: int, device: str, seed: int,
                     param_grid: Dict[str, List[Any]], epochs: int = 50, patience: int = 15,
                     completed_configs: Dict = None, best_hyperparams: Dict = None,
                     save_checkpoint: callable = None, y_window_all: np.ndarray = None,
                     n_windows: int = 0) -> Tuple[Dict, float]:
    
    print(f"\n{'='*70}")
    print(f" GRID SEARCH: {model_name.upper()}")
    print(f"{'='*70}")
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(X_all, y_all, groups))
    
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    y_window_train = y_window_all[train_idx] if y_window_all is not None else None
    y_window_val = y_window_all[val_idx] if y_window_all is not None else None
    
    param_keys = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_keys]
    combinations = list(product(*param_values))
    
    print(f"Total parameter combinations: {len(combinations)}")
    best_params = {}
    best_f1 = 0.0

    
    grid_pbar = tqdm(enumerate(combinations), total=len(combinations), desc=f"Grid Search: {model_name.upper()}", unit="config")
    
    for idx, values in grid_pbar:
        params = dict(zip(param_keys, values))
        model_params = {k: v for k, v in params.items() if k not in ['lr', 'weight_decay', 'batch_size']}
        training_params = {k: v for k, v in params.items() if k in ['lr', 'weight_decay']}
        batch_size = params.get('batch_size', 64)
        
        try:
            if model_name == 'cnn':
                model = CNN2D(n_channels, n_time, n_classes, **model_params)
            elif model_name == 'lstm':
                model = LSTMModel(n_channels, n_classes, **model_params)
            elif model_name == 'transformer':
                model = TransformerModel(n_channels, n_time, n_classes, **model_params)
            elif model_name == 'cnn_transformer':
                model = CNN_Transformer(n_channels, n_time, n_classes, **model_params)
            elif model_name == 'deep_cnn_lstm':
                model = DeepCNN_LSTM(n_channels, n_time, n_classes, **model_params)
            elif model_name == 'window_aware_deep_cnn_lstm':
                model = WindowAware_DeepCNN_LSTM(n_channels, n_time, n_classes, n_windows=n_windows, **model_params)
            elif model_name == 'window_aware_cnn_transformer':
                model = WindowAware_CNN_Transformer(n_channels, n_time, n_classes, n_windows=n_windows, **model_params)
            elif model_name == 'multitask_deep_cnn_lstm':
                model = MultiTask_DeepCNN_LSTM(n_channels, n_time, n_classes, n_windows=n_windows, **model_params)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            print(f"  ✗ Model creation failed: {e}")
            continue
        
        # Handle window-aware or multitask data loading
        is_wa = model_name in ['window_aware_deep_cnn_lstm', 'window_aware_cnn_transformer']
        is_mt = model_name == 'multitask_deep_cnn_lstm'
        
        if is_wa or is_mt:
            train_dataset = TensorDataset(X_train, torch.LongTensor(y_window_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(X_val, torch.LongTensor(y_window_val), torch.LongTensor(y_val))
        else:
            train_dataset = TensorDataset(X_train, torch.LongTensor(y_train))
            val_dataset = TensorDataset(X_val, torch.LongTensor(y_val))
            
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        try:
            model, val_f1, train_info = train_one_model(
                model, train_loader, val_loader, y_train, device,
                epochs=epochs, patience=patience, verbose=False, **training_params
            )
            
            if idx < 3 or idx >= len(combinations) - 3:
                print(f"  Result: F1={val_f1:.4f} (Acc={train_info['best_val_acc']:.4f})")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = params.copy()
                grid_pbar.set_postfix({'best_f1': f'{best_f1:.4f} ⭐'})
        except Exception as e:
            print(f"  ✗ Training failed: {e}")
        finally:
            # FIXED: Memory cleanup to prevent GPU OOM
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
            
    return best_params, best_f1

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Train DL models on 1000 Hz EEG data with grid search")
    ap.add_argument('--models', nargs='+', default=['cnn', 'lstm', 'transformer', 'cnn_transformer', 'deep_cnn_lstm', 'multitask_deep_cnn_lstm'], help='Models to train')
    ap.add_argument('--task', default='pain_threshold', help='Classification task')
    ap.add_argument('--epochs', type=int, default=50, help='Max epochs')
    ap.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    ap.add_argument('--data_root', type=str, default=None, help='Path to Data')
    ap.add_argument('--quick', action='store_true', help='Quick mode')
    ap.add_argument('--quick_n_per_subj', type=int, default=50, help='Quick mode samples')
    ap.add_argument('--output_file', type=str, default=None, help='Output file')
    ap.add_argument('--log_file', type=str, default=None, help='Log file')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--no-grid-search', action='store_true', help='Skip grid search')
    ap.add_argument('--grid-epochs', type=int, default=30, help='Grid search epochs')
    ap.add_argument('--grid-patience', type=int, default=10, help='Grid search patience')
    ap.add_argument('--resume', action='store_true', help='Resume')
    ap.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint dir')
    
    args = ap.parse_args()
    
    if args.output_file is None: args.output_file = f'results_1000hz_gridsearch_seed{args.seed}.json'
    if args.log_file is None: args.log_file = f'logs_1000hz_gridsearch_seed{args.seed}.log'
    
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(cwd, '..'))
    root = args.data_root if args.data_root else find_data_root(proj_root, 'Data')
    
    df = load_index(root)
    most_ch = get_most_common_channels(df)
    
    if args.quick:
        rnd = np.random.RandomState(args.seed)
        tmp = df.copy()
        tmp['_r'] = rnd.rand(len(tmp))
        tmp = tmp.sort_values(['participant', '_r'])
        df = tmp.groupby('participant').head(args.quick_n_per_subj).drop(columns=['_r'])
        print(f"Quick mode: using {len(df):,} rows")
    
    y_all, le, df, y_window, window_le = load_task_data(df, args.task)
    groups = df['participant'].values
    n_classes = len(le.classes_)
    n_windows = len(window_le.classes_) if window_le else 0
    
    use_window = any('window' in m or 'multitask' in m for m in args.models)
    X_all, W_all = load_all_segments(df, root, most_ch, return_window=use_window, window_le=window_le)
    n_channels, n_time = X_all.shape[1], X_all.shape[2]
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups))
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    y_window_train = W_all[train_idx] if use_window else None
    y_window_test = W_all[test_idx] if use_window else None
    
    param_grids = {
        'cnn': {'dropout': [0.4, 0.5], 'lr': [1e-3, 5e-4], 'weight_decay': [1e-5], 'batch_size': [64]},
        'lstm': {'hidden': [128, 192], 'dropout': [0.4, 0.5], 'bidirectional': [True], 'lr': [1e-3, 5e-4], 'weight_decay': [1e-5], 'batch_size': [64]},
        'transformer': {'d_model': [128], 'nhead': [8], 'num_layers': [3, 4], 'dropout': [0.3], 'lr': [5e-4, 1e-4], 'weight_decay': [1e-5], 'batch_size': [64]},
        'cnn_transformer': {'cnn_filters': [64, 128], 'd_model': [128], 'nhead': [8], 'num_layers': [2, 3], 'dropout': [0.3], 'lr': [5e-4, 1e-4], 'weight_decay': [1e-5], 'batch_size': [64]},
        'deep_cnn_lstm': {'cnn_filters': [[32, 64, 128]], 'lstm_hidden': [192, 256], 'lstm_layers': [2], 'dropout': [0.3, 0.4], 'lr': [5e-4, 1e-3], 'weight_decay': [1e-5], 'batch_size': [64]},
        'window_aware_deep_cnn_lstm': {'cnn_filters': [[32, 64, 128]], 'lstm_hidden': [192, 256], 'lstm_layers': [2], 'window_embed_dim': [16, 32], 'dropout': [0.3, 0.4], 'lr': [5e-4, 1e-3], 'weight_decay': [1e-5], 'batch_size': [64]},
        'window_aware_cnn_transformer': {'cnn_filters': [[32, 64, 128]], 'd_model': [128, 256], 'nhead': [8], 'num_layers': [2, 3], 'window_embed_dim': [16, 32], 'dropout': [0.3], 'lr': [5e-4, 1e-4], 'weight_decay': [1e-5], 'batch_size': [64]},
        'multitask_deep_cnn_lstm': {'cnn_filters': [[32, 64, 128]], 'lstm_hidden': [192, 256], 'lstm_layers': [2], 'dropout': [0.3, 0.4], 'lr': [5e-4, 1e-3], 'weight_decay': [1e-5], 'batch_size': [64]}
    }
    
    best_known_params = {
        'cnn': {'dropout': 0.4, 'lr': 0.0005, 'weight_decay': 1e-5, 'batch_size': 64},
        'lstm': {'hidden': 192, 'dropout': 0.5, 'bidirectional': True, 'lr': 0.0005, 'weight_decay': 1e-5, 'batch_size': 64},
        'transformer': {'d_model': 128, 'nhead': 8, 'num_layers': 3, 'dropout': 0.3, 'lr': 0.0001, 'weight_decay': 1e-5, 'batch_size': 64},
        'cnn_transformer': {'cnn_filters': 64, 'd_model': 128, 'nhead': 8, 'num_layers': 3, 'dropout': 0.3, 'lr': 5e-4, 'weight_decay': 1e-5, 'batch_size': 64},
        'deep_cnn_lstm': {'cnn_filters': [32, 64, 128], 'lstm_hidden': 192, 'lstm_layers': 2, 'dropout': 0.3, 'lr': 0.0003, 'weight_decay': 1e-5, 'batch_size': 64},
        'window_aware_deep_cnn_lstm': {'cnn_filters': [32, 64, 128], 'lstm_hidden': 192, 'lstm_layers': 2, 'window_embed_dim': 16, 'dropout': 0.4, 'lr': 0.0005, 'weight_decay': 1e-5, 'batch_size': 64},
        'window_aware_cnn_transformer': {'cnn_filters': [32, 64, 128], 'd_model': 128, 'nhead': 8, 'num_layers': 3, 'window_embed_dim': 16, 'dropout': 0.3, 'lr': 0.0001, 'weight_decay': 1e-5, 'batch_size': 64},
        'multitask_deep_cnn_lstm': {'cnn_filters': [32, 64, 128], 'lstm_hidden': 192, 'lstm_layers': 2, 'dropout': 0.4, 'lr': 0.0005, 'weight_decay': 1e-5, 'batch_size': 64}
    }
    
    results = {}
    best_hyperparams = {}
    best_params = {}
    best_f1 = 0.0

    
    for model_name in args.models:
        print(f"\n{'='*70}\n PROCESSING MODEL: {model_name.upper()}\n{'='*70}")
        
        if args.no_grid_search:
            best_params = best_known_params.get(model_name, {})
        else:
            if model_name in param_grids:
                best_params, _ = grid_search_model(
                    model_name, X_train, y_train, groups_train,
                    n_classes, n_channels, n_time, device, args.seed,
                    param_grids[model_name], epochs=args.grid_epochs, patience=args.grid_patience,
                    y_window_all=y_window_train if use_window else None, n_windows=n_windows
                )
                best_hyperparams[model_name] = best_params
            else:
                best_params = best_known_params.get(model_name, {})
        
        model_params = {k: v for k, v in best_params.items() if k not in ['lr', 'weight_decay', 'batch_size']}
        training_params = {k: v for k, v in best_params.items() if k in ['lr', 'weight_decay']}
        batch_size = best_params.get('batch_size', 64)
        
        if model_name == 'cnn': model = CNN2D(n_channels, n_time, n_classes, **model_params)
        elif model_name == 'lstm': model = LSTMModel(n_channels, n_classes, **model_params)
        elif model_name == 'transformer': model = TransformerModel(n_channels, n_time, n_classes, **model_params)
        elif model_name == 'cnn_transformer': model = CNN_Transformer(n_channels, n_time, n_classes, **model_params)
        elif model_name == 'deep_cnn_lstm': model = DeepCNN_LSTM(n_channels, n_time, n_classes, **model_params)
        elif model_name == 'window_aware_deep_cnn_lstm': model = WindowAware_DeepCNN_LSTM(n_channels, n_time, n_classes, n_windows=n_windows, **model_params)
        elif model_name == 'window_aware_cnn_transformer': model = WindowAware_CNN_Transformer(n_channels, n_time, n_classes, n_windows=n_windows, **model_params)
        elif model_name == 'multitask_deep_cnn_lstm': model = MultiTask_DeepCNN_LSTM(n_channels, n_time, n_classes, n_windows=n_windows, **model_params)
        
        # FIXED: Create validation split for training, then use full train+val for final model
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        tr_idx, val_idx = next(val_splitter.split(X_train, y_train, groups_train))
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        y_window_tr = y_window_train[tr_idx] if use_window else None
        y_window_val_inner = y_window_train[val_idx] if use_window else None
        
        is_wa = model_name in ['window_aware_deep_cnn_lstm', 'window_aware_cnn_transformer']
        is_mt = model_name == 'multitask_deep_cnn_lstm'
        
        if is_wa or is_mt:
            # FIXED: Use FULL training set (X_train) for final training
            train_dataset_final = TensorDataset(X_train, torch.LongTensor(y_window_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(X_val, torch.LongTensor(y_window_val_inner), torch.LongTensor(y_val))
            test_dataset = TensorDataset(X_test, torch.LongTensor(y_window_test), torch.LongTensor(y_test))
        else:
            # FIXED: Use FULL training set (X_train) for final training
            train_dataset_final = TensorDataset(X_train, torch.LongTensor(y_train))
            val_dataset = TensorDataset(X_val, torch.LongTensor(y_val))
            test_dataset = TensorDataset(X_test, torch.LongTensor(y_test))
            
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        # FIXED: train_loader now uses FULL training set
        train_loader = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # FIXED: Train on FULL training set (y_train, not y_tr)
        model, _, info = train_one_model(
            model, train_loader, val_loader, y_train, device, 
            epochs=args.epochs, patience=args.patience, class_names=le.classes_.tolist(),
            model_name=model_name.upper(), verbose=True, **training_params
        )
        
        metrics = evaluate_model(model, test_loader, device, le, model_name=model_name.upper(), window_le=window_le)
        metrics['training_info'] = info
        metrics['best_hyperparameters'] = best_params
        results[model_name] = metrics
        
        # FIXED: Cleanup after each model
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        
    output_path = os.path.join(proj_root, args.output_file)
    with open(output_path, 'w') as f:
        json.dump({'results': results, 'best_hyperparameters': best_hyperparams}, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
#!/usr/bin/env python3
"""
EEG Pain Classification (1000 Hz) - Paper Version
Baseline vs MTL (+ optional window-aware ablation)

Adds:
  - Bootstrap 95% CI for Macro-F1 and Balanced Accuracy
  - Bootstrap CI for delta improvements (MTL - Baseline)
  - Quantitative interpretability proxy: ERP Alignment Ratio
    using grad*input saliency energy (per-window)

Author: (your team)
"""

import os
import argparse
import json
import random
import time
import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple, Dict, Any, List

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score
)
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✓ Seed set to {seed}")


# =============================================================================
# Bootstrap CI helpers
# =============================================================================
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    stats = np.array(stats)
    lo = np.quantile(stats, alpha / 2)
    hi = np.quantile(stats, 1 - alpha / 2)
    return float(np.mean(stats)), float(lo), float(hi)


def bootstrap_diff_ci(y_true, pred_a, pred_b, metric_fn, n_boot=2000, seed=42, alpha=0.05):
    """
    CI of (metric(pred_b) - metric(pred_a))
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(metric_fn(y_true[idx], pred_b[idx]) - metric_fn(y_true[idx], pred_a[idx]))

    diffs = np.array(diffs)
    lo = np.quantile(diffs, alpha / 2)
    hi = np.quantile(diffs, 1 - alpha / 2)
    return float(np.mean(diffs)), float(lo), float(hi)


# =============================================================================
# Data loading
# =============================================================================
def find_data_root(proj_root: str, dataset_name: str = "Data") -> str:
    candidate = os.path.join(proj_root, dataset_name)
    return candidate if os.path.isdir(candidate) else proj_root


def load_index(root: str) -> pd.DataFrame:
    index_path = os.path.join(root, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.csv not found at {index_path}")

    df = pd.read_csv(index_path)

    if "reject_flag" in df.columns:
        before = len(df)
        df = df[df["reject_flag"] == False].copy()
        print(f"Filtered rejected epochs: {before - len(df)} removed")

    # Standardize column name for NPZ
    if "npz_file" not in df.columns:
        if "path" in df.columns:
            df = df.copy()
            df["npz_file"] = df["path"]
            print("Using 'path' as 'npz_file'")
        else:
            raise ValueError("Need 'npz_file' or 'path' column in index.csv")

    return df


def get_most_common_channels(df: pd.DataFrame) -> int:
    if "n_channels" in df.columns:
        cnt = Counter(df["n_channels"].dropna().astype(int))
        return cnt.most_common(1)[0][0] if cnt else 64
    return 64


class EEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, most_ch: int,
                 return_window: bool = False, window_le: LabelEncoder = None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.most_ch = most_ch
        self.return_window = return_window
        self.window_le = window_le

        if self.return_window and self.window_le and "window" in self.df.columns:
            self.window_indices = self.window_le.transform(self.df["window"])
        else:
            self.window_indices = np.zeros(len(self.df), dtype=int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npz_file = row.get("npz_file", "")
        npz_path = os.path.join(self.root, "npz", npz_file)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ missing: {npz_path}")

        with np.load(npz_path, allow_pickle=True) as data:
            X = data["X"]  # (channels, time)

        ch, T = X.shape
        if ch < self.most_ch:
            pad = np.zeros((self.most_ch - ch, T), dtype=X.dtype)
            X = np.vstack([X, pad])
        elif ch > self.most_ch:
            X = X[:self.most_ch, :]

        # z-score per channel
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mean) / std

        w_idx = self.window_indices[idx] if self.return_window else 0
        return torch.FloatTensor(X), int(w_idx)


def load_task_data(df: pd.DataFrame, task: str) -> Tuple[np.ndarray, LabelEncoder, pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Returns:
      y (pain labels), le (pain encoder), df_used, y_window, window_le
    """
    le = LabelEncoder()
    window_le = LabelEncoder()

    # window labels
    if "window" in df.columns:
        window_le.fit(df["window"])
        y_window = window_le.transform(df["window"])
    else:
        df = df.copy()
        df["window"] = "unknown"
        window_le.fit(["unknown"])
        y_window = np.zeros(len(df), dtype=int)

    if task == "pain_threshold":
        # No Pain = {none, low}, Pain = {mid, high, extreme}
        df = df.copy()

        def threshold_label(r):
            rb = str(r).strip().lower()
            if rb in ["none", "low", "no_pain", "no"]:
                return "no_significant_pain"
            return "significant_pain"

        df["threshold_label"] = df["rating_bin"].apply(threshold_label)
        y = le.fit_transform(df["threshold_label"])
    else:
        raise ValueError("For paper version, use --task pain_threshold")

    return y, le, df, y_window, window_le


def load_all_segments(df: pd.DataFrame, root: str, most_ch: int,
                      return_window: bool, window_le: LabelEncoder):
    dataset = EEGDataset(df, root, most_ch, return_window=return_window, window_le=window_le)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_X, all_W = [], []
    for xb, wb in tqdm(loader, desc="Loading segments", unit="batch"):
        all_X.append(xb)
        if return_window:
            all_W.append(wb)

    X = torch.cat(all_X, dim=0)
    W = torch.cat(all_W, dim=0) if return_window else torch.zeros(len(X), dtype=torch.long)
    return X, W


# =============================================================================
# Models (baseline / window-aware / multitask)
# =============================================================================
class DeepCNN_LSTM(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2,
                 cnn_filters: List[int] = [32, 64, 128],
                 lstm_hidden: int = 192, lstm_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels // 8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))

        time_reduced = n_time // 8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))

        self.lstm = nn.LSTM(
            cnn_filters[2], lstm_hidden, batch_first=True,
            num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = self.adaptive_pool(x)

        x = x.squeeze(2).transpose(1, 2)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h)


class WindowAware_DeepCNN_LSTM(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, n_windows: int = 3,
                 cnn_filters: List[int] = [32, 64, 128], lstm_hidden: int = 192,
                 lstm_layers: int = 2, window_embed_dim: int = 16, dropout: float = 0.4):
        super().__init__()
        self.window_embedding = nn.Embedding(n_windows, window_embed_dim)

        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels // 8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))

        time_reduced = n_time // 8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))

        lstm_input_dim = cnn_filters[2] + window_embed_dim
        self.lstm = nn.LSTM(
            lstm_input_dim, lstm_hidden, batch_first=True,
            num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2 + window_embed_dim, n_classes)

    def forward(self, x, window_idx):
        window_emb = self.window_embedding(window_idx)

        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = self.adaptive_pool(x)

        x = x.squeeze(2).transpose(1, 2)
        seq_len = x.size(1)
        window_emb_expanded = window_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, window_emb_expanded], dim=2)

        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = torch.cat([h, window_emb], dim=1)
        h = self.dropout(h)
        return self.fc(h)


class MultiTask_DeepCNN_LSTM(nn.Module):
    """
    MTL: shared encoder, two heads:
      - pain (main)
      - window (auxiliary)
    """
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, n_windows: int = 3,
                 cnn_filters: List[int] = [32, 64, 128],
                 lstm_hidden: int = 192, lstm_layers: int = 2, dropout: float = 0.4):
        super().__init__()

        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels // 8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))

        time_reduced = n_time // 8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))

        self.lstm = nn.LSTM(
            cnn_filters[2], lstm_hidden, batch_first=True,
            num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

        self.fc_pain = nn.Linear(lstm_hidden * 2, n_classes)
        self.fc_window = nn.Linear(lstm_hidden * 2, n_windows)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = self.adaptive_pool(x)

        x = x.squeeze(2).transpose(1, 2)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = self.dropout(h)

        return self.fc_pain(h), self.fc_window(h)


# =============================================================================
# Training / Evaluation
# =============================================================================
def class_weights_from_y(y: np.ndarray) -> torch.FloatTensor:
    unique, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(unique) * counts)
    # weights aligned by class index
    full = np.zeros(int(unique.max()) + 1, dtype=np.float32)
    for cls, w in zip(unique, weights):
        full[int(cls)] = w
    return torch.FloatTensor(full)


def train_one_model(model: nn.Module,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    y_train: np.ndarray,
                    device: str,
                    epochs: int = 50,
                    patience: int = 15,
                    lr: float = 5e-4,
                    weight_decay: float = 1e-5,
                    model_name: str = "model") -> Tuple[nn.Module, Dict[str, Any]]:

    model.to(device)

    w = class_weights_from_y(y_train).to(device)
    criterion_pain = nn.CrossEntropyLoss(weight=w)

    is_mtl = isinstance(model, MultiTask_DeepCNN_LSTM)
    is_wa = isinstance(model, WindowAware_DeepCNN_LSTM)

    if is_mtl:
        criterion_window = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()

            if is_wa:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                logits = model(xb, wb)
                loss = criterion_pain(logits, yb)

            elif is_mtl:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                pain_logits, win_logits = model(xb)
                loss_p = criterion_pain(pain_logits, yb)
                loss_w = criterion_window(win_logits, wb)
                loss = loss_p + loss_w

            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                logits = model(xb)
                loss = criterion_pain(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if is_wa:
                    xb, wb, yb = batch
                    xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                    logits = model(xb, wb)
                    loss = criterion_pain(logits, yb)

                elif is_mtl:
                    xb, wb, yb = batch
                    xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                    pain_logits, win_logits = model(xb)
                    loss_p = criterion_pain(pain_logits, yb)
                    loss_w = criterion_window(win_logits, wb)
                    loss = loss_p + loss_w

                else:
                    xb, yb = batch[0].to(device), batch[-1].to(device)
                    logits = model(xb)
                    loss = criterion_pain(logits, yb)

                val_losses.append(loss.item())

        vloss = float(np.mean(val_losses))
        scheduler.step(vloss)

        if vloss < best_val_loss - 1e-4:
            best_val_loss = vloss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"[{model_name}] epoch {epoch+1}/{epochs}  train={np.mean(train_losses):.4f}  val={vloss:.4f}")

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"✓ {model_name} best epoch = {best_epoch} (val loss {best_val_loss:.4f})")

    info = {"best_epoch": best_epoch, "best_val_loss": best_val_loss}
    return model, info


def compute_grad_input_saliency(model, loader, device, is_window_aware=False, is_multitask=False):
    """
    |grad * input| -> mean over channels => saliency over time
    Returns:
      saliency_all [N, T]
      windows_all  [N] or None
    """

    # IMPORTANT: cuDNN RNN backward requires train mode.
    was_training = model.training
    model.train()

    saliency_list = []
    windows_list = []

    for batch in tqdm(loader, desc="Saliency (grad*input)", unit="batch"):
        if is_window_aware or is_multitask:
            xb, wb, yb = batch
            xb = xb.to(device)
            wb_dev = wb.to(device)
            windows_list.append(wb.detach().cpu().numpy())
        else:
            xb, yb = batch[0], batch[-1]
            xb = xb.to(device)
            wb_dev = None

        xb = xb.requires_grad_(True)

        if is_window_aware:
            logits = model(xb, wb_dev)
        elif is_multitask:
            pain_logits, _ = model(xb)
            logits = pain_logits
        else:
            logits = model(xb)

        pred = logits.argmax(dim=1)
        chosen = logits.gather(1, pred.view(-1, 1)).sum()

        model.zero_grad(set_to_none=True)
        if xb.grad is not None:
            xb.grad.zero_()
        chosen.backward()

        attr = (xb.grad * xb).detach().abs()   # [B,C,T]
        sal = attr.mean(dim=1).cpu().numpy()   # [B,T]
        saliency_list.append(sal)

    # restore original mode
    model.train(was_training)

    saliency_all = np.concatenate(saliency_list, axis=0)
    windows_all = np.concatenate(windows_list, axis=0) if len(windows_list) else None
    return saliency_all, windows_all


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: str,
                   le: LabelEncoder,
                   window_le: LabelEncoder,
                   model_name: str) -> Dict[str, Any]:

    model.eval()
    is_wa = isinstance(model, WindowAware_DeepCNN_LSTM)
    is_mtl = isinstance(model, MultiTask_DeepCNN_LSTM)

    all_preds, all_targets, all_probs = [], [], []
    all_windows = []
    all_window_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {model_name}", unit="batch"):
            if is_wa:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                logits = model(xb, wb)
                all_windows.append(wb.detach().cpu().numpy())

            elif is_mtl:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                pain_logits, win_logits = model(xb)
                logits = pain_logits
                all_windows.append(wb.detach().cpu().numpy())
                all_window_preds.append(win_logits.argmax(dim=1).detach().cpu().numpy())

            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                logits = model(xb)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).detach().cpu().numpy()

            all_preds.append(preds)
            all_targets.append(yb.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_probs = np.concatenate(all_probs)

    acc = float((y_pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)

    # Bootstrap CI
    macro_f1_fn = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    bal_acc_fn  = lambda yt, yp: balanced_accuracy_score(yt, yp)

    f1_mean, f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, macro_f1_fn, n_boot=2000, seed=42)
    ba_mean, ba_lo, ba_hi = bootstrap_ci(y_true, y_pred, bal_acc_fn,  n_boot=2000, seed=42)

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "report": report,
        "class_names": le.classes_.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "bootstrap_ci": {
            "macro_f1": {"mean": f1_mean, "ci95": [f1_lo, f1_hi]},
            "balanced_accuracy": {"mean": ba_mean, "ci95": [ba_lo, ba_hi]},
            "n_boot": 2000
        }
    }

    # window indices + per-window metrics if available
    windows = np.concatenate(all_windows) if len(all_windows) else None
    if windows is not None and window_le is not None:
        metrics["windows"] = windows.tolist()

        per_window = {}
        for w_idx, w_name in enumerate(window_le.classes_):
            mask = (windows == w_idx)
            if np.any(mask):
                per_window[w_name] = {
                    "n": int(mask.sum()),
                    "acc": float((y_pred[mask] == y_true[mask]).mean()),
                    "f1_macro": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))
                }
        metrics["per_window_pain"] = per_window

    # auxiliary window head metrics (MTL only)
    if is_mtl and windows is not None and len(all_window_preds):
        y_win_true = windows
        y_win_pred = np.concatenate(all_window_preds)
        metrics["window_task"] = {
            "accuracy": float((y_win_pred == y_win_true).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_win_true, y_win_pred)),
            "f1_macro": float(f1_score(y_win_true, y_win_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_win_true, y_win_pred).tolist(),
            "report": classification_report(y_win_true, y_win_pred, target_names=window_le.classes_, zero_division=0),
        }

    # Quantitative interpretability proxy: ERP alignment from saliency energy
    if windows is not None and window_le is not None:
        saliency, win_idx = compute_grad_input_saliency(
            model, test_loader, device,
            is_window_aware=is_wa,
            is_multitask=is_mtl
        )
        energy = saliency.mean(axis=1)  # one value per sample

        per_win_energy = {}
        for i, name in enumerate(window_le.classes_):
            m = (win_idx == i)
            if np.any(m):
                per_win_energy[name] = {
                    "mean_energy": float(energy[m].mean()),
                    "std_energy": float(energy[m].std()),
                    "n": int(m.sum())
                }

        def find_idx_by_key(names, key):
            key = key.lower()
            for i, n in enumerate(names):
                if key in n.lower():
                    return i
            return None

        erp_i = find_idx_by_key(window_le.classes_, "erp")
        base_i = find_idx_by_key(window_le.classes_, "base")
        post_i = find_idx_by_key(window_le.classes_, "post")

        erp_align_ratio = None
        if erp_i is not None and base_i is not None and post_i is not None:
            e_erp = float(energy[win_idx == erp_i].mean())
            e_base = float(energy[win_idx == base_i].mean())
            e_post = float(energy[win_idx == post_i].mean())
            erp_align_ratio = float(e_erp / (e_base + e_post + 1e-8))

        metrics["interpretability_quant"] = {
            "method": "grad_x_input_mean_over_channels_then_mean_over_time",
            "per_window_saliency_energy": per_win_energy,
            "erp_alignment_ratio": erp_align_ratio
        }

    return metrics


# =============================================================================
# Main (baseline vs MTL + optional ablation)
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pain_threshold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--data_root", type=str, default="/home/asatsan2/Projects/EEG-Pain-Estimation/data")
    ap.add_argument("--output_file", type=str, default="paper_results_baseline_vs_mtl.json")

    # Choose what to run
    ap.add_argument("--run_window_aware_ablation", action="store_true",
                    help="Also run window-aware model (uses window label as input)")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cwd = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(cwd, ".."))
    root = args.data_root if args.data_root else find_data_root(proj_root, "Data")

    df = load_index(root)
    most_ch = get_most_common_channels(df)

    y_all, le, df, y_window_all, window_le = load_task_data(df, args.task)
    groups = df["participant"].values

    # Load all X and window indices
    X_all, W_all = load_all_segments(df, root, most_ch, return_window=True, window_le=window_le)
    n_channels, n_time = X_all.shape[1], X_all.shape[2]
    n_classes = len(le.classes_)
    n_windows = len(window_le.classes_) if window_le else 0

    # Train/test split (subject-wise)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups))

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    W_train, W_test = W_all[train_idx], W_all[test_idx]
    groups_train = groups[train_idx]

    # Inner val split from train (subject-wise)
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, val_idx = next(val_splitter.split(X_train, y_train, groups_train))

    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    W_tr, W_val = W_train[tr_idx], W_train[val_idx]

    # Loaders
    bs = args.batch_size
    gen = torch.Generator(); gen.manual_seed(args.seed)

    # baseline loader doesn't need W, but we can still package it consistently
    train_loader_base = DataLoader(TensorDataset(X_train, torch.LongTensor(y_train)),
                                   batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, generator=gen)
    val_loader_base = DataLoader(TensorDataset(X_val, torch.LongTensor(y_val)),
                                 batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_base = DataLoader(TensorDataset(X_test, torch.LongTensor(y_test)),
                                  batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # MTL needs (X, W, y)
    train_loader_mtl = DataLoader(TensorDataset(X_train, W_train.long(), torch.LongTensor(y_train)),
                                  batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, generator=gen)
    val_loader_mtl = DataLoader(TensorDataset(X_val, W_val.long(), torch.LongTensor(y_val)),
                                batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_mtl = DataLoader(TensorDataset(X_test, W_test.long(), torch.LongTensor(y_test)),
                                 batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    results = {}

    # -------------------------
    # 1) Baseline
    # -------------------------
    baseline = DeepCNN_LSTM(n_channels=n_channels, n_time=n_time, n_classes=n_classes,
                            cnn_filters=[32, 64, 128], lstm_hidden=192, lstm_layers=2, dropout=0.4)

    baseline, info_base = train_one_model(
        baseline, train_loader_base, val_loader_base, y_train,
        device=device, epochs=args.epochs, patience=args.patience,
        lr=args.lr, weight_decay=args.weight_decay, model_name="BASELINE"
    )
    metrics_base = evaluate_model(baseline, test_loader_base, device, le, window_le, "BASELINE")
    metrics_base["training_info"] = info_base
    results["baseline_deep_cnn_lstm"] = metrics_base

    del baseline
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------
    # 2) Proposed MTL
    # -------------------------
    mtl = MultiTask_DeepCNN_LSTM(n_channels=n_channels, n_time=n_time, n_classes=n_classes, n_windows=n_windows,
                                cnn_filters=[32, 64, 128], lstm_hidden=192, lstm_layers=2, dropout=0.4)

    mtl, info_mtl = train_one_model(
        mtl, train_loader_mtl, val_loader_mtl, y_train,
        device=device, epochs=args.epochs, patience=args.patience,
        lr=args.lr, weight_decay=args.weight_decay, model_name="MTL"
    )
    metrics_mtl = evaluate_model(mtl, test_loader_mtl, device, le, window_le, "MTL")
    metrics_mtl["training_info"] = info_mtl
    results["mtl_deep_cnn_lstm"] = metrics_mtl

    # -------------------------
    # 3) Delta CI (MTL - Baseline)
    # -------------------------
    y_true = np.array(metrics_base["y_true"])
    pred_base = np.array(metrics_base["y_pred"])
    pred_mtl = np.array(metrics_mtl["y_pred"])

    macro_f1_fn = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    bal_acc_fn  = lambda yt, yp: balanced_accuracy_score(yt, yp)

    d_f1_mean, d_f1_lo, d_f1_hi = bootstrap_diff_ci(y_true, pred_base, pred_mtl, macro_f1_fn, n_boot=2000, seed=42)
    d_ba_mean, d_ba_lo, d_ba_hi = bootstrap_diff_ci(y_true, pred_base, pred_mtl, bal_acc_fn,  n_boot=2000, seed=42)

    results["delta_mtl_minus_baseline_bootstrap_ci"] = {
        "macro_f1": {"mean": d_f1_mean, "ci95": [d_f1_lo, d_f1_hi]},
        "balanced_accuracy": {"mean": d_ba_mean, "ci95": [d_ba_lo, d_ba_hi]},
        "n_boot": 2000
    }

    print("\nΔ (MTL - Baseline) Bootstrap 95% CI:")
    print(f"  Macro-F1 Δ: {d_f1_mean:.4f} [{d_f1_lo:.4f}, {d_f1_hi:.4f}]")
    print(f"  BalAcc  Δ : {d_ba_mean:.4f} [{d_ba_lo:.4f}, {d_ba_hi:.4f}]")

    # -------------------------
    # 4) Optional ablation: window-aware (uses window label as input)
    # -------------------------
    if args.run_window_aware_ablation:
        wa = WindowAware_DeepCNN_LSTM(n_channels=n_channels, n_time=n_time, n_classes=n_classes, n_windows=n_windows,
                                      cnn_filters=[32, 64, 128], lstm_hidden=192, lstm_layers=2,
                                      window_embed_dim=16, dropout=0.4)
        wa, info_wa = train_one_model(
            wa, train_loader_mtl, val_loader_mtl, y_train,
            device=device, epochs=args.epochs, patience=args.patience,
            lr=args.lr, weight_decay=args.weight_decay, model_name="WINDOW_AWARE"
        )
        metrics_wa = evaluate_model(wa, test_loader_mtl, device, le, window_le, "WINDOW_AWARE")
        metrics_wa["training_info"] = info_wa
        results["window_aware_deep_cnn_lstm"] = metrics_wa

    # Save
    out_path = os.path.join(proj_root, args.output_file)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()

"""
python /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/ml.py 

screen -S eeg_dl -L -Logfile /home/asatsan2/Projects/EEG-Pain-Estimation/results/dl_run.log \
python /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/dl.py


screen -S painjob3 -dm bash -c "
source ~/.bashrc
conda activate eeg
python /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/dl_new.py \
       --task none_vs_pain \
        --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data \
        --grid-epochs 20 \
        --grid-patience 7 \
        --epochs 30 \
        --patience 10 \
        --seed 42 \
  >>  train_dl_new.log 2>&1
"

"""