


#!/usr/bin/env python3
"""
Deep Learning Models Training with Grid Search for 1000 Hz EEG Data

This script trains multiple deep learning models on the original 1000 Hz EEG dataset
with comprehensive hyperparameter grid search to find optimal parameters for each model.
All models are trained with full reproducibility using random seeds.

Author: DILANJAN DK
Email: DDIYABAL@UWO.CA

Models Included:
    - CNN2D: 2D Convolutional Neural Network
    - LSTM: Bidirectional Long Short-Term Memory
    - Transformer: Transformer Encoder
    - CNN-Transformer: Hybrid CNN-Transformer architecture
    - DeepCNN-LSTM: Deep CNN followed by LSTM

Usage Examples:
    # Train all models with grid search on none_vs_pain task
    python scripts/dl_train_1000hz_gridsearch.py --task none_vs_pain

    # Train specific models with grid search
    python scripts/dl_train_1000hz_gridsearch.py --task none_vs_pain --models cnn lstm transformer

    # Train without grid search (use best known parameters)
    python scripts/dl_train_1000hz_gridsearch.py --task none_vs_pain --no-grid-search

    # Custom seed for reproducibility
    python scripts/dl_train_1000hz_gridsearch.py --task none_vs_pain --seed 123

    # Quick test run with fewer samples
    python scripts/dl_train_1000hz_gridsearch.py --task none_vs_pain --quick --quick_n_per_subj 50

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
__version__ = "1.0.0"

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.
    
    This function ensures that all random number generators (Python, NumPy, PyTorch, CUDA)
    are seeded consistently to enable reproducible results across runs.
    
    Args:
        seed: Random seed value (default: 42)
    
    Note:
        Setting deterministic=True for CuDNN may reduce performance but ensures reproducibility.
        This is essential for scientific reproducibility.
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
    # Note: This may reduce performance but ensures reproducibility
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
    """
    Find the data root directory for the specified dataset.
    
    Args:
        proj_root: Project root directory
        dataset_name: Name of the dataset directory (default: 'Data' for 1000 Hz)
    
    Returns:
        Path to the data root directory
    """
    candidate = os.path.join(proj_root, dataset_name)
    return candidate if os.path.isdir(candidate) else proj_root


def load_index(root: str) -> pd.DataFrame:
    """
    Load index.csv file containing metadata about EEG segments.
    
    Args:
        root: Root directory containing index.csv
    
    Returns:
        DataFrame with metadata, filtered to exclude rejected epochs
    
    Raises:
        FileNotFoundError: If index.csv is not found
    """
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
    """
    Get the most common number of channels in the dataset.
    
    Args:
        df: DataFrame with n_channels column
    
    Returns:
        Most common number of channels (default: 64 if not found)
    """
    if 'n_channels' in df.columns:
        cnt = Counter(df['n_channels'].dropna().astype(int))
        return cnt.most_common(1)[0][0] if cnt else 64 
    return 64

class EEGDataset(Dataset):
    """
    Dataset for loading EEG segments from NPZ files.
    
    This dataset handles:
    - Channel padding/truncation to a common number of channels
    - Per-channel Z-score normalization
    - Loading from NPZ files stored in the npz directory
    """
    
    def __init__(self, df: pd.DataFrame, root: str, most_ch: int, return_window: bool = False, window_le: LabelEncoder = None):
        """  
        Initialize EEG dataset.
        
        Args:
            df: DataFrame with metadata (must have 'npz_file' or 'path' column)
            root: Root directory containing npz subdirectory
            most_ch: Target number of channels (for padding/truncation)
            return_window: Whether to return window index
            window_le: LabelEncoder for windows
        """
        self.df = df.reset_index(drop=True)
        self.root = root
        self.most_ch = most_ch
        self.return_window = return_window
        self.window_le = window_le
        
        # Pre-encode windows if needed 
        if self.return_window and self.window_le and 'window' in self.df.columns:
            self.window_indices = self.window_le.transform(self.df['window'])
        else: 
            self.window_indices = np.zeros(len(self.df), dtype=int)
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:   
        """  
        Load and preprocess a single EEG segment.
        
        Args:
            idx: Index of the sample to load
        
        Returns:
            Tuple of (preprocessed EEG tensor, window_idx)
            EEG tensor shape: (channels, time_points)
        """
        row = self.df.iloc[idx]
        
        # Handle both 'npz_file' and 'path' column names
        npz_file = row.get('npz_file', row.get('path', ''))
        npz_path = os.path.join(self.root, 'npz', npz_file)
        
        # Load EEG data from NPZ file
        with np.load(npz_path, allow_pickle=True) as data:
            X = data['X']  # Shape: (channels, time)
            
        # Pad or truncate channels to most_common_channels
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
    
    Raises:
        ValueError: If task is unknown or no samples found
    """
    le = LabelEncoder()
    window_le = LabelEncoder()
    y_window = np.zeros(len(df), dtype=int)
    
    # Process window labels first if available
    if 'window' in df.columns:
        y_window = window_le.fit_transform(df['window'])
    else:
        print("⚠ 'window' column not found in dataframe. Window-aware models will not work correctly.")
        # Create dummy window labels if missing
        df['window'] = 'unknown'
        window_le.fit(['unknown'])
        y_window = np.zeros(len(df), dtype=int)
    
    if task == 'pain_5class':
        y = le.fit_transform(df['rating_bin'])
    elif task == 'none_vs_pain':
        df = df.copy()
        df['binary'] = df['rating_bin'].apply(lambda x: 'none' if x == 'none' else 'pain')
        y = le.fit_transform(df['binary'])
    elif task == 'pain_only':
        pain_df = df[df['rating_bin'] != 'none'].copy()
        if len(pain_df) == 0:
            raise ValueError("No pain samples found")
        y = le.fit_transform(pain_df['rating_bin'])
        df = pain_df
        # Re-encode windows for filtered dataframe
        y_window = window_le.transform(df['window'])
    elif task == 'pain_threshold':
        # Binary: no_significant_pain (none+low) vs significant_pain (moderate+high+severe)
        df = df.copy()
        def threshold_label(rating_bin):
            if rating_bin in ['none', 'low']:
                return 'no_significant_pain'
            else:
                return 'significant_pain'
        df['threshold_label'] = df['rating_bin'].apply(threshold_label)
        y = le.fit_transform(df['threshold_label'])
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return y, le, df, y_window, window_le

def load_all_segments(df: pd.DataFrame, root: str, most_ch: int, return_window: bool = False, window_le: LabelEncoder = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load all EEG segments into memory for faster training.
    
    Args:
        df: DataFrame with metadata
        root: Root directory containing npz subdirectory
        most_ch: Target number of channels
        return_window: Whether to load window indices
        window_le: LabelEncoder for windows
    
    Returns:
        Tuple of (Tensor of EEG segments, Tensor of window indices)
    """
    print(f"\n{'='*70}")
    print(" DATA LOADING")
    print(f"{'='*70}")
    print(f"Total samples to load: {len(df):,}")
    print(f"Most common channels: {most_ch}")
    print(f"NPZ directory: {os.path.join(root, 'npz')}")
    
    # Update df with npz_file column if needed
    if 'npz_file' not in df.columns and 'path' in df.columns:
        df = df.copy()
        df['npz_file'] = df['path']
        print("Using 'path' column as 'npz_file'")
    else:
        print("Using 'npz_file' column")
    
    # Check channel distribution
    if 'n_channels' in df.columns:
        ch_counts = Counter(df['n_channels'].dropna().astype(int))
        print(f"\nChannel distribution:")
        for ch, count in sorted(ch_counts.items())[:10]:
            print(f"  {ch} channels: {count:,} samples")
        if len(ch_counts) > 10:
            print(f"  ... and {len(ch_counts) - 10} more channel configurations")
    
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
    """
    Window-aware Deep CNN-LSTM for pain threshold detection.
    - Deep CNN for spatial feature extraction from EEG
    - Window embedding to encode temporal context (baseline/erp/post)
    - Bidirectional LSTM for temporal modeling
    - Fusion of EEG features and window embeddings
    """
    
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, n_windows: int = 3,
                 cnn_filters: List[int] = [32, 64, 128], 
                 lstm_hidden: int = 192, 
                 lstm_layers: int = 2,
                 window_embed_dim: int = 16,
                 dropout: float = 0.4):
        super().__init__()
        
        # Window embedding
        self.window_embedding = nn.Embedding(n_windows, window_embed_dim)
        
        # Deep CNN for hierarchical spatial features
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels//8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))
        
        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))
        
        # Calculate output time dimension after pooling
        time_reduced = n_time // 8
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))
        
        # LSTM input: CNN features + window embedding broadcasted
        lstm_input_dim = cnn_filters[2] + window_embed_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden, batch_first=True, 
                           num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0, 
                           bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier combines LSTM output and window embedding
        self.fc = nn.Linear(lstm_hidden * 2 + window_embed_dim, n_classes)
    
    def forward(self, x, window_idx):
        # x: (batch, channels, time)
        # window_idx: (batch,) - integer indices for window type
        batch_size = x.size(0)
        
        # Get window embeddings
        window_emb = self.window_embedding(window_idx)  # (batch, window_embed_dim)
        
        # Deep CNN feature extraction
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.adaptive_pool(x)  # (batch, cnn_filters[2], 1, time_reduced)
        
        # Reshape for LSTM: (batch, time_reduced, cnn_filters[2])
        x = x.squeeze(2).transpose(1, 2)
        seq_len = x.size(1)
        
        # Concatenate window embedding to each time step
        window_emb_expanded = window_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, window_embed_dim)
        x = torch.cat([x, window_emb_expanded], dim=2)  # (batch, seq_len, cnn_filters[2] + window_embed_dim)
        
        # LSTM temporal modeling
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, lstm_hidden*2)
        
        # Concatenate LSTM output with window embedding for final classification
        h = torch.cat([h, window_emb], dim=1)  # (batch, lstm_hidden*2 + window_embed_dim)
        
        # Classification
        h = self.dropout(h)
        return self.fc(h)

class WindowAware_CNN_Transformer(nn.Module):
    """
    Window-aware CNN-Transformer for pain threshold detection.
    - Deep CNN for spatial feature extraction from EEG
    - Window embedding to encode temporal context (baseline/erp/post)
    - Transformer encoder for temporal attention
    - Fusion of EEG features and window embeddings
    """
    
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, n_windows: int = 3,
                 cnn_filters: List[int] = [32, 64, 128],
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 window_embed_dim: int = 16,
                 dropout: float = 0.3,
                 time_downsample: int = 4):
        super().__init__()
        
        # Window embedding
        self.window_embedding = nn.Embedding(n_windows, window_embed_dim)
        
        # Deep CNN for hierarchical spatial features
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(n_channels//8, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))
        
        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))
        
        # Calculate output time dimension after pooling
        time_reduced = n_time // 8
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))
        
        # Additional time downsampling for transformer
        self.time_downsample = time_downsample
        time_for_transformer = time_reduced // time_downsample
        
        # Project CNN features to transformer dimension
        self.cnn_to_transformer = nn.Linear(cnn_filters[2] + window_embed_dim, d_model)
        
        # Positional encoding
        max_len = time_for_transformer + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier combines Transformer output and window embedding
        self.fc = nn.Linear(d_model + window_embed_dim, n_classes)
    
    def forward(self, x, window_idx):
        # x: (batch, channels, time)
        # window_idx: (batch,) - integer indices for window type
        batch_size = x.size(0)
        
        # Get window embeddings
        window_emb = self.window_embedding(window_idx)  # (batch, window_embed_dim)
        
        # Deep CNN feature extraction
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.adaptive_pool(x)  # (batch, cnn_filters[2], 1, time_reduced)
        
        # Reshape: (batch, time_reduced, cnn_filters[2])
        x = x.squeeze(2).transpose(1, 2)
        
        # Downsample time dimension
        T = x.size(1)
        T_down = (T // self.time_downsample) * self.time_downsample
        x = x[:, :T_down, :]
        x = x.reshape(batch_size, T_down // self.time_downsample, -1, self.time_downsample)
        x = x.mean(dim=3)  # Average pool over downsample factor
        
        # Concatenate window embedding to each time step
        window_emb_expanded = window_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, window_emb_expanded], dim=2)  # (batch, seq_len, cnn_filters[2] + window_embed_dim)
        
        # Project to transformer dimension
        x = self.cnn_to_transformer(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer temporal attention
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Concatenate with window embedding for final classification
        x = torch.cat([x, window_emb], dim=1)  # (batch, d_model + window_embed_dim)
        
        # Classification
        x = self.dropout(x)
        return self.fc(x)

class CNN2D(nn.Module):
    """
    2D Convolutional Neural Network for EEG classification.
    
    Architecture:
        - 3 convolutional layers with increasing filters (32→64→128)
        - Batch normalization and max pooling
        - Adaptive average pooling for variable input sizes
        - Dropout and fully connected layer for classification
    """
    
    def __init__(self, n_channels: int, n_time: int, n_classes: int, dropout: float = 0.5):
        """
        Initialize CNN2D model.
        
        Args:
            n_channels: Number of EEG channels
            n_time: Number of time points
            n_classes: Number of output classes
            dropout: Dropout probability
        """
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
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Output logits of shape (batch, n_classes)
        """
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
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
    """
    Bidirectional LSTM model for EEG classification.
    
    Architecture:
        - Multi-layer bidirectional LSTM
        - Dropout for regularization
        - Fully connected layer for classification
    """
    
    def __init__(self, n_channels: int, n_classes: int, hidden: int = 128, 
                 dropout: float = 0.5, bidirectional: bool = True):
        """
        Initialize LSTM model.
        
        Args:
            n_channels: Number of EEG channels (input features)
            n_classes: Number of output classes
            hidden: Hidden state size
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True, num_layers=2, 
                           dropout=dropout if dropout > 0 else 0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        fc_in = hidden * 2 if bidirectional else hidden
        self.fc = nn.Linear(fc_in, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Output logits of shape (batch, n_classes)
        """
        x = x.transpose(1, 2)  # (batch, time, channels)
        _, (h, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h = h[-1]
        h = self.dropout(h)
        return self.fc(h)

class TransformerModel(nn.Module):
    """
    Transformer encoder model for EEG classification.
    
    Architecture:
        - Input downsampling and projection
        - Positional encoding
        - Multi-layer Transformer encoder
        - Global average pooling and classification
    """
    
    def __init__(self, n_channels: int, n_time: int, n_classes: int, 
                 d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout: float = 0.3):
        """
        Initialize Transformer model.
        
        Args:
            n_channels: Number of EEG channels
            n_time: Number of time points
            n_classes: Number of output classes
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        
        self.downsample = 4
        self.proj = nn.Linear(n_channels * self.downsample, d_model)
        
        max_len = n_time // self.downsample + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, 
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Output logits of shape (batch, n_classes)
        """
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
    """
    Hybrid CNN-Transformer model for EEG classification.
    
    Architecture:
        - CNN for spatial feature extraction
        - Transformer for temporal attention
        - Combines spatial and temporal modeling
    """
    
    def __init__(self, n_channels: int, n_time: int, n_classes: int,
                 cnn_filters: int = 64, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.3):
        """
        Initialize CNN-Transformer model.
        
        Args:
            n_channels: Number of EEG channels
            n_time: Number of time points
            n_classes: Number of output classes
            cnn_filters: Number of CNN output filters
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(n_channels//8, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))
        
        self.conv2 = nn.Conv2d(32, cnn_filters, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters)
        self.pool2 = nn.AdaptiveAvgPool2d((1, n_time // 4))
        
        self.cnn_to_transformer = nn.Linear(cnn_filters, d_model)
        
        max_len = n_time // 4 + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Output logits of shape (batch, n_classes)
        """
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
    """
    Deep CNN-LSTM model for EEG classification.
    
    Architecture:
        - Deep CNN for hierarchical spatial feature extraction
        - Bidirectional LSTM for temporal modeling
        - Best performing architecture for pain threshold detection
    """
    
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2,
                 cnn_filters: List[int] = [32, 64, 128], 
                 lstm_hidden: int = 192, 
                 lstm_layers: int = 2,
                 dropout: float = 0.4):
        """
        Initialize DeepCNN-LSTM model.
        
        Args:
            n_channels: Number of EEG channels
            n_time: Number of time points
            n_classes: Number of output classes
            cnn_filters: List of filter sizes for CNN layers
            lstm_hidden: LSTM hidden state size
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
        """
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
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Output logits of shape (batch, n_classes)
        """
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

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def class_weights_from_y(y: np.ndarray, class_names: list = None) -> Tuple[torch.FloatTensor, Dict]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y: Array of class labels
        class_names: List of class names (optional)
    
    Returns:
        Tuple of (class weights tensor, weight information dictionary)
    """
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
    """
    Train one model with early stopping.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        y_train: Training labels (for class weights)
        device: Device to train on ('cuda' or 'cpu')
        epochs: Maximum number of epochs
        patience: Early stopping patience
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        class_names: List of class names (optional)
        model_name: Name of model for logging
        verbose: Whether to print training progress
    
    Returns:
        Tuple of (trained model, best validation F1, training info dictionary)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f" TRAINING SETUP: {model_name}")
        print(f"{'='*70}")
    
    model.to(device)
    if verbose:
        print(f"Model moved to device: {device}")
    
    # Class weights
    weights, weight_info = class_weights_from_y(y_train, class_names)
    weights = weights.to(device)
    if verbose:
        print(f"\nClass weights for loss function:")
        for cls_name, info in weight_info.items():
            print(f"  {cls_name:20s}: count={info['count']:6d} ({info['percentage']:5.2f}%), weight={info['weight']:.4f}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    if verbose:
        print(f"\nOptimizer: Adam")
        print(f"  Learning rate: {lr:.6f}")
        print(f"  Weight decay: {weight_decay:.6f}")
        print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
        print(f"\nTraining configuration:")
        print(f"  Max epochs: {epochs}")
        print(f"  Early stopping patience: {patience}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
    
    best_state, best_val_loss = None, float('inf')
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    current_lr = lr
    
    # Check if model is window-aware
    is_window_aware = isinstance(model, (WindowAware_DeepCNN_LSTM, WindowAware_CNN_Transformer))
    if is_window_aware and verbose:
        print(f"  Note: Window-aware training enabled")
    
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
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                            leave=False, unit="batch")
        else:
            train_pbar = train_loader
        
        for batch in train_pbar:
            if is_window_aware:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb, wb)
            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                optimizer.zero_grad()
                out = model(xb)
                
            loss = criterion(out, yb)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            batch_loss = loss.item()
            train_losses.append(batch_loss)
            
            # Update progress bar
            if verbose:
                train_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 
                                       'avg_loss': f'{np.mean(train_losses):.4f}'})
        
        tr_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        
        if verbose:
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                          leave=False, unit="batch")
        else:
            val_pbar = val_loader
        
        with torch.no_grad():
            for batch in val_pbar:
                if is_window_aware:
                    xb, wb, yb = batch
                    xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                    out = model(xb, wb)
                else:
                    xb, yb = batch[0].to(device), batch[-1].to(device)
                    out = model(xb)
                    
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                val_preds.append(out.argmax(dim=1).cpu().numpy())
                val_targets.append(yb.cpu().numpy())
                
                # Update progress bar
                if verbose:
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch time
        if use_cuda_timing:
            epoch_end_time.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0  # Convert to seconds
        else:
            epoch_time = time.time() - epoch_start
        
        vloss = np.mean(val_losses)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        val_acc = (val_preds == val_targets).mean()
        
        # Learning rate scheduling
        old_lr = current_lr
        scheduler.step(vloss)
        current_lr = optimizer.param_groups[0]['lr']
        lr_changed = abs(current_lr - old_lr) > 1e-8
        
        # Status indicator
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
            if no_improve >= patience:
                status = "⏹ Early stop"
            else:
                status = f"No improve ({no_improve}/{patience})"
        
        # Print every epoch if verbose
        if verbose:
            lr_str = f"{current_lr:.2e}"
            if lr_changed:
                lr_str += " ↓"
            time_str = f"{epoch_time:.1f}s"
            print(f"{epoch+1:<8} {tr_loss:<12.6f} {vloss:<12.6f} {val_acc:<10.4f} {val_f1:<10.4f} {lr_str:<12} {time_str:<8} {status:<15}")
        elif epoch % 5 == 0 or epoch == epochs - 1:  # Show progress every 5 epochs even if not verbose
            lr_str = f"{current_lr:.2e}"
            if lr_changed:
                lr_str += " ↓"
            time_str = f"{epoch_time:.1f}s"
            print(f"  [{epoch+1}/{epochs}] Train Loss: {tr_loss:.4f} | Val Loss: {vloss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {time_str} | {status}")
        
        if no_improve >= patience:
            if verbose:
                print(f"\n⏹ Early stopping triggered at epoch {epoch+1}")
                print(f"   No improvement for {patience} epochs")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if verbose:
            print(f"\n✓ Loaded best model from epoch {best_epoch}")
            print(f"  Best validation loss: {best_val_loss:.6f}")
            print(f"  Best validation accuracy: {best_val_acc:.4f}")
            print(f"  Best validation F1: {best_val_f1:.4f}")
    else:
        if verbose:
            print(f"\n⚠ No best model state saved (using final model)")
    
    training_info = {
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_acc': float(best_val_acc),
        'best_val_f1': float(best_val_f1),
        'total_epochs': epoch + 1,
        'early_stopped': no_improve >= patience
    }
    
    return model, best_val_f1, training_info

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str, le: LabelEncoder, 
                   model_name: str = "Model", window_le: LabelEncoder = None) -> Dict[str, Any]:
    """
    Evaluate model on test set and return comprehensive metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        le: Label encoder
        model_name: Name of model for logging
        window_le: Window label encoder (optional, for per-window metrics)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f" EVALUATION: {model_name}")
    print(f"{'='*70}")
    print(f"Evaluating on {len(test_loader)} test batches...")
    
    model.eval()
    all_preds, all_targets = [], []
    all_probs = []
    all_windows = []
    
    # Check if model is window-aware
    is_window_aware = isinstance(model, (WindowAware_DeepCNN_LSTM, WindowAware_CNN_Transformer))
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            if is_window_aware:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                out = model(xb, wb)
                all_windows.append(wb.cpu().numpy())
            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                out = model(xb)
                if len(batch) > 2: # Has window data even if model doesn't use it
                    all_windows.append(batch[1].numpy())
            
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_probs = np.concatenate(all_probs)
    
    print(f"✓ Evaluation complete: {len(y_true):,} samples")
    
    acc = (y_pred == y_true).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    report = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate confidence statistics
    max_probs = np.max(y_probs, axis=1)
    mean_confidence = np.mean(max_probs)
    std_confidence = np.std(max_probs)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy:  {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"  Macro F1-Score:     {f1_macro:.4f}")
    print(f"  Weighted F1-Score:  {f1_weighted:.4f}")
    print(f"  Mean Confidence:    {mean_confidence:.4f} ± {std_confidence:.4f}")
    
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support.tolist(),
        'report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': le.classes_.tolist(),
        'mean_confidence': float(mean_confidence),
        'std_confidence': float(std_confidence)
    }
    
    # Per-window metrics if available
    if all_windows and window_le:
        windows = np.concatenate(all_windows)
        per_window_metrics = {}
        print(f"\nPer-Window Performance:")
        for w_idx, w_name in enumerate(window_le.classes_):
            mask = (windows == w_idx)
            if np.any(mask):
                w_acc = (y_pred[mask] == y_true[mask]).mean()
                w_f1 = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
                per_window_metrics[w_name] = {
                    'accuracy': float(w_acc),
                    'f1_macro': float(w_f1),
                    'n_samples': int(mask.sum())
                }
                print(f"  {w_name:<15}: Acc={w_acc:.4f}, F1={w_f1:.4f} ({mask.sum()} samples)")
        metrics['per_window_metrics'] = per_window_metrics
    
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
    """
    Perform grid search over hyperparameters for a single model.
    
    Args:
        model_name: Name of model ('cnn', 'lstm', 'transformer', 'cnn_transformer', 'deep_cnn_lstm')
        X_all: All training data
        y_all: All training labels
        groups: Subject groups for splitting
        n_classes: Number of classes
        n_channels: Number of channels
        n_time: Number of time points
        device: Device to train on
        seed: Random seed for reproducibility
        param_grid: Dictionary of hyperparameters to search
        epochs: Maximum epochs per configuration
        patience: Early stopping patience
        y_window_all: Window labels (optional, for window-aware models)
        n_windows: Number of window classes (optional)
    
    Returns:
        Tuple of (best parameters dictionary, best validation F1 score)
    """
    print(f"\n{'='*70}")
    print(f" GRID SEARCH: {model_name.upper()}")
    print(f"{'='*70}")
    
    # Subject-wise train/val split for tuning
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(X_all, y_all, groups))
    
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    
    y_window_train, y_window_val = None, None
    if y_window_all is not None:
        y_window_train, y_window_val = y_window_all[train_idx], y_window_all[val_idx]
        
    groups_train_split = groups[train_idx]
    groups_val_split = groups[val_idx]
    
    # Verify no subject leakage in grid search split
    train_subject_set = set(groups_train_split)
    val_subject_set = set(groups_val_split)
    overlap = train_subject_set.intersection(val_subject_set)
    
    print(f"Grid search split (subject-wise):")
    print(f"  Train: {len(y_train):,} samples ({len(np.unique(groups_train_split))} subjects)")
    print(f"  Validation: {len(y_val):,} samples ({len(np.unique(groups_val_split))} subjects)")
    if len(overlap) > 0:
        print(f"  ⚠ WARNING: {len(overlap)} subjects appear in both train and validation!")
    else:
        print(f"  ✓ No subject leakage verified")
    
    # Generate all parameter combinations
    param_keys = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_keys]
    combinations = list(product(*param_values))
    
    total_combinations = len(combinations)
    
    # Check for already completed configurations
    def config_hash(params):
        """Create a hash of configuration parameters for comparison."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    # Get completed configs for this model
    if completed_configs is None:
        completed_configs = {}
    model_completed_configs = completed_configs.get(model_name, set())
    
    # Filter out already completed configurations
    remaining_combinations = []
    skipped_count = 0
    for values in combinations:
        params = dict(zip(param_keys, values))
        config_id = config_hash(params)
        if config_id in model_completed_configs:
            skipped_count += 1
        else:
            remaining_combinations.append((values, config_id))
    
    if skipped_count > 0:
        print(f"\n✓ Found {skipped_count} already completed configurations (skipping)")
        print(f"  Remaining configurations to test: {len(remaining_combinations)}")
    
    if len(remaining_combinations) == 0:
        print(f"\n✓ All configurations for {model_name} already completed!")
        if best_hyperparams is not None:
            return best_hyperparams.get(model_name, {}), 0.0
        return {}, 0.0
    
    print(f"\nTotal parameter combinations: {total_combinations}")
    print(f"  Already completed: {skipped_count}")
    print(f"  To test: {len(remaining_combinations)}")
    print(f"Parameters being searched:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    # Calculate estimated time (rough estimate)
    print(f"\nStarting grid search...")
    print(f"Note: Each configuration will train for up to {epochs} epochs with early stopping")
    
    if best_hyperparams is not None:
        best_params = best_hyperparams.get(model_name, None)
        best_f1 = 0.0
    else:
        best_params = None
        best_f1 = 0.0
    results_list = []
    
    # Progress bar for grid search
    grid_pbar = tqdm(enumerate(remaining_combinations), total=len(remaining_combinations), 
                    desc=f"Grid Search: {model_name.upper()}", unit="config")
    
    for idx, (values, config_id) in grid_pbar:
        params = dict(zip(param_keys, values))
        i = idx + skipped_count  # Original index for display
        
        # Update progress bar
        grid_pbar.set_postfix({'best_f1': f'{best_f1:.4f}' if best_f1 > 0 else 'N/A'})
        
        # Show parameters being tested (only first few and last few to avoid clutter)
        if idx < 3 or idx >= len(remaining_combinations) - 3 or len(remaining_combinations) <= 10:
            print(f"\n[{i+1}/{total_combinations}] Testing parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        elif idx == 3:
            print(f"\n... (showing first 3 and last 3 configurations, running {len(remaining_combinations) - 6} more silently) ...")
        
        # Separate model and training parameters
        model_params = {k: v for k, v in params.items() if k not in ['lr', 'weight_decay', 'batch_size']}
        training_params = {k: v for k, v in params.items() if k in ['lr', 'weight_decay']}
        batch_size = params.get('batch_size', 64)
        
        # Determine if window-aware
        is_window_aware = model_name in ['window_aware_deep_cnn_lstm', 'window_aware_cnn_transformer']
        
        # Create model
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
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            print(f"  ✗ Model creation failed: {e}")
            continue
        
        # Create data loaders with seed for reproducibility
        if is_window_aware:
            train_dataset = TensorDataset(X_train, torch.LongTensor(y_window_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(X_val, torch.LongTensor(y_window_val), torch.LongTensor(y_val))
        else:
            train_dataset = TensorDataset(X_train, torch.LongTensor(y_train))
            val_dataset = TensorDataset(X_val, torch.LongTensor(y_val))
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=2, pin_memory=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=2, pin_memory=True)
        
        # Train with reduced verbosity (but show progress every 5 epochs)
        try:
            model, val_f1, train_info = train_one_model(
                model, train_loader, val_loader, y_train, device,
                epochs=epochs, patience=patience, 
                class_names=None, model_name=f"{model_name.upper()} (Grid Search)",
                verbose=False, **training_params
            )
            
            # Show results
            if i < 3 or i >= total_combinations - 3 or total_combinations <= 10:
                print(f"  → Validation F1: {val_f1:.4f} | Val Acc: {train_info['best_val_acc']:.4f} | Best epoch: {train_info['best_epoch']}")
            else:
                # Update progress bar silently
                grid_pbar.set_postfix({
                    'best_f1': f'{best_f1:.4f}' if best_f1 > 0 else 'N/A',
                    'current_f1': f'{val_f1:.4f}'
                })
            
            results_list.append({
                'params': params.copy(),
                'val_f1': float(val_f1),
                'val_acc': float(train_info['best_val_acc']),
                'val_loss': float(train_info['best_val_loss']),
                'best_epoch': train_info['best_epoch']
            })
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = params.copy()
                if idx < 3 or idx >= len(remaining_combinations) - 3 or len(remaining_combinations) <= 10:
                    print(f"  ✓ New best! F1: {best_f1:.4f}")
                # Always update progress bar with new best
                grid_pbar.set_postfix({'best_f1': f'{best_f1:.4f} ⭐'})
            
            # Mark this configuration as completed
            if completed_configs is not None:
                if model_name not in completed_configs:
                    completed_configs[model_name] = set()
                completed_configs[model_name].add(config_id)
            
            # Save checkpoint after each configuration
            if save_checkpoint is not None:
                save_checkpoint()
        
        except Exception as e:
            if idx < 3 or idx >= len(remaining_combinations) - 3 or len(remaining_combinations) <= 10:
                print(f"  ✗ Training failed: {e}")
            else:
                grid_pbar.write(f"  ✗ Config {i+1}/{total_combinations} failed: {e}")
            import traceback
            if idx < 3 or idx >= len(remaining_combinations) - 3 or len(remaining_combinations) <= 10:
                traceback.print_exc()
            continue
    
    # Close progress bar
    grid_pbar.close()
    
    print(f"\n{'='*70}")
    print(f" GRID SEARCH RESULTS: {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Best validation F1: {best_f1:.4f}")
    
    # Sort results by F1 score
    results_list.sort(key=lambda x: x['val_f1'], reverse=True)
    print(f"\nTop 3 configurations:")
    for i, result in enumerate(results_list[:3], 1):
        print(f"  {i}. F1: {result['val_f1']:.4f} | Params: {result['params']}")
    
    return best_params, best_f1

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run training with grid search."""
    
    # Parse command line arguments
    ap = argparse.ArgumentParser(
        description="Train DL models on 1000 Hz EEG data with grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Author: {__author__}
Email: {__email__}
Version: {__version__}

Examples:
  # Train all models with grid search
  python {sys.argv[0]} --task none_vs_pain
  
  # Train specific models
  python {sys.argv[0]} --task none_vs_pain --models cnn lstm
  
  # Skip grid search (use best known parameters)
  python {sys.argv[0]} --task none_vs_pain --no-grid-search
        """
    )
    
    ap.add_argument('--models', nargs='+', 
                    default=['cnn', 'lstm', 'transformer', 'cnn_transformer', 'deep_cnn_lstm', 'window_aware_deep_cnn_lstm', 'window_aware_cnn_transformer'],
                    choices=['cnn', 'lstm', 'transformer', 'cnn_transformer', 'deep_cnn_lstm', 'window_aware_deep_cnn_lstm', 'window_aware_cnn_transformer'],
                    help='Models to train')
    # Set default task to pain_threshold which had best results
    ap.add_argument('--task', default='pain_threshold', 
                    choices=['pain_5class', 'none_vs_pain', 'pain_only', 'pain_threshold'],
                    help='Classification task')
    ap.add_argument('--epochs', type=int, default=50, help='Max epochs for training')
    ap.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    ap.add_argument('--data_root', type=str, default=None, help='Path to Data directory')
    ap.add_argument('--quick', action='store_true', help='Quick mode with fewer samples')
    ap.add_argument('--quick_n_per_subj', type=int, default=50, help='Samples per subject in quick mode')
    ap.add_argument('--output_file', type=str, default='results_1000hz_gridsearch.json', 
                    help='Output JSON file for results')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    ap.add_argument('--no-grid-search', action='store_true', 
                    help='Skip grid search and use best known parameters')
    ap.add_argument('--grid-epochs', type=int, default=30, 
                    help='Max epochs per configuration during grid search')
    ap.add_argument('--grid-patience', type=int, default=10, 
                    help='Early stopping patience during grid search')
    ap.add_argument('--resume', action='store_true',
                    help='Resume from previous run (skip already completed models/configurations)')
    ap.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                    help='Directory to save checkpoints for resume functionality')
    
    args = ap.parse_args()
    
    # If window-aware models are requested but task is not pain_threshold, warn user
    window_aware_models = ['window_aware_deep_cnn_lstm', 'window_aware_cnn_transformer']
    if any(m in args.models for m in window_aware_models) and args.task != 'pain_threshold':
        print("⚠ Warning: Window-aware models are designed for 'pain_threshold' task.")
        print("  Other tasks might not have window labels or appropriate structure.")
    
    # Set all random seeds for reproducibility
    print(f"\n{'='*70}")
    print(" INITIALIZATION")
    print(f"{'='*70}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print(f"Version: {__version__}")
    print()
    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Paths and data
    cwd = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(cwd, '..'))
    
    if args.data_root:
        root = args.data_root
    else:
        root = find_data_root(proj_root, 'Data')
    
    print(f"\nDataset Configuration:")
    print(f"  Data root: {root}")
    print(f"  Dataset: Data (1000 Hz sampling rate)")
    print(f"  Expected duration: ~1 second segments")
    
    print(f"\nLoading index file...")
    df = load_index(root)
    print(f"✓ Loaded {len(df):,} rows from index.csv")
    
    most_ch = get_most_common_channels(df)
    print(f"  Most common channels: {most_ch}")
    
    if args.quick:
        rnd = np.random.RandomState(args.seed)
        tmp = df.copy()
        tmp['_r'] = rnd.rand(len(tmp))
        tmp = tmp.sort_values(['participant', '_r'])
        df = tmp.groupby('participant').head(args.quick_n_per_subj).drop(columns=['_r'])
        print(f"Quick mode: using {len(df):,} rows (seed={args.seed})")
    
    # Prepare task
    print(f"\n{'='*70}")
    print(" TASK PREPARATION")
    print(f"{'='*70}")
    print(f"Task: {args.task}")
    print(f"Preparing labels...")
    
    y_all, le, df, y_window, window_le = load_task_data(df, args.task)
    groups = df['participant'].values
    n_classes = len(le.classes_)
    n_windows = len(window_le.classes_) if window_le else 0
    
    print(f"✓ Task preparation complete")
    print(f"\nTask Details:")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Total samples: {len(y_all):,}")
    print(f"  Unique subjects: {len(np.unique(groups))}")
    if n_windows > 0:
        print(f"  Window types: {list(window_le.classes_)}")
    
    # Detailed class distribution
    class_dist = Counter(y_all)
    print(f"\nClass Distribution:")
    for cls_idx, count in sorted(class_dist.items()):
        cls_name = le.classes_[cls_idx]
        percentage = count / len(y_all) * 100
        print(f"  {cls_name:20s}: {count:6,} samples ({percentage:5.2f}%)")
    
    # Load all segments (with windows if needed)
    # Check if any window-aware model is requested
    use_window = any('window' in m for m in args.models)
    X_all, W_all = load_all_segments(df, root, most_ch, return_window=use_window, window_le=window_le)
    n_channels, n_time = X_all.shape[1], X_all.shape[2]
    
    print(f"\nData shape: {X_all.shape} (samples, channels, time_points)")
    print(f"Channels: {n_channels}, Time points: {n_time}")
    
    # Subject-wise train/test split
    print(f"\n{'='*70}")
    print(" DATA SPLITTING")
    print(f"{'='*70}")
    print(f"Performing subject-wise train/test split...")
    print(f"  Test size: 25%")
    print(f"  Random state: {args.seed} (for reproducibility)")
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups))
    
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    y_window_train = W_all[train_idx] if use_window else None
    y_window_test = W_all[test_idx] if use_window else None
    
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]
    
    train_subjects = len(np.unique(groups_train))
    test_subjects = len(np.unique(groups_test))
    
    # Verify no subject leakage between train and test
    train_subject_set = set(groups_train)
    test_subject_set = set(groups_test)
    overlap = train_subject_set.intersection(test_subject_set)
    
    if len(overlap) > 0:
        print(f"\n⚠ WARNING: Subject leakage detected! {len(overlap)} subjects appear in both train and test sets!")
        print(f"  Overlapping subjects: {sorted(list(overlap))[:10]}{'...' if len(overlap) > 10 else ''}")
    else:
        print(f"\n✓ Subject-wise split verified: No subject leakage between train and test sets")
    
    print(f"\n✓ Split complete")
    print(f"\nTraining Set:")
    print(f"  Samples: {len(y_train):,}")
    print(f"  Subjects: {train_subjects}")
    print(f"  Percentage: {len(y_train)/len(y_all)*100:.2f}%")
    print(f"  Class distribution:")
    train_dist = Counter(y_train)
    for cls_idx, count in sorted(train_dist.items()):
        cls_name = le.classes_[cls_idx]
        percentage = count / len(y_train) * 100
        print(f"    {cls_name:20s}: {count:6,} ({percentage:5.2f}%)")
    
    print(f"\nTest Set:")
    print(f"  Samples: {len(y_test):,}")
    print(f"  Subjects: {test_subjects}")
    print(f"  Percentage: {len(y_test)/len(y_all)*100:.2f}%")
    print(f"  Class distribution:")
    test_dist = Counter(y_test)
    for cls_idx, count in sorted(test_dist.items()):
        cls_name = le.classes_[cls_idx]
        percentage = count / len(y_test) * 100
        print(f"    {cls_name:20s}: {count:6,} ({percentage:5.2f}%)")
    
    # Define hyperparameter grids for grid search (reduced for faster training)
    param_grids = {
        'cnn': {
            'dropout': [0.4, 0.5],
            'lr': [1e-3, 5e-4],
            'weight_decay': [1e-5],
            'batch_size': [64]
        },
        'lstm': {
            'hidden': [128, 192],
            'dropout': [0.4, 0.5],
            'bidirectional': [True],
            'lr': [1e-3, 5e-4],
            'weight_decay': [1e-5],
            'batch_size': [64]
        },
        'transformer': {
            'd_model': [128],
            'nhead': [8],
            'num_layers': [3, 4],
            'dropout': [0.3],
            'lr': [5e-4, 1e-4],
            'weight_decay': [1e-5],
            'batch_size': [64]
        },
        'cnn_transformer': {
            'cnn_filters': [64, 128],
            'd_model': [128],
            'nhead': [8],
            'num_layers': [2, 3],
            'dropout': [0.3],
            'lr': [5e-4, 1e-4],
            'weight_decay': [1e-5],
            'batch_size': [64]
        },
        'deep_cnn_lstm': {
            'cnn_filters': [[32, 64, 128], [32, 64, 96]],
            'lstm_hidden': [192, 256],
            'lstm_layers': [2],
            'dropout': [0.3, 0.4],
            'lr': [5e-4, 1e-3],
            'weight_decay': [1e-5],
            'batch_size': [64]
        },
        'window_aware_deep_cnn_lstm': {
            'cnn_filters': [[32, 64, 128]],
            'lstm_hidden': [192, 256],
            'lstm_layers': [2],
            'window_embed_dim': [16, 32],
            'dropout': [0.3, 0.4],
            'lr': [5e-4, 1e-3],
            'weight_decay': [1e-5],
            'batch_size': [64]
        },
        'window_aware_cnn_transformer': {
            'cnn_filters': [[32, 64, 128]],
            'd_model': [128, 256],
            'nhead': [8],
            'num_layers': [2, 3],
            'window_embed_dim': [16, 32],
            'dropout': [0.3],
            'lr': [5e-4, 1e-4],
            'weight_decay': [1e-5],
            'batch_size': [64]
        }
    }
    
    # Calculate total combinations for each model
    print(f"\n{'='*70}")
    print(" GRID SEARCH CONFIGURATION")
    print(f"{'='*70}")
    for model_name, grid in param_grids.items():
        from itertools import product
        total = len(list(product(*grid.values())))
        print(f"  {model_name:20s}: {total:3d} combinations")
    
    # Best known parameters (if skipping grid search)
    best_known_params = {
        'cnn': {
            'dropout': 0.4,
            'lr': 0.0005,
            'weight_decay': 1e-5,
            'batch_size': 64
        },
        'lstm': {
            'hidden': 192,
            'dropout': 0.5,
            'bidirectional': True,
            'lr': 0.0005,
            'weight_decay': 1e-5,
            'batch_size': 64
        },
        'transformer': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dropout': 0.3,
            'lr': 0.0001,
            'weight_decay': 1e-5,
            'batch_size': 64
        },
        'cnn_transformer': {
            'cnn_filters': 64,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dropout': 0.3,
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'batch_size': 64
        },
        'deep_cnn_lstm': {
            'cnn_filters': [32, 64, 128],
            'lstm_hidden': 192,
            'lstm_layers': 2,
            'dropout': 0.3,
            'lr': 0.0003,
            'weight_decay': 1e-5,
            'batch_size': 64
        },
        'window_aware_deep_cnn_lstm': {
            'cnn_filters': [32, 64, 128],
            'lstm_hidden': 192,
            'lstm_layers': 2,
            'window_embed_dim': 16,
            'dropout': 0.4,
            'lr': 0.0005,
            'weight_decay': 1e-5,
            'batch_size': 64
        },
        'window_aware_cnn_transformer': {
            'cnn_filters': [32, 64, 128],
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'window_embed_dim': 16,
            'dropout': 0.3,
            'lr': 0.0001,
            'weight_decay': 1e-5,
            'batch_size': 64
        }
    }
    
    # Setup checkpoint directory for resume functionality
    checkpoint_dir = os.path.join(proj_root, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load existing results if resuming
    results = {}
    best_hyperparams = {}
    completed_models = set()
    completed_configs = {}  # model_name -> set of completed config hashes
    
    if args.resume:
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{args.task}_{args.seed}.json")
        if os.path.exists(checkpoint_file):
            print(f"\n{'='*70}")
            print(" RESUMING FROM CHECKPOINT")
            print(f"{'='*70}")
            print(f"Loading checkpoint from: {checkpoint_file}")
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    results = checkpoint.get('results', {})
                    best_hyperparams = checkpoint.get('best_hyperparameters', {})
                    completed_models = set(checkpoint.get('completed_models', []))
                    completed_configs = {k: set(v) for k, v in checkpoint.get('completed_configs', {}).items()}
                    
                    print(f"✓ Loaded checkpoint:")
                    print(f"  Completed models: {len(completed_models)}/{len(args.models)}")
                    for model_name in completed_models:
                        print(f"    - {model_name}")
                    if completed_configs:
                        for model_name, configs in completed_configs.items():
                            print(f"  {model_name}: {len(configs)} configurations already tested")
            except Exception as e:
                print(f"⚠ Failed to load checkpoint: {e}")
                print("  Starting fresh...")
                results = {}
                best_hyperparams = {}
                completed_models = set()
                completed_configs = {}
        else:
            print(f"\nNo checkpoint found at {checkpoint_file}")
            print("Starting fresh run...")
    else:
        # Clear checkpoint if not resuming
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{args.task}_{args.seed}.json")
        if os.path.exists(checkpoint_file):
            print(f"\n⚠ Not resuming - will overwrite existing checkpoint")
    
    def save_checkpoint():
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            'task': args.task,
            'seed': args.seed,
            'results': results,
            'best_hyperparameters': best_hyperparams,
            'completed_models': list(completed_models),
            'completed_configs': {k: list(v) for k, v in completed_configs.items()}
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    for model_name in args.models:
        print(f"\n{'='*70}")
        print(f" PROCESSING MODEL: {model_name.upper()}")
        print(f"{'='*70}")
        
        # Check if model is already completed
        if model_name in completed_models:
            print(f"✓ Model {model_name} already completed. Skipping...")
            print(f"  Previous results:")
            if model_name in results:
                metrics = results[model_name]
                print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"    F1-Score: {metrics.get('f1_macro', 'N/A'):.4f}")
            continue
        
        # Grid search or use best known parameters
        if args.no_grid_search:
            print(f"Using best known parameters (grid search skipped)")
            best_params = best_known_params.get(model_name, {})
        else:
            if model_name in param_grids:
                best_params, best_val_f1 = grid_search_model(
                    model_name, X_train, y_train, groups_train,
                    n_classes, n_channels, n_time, device, args.seed,
                    param_grids[model_name], 
                    epochs=args.grid_epochs, 
                    patience=args.grid_patience,
                    completed_configs=completed_configs,
                    best_hyperparams=best_hyperparams,
                    save_checkpoint=save_checkpoint,
                    y_window_all=y_window if use_window else None,
                    n_windows=n_windows
                )
                best_hyperparams[model_name] = best_params
            else:
                print(f"No grid search defined for {model_name}, using best known parameters")
                best_params = best_known_params.get(model_name, {})
        
        # Extract parameters
        model_params = {k: v for k, v in best_params.items() if k not in ['lr', 'weight_decay', 'batch_size']}
        training_params = {k: v for k, v in best_params.items() if k in ['lr', 'weight_decay']}
        batch_size = best_params.get('batch_size', 64)
        
        print(f"\n{'='*70}")
        print(f" FINAL TRAINING: {model_name.upper()}")
        print(f"{'='*70}")

        if best_params is None:
            print("⚠ Grid search returned no valid parameters.")
            print("✅ Falling back to best known default parameters.")
            best_params = best_known_params.get(model_name, {})
        else:
            print(f"Best parameters found:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")

        
        # Build model
        print(f"\nBuilding {model_name.upper()} model...")
        print(f"  Input shape: (batch, channels={n_channels}, time={n_time})")
        print(f"  Output classes: {n_classes}")
        print(f"\n{'='*70}")
        
        if model_name == 'cnn':
            model = CNN2D(n_channels, n_time, n_classes, **model_params)
            print(f"  Architecture: 2D CNN with 3 conv layers (32→64→128 filters)")
        elif model_name == 'lstm':
            model = LSTMModel(n_channels, n_classes, **model_params)
            hidden = model_params.get('hidden', 128)
            bidirectional = model_params.get('bidirectional', True)
            print(f"  Architecture: LSTM (hidden={hidden}, bidirectional={bidirectional}, 2 layers)")
        elif model_name == 'transformer':
            model = TransformerModel(n_channels, n_time, n_classes, **model_params)
            d_model = model_params.get('d_model', 128)
            nhead = model_params.get('nhead', 8)
            num_layers = model_params.get('num_layers', 4)
            print(f"  Architecture: Transformer (d_model={d_model}, nhead={nhead}, layers={num_layers})")
        elif model_name == 'cnn_transformer':
            model = CNN_Transformer(n_channels, n_time, n_classes, **model_params)
            cnn_filters = model_params.get('cnn_filters', 64)
            d_model = model_params.get('d_model', 128)
            print(f"  Architecture: Hybrid CNN-Transformer (CNN filters={cnn_filters}, d_model={d_model})")
        elif model_name == 'deep_cnn_lstm':
            model = DeepCNN_LSTM(n_channels, n_time, n_classes, **model_params)
            cnn_filters = model_params.get('cnn_filters', [32, 64, 128])
            lstm_hidden = model_params.get('lstm_hidden', 192)
            print(f"  Architecture: Deep CNN-LSTM (CNN filters={cnn_filters}, LSTM hidden={lstm_hidden})")
        elif model_name == 'window_aware_deep_cnn_lstm':
            model = WindowAware_DeepCNN_LSTM(n_channels, n_time, n_classes, n_windows=n_windows, **model_params)
            cnn_filters = model_params.get('cnn_filters', [32, 64, 128])
            lstm_hidden = model_params.get('lstm_hidden', 192)
            window_embed_dim = model_params.get('window_embed_dim', 16)
            print(f"  Architecture: Window-Aware Deep CNN-LSTM (Window embed={window_embed_dim})")
        elif model_name == 'window_aware_cnn_transformer':
            model = WindowAware_CNN_Transformer(n_channels, n_time, n_classes, n_windows=n_windows, **model_params)
            cnn_filters = model_params.get('cnn_filters', [32, 64, 128])
            d_model = model_params.get('d_model', 128)
            window_embed_dim = model_params.get('window_embed_dim', 16)
            print(f"  Architecture: Window-Aware CNN-Transformer (Window embed={window_embed_dim})")
        else:
            print(f"Unknown model: {model_name}, skipping...")
            continue
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Estimate model size
        model_size_mb = total_params * 4 / 1024**2  # Assuming float32
        print(f"  Estimated model size: ~{model_size_mb:.2f} MB")
        
        # Create train/val split for early stopping
        print(f"\nCreating train/validation split for early stopping...")
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_inner_idx, val_idx = next(val_splitter.split(X_train, y_train, groups_train))
        X_tr, X_val = X_train[train_inner_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_inner_idx], y_train[val_idx]
        groups_tr = groups_train[train_inner_idx]
        groups_val = groups_train[val_idx]
        
        y_window_tr = y_window_train[train_inner_idx] if use_window else None
        y_window_val_inner = y_window_train[val_idx] if use_window else None
        
        # Verify no subject leakage in train/val split
        train_subject_set = set(groups_tr)
        val_subject_set = set(groups_val)
        overlap = train_subject_set.intersection(val_subject_set)
        
        print(f"  Train: {len(y_tr):,} samples ({len(np.unique(groups_tr))} subjects)")
        print(f"  Validation: {len(y_val):,} samples ({len(np.unique(groups_val))} subjects)")
        if len(overlap) > 0:
            print(f"  ⚠ WARNING: {len(overlap)} subjects appear in both train and validation!")
        else:
            print(f"  ✓ No subject leakage verified")
        
        # Create loaders
        print(f"\nCreating data loaders...")
        is_window_aware = model_name in ['window_aware_deep_cnn_lstm', 'window_aware_cnn_transformer']
        
        if is_window_aware:
            train_dataset = TensorDataset(X_tr, torch.LongTensor(y_window_tr), torch.LongTensor(y_tr))
            val_dataset = TensorDataset(X_val, torch.LongTensor(y_window_val_inner), torch.LongTensor(y_val))
            test_dataset = TensorDataset(X_test, torch.LongTensor(y_window_test), torch.LongTensor(y_test))
        else:
            train_dataset = TensorDataset(X_tr, torch.LongTensor(y_tr))
            val_dataset = TensorDataset(X_val, torch.LongTensor(y_val))
            test_dataset = TensorDataset(X_test, torch.LongTensor(y_test))
        
        # Set generator seed for DataLoader reproducibility
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=4, pin_memory=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=4, pin_memory=True)
        
        print(f"  Train batches: {len(train_loader)} (batch_size={batch_size})")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Train
        model, best_val_f1, training_info = train_one_model(
            model, train_loader, val_loader, y_tr, device,
            epochs=args.epochs, patience=args.patience, 
            class_names=le.classes_.tolist(), model_name=model_name.upper(),
            **training_params
        )
        
        # Evaluate on test set
        metrics = evaluate_model(model, test_loader, device, le, model_name=model_name.upper(), window_le=window_le)
        if model_name == "window_aware_deep_cnn_lstm":
            run_window_aware_interpretability(model, test_loader, device, window_le)

        metrics['training_info'] = training_info
        metrics['best_hyperparameters'] = best_params
        results[model_name] = metrics
        
        # Mark model as completed and save checkpoint
        completed_models.add(model_name)
        save_checkpoint()
        print(f"\n✓ Checkpoint saved. Model {model_name} completed.")
        
        print(f"\n{'='*70}")
        print(f" DETAILED RESULTS: {model_name.upper()}")
        print(f"{'='*70}")
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
        print(f"  Macro F1-Score:     {metrics['f1_macro']:.4f}")
        print(f"  Weighted F1-Score:  {metrics['f1_weighted']:.4f}")
        if 'mean_confidence' in metrics:
            print(f"  Mean Confidence:    {metrics['mean_confidence']:.4f} ± {metrics['std_confidence']:.4f}")
        
        print(f"\nPer-Class Performance:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        for i, class_name in enumerate(le.classes_):
            print(f"{class_name:<20} {metrics['precision_per_class'][i]:<12.4f} "
                  f"{metrics['recall_per_class'][i]:<12.4f} "
                  f"{metrics['f1_per_class'][i]:<12.4f} "
                  f"{metrics['support_per_class'][i]:<10}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"  True\\Pred", end="")
        for cls_name in le.classes_:
            print(f"{cls_name:>10}", end="")
        print()
        for i, cls_name in enumerate(le.classes_):
            print(f"  {cls_name:<10}", end="")
            for j in range(len(le.classes_)):
                print(f"{cm[i,j]:>10}", end="")
            print()
        
        print(f"\nClassification Report:")
        print(metrics['report'])
        
        if 'training_info' in metrics:
            train_info = metrics['training_info']
            print(f"\nTraining Summary:")
            print(f"  Best epoch: {train_info['best_epoch']}")
            print(f"  Total epochs trained: {train_info['total_epochs']}")
            print(f"  Early stopped: {train_info['early_stopped']}")
            print(f"  Best validation loss: {train_info['best_val_loss']:.6f}")
            print(f"  Best validation accuracy: {train_info['best_val_acc']:.4f}")
            print(f"  Best validation F1: {train_info['best_val_f1']:.4f}")
    
    # Final Summary
    print(f"\n{'='*70}")
    print(" FINAL SUMMARY - 1000 Hz Results")
    print(f"{'='*70}")
    print(f"Task: {args.task}")
    print(f"Dataset: Data (1000 Hz)")
    print(f"Grid Search: {'Disabled' if args.no_grid_search else 'Enabled'}")
    print(f"\nModel Performance Comparison:")
    print(f"{'Model':<20} {'Accuracy':<12} {'Bal-Acc':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['balanced_accuracy']:<12.4f} "
              f"{metrics['f1_macro']:<12.4f} {metrics['f1_weighted']:<12.4f}")
    
    # Save results to JSON file
    output_path = os.path.join(proj_root, args.output_file)
    with open(output_path, 'w') as f:
        json.dump({
            'author': __author__,
            'email': __email__,
            'version': __version__,
            'task': args.task,
            'dataset': 'Data_1000Hz',
            'reproducibility': {
                'random_seed': args.seed,
                'note': 'All random number generators (Python, NumPy, PyTorch, CUDA) were seeded for reproducibility'
            },
            'grid_search': {
                'enabled': not args.no_grid_search,
                'grid_epochs': args.grid_epochs if not args.no_grid_search else None,
                'grid_patience': args.grid_patience if not args.no_grid_search else None
            },
            'best_hyperparameters': best_hyperparams if not args.no_grid_search else best_known_params,
            'dataset_info': {
                'sampling_rate': 1000,
                'time_points': int(n_time),
                'channels': int(n_channels),
                'n_samples': len(y_all),
                'n_subjects': len(np.unique(groups)),
                'n_classes': n_classes,
                'class_names': le.classes_.tolist()
            },
            'train_test_split': {
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'train_subjects': len(np.unique(groups_train)),
                'test_subjects': len(np.unique(groups_test)),
                'test_size': 0.25,
                'random_state': args.seed
            },
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"\n{'='*70}")
    print(" TRAINING COMPLETE")
    print(f"{'='*70}")
    
    return results


import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# ✅ GRAD-CAM CLASS
# ================================
class GradCAM_WindowAware:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, window_idx, class_idx=None):
        self.model.zero_grad()
        output = self.model(x, window_idx)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


# ================================
# ✅ MAIN INTERPRETABILITY FUNCTION
# ================================
def run_window_aware_interpretability(model, test_loader, device, window_le):
    print("\n" + "="*70)
    print(" RUNNING WINDOW-AWARE INTERPRETABILITY ")
    print("="*70)

    model.eval()
    grad_cam = GradCAM_WindowAware(model, model.conv3)

    # Take ONE batch from test set
    sample_x, sample_w, sample_y = next(iter(test_loader))
    sample_x = sample_x.to(device)
    sample_w = sample_w.to(device)

    # ✅ Generate Grad-CAM
    cam = grad_cam.generate(sample_x[:1], sample_w[:1])

    # ✅ Save raw Grad-CAM
    np.save("gradcam_windowaware.npy", cam)
    print("✅ Grad-CAM saved as gradcam_windowaware.npy")

    # ✅ Plot Grad-CAM (Channel × Time)
    plt.figure(figsize=(10, 5))
    plt.imshow(cam[0], aspect="auto", cmap="jet")
    plt.colorbar()
    plt.title("Grad-CAM: Window-Aware CNN-LSTM")
    plt.xlabel("Time")
    plt.ylabel("Channels")
    plt.tight_layout()
    plt.savefig("gradcam_windowaware.png", dpi=300)
    plt.close()
    print("✅ Grad-CAM heatmap saved as gradcam_windowaware.png")

    # ================================
    # ✅ WINDOW EMBEDDING INTERPRETABILITY
    # ================================
    window_embeddings = model.window_embedding.weight.detach().cpu().numpy()
    np.save("window_embeddings.npy", window_embeddings)

    plt.figure(figsize=(6, 4))
    sns.heatmap(window_embeddings, annot=True, fmt=".3f",
                xticklabels=[f"Dim-{i}" for i in range(window_embeddings.shape[1])],
                yticklabels=window_le.classes_)
    plt.title("Window Embedding Importance")
    plt.xlabel("Embedding Dimensions")
    plt.ylabel("Window Type")
    plt.tight_layout()
    plt.savefig("window_embedding_heatmap.png", dpi=300)
    plt.close()

    print("✅ Window embeddings saved as window_embeddings.npy")
    print("✅ Window embedding heatmap saved as window_embedding_heatmap.png")

    print("\n✅ INTERPRETABILITY COMPLETE\n")


if __name__ == '__main__':
    main()
"""
  screen -S painjob_windowaware -dm bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg && python /home/asatsan2/Projects/EEG-Pain-Estimation/notebooks/new.train.py --task pain_threshold --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data --grid-epochs 20 --grid-patience 7 --epochs 30 --patience 10 --models window_aware_deep_cnn_lstm --seed 42 > /home/asatsan2/Projects/EEG-Pain-Estimation/train_windowaware.log 2>&1"
 
 tail -f /home/asatsan2/Projects/EEG-Pain-Estimation/train_windowaware.log

  """ 