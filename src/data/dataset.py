# src/data/datasets.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, df, root, most_ch, return_window=False, window_le=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.most_ch = most_ch
        self.return_window = return_window

        self.window_indices = (
            window_le.transform(df["window"])
            if return_window and window_le is not None
            else np.zeros(len(df), dtype=int)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = os.path.join(self.root, "npz", row["npz_file"])

        with np.load(npz_path) as data:
            X = data["X"]  # (channels, time)

        ch, T = X.shape

        if ch < self.most_ch:
            X = np.vstack([X, np.zeros((self.most_ch - ch, T))])
        else:
            X = X[: self.most_ch]

        # normalize per channel
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mean) / std

        window = self.window_indices[idx]
        return torch.FloatTensor(X), torch.LongTensor([window])

def load_all_segments(df, root, most_ch, return_window, window_le, batch_size=32):
    dataset = EEGDataset(df, root, most_ch, return_window, window_le)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    X_all, W_all = [], []
    for X, W in loader:
        X_all.append(X)
        W_all.append(W)

    return torch.cat(X_all), torch.cat(W_all).squeeze()
