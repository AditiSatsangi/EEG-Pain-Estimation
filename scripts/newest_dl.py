# ===========================
# FIXED VERSION (drop-in)
# - Baseline interpretability no longer NaN
# - Saliency records W whenever loader provides it (X,W,y), regardless of model type
# - Adds guards if win_idx is None
# - Keeps your MTL lambda schedule + detach head + optional weighted window loss
# ===========================

import os
import argparse
import json 
import random
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

    # More robust reject_flag handling (bool or str)
    if "reject_flag" in df.columns:
        before = len(df)
        rej = df["reject_flag"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[~rej].copy()
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

    if "window" in df.columns:
        window_le.fit(df["window"])
        y_window = window_le.transform(df["window"])
    else:
        df = df.copy()
        df["window"] = "unknown"
        window_le.fit(["unknown"])
        y_window = np.zeros(len(df), dtype=int)

    if task == "pain_threshold":
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
# Models
# =============================================================================
class DeepCNN_LSTM(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2,
                 cnn_filters: List[int] = [32, 64, 128],
                 lstm_hidden: int = 192, lstm_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(max(1, n_channels // 8), 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))

        time_reduced = max(1, n_time // 8)
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

        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(max(1, n_channels // 8), 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))

        time_reduced = max(1, n_time // 8)
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

        self.conv1 = nn.Conv2d(1, cnn_filters[0], kernel_size=(max(1, n_channels // 8), 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))

        time_reduced = max(1, n_time // 8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))

        self.lstm = nn.LSTM(
            cnn_filters[2], lstm_hidden, batch_first=True,
            num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

        self.fc_pain = nn.Linear(lstm_hidden * 2, n_classes)
        self.fc_window = nn.Linear(lstm_hidden * 2, n_windows)

    def encode(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = self.adaptive_pool(x)

        x = x.squeeze(2).transpose(1, 2)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = self.dropout(h)
        return h

    def forward(self, x, detach_window_head: bool = False):
        feat = self.encode(x)
        pain_logits = self.fc_pain(feat)
        win_feat = feat.detach() if detach_window_head else feat
        win_logits = self.fc_window(win_feat)
        return pain_logits, win_logits


# =============================================================================
# Training / Evaluation
# =============================================================================
def class_weights_from_y(y: np.ndarray) -> torch.FloatTensor:
    unique, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(unique) * counts)
    full = np.zeros(int(unique.max()) + 1, dtype=np.float32)
    for cls, w in zip(unique, weights):
        full[int(cls)] = w
    return torch.FloatTensor(full)


def mtl_lambda_schedule(epoch_idx: int, total_epochs: int, base_lambda: float, warmup_epochs: int) -> float:
    if base_lambda <= 0:
        return 0.0
    if epoch_idx < warmup_epochs:
        return float(base_lambda)
    remaining = max(total_epochs - warmup_epochs, 1)
    t = (epoch_idx - warmup_epochs) / remaining
    lam = base_lambda * max(0.0, (1.0 - t))
    return float(lam)


def _compute_window_class_weights_from_loader(train_loader: DataLoader) -> torch.FloatTensor:
    all_w = []
    for batch in train_loader:
        wb = batch[1]
        all_w.append(wb.detach().cpu().numpy())
    all_w = np.concatenate(all_w)
    unique, counts = np.unique(all_w, return_counts=True)
    weights = len(all_w) / (len(unique) * counts)
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
                    model_name: str = "model",
                    mtl_lambda: float = 0.2,
                    mtl_lambda_warmup_epochs: int = 10,
                    mtl_detach_window_head: bool = False,
                    window_loss_class_weighted: bool = False) -> Tuple[nn.Module, Dict[str, Any]]:

    model.to(device)

    w = class_weights_from_y(y_train).to(device)
    criterion_pain = nn.CrossEntropyLoss(weight=w)

    is_mtl = isinstance(model, MultiTask_DeepCNN_LSTM)
    is_wa = isinstance(model, WindowAware_DeepCNN_LSTM)

    if is_mtl:
        if window_loss_class_weighted:
            w_win = _compute_window_class_weights_from_loader(train_loader).to(device)
            criterion_window = nn.CrossEntropyLoss(weight=w_win).to(device)
        else:
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
            optimizer.zero_grad(set_to_none=True)

            if is_wa:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                logits = model(xb, wb)
                loss = criterion_pain(logits, yb)

            elif is_mtl:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                pain_logits, win_logits = model(xb, detach_window_head=mtl_detach_window_head)

                loss_p = criterion_pain(pain_logits, yb)
                loss_w = criterion_window(win_logits, wb)

                lam = mtl_lambda_schedule(epoch, epochs, mtl_lambda, mtl_lambda_warmup_epochs)
                loss = loss_p + lam * loss_w

            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                logits = model(xb)
                loss = criterion_pain(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

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
                    pain_logits, win_logits = model(xb, detach_window_head=False)

                    loss_p = criterion_pain(pain_logits, yb)
                    loss_w = criterion_window(win_logits, wb)

                    lam = mtl_lambda_schedule(epoch, epochs, mtl_lambda, mtl_lambda_warmup_epochs)
                    loss = loss_p + lam * loss_w

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
            if is_mtl:
                lam_now = mtl_lambda_schedule(epoch, epochs, mtl_lambda, mtl_lambda_warmup_epochs)
                print(f"[{model_name}] epoch {epoch+1}/{epochs}  train={np.mean(train_losses):.4f}  val={vloss:.4f}  lambda={lam_now:.3f}")
            else:
                print(f"[{model_name}] epoch {epoch+1}/{epochs}  train={np.mean(train_losses):.4f}  val={vloss:.4f}")

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"✓ {model_name} best epoch = {best_epoch} (val loss {best_val_loss:.4f})")

    info = {"best_epoch": best_epoch, "best_val_loss": best_val_loss}
    if is_mtl:
        info.update({
            "mtl_lambda": float(mtl_lambda),
            "mtl_lambda_warmup_epochs": int(mtl_lambda_warmup_epochs),
            "mtl_detach_window_head": bool(mtl_detach_window_head),
            "window_loss_class_weighted": bool(window_loss_class_weighted),
        })
    return model, info


# ===========================
# IMPORTANT FIX IS HERE
# ===========================
def compute_grad_input_saliency(model, loader, device, is_window_aware=False, is_multitask=False):
    """
    |grad * input| -> mean over channels => saliency over time
    Works for loaders yielding (X,y) OR (X,W,y).
    Returns:
      saliency_all [N, T]
      windows_all  [N] or None
    """
    was_training = model.training
    model.train()  # IMPORTANT: cuDNN RNN backward requires train mode.

    saliency_list = []
    windows_list = []

    for batch in tqdm(loader, desc="Saliency (grad*input)", unit="batch"):
        # Accept both (X,y) and (X,W,y)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
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
            if wb_dev is None:
                raise RuntimeError("Window-aware saliency requires W, but loader provided no W.")
            logits = model(xb, wb_dev)
        elif is_multitask:
            pain_logits, _ = model(xb, detach_window_head=False)
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
                pain_logits, win_logits = model(xb, detach_window_head=False)
                logits = pain_logits
                all_windows.append(wb.detach().cpu().numpy())
                all_window_preds.append(win_logits.argmax(dim=1).detach().cpu().numpy())

            else:
                # Baseline can be evaluated on either (X,y) OR (X,W,y)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    xb, wb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    all_windows.append(wb.detach().cpu().numpy())
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

    acc = float((y_pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)

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

    # Interpretability proxy for ANY model if windows exist in loader
    if windows is not None and window_le is not None:
        saliency, win_idx = compute_grad_input_saliency(
            model, test_loader, device,
            is_window_aware=is_wa,
            is_multitask=is_mtl
        )

        # Guard: if win_idx didn't get collected for some reason, skip gracefully
        if win_idx is None:
            metrics["interpretability_quant"] = {
                "method": "grad_x_input_mean_over_channels_then_mean_over_time",
                "per_window_saliency_energy": {},
                "erp_alignment_ratio": None,
                "note": "win_idx was None (loader did not provide window indices)."
            }
            return metrics

        energy = saliency.mean(axis=1)

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
# Main
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

    ap.add_argument("--mtl_lambda", type=float, default=0.2, help="loss = pain + lambda * window")
    ap.add_argument("--mtl_lambda_warmup_epochs", type=int, default=10,
                    help="epochs to keep lambda fixed before decaying to 0")
    ap.add_argument("--mtl_detach_window_head", action="store_true",
                    help="detach shared features for window head")
    ap.add_argument("--window_loss_class_weighted", action="store_true",
                    help="use class-weighted CE for window task")

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

    y_all, le, df, _, window_le = load_task_data(df, args.task)
    groups = df["participant"].values

    X_all, W_all = load_all_segments(df, root, most_ch, return_window=True, window_le=window_le)
    n_channels, n_time = X_all.shape[1], X_all.shape[2]
    n_classes = len(le.classes_)
    n_windows = len(window_le.classes_) if window_le else 0

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups))

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    W_train, W_test = W_all[train_idx], W_all[test_idx]

    bs = args.batch_size
    gen = torch.Generator(); gen.manual_seed(args.seed)

    train_loader_base = DataLoader(TensorDataset(X_train, torch.LongTensor(y_train)),
                                   batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, generator=gen)
    val_loader_base = DataLoader(TensorDataset(X_train, torch.LongTensor(y_train)),
                                 batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    test_loader_mtl = DataLoader(TensorDataset(X_test, W_test.long(), torch.LongTensor(y_test)),
                                 batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    train_loader_mtl = DataLoader(TensorDataset(X_train, W_train.long(), torch.LongTensor(y_train)),
                                  batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, generator=gen)
    val_loader_mtl = DataLoader(TensorDataset(X_train, W_train.long(), torch.LongTensor(y_train)),
                                batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    results = {}

    # 1) Baseline
    baseline = DeepCNN_LSTM(n_channels=n_channels, n_time=n_time, n_classes=n_classes,
                            cnn_filters=[32, 64, 128], lstm_hidden=192, lstm_layers=2, dropout=0.4)

    baseline, info_base = train_one_model(
        baseline, train_loader_base, val_loader_base, y_train,
        device=device, epochs=args.epochs, patience=args.patience,
        lr=args.lr, weight_decay=args.weight_decay, model_name="BASELINE"
    )

    # Evaluate baseline on (X,W,y) loader -> interpretability fair + NOT NaN
    metrics_base = evaluate_model(baseline, test_loader_mtl, device, le, window_le, "BASELINE")
    metrics_base["training_info"] = info_base
    results["baseline_deep_cnn_lstm"] = metrics_base

    del baseline
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2) MTL
    mtl = MultiTask_DeepCNN_LSTM(n_channels=n_channels, n_time=n_time, n_classes=n_classes, n_windows=n_windows,
                                cnn_filters=[32, 64, 128], lstm_hidden=192, lstm_layers=2, dropout=0.4)

    mtl, info_mtl = train_one_model(
        mtl, train_loader_mtl, val_loader_mtl, y_train,
        device=device, epochs=args.epochs, patience=args.patience,
        lr=args.lr, weight_decay=args.weight_decay, model_name="MTL",
        mtl_lambda=args.mtl_lambda,
        mtl_lambda_warmup_epochs=args.mtl_lambda_warmup_epochs,
        mtl_detach_window_head=args.mtl_detach_window_head,
        window_loss_class_weighted=args.window_loss_class_weighted
    )

    metrics_mtl = evaluate_model(mtl, test_loader_mtl, device, le, window_le, "MTL")
    metrics_mtl["training_info"] = info_mtl
    results["mtl_deep_cnn_lstm"] = metrics_mtl

    # 3) Delta CI
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

    out_path = os.path.join(proj_root, args.output_file)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {out_path}")



if __name__ == "__main__":
    main()


"""
# Best first run: lambda=0.2 warmup 10 + detach window head
python /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/newest_dl.py \
  --task pain_threshold --seed 42 --epochs 50 --patience 15 --batch_size 64 \
  --mtl_lambda 0.2 --mtl_lambda_warmup_epochs 10 --mtl_detach_window_head \
  --output_file paper_baseline_vs_mtl_seed42_lam02_detach.json \
  >> train_newest_lam02_detach.log 2>&1

# Try lambda=0.1 too (often improves)
python /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/newest_dl.py \
  --task pain_threshold --seed 42 --epochs 50 --patience 15 --batch_size 64 \
  --mtl_lambda 0.1 --mtl_lambda_warmup_epochs 10 --mtl_detach_window_head \
  --output_file paper_baseline_vs_mtl_seed42_lam01_detach.json \
  >> train_newest_lam01_detach.log 2>&1

 
screen -S painjob3 -dm bash -c "
source ~/.bashrc
conda activate eeg
python /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/newest_dl.py \
  --task pain_threshold --seed 42 --epochs 50 --patience 15 --batch_size 64 \
  --mtl_lambda 0.2 --mtl_lambda_warmup_epochs 10 --mtl_detach_window_head \
  --output_file paper_baseline_vs_mtl_seed42_lam02_detach.json \
  >> train_lam02_detach.log 2>&1
"

screen -S painjob -dm bash -c "
source ~/.bashrc
conda activate eeg
python /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/newest_dl.py \
  --task pain_threshold --seed 42 --epochs 50 --patience 15 --batch_size 64 \
  --mtl_lambda 0.1 --mtl_lambda_warmup_epochs 10 --mtl_detach_window_head \
  --output_file paper_baseline_vs_mtl_seed42_lam01_detach.json \
  >> train_lam01_detach.log 2>&1
"

screen -S painjob3 -dm bash -c "
source ~/.bashrc
conda activate eeg

python scripts/dl_train_1000hz_gridsearch_fixed.py --task pain_threshold --models deep_cnn_lstm --no-grid-search --seed 42 --output_file baseline.json
  >>  train_dl_newest.log 2>&1
"
python /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/newest_dl.py --task pain_threshold --models deep_cnn_lstm --no-grid-search --seed 42 --output_file baseline.json


python /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/newest_dl.py --task pain_threshold --models multitask_deep_cnn_lstm --no-grid-search --seed 42 --output_file mtl.json

"""