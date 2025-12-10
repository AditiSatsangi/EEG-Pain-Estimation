import os
import sys
import argparse
import json
import random
import time
import hashlib
from pathlib import Path
from collections import Counter
from itertools import product
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# REPRODUCIBILITY
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
# DATA LOADING
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
        print(f"  Filtered rejected epochs: {before - len(df)} removed")
    return df


def get_most_common_channels(df: pd.DataFrame) -> int:
    if "n_channels" in df.columns:
        cnt = Counter(df["n_channels"].dropna().astype(int))
        if cnt:
            return cnt.most_common(1)[0][0]
    return 64


class EEGDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root: str,
        most_ch: int,
        return_window: bool = False,
        window_le: LabelEncoder = None,
    ):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.most_ch = most_ch
        self.return_window = return_window
        self.window_le = window_le

        if self.return_window and self.window_le and "window" in self.df.columns:
            self.window_indices = self.window_le.transform(self.df["window"])
        else:
            self.window_indices = np.zeros(len(self.df), dtype=int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npz_file = row.get("npz_file", row.get("path", ""))
        npz_path = os.path.join(self.root, "npz", npz_file)

        with np.load(npz_path, allow_pickle=True) as data:
            X = data["X"]  # (channels, time)

        ch, T = X.shape
        if ch < self.most_ch:
            pad = np.zeros((self.most_ch - ch, T), dtype=X.dtype)
            X = np.vstack([X, pad])
        elif ch > self.most_ch:
            X = X[: self.most_ch, :]

        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mean) / std

        window_idx = self.window_indices[idx] if self.return_window else 0
        return torch.FloatTensor(X), int(window_idx)


def load_task_data(
    df: pd.DataFrame, task: str
) -> Tuple[np.ndarray, LabelEncoder, pd.DataFrame, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    window_le = LabelEncoder()

    if "window" not in df.columns:
        print("⚠ 'window' column missing — creating dummy 'unknown' window.")
        df = df.copy()
        df["window"] = "unknown"

    y_window = window_le.fit_transform(df["window"])

    if task == "pain_5class":
        y = le.fit_transform(df["rating_bin"])
    elif task == "none_vs_pain":
        df = df.copy()
        df["binary"] = df["rating_bin"].apply(
            lambda x: "none" if x == "none" else "pain"
        )
        y = le.fit_transform(df["binary"])
    elif task == "pain_only":
        pain_df = df[df["rating_bin"] != "none"].copy()
        if len(pain_df) == 0:
            raise ValueError("No pain samples found")
        y = le.fit_transform(pain_df["rating_bin"])
        df = pain_df
        y_window = window_le.transform(df["window"])
    elif task == "pain_threshold":
        df = df.copy()

        def threshold_label(rb):
            return "no_significant_pain" if rb in ["none", "low"] else "significant_pain"

        df["threshold_label"] = df["rating_bin"].apply(threshold_label)
        y = le.fit_transform(df["threshold_label"])
    else:
        raise ValueError(f"Unknown task: {task}")

    return y, le, df, y_window, window_le


def load_all_segments(
    df: pd.DataFrame,
    root: str,
    most_ch: int,
    return_window: bool = False,
    window_le: LabelEncoder = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    print("\n" + "=" * 70)
    print(" DATA LOADING")
    print("=" * 70)
    print(f"Total samples to load: {len(df):,}")
    print(f"Most common channels: {most_ch}")
    print(f"NPZ directory: {os.path.join(root, 'npz')}")

    if "npz_file" not in df.columns and "path" in df.columns:
        df = df.copy()
        df["npz_file"] = df["path"]
        print("Using 'path' column as 'npz_file'")
    else:
        print("Using 'npz_file' column")

    dataset = EEGDataset(df, root, most_ch, return_window=return_window, window_le=window_le)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_X, all_W = [], []
    for xb, wb in tqdm(loader, desc="Loading segments", unit="batch"):
        all_X.append(xb)
        if return_window:
            all_W.append(wb)

    X_tensor = torch.cat(all_X, dim=0)
    W_tensor = (
        torch.cat(all_W, dim=0)
        if return_window
        else torch.zeros(len(X_tensor), dtype=torch.long)
    )

    print(f"\n✓ Loaded {X_tensor.shape[0]:,} segments")
    print(
        f"  Shape: (samples={X_tensor.shape[0]}, channels={X_tensor.shape[1]}, time={X_tensor.shape[2]})"
    )
    return X_tensor, W_tensor


# =============================================================================
# MODEL: WINDOW-AWARE DEEP CNN-LSTM
# =============================================================================

class WindowAware_DeepCNN_LSTM(nn.Module):
    """
    Window-aware Deep CNN-LSTM:
      - Deep CNN over (channels, time)
      - Window embedding (baseline/ERP/post) fused at sequence level
      - BiLSTM over time
    """

    def __init__(
        self,
        n_channels: int,
        n_time: int,
        n_classes: int = 2,
        n_windows: int = 3,
        cnn_filters: List[int] = [32, 64, 128],
        lstm_hidden: int = 192,
        lstm_layers: int = 2,
        window_embed_dim: int = 16,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.window_embedding = nn.Embedding(n_windows, window_embed_dim)

        self.conv1 = nn.Conv2d(
            1, cnn_filters[0], kernel_size=(n_channels // 8, 3), padding=(0, 1)
        )
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(
            cnn_filters[0], cnn_filters[1], kernel_size=(1, 3), padding=(0, 1)
        )
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(
            cnn_filters[1], cnn_filters[2], kernel_size=(1, 3), padding=(0, 1)
        )
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        self.pool3 = nn.MaxPool2d((1, 2))

        time_reduced = n_time // 8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, time_reduced))

        lstm_input_dim = cnn_filters[2] + window_embed_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            lstm_hidden,
            batch_first=True,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2 + window_embed_dim, n_classes)

    def forward(self, x, window_idx):
        # x: (B, C, T)
        B = x.size(0)
        w_emb = self.window_embedding(window_idx)  # (B, D_w)

        x = x.unsqueeze(1)  # (B,1,C,T)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.adaptive_pool(x)  # (B, F3, 1, T')
        x = x.squeeze(2).transpose(1, 2)  # (B, T', F3)
        seq_len = x.size(1)

        w_exp = w_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, w_exp], dim=2)  # (B, T', F3 + D_w)

        _, (h, _) = self.lstm(x)  # h: (num_layers*2, B, H)
        h = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2H)

        h = torch.cat([h, w_emb], dim=1)  # (B, 2H + D_w)
        h = self.dropout(h)
        return self.fc(h)


# =============================================================================
# TRAINING / EVAL
# =============================================================================

def class_weights_from_y(y: np.ndarray, class_names: list = None):
    unique, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(unique) * counts)
    info = {}
    for i, (cls, cnt) in enumerate(zip(unique, counts)):
        name = class_names[cls] if class_names is not None else str(cls)
        info[name] = {
            "count": int(cnt),
            "weight": float(weights[i]),
            "pct": float(cnt / len(y) * 100),
        }
    return torch.FloatTensor(weights), info


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    device: str,
    epochs: int = 50,
    patience: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    class_names: list = None,
    model_name: str = "WINDOW_AWARE_DCNN_LSTM",
) -> Tuple[nn.Module, float, Dict]:
    print("\n" + "=" * 70)
    print(f" TRAINING: {model_name}")
    print("=" * 70)

    model.to(device)

    weights, info = class_weights_from_y(y_train, class_names)
    weights = weights.to(device)
    print("\nClass weights:")
    for k, v in info.items():
        print(f"  {k:20s} count={v['count']:6d} ({v['pct']:5.2f}%)  w={v['weight']:.4f}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )  # <-- no verbose kwarg (fix)

    best_state = None
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    print(
        f"\n{'Epoch':<6} {'TrainLoss':<10} {'ValLoss':<10} {'ValAcc':<8} "
        f"{'ValF1':<8} {'LR':<10} {'Status'}"
    )
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, wb, yb in train_loader:
            xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, wb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        tr_loss = float(np.mean(train_losses))

        # validation
        model.eval()
        val_losses = []
        preds, targets = [], []
        with torch.no_grad():
            for xb, wb, yb in val_loader:
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                out = model(xb, wb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                preds.append(out.argmax(dim=1).cpu().numpy())
                targets.append(yb.cpu().numpy())

        vloss = float(np.mean(val_losses))
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        val_acc = float((preds == targets).mean())
        val_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))

        scheduler.step(vloss)
        lr_now = optimizer.param_groups[0]["lr"]

        status = ""
        if vloss < best_val_loss - 1e-4:
            best_val_loss = vloss
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            status = "✓ new best"
        else:
            no_improve += 1
            status = f"no improve ({no_improve}/{patience})"
            if no_improve >= patience:
                status = "⏹ early stop"

        print(
            f"{epoch:<6d} {tr_loss:<10.4f} {vloss:<10.4f} {val_acc:<8.4f} "
            f"{val_f1:<8.4f} {lr_now:<10.2e} {status}"
        )

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(
            f"\n✓ Loaded best epoch {best_epoch} | "
            f"ValLoss={best_val_loss:.4f} ValAcc={best_val_acc:.4f} ValF1={best_val_f1:.4f}"
        )
    else:
        print("\n⚠ No best_state saved; using final weights")

    info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "total_epochs": epoch,
        "early_stopped": no_improve >= patience,
    }

    return model, best_val_f1, info


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    le: LabelEncoder,
    window_le: LabelEncoder,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" EVALUATION (TEST SET)")
    print("=" * 70)

    model.eval()
    all_preds, all_targets, all_probs, all_windows = [], [], [], []

    with torch.no_grad():
        for xb, wb, yb in tqdm(test_loader, desc="Evaluating", unit="batch"):
            xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
            out = model(xb, wb)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_windows.append(wb.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_probs = np.concatenate(all_probs)
    windows = np.concatenate(all_windows)

    acc = float((y_pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision, recall, f1_pc, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    report = classification_report(
        y_true, y_pred, target_names=le.classes_, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    max_probs = np.max(y_probs, axis=1)
    mean_conf = float(max_probs.mean())
    std_conf = float(max_probs.std())

    print(f"\nAccuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"Macro F1:          {f1_macro:.4f}")
    print(f"Weighted F1:       {f1_weighted:.4f}")
    print(f"Mean confidence:   {mean_conf:.4f} ± {std_conf:.4f}")

    per_window_metrics = {}
    print("\nPer-Window Performance:")
    for w_idx, w_name in enumerate(window_le.classes_):
        mask = windows == w_idx
        if np.any(mask):
            w_acc = float((y_pred[mask] == y_true[mask]).mean())
            w_f1 = float(
                f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            )
            per_window_metrics[w_name] = {
                "accuracy": w_acc,
                "f1_macro": w_f1,
                "n_samples": int(mask.sum()),
            }
            print(f"  {w_name:12s}: Acc={w_acc:.4f}, F1={w_f1:.4f} ({mask.sum()} samples)")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1_pc.tolist(),
        "support_per_class": support.tolist(),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": le.classes_.tolist(),
        "mean_confidence": mean_conf,
        "std_confidence": std_conf,
        "per_window_metrics": per_window_metrics,
    }


# =============================================================================
# GRID SEARCH (ONLY window_aware_deep_cnn_lstm)
# =============================================================================

def grid_search_window_aware(
    X_all: torch.Tensor,
    y_all: np.ndarray,
    y_window_all: np.ndarray,
    groups: np.ndarray,
    n_channels: int,
    n_time: int,
    n_classes: int,
    n_windows: int,
    device: str,
    seed: int,
    param_grid: Dict[str, List[Any]],
    epochs: int = 30,
    patience: int = 10,
) -> Tuple[Dict, float]:
    print("\n" + "=" * 70)
    print(" GRID SEARCH: WINDOW_AWARE_DEEP_CNN_LSTM")
    print("=" * 70)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(X_all, y_all, groups))

    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    w_train, w_val = y_window_all[train_idx], y_window_all[val_idx]

    print(
        f"Grid search split: Train {len(y_train):,} (subjects={len(np.unique(groups[train_idx]))}), "
        f"Val {len(y_val):,} (subjects={len(np.unique(groups[val_idx]))})"
    )

    param_keys = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_keys]
    combinations = list(product(*param_values))

    print(f"\nTotal combinations: {len(combinations)}")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")

    best_f1 = 0.0
    best_params = None

    for i, values in enumerate(combinations, 1):
        params = dict(zip(param_keys, values))
        print(f"\n[{i}/{len(combinations)}] Testing params:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        model_params = {
            k: v
            for k, v in params.items()
            if k not in ["lr", "weight_decay", "batch_size"]
        }
        train_params = {k: v for k, v in params.items() if k in ["lr", "weight_decay"]}
        batch_size = params.get("batch_size", 64)

        model = WindowAware_DeepCNN_LSTM(
            n_channels,
            n_time,
            n_classes=n_classes,
            n_windows=n_windows,
            **model_params,
        )

        train_ds = TensorDataset(
            X_train, torch.LongTensor(w_train), torch.LongTensor(y_train)
        )
        val_ds = TensorDataset(
            X_val, torch.LongTensor(w_val), torch.LongTensor(y_val)
        )

        gen = torch.Generator()
        gen.manual_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            generator=gen,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        try:
            model, val_f1, info = train_one_model(
                model,
                train_loader,
                val_loader,
                y_train,
                device,
                epochs=epochs,
                patience=patience,
                model_name="WINDOW_AWARE_DCNN_LSTM (GRID)",
                **train_params,
            )
            print(
                f"  → ValF1={val_f1:.4f}  ValAcc={info['best_val_acc']:.4f}  best_epoch={info['best_epoch']}"
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = params
                print("  ✓ New best hyperparams!")
        except Exception as e:
            print(f"  ✗ Config failed: {e}")
            import traceback

            traceback.print_exc()

    if best_params is None:
        print("\n⚠ Grid search failed; returning empty params.")
        return {}, 0.0

    print("\nBest params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best grid-search F1: {best_f1:.4f}")
    return best_params, best_f1


# =============================================================================
# GRAD-CAM FOR WINDOW-AWARE MODEL
# =============================================================================

class GradCAM_WindowAware:
    """
    Grad-CAM over conv3 feature maps for WindowAware_DeepCNN_LSTM.

    Fixes:
      - Uses register_full_backward_hook (no deprecation warning)
      - Forces model into train() for cuDNN RNN backward
    """

    def __init__(self, model: WindowAware_DeepCNN_LSTM, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out

        def bwd_hook(module, grad_input, grad_output):
            # grad_output is a tuple; take grad wrt output
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, x: torch.Tensor, window_idx: torch.Tensor, class_idx: int = None):
        """
        x: (1, C, T)
        window_idx: (1,)
        """
        # Ensure gradients are tracked
        was_training = self.model.training
        self.model.zero_grad()
        self.model.train()  # IMPORTANT: cuDNN RNN backward requires training mode

        x = x.requires_grad_(True)

        out = self.model(x, window_idx)  # (1, num_classes)

        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())

        score = out[0, class_idx]
        score.backward(retain_graph=False)

        # activations: (B, F, H, W), gradients: same
        grads = self.gradients  # (1, F, H, W)
        acts = self.activations

        # Global average pooling over spatial dims
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, F, 1, 1)
        cam = (weights * acts).sum(dim=1)  # (1, H, W)

        cam = torch.relu(cam)
        cam_min = cam.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        cam_max = cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Restore training flag
        self.model.train(was_training)

        return cam.detach().cpu().numpy()  # (1, H, W)

import torch.nn.functional as F

def project_cam_to_eeg_time(cam, original_time_len):
    """
    cam: numpy array of shape (H, W) or (1, H, W)
    original_time_len: T of raw EEG
    """
    if cam.ndim == 3:
        cam = cam[0]

    # Average over conv-channel dimension → (W,)
    cam_time = cam.mean(axis=0)

    # Normalize
    cam_time = (cam_time - cam_time.min()) / (cam_time.max() - cam_time.min() + 1e-8)

    # Convert to torch for interpolation
    cam_time_torch = torch.tensor(cam_time)[None, None, :]  # (1,1,W)

    cam_up = F.interpolate(
        cam_time_torch,
        size=original_time_len,
        mode="linear",
        align_corners=False
    )

    return cam_up.squeeze().numpy()  # (T,)

def run_window_aware_interpretability(
    model: WindowAware_DeepCNN_LSTM,
    test_loader: DataLoader,
    device: str,
    window_le: LabelEncoder,
):
    print("\n" + "=" * 70)
    print(" RUNNING WINDOW-AWARE INTERPRETABILITY ")
    print("=" * 70)

    # Take one batch
    sample_x, sample_w, sample_y = next(iter(test_loader))
    sample_x = sample_x.to(device)
    sample_w = sample_w.to(device)

    # Use first sample
    x0 = sample_x[:1]
    w0 = sample_w[:1]

    grad_cam = GradCAM_WindowAware(model, model.conv3)
    cam = grad_cam.generate(x0, w0)  # (1, H, W)

    np.save("gradcam_windowaware.npy", cam)
    print("✓ Grad-CAM array saved: gradcam_windowaware.npy")

    plt.figure(figsize=(8, 4))
    plt.imshow(cam[0], aspect="auto", cmap="jet")
    plt.colorbar()
    plt.title("Grad-CAM (Window-Aware Deep CNN-LSTM)")
    plt.xlabel("Conv-time dimension")
    plt.ylabel("Conv-channel dimension")
    plt.tight_layout()
    plt.savefig("gradcam_windowaware.png", dpi=300)
    plt.close()
    print("✓ Grad-CAM heatmap saved: gradcam_windowaware.png")

    # Window embeddings
    w_emb = model.window_embedding.weight.detach().cpu().numpy()
    np.save("window_embeddings.npy", w_emb)
    print("✓ Window embedding matrix saved: window_embeddings.npy")

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        w_emb,
        annot=False,
       xticklabels=[f"dim-{i}" for i in range(w_emb.shape[1])],
        yticklabels=window_le.classes_,
        cmap="viridis",
    )
    plt.title("Window Embeddings (Baseline / ERP / Post)")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Window type")
    plt.tight_layout()
    plt.savefig("window_embedding_heatmap.png", dpi=300)
    plt.close()
    print("✓ Window embedding heatmap saved: window_embedding_heatmap.png")


    # Get original EEG signal (one channel for visualization)
    raw_eeg = x0[0, 0].detach().cpu().numpy()   # channel-0 for example
    T = raw_eeg.shape[0]

    cam_time = project_cam_to_eeg_time(cam, T)

    # Plot overlay
    plt.figure(figsize=(10,4))
    plt.plot(raw_eeg, label="Raw EEG", alpha=0.7)
    plt.plot(cam_time * np.max(np.abs(raw_eeg)), 
            label="Projected Grad-CAM", linewidth=2)
    plt.legend()
    plt.title("Grad-CAM projected back to original EEG time")
    plt.xlabel("Time samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("gradcam_projected_on_eeg.png", dpi=300)
    plt.close()

    print("✓ Projected Grad-CAM saved: gradcam_projected_on_eeg.png")

    print("\n✓ INTERPRETABILITY COMPLETE\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        default="pain_threshold",
        choices=["pain_5class", "none_vs_pain", "pain_only", "pain_threshold"],
    )
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grid-epochs", type=int, default=20)
    ap.add_argument("--grid-patience", type=int, default=7)
    ap.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip grid search and use fixed hyperparams",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_file", type=str, default="results_windowaware.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    cwd = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(cwd, ".."))
    root = args.data_root if args.data_root else find_data_root(proj_root, "Data")

    print("\n" + "=" * 70)
    print(" DATASET SETUP")
    print("=" * 70)
    print(f"Data root: {root}")
    df = load_index(root)
    print(f"✓ index.csv rows: {len(df):,}")

    most_ch = get_most_common_channels(df)
    print(f"Most common channels: {most_ch}")

    print("\n" + "=" * 70)
    print(" TASK PREPARATION")
    print("=" * 70)
    print(f"Task: {args.task}")
    y_all, le, df, y_window, window_le = load_task_data(df, args.task)
    groups = df["participant"].values
    n_classes = len(le.classes_)
    n_windows = len(window_le.classes_)

    print(f"Classes: {list(le.classes_)}")
    print(f"Windows: {list(window_le.classes_)}")
    print(f"Total samples: {len(y_all):,}")
    print(f"Subjects: {len(np.unique(groups))}")

    print("\nClass distribution:")
    cd = Counter(y_all)
    for idx, cnt in sorted(cd.items()):
        name = le.classes_[idx]
        pct = cnt / len(y_all) * 100
        print(f"  {name:20s}: {cnt:6d} ({pct:5.2f}%)")

    X_all, W_all = load_all_segments(
        df, root, most_ch, return_window=True, window_le=window_le
    )
    n_channels, n_time = X_all.shape[1], X_all.shape[2]

    print("\n" + "=" * 70)
    print(" TRAIN/TEST SPLIT (SUBJECT-WISE)")
    print("=" * 70)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups))

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    W_train, W_test = W_all[train_idx], W_all[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    print(
        f"Train: {len(y_train):,} samples, {len(np.unique(groups_train))} subjects\n"
        f"Test:  {len(y_test):,} samples, {len(np.unique(groups_test))} subjects"
    )

    # ----------------- HYPERPARAMS -----------------
    param_grid = {
        "cnn_filters": [[32, 64, 128]],
        "lstm_hidden": [192, 256],
        "lstm_layers": [2],
        "window_embed_dim": [16, 32],
        "dropout": [0.3, 0.4],
        "lr": [5e-4, 1e-3],
        "weight_decay": [1e-5],
        "batch_size": [64],
    }

    best_known = {
        "cnn_filters": [32, 64, 128],
        "lstm_hidden": 192,
        "lstm_layers": 2,
        "window_embed_dim": 16,
        "dropout": 0.4,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "batch_size": 64,
    }

    if args.no_grid_search:
        print("\nSkipping grid search, using fixed hyperparameters.")
        best_params = best_known
        grid_best_f1 = None
    else:
        best_params, grid_best_f1 = grid_search_window_aware(
            X_train,
            y_train,
            W_train.numpy(),
            groups_train,
            n_channels,
            n_time,
            n_classes,
            n_windows,
            device,
            args.seed,
            param_grid,
            epochs=args.grid_epochs,
            patience=args.grid_patience,
        )
        if not best_params:
            print("\n⚠ Grid search failed; falling back to best_known defaults.")
            best_params = best_known

    print("\n" + "=" * 70)
    print(" FINAL TRAINING WITH BEST HYPERPARAMS")
    print("=" * 70)
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    model_params = {
        k: v
        for k, v in best_params.items()
        if k not in ["lr", "weight_decay", "batch_size"]
    }
    train_params = {k: v for k, v in best_params.items() if k in ["lr", "weight_decay"]}
    batch_size = best_params.get("batch_size", 64)

    model = WindowAware_DeepCNN_LSTM(
        n_channels,
        n_time,
        n_classes=n_classes,
        n_windows=n_windows,
        **model_params,
    )

    inner_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=args.seed
    )
    tr_idx, val_idx = next(inner_splitter.split(X_train, y_train, groups_train))
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    W_tr, W_val = W_train[tr_idx], W_train[val_idx]

    train_ds = TensorDataset(X_tr, torch.LongTensor(W_tr.numpy()), torch.LongTensor(y_tr))
    val_ds = TensorDataset(X_val, torch.LongTensor(W_val.numpy()), torch.LongTensor(y_val))
    test_ds = TensorDataset(
        X_test, torch.LongTensor(W_test.numpy()), torch.LongTensor(y_test)
    )

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(
        f"\nTrain batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    model, best_val_f1, train_info = train_one_model(
        model,
        train_loader,
        val_loader,
        y_tr,
        device,
        epochs=args.epochs,
        patience=args.patience,
        class_names=le.classes_.tolist(),
        **train_params,
    )

    metrics = evaluate_model(model, test_loader, device, le, window_le)
    metrics["training_info"] = train_info
    metrics["grid_best_f1"] = grid_best_f1
    metrics["best_hyperparameters"] = best_params

    # ------------ Interpretability ------------
    run_window_aware_interpretability(model, test_loader, device, window_le)

    # ------------ Save results ------------
    out_path = os.path.join(proj_root, args.output_file)
    with open(out_path, "w") as f:
        json.dump(
            {
                "task": args.task,
                "seed": args.seed,
                "n_channels": int(n_channels),
                "n_time": int(n_time),
                "n_classes": n_classes,
                "class_names": le.classes_.tolist(),
                "grid_search": not args.no_grid_search,
                "results": metrics,
            },
            f,
            indent=2,
        )
    print(f"\n✓ Results saved to: {out_path}")
    print("\nTRAINING + INTERPRETABILITY COMPLETE\n")


if __name__ == "__main__":
    main()

"""
screen -dmS pain_window bash -lc "
source ~/miniconda3/etc/profile.d/conda.sh &&
conda activate eeg &&
python /home/asatsan2/Projects/EEG-Pain-Estimation/notebooks/new.train.py \
  --task pain_threshold \
  --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data \
  --epochs 30 \
  --patience 10 \
  --seed 42 \
  --no-grid-search \
  > /home/asatsan2/Projects/EEG-Pain-Estimation/train_window_intereoret.log 2>&1
"
screen -dmS pain bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg && python /home/asatsan2/Projects/EEG-Pain-Estimation/notebooks/new.train.py --task none_vs_pain --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data --epochs 30 --patience 10 --seed 42 --no-grid-search > /home/asatsan2/Projects/EEG-Pain-Estimation/train_window.log 2>&1"

screen -S painjob_windowaware -dm bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg && python /home/asatsan2/Projects/EEG-Pain-Estimation/notebooks/new.train.py --task pain_threshold --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data --epochs 30 --no-grid-search --patience 10 --models window_aware_deep_cnn_lstm --seed 42 > /home/asatsan2/Projects/EEG-Pain-Estimation/train_window.log 2>&1"
 
"""