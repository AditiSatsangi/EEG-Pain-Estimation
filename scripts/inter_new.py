# ===========================
# MULTI-THRESHOLD VERSION (stable / no-random-stop)
# Key stability fixes:
# 1) NO preloading all segments into RAM (removes huge memory spikes)
# 2) num_workers=0 + pin_memory=False by default (avoids dataloader dead/worker OOM)
# 3) Saliency runs with a SMALL batch size + optional max_samples cap
# 4) Robust error prints + unbuffered-friendly logs
# ===========================

import os
import argparse
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple, Dict, Any, List, Optional

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
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✓ Seed set to {seed}", flush=True)


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
# Threshold mapping
# =============================================================================
def create_threshold_label_categorical(rating_bin: str, threshold: int) -> str:
    """
    Categorical threshold mapping:
    - threshold=3: none/low/mid vs high/extreme
    - threshold=5: none/low vs mid/high/extreme (default)
    - threshold=7: none/low/mid/high vs extreme
    """
    rb = str(rating_bin).strip().lower()

    pain_levels = {
        'none': 0, 'no_pain': 0, 'no': 0,
        'low': 1,
        'mid': 2, 'medium': 2, 'moderate': 2,
        'high': 3,
        'extreme': 4, 'severe': 4
    }
    level = pain_levels.get(rb, 2)  # default mid

    if threshold == 3:
        return 'no_significant_pain' if level <= 2 else 'significant_pain'
    elif threshold == 5:
        return 'no_significant_pain' if level <= 1 else 'significant_pain'
    elif threshold == 7:
        return 'no_significant_pain' if level <= 3 else 'significant_pain'
    else:
        # fallback: treat threshold as integer boundary on "level"
        return 'no_significant_pain' if level < threshold else 'significant_pain'


# =============================================================================
# Data loading
# =============================================================================
def load_index(root: str) -> pd.DataFrame:
    index_path = os.path.join(root, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.csv not found at {index_path}")

    df = pd.read_csv(index_path)

    if "reject_flag" in df.columns:
        before = len(df)
        rej = df["reject_flag"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[~rej].copy()
        print(f"Filtered rejected epochs: {before - len(df)} removed", flush=True)

    if "npz_file" not in df.columns:
        if "path" in df.columns:
            df = df.copy()
            df["npz_file"] = df["path"]
            print("Using 'path' as 'npz_file'", flush=True)
        else:
            raise ValueError("Need 'npz_file' or 'path' column in index.csv")

    if "window" not in df.columns:
        df = df.copy()
        df["window"] = "unknown"

    if "participant" not in df.columns:
        raise ValueError("index.csv must contain 'participant' column for group split")

    if "rating_bin" not in df.columns:
        raise ValueError("index.csv must contain 'rating_bin' column for thresholding")

    return df


def get_most_common_channels(df: pd.DataFrame) -> int:
    if "n_channels" in df.columns:
        cnt = Counter(df["n_channels"].dropna().astype(int))
        return cnt.most_common(1)[0][0] if cnt else 64
    return 64


def make_labels_for_threshold(df: pd.DataFrame, threshold: int) -> Tuple[np.ndarray, LabelEncoder]:
    df = df.copy()
    df["threshold_label"] = df["rating_bin"].apply(lambda x: create_threshold_label_categorical(x, threshold))
    le = LabelEncoder()
    y = le.fit_transform(df["threshold_label"].values)
    return y, le


class EEGDataset(Dataset):
    """
    Lazy loads NPZ on-the-fly. This avoids preloading all X into RAM,
    which is the biggest reason your process can get killed.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        root: str,
        most_ch: int,
        window_le: LabelEncoder,
        y: Optional[np.ndarray] = None,           # pain labels for current threshold split
        return_window: bool = False,              # whether to return window index
        return_label: bool = True                 # whether to return y
    ):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.most_ch = int(most_ch)
        self.window_le = window_le
        self.return_window = bool(return_window)
        self.return_label = bool(return_label)
        self.y = y if y is not None else None

        # precompute window indices (cheap)
        self.window_idx = self.window_le.transform(self.df["window"].values)

        if self.return_label and self.y is None:
            raise ValueError("return_label=True but y=None. Pass y for this split.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npz_path = os.path.join(self.root, "npz", row["npz_file"])
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
        X = torch.from_numpy(X).float()

        out = [X]

        if self.return_window:
            out.append(torch.tensor(int(self.window_idx[idx]), dtype=torch.long))

        if self.return_label:
            out.append(torch.tensor(int(self.y[idx]), dtype=torch.long))

        return tuple(out)


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


class MultiTask_DeepCNN_LSTM(nn.Module):
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
    return float(base_lambda * max(0.0, (1.0 - t)))


def _compute_window_class_weights_from_loader(train_loader: DataLoader) -> torch.FloatTensor:
    all_w = []
    for batch in train_loader:
        # batch = (X, W, y)
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
                    epochs: int,
                    patience: int,
                    lr: float,
                    weight_decay: float,
                    model_name: str,
                    mtl_lambda: float = 0.2,
                    mtl_lambda_warmup_epochs: int = 10,
                    mtl_detach_window_head: bool = False,
                    window_loss_class_weighted: bool = False) -> Tuple[nn.Module, Dict[str, Any]]:

    model.to(device)
    w = class_weights_from_y(y_train).to(device)
    criterion_pain = nn.CrossEntropyLoss(weight=w)

    is_mtl = isinstance(model, MultiTask_DeepCNN_LSTM)

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

            if is_mtl:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                pain_logits, win_logits = model(xb, detach_window_head=mtl_detach_window_head)

                loss_p = criterion_pain(pain_logits, yb)
                loss_w = criterion_window(win_logits, wb)
                lam = mtl_lambda_schedule(epoch, epochs, mtl_lambda, mtl_lambda_warmup_epochs)
                loss = loss_p + lam * loss_w
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
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
                if is_mtl:
                    xb, wb, yb = batch
                    xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                    pain_logits, win_logits = model(xb, detach_window_head=False)
                    loss_p = criterion_pain(pain_logits, yb)
                    loss_w = criterion_window(win_logits, wb)
                    lam = mtl_lambda_schedule(epoch, epochs, mtl_lambda, mtl_lambda_warmup_epochs)
                    loss = loss_p + lam * loss_w
                else:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
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

        # print every 5 epochs, and ALWAYS flush
        if (epoch + 1) % 5 == 0:
            if is_mtl:
                lam_now = mtl_lambda_schedule(epoch, epochs, mtl_lambda, mtl_lambda_warmup_epochs)
                print(f"[{model_name}] epoch {epoch+1}/{epochs} train={np.mean(train_losses):.4f} val={vloss:.4f} lambda={lam_now:.3f}", flush=True)
            else:
                print(f"[{model_name}] epoch {epoch+1}/{epochs} train={np.mean(train_losses):.4f} val={vloss:.4f}", flush=True)

        if no_improve >= patience:
            print(f"[{model_name}] early stop: no_improve={no_improve} >= patience={patience}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"✓ {model_name} best epoch = {best_epoch} (val loss {best_val_loss:.4f})", flush=True)

    info = {"best_epoch": best_epoch, "best_val_loss": best_val_loss}
    if is_mtl:
        info.update({
            "mtl_lambda": float(mtl_lambda),
            "mtl_lambda_warmup_epochs": int(mtl_lambda_warmup_epochs),
            "mtl_detach_window_head": bool(mtl_detach_window_head),
            "window_loss_class_weighted": bool(window_loss_class_weighted),
        })
    return model, info


def compute_grad_input_saliency(model, loader, device, is_multitask=False, max_batches: Optional[int] = None):
    """
    Stable saliency:
    - Uses SMALL batch loader
    - Optional max_batches cap to avoid huge runtime/memory
    """
    was_training = model.training
    model.train()  # cuDNN LSTM backward sometimes needs train()

    saliency_list = []
    windows_list = []

    for bi, batch in enumerate(tqdm(loader, desc="Saliency (grad*input)", unit="batch")):
        if max_batches is not None and bi >= max_batches:
            break

        xb, wb, yb = batch  # (X,W,y) required
        xb = xb.to(device).requires_grad_(True)
        windows_list.append(wb.detach().cpu().numpy())

        if is_multitask:
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

        attr = (xb.grad * xb).detach().abs()     # [B,C,T]
        sal = attr.mean(dim=1).cpu().numpy()     # [B,T]
        saliency_list.append(sal)

        # prevent GPU accumulation
        del xb, logits, chosen, pred, attr
        if device == "cuda":
            torch.cuda.empty_cache()

    model.train(was_training)
    saliency_all = np.concatenate(saliency_list, axis=0) if len(saliency_list) else np.zeros((0, 1), dtype=np.float32)
    windows_all = np.concatenate(windows_list, axis=0) if len(windows_list) else np.zeros((0,), dtype=np.int64)
    return saliency_all, windows_all


def evaluate_model(model: nn.Module,
                   test_loader_xy: DataLoader,
                   test_loader_xwy: DataLoader,
                   device: str,
                   le: LabelEncoder,
                   window_le: LabelEncoder,
                   model_name: str,
                   compute_interpretability: bool,
                   saliency_max_batches: Optional[int] = None) -> Dict[str, Any]:

    model.eval()
    is_mtl = isinstance(model, MultiTask_DeepCNN_LSTM)

    all_preds, all_targets = [], []
    all_windows = []
    all_window_preds = []

    with torch.no_grad():
        loader = test_loader_xwy if is_mtl else test_loader_xy
        for batch in tqdm(loader, desc=f"Eval {model_name}", unit="batch"):
            if is_mtl:
                xb, wb, yb = batch
                xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
                pain_logits, win_logits = model(xb, detach_window_head=False)
                logits = pain_logits
                all_windows.append(wb.detach().cpu().numpy())
                all_window_preds.append(win_logits.argmax(dim=1).detach().cpu().numpy())
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.detach().cpu().numpy())

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

    if is_mtl and len(all_windows):
        windows = np.concatenate(all_windows)
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

        y_win_true = windows
        y_win_pred = np.concatenate(all_window_preds)
        metrics["window_task"] = {
            "accuracy": float((y_win_pred == y_win_true).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_win_true, y_win_pred)),
            "f1_macro": float(f1_score(y_win_true, y_win_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_win_true, y_win_pred).tolist(),
            "report": classification_report(y_win_true, y_win_pred, target_names=window_le.classes_, zero_division=0),
        }

    # interpretability only if requested (primary threshold only)
    if compute_interpretability:
        # IMPORTANT: use a SMALL batch saliency loader (avoid OOM)
        sal_loader = test_loader_xwy  # already X,W,y
        saliency, win_idx = compute_grad_input_saliency(
            model, sal_loader, device,
            is_multitask=is_mtl,
            max_batches=saliency_max_batches
        )
        # if capped batches, this is a subset interpretability (still OK)
        energy = saliency.mean(axis=1) if saliency.shape[0] else np.array([])

        per_win_energy = {}
        for i, name in enumerate(window_le.classes_):
            m = (win_idx == i)
            if np.any(m):
                per_win_energy[name] = {
                    "mean_energy": float(energy[m].mean()),
                    "std_energy": float(energy[m].std()),
                    "n": int(m.sum())
                }

        def find_idx(names, key):
            key = key.lower()
            for i, n in enumerate(names):
                if key in n.lower():
                    return i
            return None

        erp_i = find_idx(window_le.classes_, "erp")
        base_i = find_idx(window_le.classes_, "base")
        post_i = find_idx(window_le.classes_, "post")

        erp_align_ratio = None
        if energy.size and erp_i is not None and base_i is not None and post_i is not None:
            e_erp = float(energy[win_idx == erp_i].mean())
            e_base = float(energy[win_idx == base_i].mean())
            e_post = float(energy[win_idx == post_i].mean())
            erp_align_ratio = float(e_erp / (e_base + e_post + 1e-8))

        metrics["interpretability_quant"] = {
            "method": "grad_x_input_mean_over_channels_then_mean_over_time",
            "per_window_saliency_energy": per_win_energy,
            "erp_alignment_ratio": erp_align_ratio,
            "note": "Saliency computed with small-batch loader for stability."
        }

    return metrics


# =============================================================================
# Main (multi-threshold) - LAZY LOAD VERSION
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--output_file", type=str, default="paper_results_multi_threshold.json")

    # Threshold options
    ap.add_argument("--thresholds", type=int, nargs="+", default=[5, 3, 7],
                    help="Thresholds to evaluate, e.g., --thresholds 5 3 7")
    ap.add_argument("--primary_threshold", type=int, default=5,
                    help="Only this threshold computes interpretability; others are perf-only")

    # MTL knobs
    ap.add_argument("--mtl_lambda", type=float, default=0.2)
    ap.add_argument("--mtl_lambda_warmup_epochs", type=int, default=10)
    ap.add_argument("--mtl_detach_window_head", action="store_true")
    ap.add_argument("--window_loss_class_weighted", action="store_true")

    # Stability knobs
    ap.add_argument("--num_workers", type=int, default=0, help="Use 0 for max stability")
    ap.add_argument("--pin_memory", action="store_true", help="Off by default; enable only if stable")
    ap.add_argument("--saliency_batch_size", type=int, default=8, help="Small batch for saliency")
    ap.add_argument("--saliency_max_batches", type=int, default=0,
                    help="0 = no cap; else compute saliency only for first N batches")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    df = load_index(args.data_root)
    most_ch = get_most_common_channels(df)

    # window encoder over ALL data (stable)
    window_le = LabelEncoder().fit(df["window"].values)
    n_windows = len(window_le.classes_)

    # group split indices (fixed across thresholds)
    groups = df["participant"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    dummy_y = np.zeros(len(df), dtype=int)
    train_idx, test_idx = next(splitter.split(np.zeros((len(df), 1)), dummy_y, groups))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    # val split (group-wise)
    groups_train = df_train["participant"].values
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, val_idx = next(val_splitter.split(np.zeros((len(df_train), 1)), np.zeros(len(df_train)), groups_train))

    df_tr  = df_train.iloc[tr_idx].reset_index(drop=True)
    df_val = df_train.iloc[val_idx].reset_index(drop=True)

    # Determine n_time from ONE sample (no preload) 
    sample_path = os.path.join(args.data_root, "npz", df.iloc[0]["npz_file"])
    with np.load(sample_path, allow_pickle=True) as d:
        X0 = d["X"]
    n_time = int(X0.shape[1])
    n_channels = int(most_ch)

    results_all: Dict[str, Any] = {
        "notes": {
            "interpretability_policy": (
                "Interpretability analysis is reported for the primary threshold; "
                "additional thresholds are included as robustness checks for predictive performance."
            ),
            "primary_threshold": int(args.primary_threshold),
            "all_thresholds": [int(t) for t in args.thresholds],
            "stability_fixes": [
                "lazy_npz_loading_no_X_all_preload",
                f"num_workers={int(args.num_workers)}",
                f"pin_memory={bool(args.pin_memory)}",
                f"saliency_batch_size={int(args.saliency_batch_size)}",
                f"saliency_max_batches={int(args.saliency_max_batches)}",
            ]
        },
        "by_threshold": {}
    }

    for thr in args.thresholds:
        thr = int(thr)
        print("\n" + "="*80, flush=True)
        print(f"THRESHOLD = {thr}  (primary={args.primary_threshold})", flush=True)
        print("="*80, flush=True)

        # labels for this threshold (for each split)
        y_train, le = make_labels_for_threshold(df_train, thr)
        y_test,  _  = make_labels_for_threshold(df_test,  thr)
        y_tr,    _  = make_labels_for_threshold(df_tr,    thr)
        y_val,   _  = make_labels_for_threshold(df_val,   thr)

        n_classes = len(le.classes_)
        compute_interp = (thr == int(args.primary_threshold))

        # datasets:
        ds_tr_base  = EEGDataset(df_tr,  args.data_root, most_ch, window_le, y=y_tr,  return_window=False, return_label=True)
        ds_val_base = EEGDataset(df_val, args.data_root, most_ch, window_le, y=y_val, return_window=False, return_label=True)
        ds_test_xy  = EEGDataset(df_test,args.data_root, most_ch, window_le, y=y_test,return_window=False, return_label=True)

        ds_tr_mtl   = EEGDataset(df_tr,  args.data_root, most_ch, window_le, y=y_tr,  return_window=True, return_label=True)
        ds_val_mtl  = EEGDataset(df_val, args.data_root, most_ch, window_le, y=y_val, return_window=True, return_label=True)
        ds_test_xwy = EEGDataset(df_test,args.data_root, most_ch, window_le, y=y_test,return_window=True, return_label=True)

        # loaders (stable defaults)
        common_loader_kwargs = dict(
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=False
        )

        train_loader_base = DataLoader(ds_tr_base, batch_size=args.batch_size, shuffle=True,  **common_loader_kwargs)
        val_loader_base   = DataLoader(ds_val_base,batch_size=args.batch_size, shuffle=False, **common_loader_kwargs)
        test_loader_xy    = DataLoader(ds_test_xy, batch_size=args.batch_size, shuffle=False, **common_loader_kwargs)

        train_loader_mtl  = DataLoader(ds_tr_mtl,  batch_size=args.batch_size, shuffle=True,  **common_loader_kwargs)
        val_loader_mtl    = DataLoader(ds_val_mtl, batch_size=args.batch_size, shuffle=False, **common_loader_kwargs)
        test_loader_xwy   = DataLoader(ds_test_xwy,batch_size=args.batch_size, shuffle=False, **common_loader_kwargs)

        # saliency loader: SMALL batch size always
        sal_loader = DataLoader(
            ds_test_xwy,
            batch_size=int(args.saliency_batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )

        # ---------- Train baseline ----------
        baseline = DeepCNN_LSTM(n_channels=n_channels, n_time=n_time, n_classes=n_classes)
        baseline, info_base = train_one_model(
            baseline, train_loader_base, val_loader_base, y_tr,
            device=device, epochs=args.epochs, patience=args.patience,
            lr=args.lr, weight_decay=args.weight_decay, model_name=f"BASELINE_T{thr}"
        )

        metrics_base = evaluate_model(
            baseline,
            test_loader_xy=test_loader_xy,
            test_loader_xwy=sal_loader if compute_interp else test_loader_xwy,  # interpretability uses sal_loader
            device=device,
            le=le,
            window_le=window_le,
            model_name="BASELINE",
            compute_interpretability=compute_interp,
            saliency_max_batches=(None if int(args.saliency_max_batches) == 0 else int(args.saliency_max_batches))
        )
        metrics_base["training_info"] = info_base

        del baseline
        if device == "cuda":
            torch.cuda.empty_cache()

        # ---------- Train MTL ----------
        mtl = MultiTask_DeepCNN_LSTM(n_channels=n_channels, n_time=n_time, n_classes=n_classes, n_windows=n_windows)
        mtl, info_mtl = train_one_model(
            mtl, train_loader_mtl, val_loader_mtl, y_tr,
            device=device, epochs=args.epochs, patience=args.patience,
            lr=args.lr, weight_decay=args.weight_decay, model_name=f"MTL_T{thr}",
            mtl_lambda=args.mtl_lambda,
            mtl_lambda_warmup_epochs=args.mtl_lambda_warmup_epochs,
            mtl_detach_window_head=args.mtl_detach_window_head,
            window_loss_class_weighted=args.window_loss_class_weighted
        )

        metrics_mtl = evaluate_model(
            mtl,
            test_loader_xy=test_loader_xy,
            test_loader_xwy=sal_loader if compute_interp else test_loader_xwy,
            device=device,
            le=le,
            window_le=window_le,
            model_name="MTL",
            compute_interpretability=compute_interp,
            saliency_max_batches=(None if int(args.saliency_max_batches) == 0 else int(args.saliency_max_batches))
        )
        metrics_mtl["training_info"] = info_mtl

        # ---------- Delta CI ----------
        y_true = np.array(metrics_base["y_true"])
        pred_base = np.array(metrics_base["y_pred"])
        pred_mtl  = np.array(metrics_mtl["y_pred"])

        macro_f1_fn = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
        bal_acc_fn  = lambda yt, yp: balanced_accuracy_score(yt, yp)

        d_f1_mean, d_f1_lo, d_f1_hi = bootstrap_diff_ci(y_true, pred_base, pred_mtl, macro_f1_fn, n_boot=2000, seed=42)
        d_ba_mean, d_ba_lo, d_ba_hi = bootstrap_diff_ci(y_true, pred_base, pred_mtl, bal_acc_fn,  n_boot=2000, seed=42)

        delta = {
            "macro_f1": {"mean": d_f1_mean, "ci95": [d_f1_lo, d_f1_hi]},
            "balanced_accuracy": {"mean": d_ba_mean, "ci95": [d_ba_lo, d_ba_hi]},
            "n_boot": 2000
        }

        results_all["by_threshold"][str(thr)] = {
            "label_definition": {"threshold": thr, "classes": le.classes_.tolist()},
            "baseline_deep_cnn_lstm": metrics_base,
            "mtl_deep_cnn_lstm": metrics_mtl,
            "delta_mtl_minus_baseline_bootstrap_ci": delta,
            "computed_interpretability": bool(compute_interp)
        }

        print(results_all)

        del mtl
        if device == "cuda":
            torch.cuda.empty_cache()

    out_path = os.path.join(args.data_root, args.output_file)
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n✓ Saved multi-threshold results to: {out_path}", flush=True)


if __name__ == "__main__":
    main()


"""
screen -S pain2 -dm bash -lc '
source ~/.bashrc
conda activate eeg

python -u /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/inter_new.py \
  --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data \
  --thresholds 3 7 5 \
  --primary_threshold 5 \
  --mtl_lambda 0.2 \
  --mtl_detach_window_head \
  --num_workers 0 \
  --saliency_batch_size 8 \
  --saliency_max_batches 0 \
  --output_file paper_baseline_vs_mtl_2.json \
  > new22.log 2>&1
'

screen -S pain1 -dm bash -lc '
source ~/.bashrc
conda activate eeg

python -u /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/inter_new.py \
  --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data \
  --thresholds 3 7 5 \
  --primary_threshold 5 \
  --mtl_lambda 0.1 \
  --mtl_detach_window_head \
  --num_workers 0 \
  --saliency_batch_size 8 \
  --saliency_max_batches 0 \
  --output_file paper_baseline_vs_mtl_1.json \
  > new11.log 2>&1
'

screen -S pain3 -dm bash -lc '
source ~/.bashrc
conda activate eeg

python -u /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/inter_new.py \
  --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data \
  --thresholds 3 7 5 \
  --primary_threshold 5 \
  --mtl_lambda 0.1 \
  --mtl_lambda_warmup_epochs 10 \
  --mtl_detach_window_head \
  --window_loss_class_weighted \
  --num_workers 0 \
  --saliency_batch_size 8 \
  --saliency_max_batches 0 \
  --output_file paper_baseline_vs_mtl_111111.json \
  --seed 42 \
  --epochs 50 \
  --patience 15 \
  > new11111.log 2>&1
'


screen -S pain4 -dm bash -lc '
source ~/.bashrc
conda activate eeg

python -u /home/asatsan2/Projects/EEG-Pain-Estimation/scripts/inter_new.py \
  --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data \
  --thresholds 3 7 5 \
  --primary_threshold 5 \
  --mtl_lambda 0.2 \
  --mtl_lambda_warmup_epochs 10 \
  --mtl_detach_window_head \
  --window_loss_class_weighted \
  --num_workers 0 \
  --saliency_batch_size 8 \
  --saliency_max_batches 0 \
  --output_file paper_baseline_vs_mtl_2222.json \
  --seed 42 \
  --epochs 50 \
  --patience 15 \
  > new22222.log 2>&1
'
"""