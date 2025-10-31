# Step 1: Load and Explore EEG Pain Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Set dataset path
root = '/path/to/dataset/Analysis/segments_allstim_1000hz_v2'

# Load index
index_df = pd.read_csv(f'{root}/index.csv')
print("Total epochs:", len(index_df))
print(index_df.head())

# Basic cleaning — remove flagged epochs
clean_df = index_df[index_df['reject_flag'] == False]
print("Clean epochs after removing flagged:", len(clean_df))

# Distribution of pain categories
plt.figure(figsize=(8,4))
sns.countplot(data=clean_df, x='rating_bin', order=clean_df['rating_bin'].value_counts().index)
plt.title("Distribution of Pain Rating Bins")
plt.xticks(rotation=45)
plt.show()

# Distribution of stimulus types
plt.figure(figsize=(6,4))
sns.countplot(data=clean_df, x='stimulus_category')
plt.title("Stimulus Category Distribution")
plt.show()

# Load and visualize one EEG epoch
example = clean_df.sample(1).iloc[0]
npz = np.load(f"{root}/npz/{example['path']}")
X = npz['X']
ch_names = npz['ch_names']
print(f"Example EEG shape: {X.shape}")

# Plot signal from 5 random channels
plt.figure(figsize=(10,6))
for i, ch in enumerate(np.random.choice(range(X.shape[0]), 5, replace=False)):
    plt.plot(X[ch] + i*10, label=ch_names[ch])  # offset for visibility
plt.title(f"EEG Segment Example (rating_bin={example['rating_bin']})")
plt.xlabel("Time (ms)")
plt.legend()
plt.show()


# EEG Pain Classification - End-to-end pipeline

import os, gc, time, random
from glob import glob
from collections import Counter, defaultdict
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

# Machine learning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib

# PyTorch for deep models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# For interpretability
import shap
try:
    import captum
    from captum.attr import IntegratedGradients, Saliency
except Exception as e:
    captum = None
    IntegratedGradients = None
    Saliency = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- USER EDIT: set dataset root ----------
ROOT = '/path/to/dataset/Analysis/segments_allstim_1000hz_v2'  # <- change to your path
# -------------------------------------------------

# -------------- Utility functions --------------
def load_index(root=ROOT):
    idx = pd.read_csv(os.path.join(root, 'index.csv'))
    return idx

def load_npz_epoch(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data['X'].astype(np.float32)  # shape: [n_channels, n_times]
    # tmin/tmax/sfreq/ch_names maybe present
    meta = {k: data[k] for k in data.files if k not in ['X']}
    return X, meta

# Quick look
index_df = load_index()
print("Total epochs:", len(index_df))
index_df.head()

# -------------- Preprocess / filter metadata --------------
# Filter policy: drop unlabeled & drop flagged epochs by default
def prepare_metadata(df, drop_unlabeled=True, drop_flagged=True):
    df2 = df.copy()
    if drop_unlabeled:
        df2 = df2[df2['rating_bin'].notna() & (df2['rating_bin'] != 'unlabeled')]
    if drop_flagged:
        if 'reject_flag' in df2.columns:
            df2 = df2[df2['reject_flag'] == False]
    df2 = df2.reset_index(drop=True)
    return df2

meta_df = prepare_metadata(index_df, drop_unlabeled=True, drop_flagged=True)
print("After filtering:", len(meta_df))
meta_df['rating_bin'].value_counts()

# -------------- Basic EDA --------------
def plot_distributions(df):
    plt.figure(figsize=(8,4))
    sns.countplot(df['rating_bin'], order=df['rating_bin'].value_counts().index)
    plt.title("rating_bin distribution")
    plt.show()
    plt.figure(figsize=(6,4))
    sns.countplot(df['stimulus_category'])
    plt.title("stimulus_category")
    plt.show()
    # per-subject counts
    cnt = df['participant'].value_counts()
    plt.figure(figsize=(10,4)); sns.histplot(cnt, bins=30); plt.title("epochs per participant"); plt.show()

plot_distributions(meta_df)

# -------------- Simple functions to extract features --------------
# 1) Band power features (delta/theta/alpha/beta/gamma)
from scipy.signal import welch

def bandpower_features(X, sf=1000.0, bands=None):
    # X: [n_channels, n_times]
    if bands is None:
        bands = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)}
    n_ch = X.shape[0]
    feats = []
    # compute PSD via Welch for each channel
    f, Pxx = welch(X, fs=sf, nperseg=min(1024, X.shape[1]))
    for bname, (l,h) in bands.items():
        # integrate PSD between l and h
        mask = (f >= l) & (f <= h)
        bp = Pxx[:, mask].mean(axis=1)  # mean power in band per channel
        feats.append(bp)
    # feats shape (#bands, n_ch) -> flatten
    feats = np.concatenate(feats, axis=0)
    return feats  # length = n_ch * n_bands

def time_domain_feats(X):
    # simple time-domain stats per channel
    mu = X.mean(axis=1)
    std = X.std(axis=1)
    mad = np.mean(np.abs(X - mu[:,None]), axis=1)
    return np.concatenate([mu, std, mad], axis=0)

def extract_features_from_npz(path, root=ROOT):
    X, meta = load_npz_epoch(os.path.join(root, 'npz', path))
    sfreq = meta.get('sfreq', 1000.0).astype(float) if 'sfreq' in meta else 1000.0
    bp = bandpower_features(X, sf=sfreq)
    td = time_domain_feats(X)
    feats = np.concatenate([bp, td], axis=0)
    return feats

# Test speed on small subset
sample_paths = meta_df['path'].sample(50, random_state=SEED).values
t0 = time.time()
_ = [extract_features_from_npz(p) for p in sample_paths]
print("Feature extraction time for 50 samples:", time.time() - t0, "seconds")

# -------------- Build dataset for classical ML --------------
def build_feature_matrix(meta, root=ROOT, n_jobs=6, verbose=True):
    # returns X (n_samples, n_feats), y (n_samples)
    paths = meta['path'].values
    labels = meta['rating_bin'].values
    feats_list = []
    for p in tqdm(paths, disable=not verbose):
        feats_list.append(extract_features_from_npz(p, root))
    X = np.vstack(feats_list)
    y = labels
    return X, y

# WARNING: If your dataset is large (~80k) this will take time & RAM.
# You might want to test on a smaller subset first:
meta_small = meta_df.sample(n=2000, random_state=SEED) if len(meta_df) > 5000 else meta_df
X_feat, y_feat = build_feature_matrix(meta_small)
print("X shape:", X_feat.shape)
le = LabelEncoder()
y_enc = le.fit_transform(y_feat)
print("Classes:", le.classes_)

# -------------- Baseline classical ML: RandomForest & XGBoost --------------
from sklearn.model_selection import StratifiedKFold
def run_classical_ml(X, y, model='rf', n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    fold = 0
    for train_idx, test_idx in skf.split(X, y):
        fold += 1
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
        if model == 'rf':
            clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
            clf.fit(Xtr_s, ytr)
        elif model == 'xgb':
            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200, random_state=SEED, n_jobs=6)
            clf.fit(Xtr_s, ytr)
        ypred = clf.predict(Xte_s)
        accs.append(accuracy_score(yte, ypred))
        f1s.append(f1_score(yte, ypred, average='weighted'))
        print(f"Fold {fold}: acc={accs[-1]:.4f}, f1={f1s[-1]:.4f}")
    print("Mean acc:", np.mean(accs), "Mean f1:", np.mean(f1s))
    return clf, scaler

clf_rf, scaler_rf = run_classical_ml(X_feat, y_enc, model='rf', n_splits=3)

# Save classical model example
joblib.dump({'clf':clf_rf, 'scaler':scaler_rf, 'label_encoder':le}, "rf_bandpower_model.joblib")

# -------------- SHAP explainability for RandomForest (sample) --------------
# Compute SHAP values on a small subset
explainer = shap.TreeExplainer(clf_rf)
# use small background subset
bg = X_feat[np.random.choice(len(X_feat), min(200, len(X_feat)), replace=False)]
shap_values = explainer.shap_values(bg)
# For multiclass shap_values is list per class; visualize feature importance for class 1
shap.summary_plot(shap_values, bg, feature_names=[f'feat_{i}' for i in range(X_feat.shape[1])], show=True)

# -------------- Deep learning baseline: 1D EEGNet-like network --------------
# We'll create a compact model that first applies spatial conv across channels,
# then temporal convs + global pooling -> classifier.
class EEGDataset(Dataset):
    def __init__(self, meta, root=ROOT, label_encoder=None, window='erp', transform=None):
        self.meta = meta.reset_index(drop=True)
        self.root = root
        self.le = label_encoder
        self.transform = transform
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        npz = np.load(os.path.join(self.root, 'npz', row['path']), allow_pickle=True)
        X = npz['X'].astype(np.float32)  # [n_ch, n_t]
        # Option: downsample time dimension to lower length for faster training:
        # X = X[:, ::2]  # half the time
        y = row['rating_bin']
        if self.le is not None:
            y = int(self.le.transform([y])[0])
        # normalize per-sample (z-scored already, but ensure zero-mean unit-std)
        X = (X - X.mean()) / (X.std() + 1e-6)
        # convert to tensor shape (1, n_ch, n_t) for conv2d or (n_ch, n_t) for conv1d
        return torch.tensor(X), torch.tensor(y, dtype=torch.long)

class EEGNet1D(nn.Module):
    def __init__(self, n_ch=64, n_time=1001, n_classes=4, dropout=0.5):
        super().__init__()
        # Spatial conv: combine channel information (1D conv along time but with channel mixing first using 1x1)
        # We'll treat input as (batch, channels, time)
        self.spatial = nn.Conv1d(in_channels=n_ch, out_channels=32, kernel_size=1)  # mixes channels
        # Temporal feature extraction (stack of conv blocks)
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # expect x: (batch, channels, time)
        x = self.spatial(x)       # -> (batch, 32, time)
        x = self.temporal(x)      # -> (batch, 128, 1)
        x = self.classifier(x)    # -> (batch, n_classes)
        return x

# Quick training utilities
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot_loss = 0.0
    preds, trues = [], []
    for X, y in loader:
        X = X.to(device)         # X: (batch, n_ch, n_t)
        y = y.to(device)
        if X.ndim == 3:
            pass
        else:
            X = X.unsqueeze(0)
        # Note: our dataset returns (n_ch, n_t) -> batch size handled by loader
        # but conv1d expects (batch, channels, time)
        if X.shape[1] != model.spatial.in_channels:
            # If orientation mismatch, transpose
            X = X.permute(0,2,1)
        out = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * X.shape[0]
        preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
        trues.extend(y.cpu().numpy().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    acc = accuracy_score(trues, preds)
    return avg_loss, acc

def eval_model(model, loader, criterion, device):
    model.eval()
    tot_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device); y = y.to(device)
            if X.shape[1] != model.spatial.in_channels:
                X = X.permute(0,2,1)
            out = model(X)
            loss = criterion(out, y)
            tot_loss += loss.item() * X.shape[0]
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            trues.extend(y.cpu().numpy().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='weighted')
    return avg_loss, acc, f1, preds, trues

# -------------- Prepare DataLoaders for deep model (LOSO example) --------------
def loso_train_eval(meta_df, le, device='cuda' if torch.cuda.is_available() else 'cpu', max_subjects=None):
    subjects = sorted(meta_df['participant'].unique())
    if max_subjects:
        subjects = subjects[:max_subjects]
    results = []
    for subj in subjects:
        print("LOSO subject:", subj)
        train_meta = meta_df[meta_df['participant'] != subj].reset_index(drop=True)
        test_meta = meta_df[meta_df['participant'] == subj].reset_index(drop=True)
        # Create datasets
        train_ds = EEGDataset(train_meta, root=ROOT, label_encoder=le)
        test_ds = EEGDataset(test_meta, root=ROOT, label_encoder=le)
        # Small validation split from train for early stop
        val_size = int(0.1 * len(train_ds))
        train_size = len(train_ds) - val_size
        if train_size <= 0:
            print("Too few samples for subject", subj)
            continue
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        # instantiate model
        # get channel/time dims from a sample
        sampleX, _ = train_ds[0]
        n_ch, n_t = sampleX.shape
        model = EEGNet1D(n_ch=n_ch, n_time=n_t, n_classes=len(le.classes_)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # training loop
        best_val_f1 = 0.0
        best_model_state = None
        for epoch in range(1, 21):  # 20 epochs typical for initial test
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, _, _ = eval_model(model, val_loader, criterion, device)
            print(f"Epoch {epoch}: train_acc {tr_acc:.3f} val_acc {val_acc:.3f} val_f1 {val_f1:.3f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {k:v.cpu() for k,v in model.state_dict().items()}
        # load best and eval on test
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        test_loss, test_acc, test_f1, preds, trues = eval_model(model, test_loader, criterion, device)
        print(f"LOSO {subj} -> test_acc {test_acc:.3f} test_f1 {test_f1:.3f}")
        results.append({'subject':subj, 'acc':test_acc, 'f1':test_f1, 'n_test':len(test_ds)})
        # cleanup
        del model, optimizer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(results)

# label encoder
le_full = LabelEncoder()
le_full.fit(meta_df['rating_bin'].values)
print("Classes:", le_full.classes_)

# Warning: LOSO is expensive if many subjects. For quick run you can set max_subjects=5
# Example short LOSO run (uncomment to run)
# loso_results = loso_train_eval(meta_df, le_full, device='cuda', max_subjects=5)
# print(loso_results.describe())

# -------------- Interpretability for CNN (example per-sample integrated gradients) --------------
def run_integrated_gradients(model, sample_X, target_label, device='cuda'):
    # sample_X: numpy array shape (n_ch, n_t)
    if IntegratedGradients is None:
        raise RuntimeError("Captum not installed; run `pip install captum`")
    model.eval()
    X = torch.tensor(sample_X).float().unsqueeze(0).to(device)  # (1, n_ch, n_t)
    if X.shape[1] != model.spatial.in_channels:
        X = X.permute(0,2,1)
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(X, target=target_label, return_convergence_delta=True)
    # attributions: same shape as input
    return attributions.detach().cpu().numpy()[0], delta.detach().cpu().numpy()
"""
# -------------- Save & logging helpers --------------
def save_results(df, fname='results_summary.csv'):
    df.to_csv(fname, index=False)
    print("Saved", fname)

# -------------- Extra: Quick simple test run for a single subject (debug) --------------
# If you want a minimal quick test (faster): sample small subset per subject and run LOSO with max_subjects=3
# loso_small = loso_train_eval(meta_df, le_full, device='cpu', max_subjects=3)
# print(loso_small)"""