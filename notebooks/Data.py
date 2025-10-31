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
