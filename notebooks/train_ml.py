
import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from scipy.signal import welch, butter, filtfilt, stft

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    make_scorer,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Optional XGBoost (handled gracefully if not available)
try:
    from xgboost import XGBClassifier  # noqa
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


# ---------------------------- Constants ----------------------------

EPS = 1e-12
FS_DEFAULT = 1000
N_PERSEG = 256
N_OVERLAP = 128

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

ROI_GROUPS = {
    'frontal':    ['fp1','fp2','f1','f2','f3','f4','f5','f6','f7','f8','fz'],
    'frontocent': ['fc1','fc2','fc3','fc4','fc5','fc6','fcz'],
    'central':    ['c1','c2','c3','c4','c5','c6','cz'],
    'parietal':   ['p1','p2','p3','p4','p5','p6','pz'],
    'occipital':  ['o1','o2','oz'],
    'temp_left':  ['t7','t3','ft7','tp7'],
    'temp_right': ['t8','t4','ft8','tp8'],
}
ROI_KEYS = list(ROI_GROUPS.keys())
TIME_BINS_MS = [(0, 200), (200, 400), (400, 800)]


# ---------------------------- Data Utils ----------------------------

def find_data_root(start_dir: str) -> str:
    candidates = [
        os.path.join(start_dir, 'Data'),
        os.path.join(start_dir, '../Data'),
        os.path.join(start_dir, './Data'),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, '/home/asatsan2/Projects/EEG-Pain-Estimation/data/index.csv')):
            return os.path.abspath(c)
    raise FileNotFoundError("index.csv not found. Expected in one of: ./Data, ../Data")


def load_index(root: str) -> pd.DataFrame:
    index_path = os.path.join(root, 'index.csv')
    df = pd.read_csv(index_path)
    if 'reject_flag' in df.columns:
        df = df[df['reject_flag'] == False].copy()
    if len(df) == 0:
        raise RuntimeError("No clean epochs found after filtering reject_flag.")
    return df


def most_common_channel_count(root: str, df: pd.DataFrame) -> int:
    if 'n_channels' in df.columns and df['n_channels'].notna().any():
        try:
            return int(df['n_channels'].mode().iloc[0])
        except Exception:
            pass
    return 64


def cache_dir(root: str) -> str:
    d = os.path.join(root, 'feature_cache')
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------- v2 Features ----------------------------

def extract_features_epoch_v2(X: np.ndarray, fs: int) -> np.ndarray:
    f, Pxx = welch(
        X, fs=fs, nperseg=N_PERSEG, noverlap=N_OVERLAP,
        window='hann', detrend='constant', scaling='density', axis=1
    )
    m = (f >= 1) & (f <= 45)
    f = f[m]
    P = Pxx[:, m]

    total_power_vec = P.sum(axis=1) + EPS
    total_power = total_power_vec[:, None]

    rel_bp = {}
    for name, (lo, hi) in BANDS.items():
        mb = (f >= lo) & (f < hi)
        if np.any(mb):
            # numpy trapezoid available in newer versions
            if hasattr(np, 'trapezoid'):
                bp = np.trapezoid(P[:, mb], f[mb], axis=1)
            else:
                bp = np.trapz(P[:, mb], f[mb], axis=1)
        else:
            bp = np.zeros(P.shape[0], dtype=np.float64)
        rel_bp[f'rel_{name}'] = bp / total_power_vec

    p_norm = P / total_power
    spec_entropy = -(p_norm * np.log(p_norm + EPS)).sum(axis=1) / np.log(P.shape[1])

    cs = P.cumsum(axis=1)
    tot = cs[:, -1]
    idx_med = (cs >= (0.5 * tot)[:, None]).argmax(axis=1)
    idx_sef75 = (cs >= (0.75 * tot)[:, None]).argmax(axis=1)
    idx_sef95 = (cs >= (0.95 * tot)[:, None]).argmax(axis=1)
    median_freq = f[idx_med]
    sef75 = f[idx_sef75]
    sef95 = f[idx_sef95]

    ms = (f >= 3) & (f <= 45)
    xf = np.log10(f[ms] + EPS)
    n = xf.size
    sum_x = xf.sum()
    sum_x2 = (xf ** 2).sum()
    Y = np.log10(P[:, ms] + EPS)
    sum_y = Y.sum(axis=1)
    sum_xy = (Y * xf).sum(axis=1)
    mean_x = sum_x / n
    denom = (sum_x2 - n * (mean_x ** 2)) + EPS
    slope = (sum_xy - mean_x * sum_y) / denom
    intercept = (sum_y / n) - slope * mean_x

    var0 = X.var(axis=1)
    dx = np.diff(X, axis=1)
    var1 = dx.var(axis=1)
    mob = np.sqrt((var1 + EPS) / (var0 + EPS))
    ddx = np.diff(dx, axis=1)
    var2 = ddx.var(axis=1)
    comp = np.sqrt((var2 + EPS) / (var1 + EPS)) / (mob + EPS)
    ll = np.abs(dx).sum(axis=1)

    gm = np.exp(np.mean(np.log(P + EPS), axis=1))
    am = np.mean(P + EPS, axis=1)
    flatness = gm / (am + EPS)

    theta_alpha = rel_bp['rel_theta'] / (rel_bp['rel_alpha'] + EPS)
    beta_alpha = rel_bp['rel_beta'] / (rel_bp['rel_alpha'] + EPS)
    alpha_beta = rel_bp['rel_alpha'] / (rel_bp['rel_beta'] + EPS)
    beta_theta = rel_bp['rel_beta'] / (rel_bp['rel_theta'] + EPS)
    log_total_power = np.log(total_power_vec + EPS)

    M_cols = [
        rel_bp['rel_delta'], rel_bp['rel_theta'], rel_bp['rel_alpha'], rel_bp['rel_beta'], rel_bp['rel_gamma'],
        spec_entropy, median_freq, sef75, sef95, slope, intercept,
        var0, mob, comp, ll, flatness,
        theta_alpha, beta_alpha, alpha_beta, beta_theta,
        log_total_power,
    ]
    M = np.stack(M_cols, axis=1).astype(np.float32)
    per_channel_flat = M.reshape(-1)
    return per_channel_flat


def build_v2(root: str, df: pd.DataFrame, fs_default: int, most_ch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cdir = cache_dir(root)
    cache_path = os.path.join(cdir, f"welch_v2_fs{fs_default}_seg{N_PERSEG}_ov{N_OVERLAP}_pc1_agg1_ch{most_ch}_n{len(df)}.npz")
    if os.path.exists(cache_path):
        npz = np.load(cache_path, allow_pickle=True)
        return npz['X_features'], npz['y_labels'], npz['groups'], npz['window_labels'] if 'window_labels' in npz else None
    feats, labels, groups, windows = [], [], [], []
    for _, row in df.iterrows():
        npz_path = os.path.join(root, 'npz', row['path'])
        if not os.path.exists(npz_path):
            continue
        npzf = np.load(npz_path, allow_pickle=True)
        X = npzf['X']
        if X.shape[0] != most_ch:
            continue
        fs = int(row.get('sfreq', fs_default)) if 'sfreq' in row else fs_default
        feat = extract_features_epoch_v2(X, fs=fs)
        feats.append(feat)
        labels.append(row['rating_bin'])
        groups.append(row['participant'])
        windows.append(row['window'] if 'window' in row else 'unknown')
    X_features = np.vstack(feats).astype(np.float32)
    y_labels = np.array(labels)
    groups = np.array(groups)
    window_labels = np.array(windows)
    np.savez_compressed(cache_path, X_features=X_features, y_labels=y_labels, groups=groups, window_labels=window_labels)
    return X_features, y_labels, groups, window_labels


# ---------------------------- v3 Features ----------------------------

def _find_roi_indices(ch_names):
    idxs = []
    ch_l = [str(c).lower() for c in ch_names]
    for key in ROI_KEYS:
        want = ROI_GROUPS[key]
        grp = [i for i, name in enumerate(ch_l) if any(w in name for w in want)]
        idxs.append(grp)
    return idxs


def _roi_ts(X, ch_names):
    idxs = _find_roi_indices(ch_names)
    rois = []
    for grp in idxs:
        if len(grp) == 0:
            rois.append(np.zeros(X.shape[1], dtype=np.float32))
        else:
            rois.append(X[grp, :].mean(axis=0))
    return np.vstack(rois)


def _butter_band(lo, hi, fs, order=4):
    nyq = 0.5 * fs
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype='band')
    return b, a


def _roi_band_cov_features(roi_ts, fs, bands):
    n_roi = roi_ts.shape[0]
    feats = []
    tri = np.triu_indices(n_roi, k=1)
    for _, (lo, hi) in bands.items():
        b, a = _butter_band(lo, hi, fs)
        Xf = filtfilt(b, a, roi_ts, axis=1)
        C = np.corrcoef(Xf)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        feats.append(C[tri])
    return np.concatenate(feats, axis=0)


def _roi_bandpowers(roi_ts, fs, bands):
    f, Pxx = welch(roi_ts, fs=fs, nperseg=N_PERSEG, noverlap=N_OVERLAP, axis=1)
    m = (f >= 1) & (f <= 45)
    f = f[m]
    P = Pxx[:, m]
    tot = P.sum(axis=1, keepdims=True) + EPS
    rels = []
    for _, (lo, hi) in bands.items():
        mb = (f >= lo) & (f < hi)
        if np.any(mb):
            if hasattr(np, 'trapezoid'):
                bp = np.trapezoid(P[:, mb], f[mb], axis=1)
            else:
                bp = np.trapz(P[:, mb], f[mb], axis=1)
        else:
            bp = np.zeros(P.shape[0])
        rels.append((bp / tot[:, 0]).astype(np.float32))
    return np.concatenate(rels, axis=0)


def _roi_erp_features(roi_ts, fs):
    feats = []
    for (t0, t1) in TIME_BINS_MS:
        s0 = int(t0 * fs / 1000.0)
        s1 = int(t1 * fs / 1000.0)
        seg = roi_ts[:, s0:s1]
        feats.append(seg.mean(axis=1))
        feats.append(seg.max(axis=1))
        feats.append(seg.min(axis=1))
    return np.concatenate(feats, axis=0).astype(np.float32)


def _roi_ersp_features(roi_ts, fs, bands):
    f, t, Z = stft(roi_ts, fs=fs, nperseg=128, noverlap=64, axis=1, padded=False, boundary=None)
    P = (np.abs(Z) ** 2)
    feats = []
    for (t0, t1) in TIME_BINS_MS:
        m_t = (t >= (t0/1000.0)) & (t < (t1/1000.0))
        if not np.any(m_t):
            feats.append(np.zeros(roi_ts.shape[0] * len(bands), dtype=np.float32))
            continue
        for _, (lo, hi) in bands.items():
            m_f = (f >= lo) & (f < hi)
            if np.any(m_f):
                val = P[:, m_f][:, :, m_t].mean(axis=(1,2))
            else:
                val = np.zeros(roi_ts.shape[0])
            feats.append(val.astype(np.float32))
    return np.concatenate(feats, axis=0)


def build_v3(root: str, df: pd.DataFrame, fs_default: int, most_ch: int, quick: bool = False, quick_n_per_subj: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cdir = cache_dir(root)
    cache_path = os.path.join(cdir, f"feat_v3_fs{fs_default}_seg{N_PERSEG}_ov{N_OVERLAP}_roi{len(ROI_KEYS)}_bands{len(BANDS)}_ch{most_ch}_n{len(df)}.npz")
    if os.path.exists(cache_path) and not quick:
        npz = np.load(cache_path, allow_pickle=True)
        return npz['X_features_v3'], npz['y_labels'], npz['groups'], npz['window_labels']
    feats, labels, groups, windows = [], [], [], []
    use_df = df
    if quick:
        rnd = np.random.RandomState(42)
        tmp = df.copy()
        tmp['_r'] = rnd.rand(len(tmp))
        tmp = tmp.sort_values(['participant', '_r'])
        use_df = tmp.groupby('participant').head(quick_n_per_subj)
    for _, row in use_df.iterrows():
        npz_path = os.path.join(root, 'npz', row['path'])
        if not os.path.exists(npz_path):
            continue
        npzf = np.load(npz_path, allow_pickle=True)
        X = npzf['X']
        if X.shape[0] != most_ch:
            continue
        ch_names = [str(c) for c in npzf.get('ch_names', [f'ch{i}' for i in range(X.shape[0])])]
        fs = int(row.get('sfreq', fs_default)) if 'sfreq' in row else fs_default
        roi = _roi_ts(X, ch_names)
        f_cov = _roi_band_cov_features(roi, fs, BANDS)
        f_bp = _roi_bandpowers(roi, fs, BANDS)
        f_erp = _roi_erp_features(roi, fs)
        f_ersp = _roi_ersp_features(roi, fs, BANDS)
        feat = np.concatenate([f_cov, f_bp, f_erp, f_ersp], axis=0).astype(np.float32)
        feats.append(feat)
        labels.append(row['rating_bin'])
        groups.append(row['participant'])
        windows.append(row['window'] if 'window' in row else 'unknown')
    X_raw = np.vstack(feats).astype(np.float32)
    y_labels = np.array(labels)
    groups = np.array(groups)
    window_labels = np.array(windows)
    # subject-wise baseline normalization
    X_norm = np.empty_like(X_raw)
    for s in np.unique(groups):
        idx_sub = (groups == s)
        idx_base = idx_sub & (window_labels == 'baseline')
        mu = X_raw[idx_base].mean(axis=0) if np.any(idx_base) else X_raw[idx_sub].mean(axis=0)
        X_norm[idx_sub] = X_raw[idx_sub] - mu
    if not quick:
        np.savez_compressed(cache_path, X_features_v3=X_norm, y_labels=y_labels, groups=groups, window_labels=window_labels)
    return X_norm, y_labels, groups, window_labels


# ---------------------------- v4 Features ----------------------------

def _vec_spd_upper(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    iu = np.triu_indices(n)
    return mat[iu].astype(np.float32)


def _log_euclidean_spd(C: np.ndarray) -> np.ndarray:
    C = (C + C.T) * 0.5
    tr = np.trace(C) + EPS
    Cn = C / tr
    w, V = np.linalg.eigh(Cn)
    w = np.maximum(w, 1e-10)
    L = V @ np.diag(np.log(w)) @ V.T
    return (L + L.T) * 0.5


def build_v4(root: str, df: pd.DataFrame, fs_default: int, most_ch: int, quick: bool = False, quick_n_per_subj: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cdir = cache_dir(root)
    cache_path = os.path.join(cdir, f"feat_v4_riotsp_fs{fs_default}_seg{N_PERSEG}_ov{N_OVERLAP}_roi{len(ROI_KEYS)}_bands{len(BANDS)}_ch{most_ch}_n{len(df)}.npz")
    if os.path.exists(cache_path) and not quick:
        npz = np.load(cache_path, allow_pickle=True)
        return npz['X_features_v4'], npz['y_labels'], npz['groups'], npz['window_labels']
    feats, labels, groups, windows = [], [], [], []
    use_df = df
    if quick:
        rnd = np.random.RandomState(42)
        tmp = df.copy()
        tmp['_r'] = rnd.rand(len(tmp))
        tmp = tmp.sort_values(['participant', '_r'])
        use_df = tmp.groupby('participant').head(quick_n_per_subj)
    for _, row in use_df.iterrows():
        npz_path = os.path.join(root, 'npz', row['path'])
        if not os.path.exists(npz_path):
            continue
        npzf = np.load(npz_path, allow_pickle=True)
        X = npzf['X']
        if X.shape[0] != most_ch:
            continue
        ch_names = [str(c) for c in npzf.get('ch_names', [f'ch{i}' for i in range(X.shape[0])])]
        fs = int(row.get('sfreq', fs_default)) if 'sfreq' in row else fs_default
        roi = _roi_ts(X, ch_names)
        feat_vecs = []
        for _, (lo, hi) in BANDS.items():
            b, a = _butter_band(lo, hi, fs)
            Xf = filtfilt(b, a, roi, axis=1)
            C = np.cov(Xf, bias=True)
            C += np.eye(C.shape[0]) * 1e-6
            L = _log_euclidean_spd(C)
            feat_vecs.append(_vec_spd_upper(L))
        v = np.concatenate(feat_vecs, axis=0)
        feats.append(v)
        labels.append(row['rating_bin'])
        groups.append(row['participant'])
        windows.append(row['window'] if 'window' in row else 'unknown')
    X_raw = np.vstack(feats).astype(np.float32)
    y_labels = np.array(labels)
    groups = np.array(groups)
    window_labels = np.array(windows)
    # subject-wise baseline centering
    X_centered = np.empty_like(X_raw)
    for s in np.unique(groups):
        idx_sub = (groups == s)
        idx_base = idx_sub & (window_labels == 'baseline')
        mu = X_raw[idx_base].mean(axis=0) if np.any(idx_base) else X_raw[idx_sub].mean(axis=0)
        X_centered[idx_sub] = X_raw[idx_sub] - mu
    if not quick:
        np.savez_compressed(cache_path, X_features_v4=X_centered, y_labels=y_labels, groups=groups, window_labels=window_labels)
    return X_centered, y_labels, groups, window_labels


def build_all(root: str, df: pd.DataFrame, fs_default: int, most_ch: int, quick: bool = False, quick_n_per_subj: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a concatenated feature vector per epoch: [v2 | v3 | v4].
    - v3 and v4 parts receive subject-wise baseline normalization/centering.
    - v2 part is left as-is (global scaler is applied in the ML pipeline).
    """
    cdir = cache_dir(root)
    cache_path = os.path.join(cdir, f"feat_all_v2v3v4_fs{fs_default}_seg{N_PERSEG}_ov{N_OVERLAP}_roi{len(ROI_KEYS)}_bands{len(BANDS)}_ch{most_ch}_n{len(df)}.npz")
    if os.path.exists(cache_path) and not quick:
        npz = np.load(cache_path, allow_pickle=True)
        return npz['X_features_all'], npz['y_labels'], npz['groups'], npz['window_labels']
    v2_list, v3_list, v4_list = [], [], []
    labels, groups, windows, ch_name_cache = [], [], [], []
    use_df = df
    if quick:
        rnd = np.random.RandomState(42)
        tmp = df.copy()
        tmp['_r'] = rnd.rand(len(tmp))
        tmp = tmp.sort_values(['participant', '_r'])
        use_df = tmp.groupby('participant').head(quick_n_per_subj)
    for _, row in use_df.iterrows():
        npz_path = os.path.join(root, 'npz', row['path'])
        if not os.path.exists(npz_path):
            continue
        npzf = np.load(npz_path, allow_pickle=True)
        X = npzf['X']
        if X.shape[0] != most_ch:
            continue
        fs = int(row.get('sfreq', fs_default)) if 'sfreq' in row else fs_default
        # v2 per-channel features
        v2_feat = extract_features_epoch_v2(X, fs=fs)
        # v3 ROI features
        ch_names = [str(c) for c in npzf.get('ch_names', [f'ch{i}' for i in range(X.shape[0])])]
        roi = _roi_ts(X, ch_names)
        v3_cov = _roi_band_cov_features(roi, fs, BANDS)
        v3_bp = _roi_bandpowers(roi, fs, BANDS)
        v3_erp = _roi_erp_features(roi, fs)
        v3_ersp = _roi_ersp_features(roi, fs, BANDS)
        v3_feat = np.concatenate([v3_cov, v3_bp, v3_erp, v3_ersp], axis=0).astype(np.float32)
        # v4 SPD features
        feat_vecs = []
        for _, (lo, hi) in BANDS.items():
            b, a = _butter_band(lo, hi, fs)
            Xf = filtfilt(b, a, roi, axis=1)
            C = np.cov(Xf, bias=True)
            C += np.eye(C.shape[0]) * 1e-6
            L = _log_euclidean_spd(C)
            feat_vecs.append(_vec_spd_upper(L))
        v4_feat = np.concatenate(feat_vecs, axis=0)
        # collect
        v2_list.append(v2_feat.astype(np.float32))
        v3_list.append(v3_feat.astype(np.float32))
        v4_list.append(v4_feat.astype(np.float32))
        labels.append(row['rating_bin'])
        groups.append(row['participant'])
        windows.append(row['window'] if 'window' in row else 'unknown')
    if len(v2_list) == 0:
        raise RuntimeError("No features extracted for 'all'.")
    V2 = np.vstack(v2_list)
    V3_raw = np.vstack(v3_list)
    V4_raw = np.vstack(v4_list)
    y_labels = np.array(labels)
    groups = np.array(groups)
    window_labels = np.array(windows)
    # subject-wise baseline normalization on v3 and v4 parts only
    V3_norm = np.empty_like(V3_raw)
    V4_center = np.empty_like(V4_raw)
    for s in np.unique(groups):
        idx_sub = (groups == s)
        idx_base = idx_sub & (window_labels == 'baseline')
        mu3 = V3_raw[idx_base].mean(axis=0) if np.any(idx_base) else V3_raw[idx_sub].mean(axis=0)
        mu4 = V4_raw[idx_base].mean(axis=0) if np.any(idx_base) else V4_raw[idx_sub].mean(axis=0)
        V3_norm[idx_sub] = V3_raw[idx_sub] - mu3
        V4_center[idx_sub] = V4_raw[idx_sub] - mu4
    X_all = np.concatenate([V2, V3_norm, V4_center], axis=1).astype(np.float32)
    if not quick:
        np.savez_compressed(cache_path, X_features_all=X_all, y_labels=y_labels, groups=groups, window_labels=window_labels)
    return X_all, y_labels, groups, window_labels


def build_pain_threshold_task(
    X: np.ndarray, y_labels: np.ndarray, groups: np.ndarray
) -> Dict[str, Any]:
    """
    Map 5-class rating_bin labels to pain-threshold binary:
      - no_significant_pain: 'none', 'low'
      - significant_pain   : 'mid'/'moderate', 'high', 'extreme'/'severe'
    """
    nopain = {'none', 'low', 'no', 'no_pain', 'nopain'}
    pain = {'mid', 'moderate', 'high', 'severe', 'extreme'}

    def map_label(lbl: str) -> str:
        l = str(lbl).strip().lower()
        if l in nopain:
            return 'no_significant_pain'
        # Anything explicitly in pain set or not in nopain defaults to pain
        return 'significant_pain'

    y_text = np.array([map_label(str(y)) for y in y_labels], dtype=object)
    le = LabelEncoder().fit(y_text)
    y = le.transform(y_text)
    return dict(X=X, y=y, le=le, groups=groups)


def build_models(seed: int = 42) -> Dict[str, Any]:
    """
    Define ML pipelines consistent with existing project choices.
    - RF and XGBoost use SMOTE to balance classes
    - LogisticRegression and LinearSVC use class_weight='balanced' and PCA
    """
    models: Dict[str, Any] = {}

    models['RandomForest'] = ImbPipeline(steps=[
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=seed)),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1
        ))
    ])

    models['LogReg_bal'] = ImbPipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, svd_solver='full')),
        ('clf', LogisticRegression(
            max_iter=5000, class_weight='balanced', solver='saga', n_jobs=-1, random_state=seed
        ))
    ])

    models['LinearSVC_bal'] = ImbPipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, svd_solver='full')),
        ('clf', LinearSVC(C=1.0, class_weight='balanced', random_state=seed))
    ])

    if XGB_AVAILABLE:
        # Try GPU; fallback to CPU
        try:
            _probe = XGBClassifier(
                tree_method='gpu_hist', predictor='gpu_predictor',
                n_estimators=1, verbosity=0, random_state=seed
            )
            # Only to validate GPU works (fit tiny batch later inside CV fit)
            use_gpu = True
        except Exception:
            use_gpu = False
        if use_gpu:
            xgb = XGBClassifier(
                n_estimators=600, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, random_state=seed,
                tree_method='gpu_hist', predictor='gpu_predictor', verbosity=0,
                reg_lambda=1.0, min_child_weight=1
            )
        else:
            xgb = XGBClassifier(
                n_estimators=600, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, random_state=seed,
                tree_method='hist', n_jobs=-1, verbosity=0,
                reg_lambda=1.0, min_child_weight=1
            )
        models['XGBoost'] = ImbPipeline(steps=[
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=seed)),
            ('clf', xgb)
        ])

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Train classical ML models on 1000 Hz EEG features with subject-wise splitting"
    )
    parser.add_argument('--feature-set', choices=['all'], default='all',
                        help='Feature set to extract (only "all": v2+v3+v4 combined)')
    parser.add_argument('--test-size', type=float, default=0.25,
                        help='Proportion of subjects held out for test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode for v3/v4 feature extraction (subset per subject)')
    parser.add_argument('--quick_n_per_subj', type=int, default=50,
                        help='Samples per subject in quick mode (v3/v4)')
    parser.add_argument('--data-root', type=str, default="/home/asatsan2/Projects/EEG-Pain-Estimation/data",
                        help='Path to Data directory (defaults to project Data/)')
    parser.add_argument('--output-file', type=str, default='results/results_ml_1000hz_pain_threshold.json',
                        help='Where to save the final JSON results')
    parser.add_argument('--models', nargs='+',
                        choices=['RandomForest', 'LogReg_bal', 'LinearSVC_bal', 'XGBoost'],
                        default=['RandomForest', 'LogReg_bal','LinearSVC_bal', 'XGBoost'],
                        help='Which models to run')
    args = parser.parse_args()

    set_seed(args.seed)

    print("\n" + "=" * 70)
    print(" INITIALIZATION")
    print("=" * 70)
    print(f"Author : DILANJAN DK")
    print(f"Email  : DDIYABAL@UWO.CA")
    print(f"Seed   : {args.seed}")
    print(f"Feat   : {args.feature_set}")
    print(f"Quick  : {args.quick} (n_per_subj={args.quick_n_per_subj})")

    # Resolve data root and load index
    data_root = args.data_root or find_data_root(PROJ_ROOT)
    print(f"\nData root: {data_root}")
    df = load_index(data_root)
    print(f"Clean epochs: {len(df):,}")
    most_ch = most_common_channel_count(data_root, df)
    print(f"Most common channel count: {most_ch}")

    # Feature extraction (reuse exact implementations)
    t0 = time.time()
    X, y_labels, groups, window_labels = build_all(
        data_root, df, FS_DEFAULT, most_ch, quick=args.quick, quick_n_per_subj=args.quick_n_per_subj
    )
    print(f"Features shape: {X.shape} | built in {(time.time()-t0):.1f}s")

    # Pain-threshold task mapping
    task = build_pain_threshold_task(X, y_labels, groups)
    X_all, y_all, le, groups_all = task['X'], task['y'], task['le'], task['groups']
    class_names = le.classes_.tolist()

    print("\n" + "=" * 70)
    print(" TASK: pain_threshold (none+low vs mid/high/extreme)")
    print("=" * 70)
    unique, counts = np.unique(y_all, return_counts=True)
    print("Classes:", class_names)
    for cls_idx, cnt in zip(unique, counts):
        print(f"  {class_names[cls_idx]:22s}: {cnt:6d} ({100.0*cnt/len(y_all):5.2f}%)")

    # Subject-wise split
    print("\n" + "=" * 70)
    print(" SUBJECT-WISE SPLIT")
    print("=" * 70)
    gss = GroupShuffleSplit(test_size=args.test_size, n_splits=1, random_state=args.seed)
    tr_idx, te_idx = next(gss.split(X_all, y_all, groups_all))
    X_train, X_test = X_all[tr_idx], X_all[te_idx]
    y_train, y_test = y_all[tr_idx], y_all[te_idx]
    groups_train, groups_test = groups_all[tr_idx], groups_all[te_idx]
    print(f"Train: {len(y_train):,} samples | {len(np.unique(groups_train))} subjects")
    print(f"Test : {len(y_test):,} samples | {len(np.unique(groups_test))} subjects")
    if len(set(groups_train).intersection(set(groups_test))) > 0:
        print("⚠ WARNING: Subject leakage detected!")
    else:
        print("✓ No subject leakage (strict subject-wise split)")

    # Define models
    models = build_models(seed=args.seed)
    # Filter requested models
    models = {k: v for k, v in models.items() if k in args.models}
    if not models:
        raise ValueError("No valid models selected.")

    scoring = {
        'macro_f1': make_scorer(f1_score, average='macro'),
        'bal_acc': make_scorer(balanced_accuracy_score),
        'acc': make_scorer(accuracy_score),
    }
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)

    results: Dict[str, Any] = {}
    for name, pipe in models.items():
        print("\n" + "=" * 70)
        print(f" {name}: 5-fold StratifiedGroupKFold CV (subject-wise)")
        print("=" * 70)
        cv_res = cross_validate(
            pipe, X_train, y_train, groups=groups_train,
            scoring=scoring, cv=cv, n_jobs=-1, return_train_score=False
        )
        print(f"CV macro-F1: {cv_res['test_macro_f1'].mean():.3f} ± {cv_res['test_macro_f1'].std():.3f}")
        print(f"CV bal-acc : {cv_res['test_bal_acc'].mean():.3f} ± {cv_res['test_bal_acc'].std():.3f}")
        print(f"CV acc     : {cv_res['test_acc'].mean():.3f} ± {cv_res['test_acc'].std():.3f}")

        # Fit on full train, evaluate on held-out test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(
            le.inverse_transform(y_test),
            le.inverse_transform(y_pred),
            digits=3, zero_division=0
        )
        cm = confusion_matrix(
            le.inverse_transform(y_test),
            le.inverse_transform(y_pred),
        ).tolist()

        print("\nHoldout (subject-wise) classification report:")
        print(report)
        print("Confusion Matrix:")
        print(np.array(cm))

        results[name] = {
            'cv': {
                'macro_f1_mean': float(cv_res['test_macro_f1'].mean()),
                'macro_f1_std': float(cv_res['test_macro_f1'].std()),
                'bal_acc_mean': float(cv_res['test_bal_acc'].mean()),
                'bal_acc_std': float(cv_res['test_bal_acc'].std()),
                'acc_mean': float(cv_res['test_acc'].mean()),
                'acc_std': float(cv_res['test_acc'].std()),
                'n_splits': int(cv.get_n_splits()),
            },
            'test': {
                'accuracy': float(acc),
                'balanced_accuracy': float(bal_acc),
                'f1_macro': float(f1m),
                'report': report,
                'confusion_matrix': cm,
                'class_names': class_names,
            },
        }

    # Save results
    out_path = os.path.join(PROJ_ROOT, args.output_file)
    payload = {
        'author': "DILANJAN DK",
        'email': "DDIYABAL@UWO.CA",
        'task': 'pain_threshold',
        'dataset': 'Data (1000 Hz)',
        'feature_set': args.feature_set,
        'reproducibility': {
            'random_seed': args.seed
        },
        'subject_split': {
            'test_size': args.test_size,
            'train_samples': int(len(y_train)),
            'test_samples': int(len(y_test)),
            'train_subjects': int(len(np.unique(groups_train))),
            'test_subjects': int(len(np.unique(groups_test))),
        },
        'class_distribution_overall': {
            class_names[i]: int(c) for i, c in zip(unique, counts)
        },
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print("\n" + "=" * 70)
    print(" COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()


"""python /home/asatsan2/Projects/EEG-Pain-Estimation/notebooks/train_ml.py \
         --task none_vs_pain \
        --data_root /home/asatsan2/Projects/EEG-Pain-Estimation/data \
        --grid-epochs 20 \
        --grid-patience 7 \
        --epochs 30 \
        --patience 10 \
        --seed 42 
        
        python /home/asatsan2/Projects/EEG-Pain-Estimation/notebooks/train_ml.py  \
  --test-size 0.25 \
  --seed 42 \
  -- models RandomForest, LinearSVC_bal, XGBoost


  python /home/asatsan2/Projects/EEG-Pain-Estimation/notebooks/train_ml.py \
  --test-size 0.25 \
  --seed 42 \
  --models RandomForest LinearSVC_bal XGBoost

  --output-file results_ml_1000hz_pain_threshold_all.json
  
  screen -S myjob -dm python /home/asatsan2/Projects/EEG-Pain-Estimation/notebooks/train_ml.py
  """