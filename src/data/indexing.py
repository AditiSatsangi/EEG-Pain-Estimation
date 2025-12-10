# src/data/indexing.py
import os
import pandas as pd
from collections import Counter

def find_data_root(project_root, dataset_folder="Data"):
    candidate = os.path.join(project_root, dataset_folder)
    return candidate if os.path.isdir(candidate) else project_root

def load_index(root):
    path = os.path.join(root, "index.csv")
    df = pd.read_csv(path)
    if "reject_flag" in df.columns:
        df = df[df["reject_flag"] == False].copy()
    return df

def get_most_common_channels(df):
    if "n_channels" in df.columns:
        return Counter(df["n_channels"]).most_common(1)[0][0]
    return 64
