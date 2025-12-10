# src/utils/io_utils.py
import os
import pickle
import torch
from datetime import datetime

def create_model_directory(base_dir="saved_models", model_name="model"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"{model_name}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path

def save_dl_model(model, path):
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    print(f"[✓] DL model saved → {path}/model.pt")

def save_classical_model(model, path, filename="model.pkl"):
    with open(os.path.join(path, filename), "wb") as f:
        pickle.dump(model, f)
    print(f"[✓] Classical model saved → {path}/{filename}")
