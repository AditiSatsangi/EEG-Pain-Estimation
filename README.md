# MT-WAN: Multi-Task Window-Aware Network for Phase-Aware EEG Pain Decoding

This repository implements a complete pipeline for EEG-based pain estimation using:

• Deep Learning: Hybrid Deep CNN–BiLSTM baseline and MT-WAN (multi-task pain + window supervision)  
• Classical ML: XGBoost / SVM / Logistic Regression / Random Forest baselines  
• Strict subject-wise evaluation with no participant leakage  
• Phase-aware learning across three EEG windows:  
  - Baseline (−1.0–0.0 s)  
  - ERP (−0.2–0.8 s)  
  - Post (0.0–1.0 s)

---

## Core Idea

**MT-WAN** extends the pain-only Deep CNN–BiLSTM with an auxiliary window head:

L = L_pain + λ L_window

• Encourages phase-sensitive representations  
• No window labels required at inference  
• Improves generalization and interpretability

---

## Data Format

Expected layout:

data/
├── index.csv
└── npz/
    ├── sample_01.npz
    ├── sample_02.npz
    └── ...

Each .npz contains EEG epoch:
• Shape: (64, 1001)  → channels × time (1000 Hz)

index.csv must contain at least:
• participant id (for group split)  
• window label (Baseline/ERP/Post)  
• rating_bin (pain label)  
• file identifier

---

## Installation

### Conda
conda env create -f environment.yml
conda activate eeg

### Pip
pip install -r requirements.txt

---

## Training

### 1) Final Deep Learning Run
python scripts/DL_final_training.py

### 2) Grid Search / Tuning
python scripts/dl_train_1000hz_gridsearch.py

### 3) Classical ML Baselines
python scripts/ml_final_train.py

---

## Evaluation Protocol

• Strict subject-wise split using GroupShuffleSplit  
• Multi-threshold evaluation T ∈ {3,5,7}  
• Primary threshold: T = 5  
• Metrics: Accuracy, Balanced Accuracy, Macro-F1  
• Bootstrap CI (2000 resamples) for reliability

---

## Interpretability (T = 5 only)

Grad×Input saliency and ERP-alignment proxy:

python scripts/interpretability_dl.py

Outputs saved to:
results/

---

## Outputs

saved_models/
  ├── *.pth

results/
  ├── *.json
  ├── *.png
  └── logs

data/
  ├── paper_baseline_vs_mtl_*.json

---

## Notes

• Baseline DL = pain-only Deep CNN–BiLSTM  
• Final MT-WAN uses λ = 0.2  
• All experiments prevent subject leakage  
• Window supervision is used only during training

---

## Authors

Dilanjan Diyabalanage – PhD Physics, Western University
Aditi Satsangi – MSc Computer Science, Western University  

---

## Contact

Open an issue on the repository for questions or collaboration.
