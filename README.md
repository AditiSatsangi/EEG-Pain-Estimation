# MT-WAN: Phase-Aware EEG Pain Decoding via Multi-Task Learning

This repository implements a complete and reproducible pipeline for **EEG-based pain classification** using both **classical machine learning** and **deep learning**.  
The proposed model, **MT-WAN (Multi-Task Window-Aware Network)**, improves a strong Deep CNN–BiLSTM baseline by adding **auxiliary supervision on EEG temporal phases (Baseline / ERP / Post)** to encourage physiologically meaningful representations.

---

## Key Contributions

• **Multi-task learning for phase awareness**  
  Jointly learns pain classification + window identification  
• **No window labels required at inference**  
• **Strict subject-wise evaluation (no leakage)**  
• **Robust across thresholds T ∈ {3,5,7}**  
• **Interpretability with Grad×Input saliency**

---

## Data Format

Each `.npz` file contains one EEG epoch:

• Shape: **(64 channels × 1001 time points)**  
• Sampling rate: **1000 Hz**

`index.csv` must include:

• participant ID (for group split)  
• window label (Baseline / ERP / Post)  
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

### Train Final MT-WAN Model
python scripts/DL_final_training.py

### Hyperparameter Search
python scripts/dl_train_1000hz_gridsearch.py

### Classical ML Baselines
python scripts/ml_final_train.py

---

## Evaluation Protocol

• **Subject-wise split** using GroupShuffleSplit  
• **Multi-threshold** evaluation: T ∈ {3,5,7}  
• **Primary threshold: T = 5**  
• Metrics: Accuracy, Balanced Accuracy, Macro-F1  
• **Bootstrap CI (2000 resamples)** for reliability

---

## Interpretability

Grad×Input saliency is used at **T = 5** to assess:

• phase-specific attribution  
• ERP alignment ratio  
• window-wise relevance

Run:

python scripts/interpretability_dl.py

Outputs saved in:

results/

---

## Outputs

saved_models/    → trained checkpoints  
results/         → figures, metrics, JSON logs  
data/            → experiment result files

---

## Model Overview

**Baseline DL:** Deep CNN–BiLSTM (pain-only)  
**Proposed:** MT-WAN (pain + window supervision)

Loss:

L = L_pain + λ L_window

Final model uses **λ = 0.2**

---

## Authors

**Aditi Satsangi** – MSc Computer Science, Western University  
**Dilanjan Diyabalanage** – PhD Physics, Western University

---

## Contact

For questions or collaboration, please open an issue in this repository.



