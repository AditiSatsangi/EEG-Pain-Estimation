# MT-WAN: Implicit Window Supervision for Phase-Aware EEG Pain Decoding

This repository contains the implementation of **EEG-based pain estimation using multi-task learning**.  
We propose **MT-WAN (Multi-Task Window-Aware Network)**, which extends a strong **Deep CNN–BiLSTM pain classifier** with **implicit supervision on EEG temporal phases** (Baseline / ERP / Post-Stimulus).

🟢 **Key idea:**  
The model is trained to *predict the EEG window as an auxiliary task*, which acts as a physiological regularizer.  
At inference time, **no window labels are required** — making the approach practical for real-world clinical deployment.

---

## 🧩 Overview of the Approach

Our pipeline follows the workflow illustrated in **Fig. 1 (Overall MT-WAN Framework)**:

- EEG epochs and metadata are preprocessed with:
  - epoch rejection flags  
  - channel padding/truncation to a fixed 64-channel montage  
  - per-channel z-score normalization  
- A **strict subject-wise split** (GroupShuffleSplit) prevents participant leakage  
- Experiments are repeated for **pain thresholds T ∈ {3, 5, 7}**

### Modeling Streams

1. **Classical ML (engineered features)**
   - PSD / ROI / spectral–temporal descriptors  
   - RF / XGBoost / SVM  
   - Window-aware ML with feature × window interactions

2. **Deep Learning**
   - **Baseline:** Deep CNN–BiLSTM (pain-only)
   - **Proposed MT-WAN:**  
     - Shared CNN–BiLSTM encoder  
     - Pain head + Window head  
     - Multi-task loss:

```
L = L_pain + λ L_window
```

   - Final model uses **λ = 0.2**

3. **Evaluation**
   - Acc, BalAcc, Macro-F1  
   - Bootstrap 95% CI for Δ(MTL–Baseline)  
   - Interpretability (T = 5) using **Grad×Input saliency + ERP alignment ratio**

---

## 🔬 What Is New in This Work

- **Implicit window supervision** instead of manual window input  
- **Zero-label inference** – model does not need phase tags at test time  
- **Physiologically grounded learning** through auxiliary window task  
- **Statistical reliability** via bootstrap CIs  
- **Phase-aware interpretability** without Grad-CAM

---

## 📁 Data Format

Each `.npz` file represents one EEG epoch:

- **Shape:** `(64 × 1001)`  
- **Sampling rate:** `1000 Hz`

### Required `index.csv` fields

- `participant_id` – for subject-wise split  
- `window` – {Baseline, ERP, Post} *(training only)*  
- `rating_bin` – binary pain label  
- `file_id` – maps to .npz file

```
data/
 ├── index.csv
 └── npz/
      ├── sub01_ep01.npz
      └── ...
```

---

## ⚙️ Installation

### Conda

```bash
conda env create -f environment.yml
conda activate eeg
```

### Pip

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

### Final MT-WAN (λ = 0.2)

```bash
python scripts/DL_final_training.py
```

### Hyperparameter Search

```bash
python scripts/dl_train_1000hz_gridsearch.py
```

### Classical ML Baselines

```bash
python scripts/ml_final_train.py
```

---

## 📊 Evaluation Protocol

- **Subject-wise split:** GroupShuffleSplit  
- **Multi-threshold:** T ∈ {3,5,7}  
- **Primary threshold:** T = 5  
- Metrics:  
  - Accuracy  
  - Balanced Accuracy  
  - Macro-F1  
- **Bootstrap CI (2000 resamples)** for Δ(MTL – Baseline)

---

## 🧠 Interpretability

Evaluated only at **T = 5**:

- Grad×Input saliency per epoch  
- Saliency energy in Baseline / ERP / Post  
- **ERP alignment ratio**

Run:

```bash
python scripts/interpretability_dl.py
```

Results saved to:

```
results/
```

---

## 🧪 Models Included

### Deep Learning
- CNN  
- BiLSTM  
- Transformer  
- **Deep CNN–BiLSTM (Baseline)**  
- **MT-WAN (Proposed)**

### Classical ML
- Logistic Regression (balanced)  
- Linear SVM (balanced)  
- Random Forest  
- XGBoost  
- Window-aware ML variants

---

## 📦 Outputs

```
saved_models/   → checkpoints  
results/        → metrics, figures, JSON logs  
data/           → dataset and experiment files
```

---

## 📬 Contact

For questions or collaboration, please open an issue in this repository.

⭐ If this work is useful, consider starring the repo!


