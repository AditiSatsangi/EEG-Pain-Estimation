# MT-WAN: Phase-Aware EEG Pain Decoding via Multi-Task Learning

This repository implements a complete and reproducible pipeline for **EEG-based pain classification** using both **classical machine learning** and **deep learning**.  

We propose **MT-WAN (Multi-Task Window-Aware Network)**, which improves a strong **pain-only Deep CNN–BiLSTM** baseline via **implicit window supervision**.  
The model is trained to predict both:

- **Pain label** (binary: no significant pain vs. significant pain)  
- **EEG temporal phase** (Baseline / ERP / Post)

🟢 **Key idea:**  
Window labels are used **only during training** as an auxiliary task.  
At inference time, the model requires **no window information** → practical for real clinical use.

---

## 🔬 Key Contributions

- **Implicit phase supervision via multi-task learning**  
  → joint learning of pain + window identity  
- **Zero-label inference** – no window tags required at deployment  
- **Strict subject-wise evaluation** (no participant leakage)  
- **Robust across thresholds** T ∈ {3, 5, 7}  
- **Physiological interpretability** using Grad×Input saliency

---

## 📁 Data Format

Each EEG epoch is stored as a `.npz` file:

- **Shape:** `(64 channels × 1001 time points)`  
- **Sampling rate:** `1000 Hz`

### Expected Structure

```
data/
├── index.csv
└── npz/
    ├── sample_0001.npz
    ├── sample_0002.npz
    └── ...
```

### Required Columns in `index.csv`

- `participant_id` → used for subject-wise split  
- `window` → {Baseline, ERP, Post} (used only for training auxiliary task)  
- `rating_bin` → binary pain label  
- `file_id` → maps to `.npz` file

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

### Train Final MT-WAN Model (λ = 0.2)

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
- **Multi-threshold evaluation:** T ∈ {3, 5, 7}  
- **Primary threshold:** T = 5  
- **Metrics**
  - Accuracy  
  - Balanced Accuracy  
  - Macro-F1  

- **Statistical reliability:**  
  Bootstrap CI with **2000 resamples** (primary threshold)

---

## 🧠 Interpretability (T = 5 Only)

Interpretability is evaluated using **Grad×Input saliency**:

- Phase-specific attribution (Baseline / ERP / Post)  
- ERP alignment ratio  
- Window-wise saliency energy

Run:

```bash
python scripts/interpretability_dl.py
```

Outputs saved to:

```
results/
```

---

## 🧩 Model Overview

### Baseline

- **Deep CNN–BiLSTM (pain-only)**

### Proposed Model: MT-WAN

- Shared encoder  
- Pain classification head  
- Auxiliary window head (training only)

#### Training Objective

```
L = L_pain + λ L_window
```

Final model uses:

👉 **λ = 0.2**

---

## 📦 Outputs

```
saved_models/    → trained checkpoints  
results/         → figures, metrics, JSON logs  
data/            → dataset files  
```

---

## 🧪 Classical ML Models

- Logistic Regression (balanced)  
- Linear SVM (balanced)  
- Random Forest  
- XGBoost  
- Window-aware ML variants with feature–window interactions


---

## 📬 Contact

For questions, issues, or collaboration:

👉 Please open an issue in this repository.

---

⭐ If you find this work useful, consider giving the repository a star!





