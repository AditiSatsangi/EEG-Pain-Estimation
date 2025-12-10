# src/training/classical_ml.py
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def prepare_features_for_classical_ml(X):
    return X.mean(axis=2)   # simple baseline feature

def train_svm(X, y, C=1.0, kernel="rbf"):
    model = SVC(C=C, kernel=kernel, probability=True)
    model.fit(X, y)
    return model

def train_random_forest(X, y, n_estimators=200):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    return model

def evaluate_classical_model(model, X, y, label_names):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    rep = classification_report(y, preds, target_names=label_names)
    return {"accuracy": acc, "report": rep}
