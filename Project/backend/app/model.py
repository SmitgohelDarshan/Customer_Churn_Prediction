"""
model.py — Loads all PKLs at startup; exposes predict helpers
"""
import os
import joblib
import numpy as np

_BASE = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _load(name: str):
    return joblib.load(os.path.join(_BASE, name))


# Models dictionary
models = {
    "LogisticRegression": _load("model_logreg.pkl"),
    "RandomForest": _load("model_rf.pkl"),
    "DecisionTree": _load("model_dt.pkl"),
    "KNN": _load("model_knn.pkl"),
}

scaler      = _load("scaler.pkl")
train_cols  = _load("train_columns.pkl")   # list[str], 30 features

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def predict(X: np.ndarray, model_name: str = "LogisticRegression"):
    """Returns (label_str, probability_float, actual_model_name)."""
    if model_name not in models:
        model_name = "LogisticRegression"
    clf = models[model_name]
    
    prob = float(clf.predict_proba(X)[0][1])
    label = "Yes" if prob >= 0.5 else "No"
    return label, prob, model_name
