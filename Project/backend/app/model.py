"""
model.py — Loads all PKLs at startup; exposes predict helpers
"""
import os
import joblib
import numpy as np

_BASE = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _load(name: str):
    return joblib.load(os.path.join(_BASE, name))


# Primary production model (LogisticRegression)
model       = _load("model_logreg.pkl")
scaler      = _load("scaler.pkl")
train_cols  = _load("train_columns.pkl")   # list[str], 30 features

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
MODEL_NAME = "LogisticRegression"


def predict(X: np.ndarray):
    """Returns (label_str, probability_float)."""
    prob = float(model.predict_proba(X)[0][1])
    label = "Yes" if prob >= 0.5 else "No"
    return label, prob
