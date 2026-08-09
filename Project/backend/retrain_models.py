"""
retrain_models.py
=================
Trains 3 additional comparison models (RandomForest, DecisionTree, KNN)
on the SAME preprocessed data as the original LogisticRegression.
Keeps the existing model_logreg.pkl + scaler.pkl + train_columns.pkl untouched.
Extends artifacts/metrics.json to include all 4 models.

Run from the backend/ directory:
    python retrain_models.py
"""

import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── 1. Load raw data ──────────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

# ── 2. Preprocessing (mirrors EDA.ipynb exactly) ──────────────────────────────
# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop customerID
df.drop(columns=['customerID'], inplace=True)

# Label-encode binary columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-category columns with drop_first=True
onehot_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaymentMethod', 'Contract'
]
df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

# Train/test split
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric columns (fit on train only)
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Ensure no NaNs remain before training
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ── 3. Load existing LogisticRegression metrics ───────────────────────────────
metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
with open(metrics_path, "r") as f:
    all_metrics = json.load(f)

# ── 4. Train comparison models ────────────────────────────────────────────────
comparison_models = {
    "RandomForest": {
        "clf": RandomForestClassifier(random_state=42, n_estimators=100),
        "display_name": "Random Forest",
        "params": {"n_estimators": 100, "random_state": 42},
        "pkl_name": "model_rf.pkl",
    },
    "DecisionTree": {
        "clf": DecisionTreeClassifier(random_state=42),
        "display_name": "Decision Tree",
        "params": {"random_state": 42},
        "pkl_name": "model_dt.pkl",
    },
    "KNN": {
        "clf": KNeighborsClassifier(),
        "display_name": "K-Nearest Neighbors",
        "params": {"n_neighbors": 5},
        "pkl_name": "model_knn.pkl",
    },
}

for key, cfg in comparison_models.items():
    print(f"\nTraining {cfg['display_name']}...")
    clf = cfg["clf"]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc  = round(float(accuracy_score(y_test, y_pred)), 4)
    prec = round(float(precision_score(y_test, y_pred)), 4)
    rec  = round(float(recall_score(y_test, y_pred)), 4)
    f1   = round(float(f1_score(y_test, y_pred)), 4)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    print(f"  Accuracy={acc}, Precision={prec}, Recall={rec}, F1={f1}")
    print(f"  Confusion matrix: {cm}")

    # Save model PKL
    pkl_path = os.path.join(ARTIFACTS_DIR, cfg["pkl_name"])
    joblib.dump(clf, pkl_path)
    print(f"  Saved {pkl_path}")

    # Add to metrics dict
    all_metrics[key] = {
        "display_name": cfg["display_name"],
        "is_active": False,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "params": cfg["params"],
    }

# ── 5. Save updated metrics.json ──────────────────────────────────────────────
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nSaved updated metrics.json with {len(all_metrics)} models.")

print("\nDone! All comparison model artifacts saved.")
