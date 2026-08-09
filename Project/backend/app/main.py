"""
main.py — FastAPI application
Endpoints:
  POST /predict
  GET  /dataset-info
  GET  /model-info
  GET  /metrics
  GET  /metrics/{model_name}
"""
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import CustomerInput, PredictionResponse, ModelInfoResponse
from app.preprocessing import preprocess
from app import model as mdl

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    description="Serves churn predictions, dataset info, and model metrics for the Telco Churn ML project.",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Artifact paths ────────────────────────────────────────────────────────────
_ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _load_json(filename: str) -> dict:
    path = os.path.join(_ARTIFACTS, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """Preprocess raw customer data and return churn prediction + probability."""
    try:
        X = preprocess(customer)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    label, prob, actual_model = mdl.predict(X, model_name=customer.model_name)
    return PredictionResponse(
        churn_label=label,
        churn_probability=round(prob, 4),
        model_used=actual_model,
    )


@app.get("/dataset-info")
def dataset_info():
    """Returns dataset_info.json — column metadata, churn stats, cleaning notes."""
    return _load_json("dataset_info.json")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(model_name: str = "LogisticRegression"):
    """Returns metadata about the requested model."""
    metrics = _load_json("metrics.json")
    if model_name not in metrics:
        model_name = "LogisticRegression"
    
    model_data = metrics[model_name]
    
    descriptions = {
        "LogisticRegression": "Logistic Regression estimates the probability of a customer churning by computing a weighted sum of their feature values and passing it through a sigmoid function that maps any real number to a value between 0 and 1. A threshold of 0.5 is then applied: customers with a predicted probability above 0.5 are classified as likely to churn. The model is interpretable — each feature's coefficient directly reflects its contribution to the churn risk.",
        "RandomForest": "Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes for classification. It corrects for decision trees' habit of overfitting to their training set.",
        "DecisionTree": "Decision Tree builds a classification model in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.",
        "KNN": "K-Nearest Neighbors is a non-parametric method used for classification. A customer is classified by a plurality vote of their neighbors, with the customer being assigned to the class most common among their k nearest neighbors (k is a positive integer, typically small)."
    }
    
    return ModelInfoResponse(
        algorithm=model_data.get("display_name", model_name),
        params=model_data.get("params", {}),
        label_encoded_cols=["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"],
        onehot_cols=[
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "PaymentMethod", "Contract",
        ],
        scaled_cols=["tenure", "MonthlyCharges", "TotalCharges"],
        description=descriptions.get(model_name, "Machine Learning model.")
    )


@app.get("/metrics")
def all_metrics():
    """Returns metrics for all trained models (accuracy, precision, recall, f1, confusion matrix)."""
    return _load_json("metrics.json")


@app.get("/metrics/{model_name}")
def single_model_metrics(model_name: str):
    """Returns metrics for a single model by key (e.g. LogisticRegression, RandomForest)."""
    data = _load_json("metrics.json")
    if model_name not in data:
        valid = list(data.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Valid keys: {valid}",
        )
    return data[model_name]


@app.get("/health")
def health():
    return {"status": "ok", "model": mdl.MODEL_NAME}
