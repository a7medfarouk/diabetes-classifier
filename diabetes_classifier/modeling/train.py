from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier
import typer

from diabetes_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

DATASETS = {
    "brfss":      "Diabetes_binary",
    "prediction": "diabetes",
}


# ──────────────────────────────
# Load
# ──────────────────────────────
def load_data(dataset: str) -> tuple:
    logger.info(f"Loading {dataset} datasets...")
    train = pd.read_csv(PROCESSED_DATA_DIR / f"featured_train_{dataset}.csv")
    val   = pd.read_csv(PROCESSED_DATA_DIR / f"featured_val_{dataset}.csv")

    target  = DATASETS[dataset]
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_val   = val.drop(columns=[target])
    y_val   = val[target]

    logger.success(f"Loaded {dataset} — train: {X_train.shape}, val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


# ──────────────────────────────
# Define Models
# ──────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=50000, random_state=42, n_jobs=-1),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1),
        "KNN":                 KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "MLP":                 MLPClassifier(hidden_layer_sizes=(100,), max_iter=50000, random_state=42),
    }


# ──────────────────────────────
# Train and Evaluate
# ──────────────────────────────
def train_and_evaluate(models: dict, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred      = model.predict(X_val)
        y_pred_prob = model.predict_proba(X_val)[:, 1]

        results[name] = {
            "Accuracy":  accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred),
            "Recall":    recall_score(y_val, y_pred),
            "F1":        f1_score(y_val, y_pred),
            "ROC-AUC":   roc_auc_score(y_val, y_pred_prob),
        }

        print(f"\n{name}:")
        print(classification_report(y_val, y_pred))

    return pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)


# ──────────────────────────────
# Main
# ──────────────────────────────
@app.command()
def main(output_path: Path = MODELS_DIR):
    for dataset in DATASETS:
        logger.info(f"\n{'='*50}\nDataset: {dataset.upper()}\n{'='*50}")
        X_train, y_train, X_val, y_val = load_data(dataset)
        results = train_and_evaluate(get_models(), X_train, y_train, X_val, y_val)
        logger.success(f"\n{dataset.upper()} Results:\n{results}")


if __name__ == "__main__":
    app()