import json

import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from diabetes_classifier.config import PROCESSED_DATA_DIR, REPORTS_DIR
from diabetes_classifier.modeling.train import (
    DATASETS,
    load_model_and_params,
    results_to_json,
)

app = typer.Typer()

# Best model per dataset
BEST_MODELS = {
    "brfss": ("XGBoost", False),
    "prediction": ("XGBoost", True),
}


# ──────────────────────────────
# Load Test Data
# ──────────────────────────────
def load_test_data(dataset: str) -> tuple:
    logger.info(f"Loading {dataset} test set...")
    test = pd.read_csv(PROCESSED_DATA_DIR / f"featured_test_{dataset}.csv")
    target = DATASETS[dataset]
    X_test = test.drop(columns=[target])
    y_test = test[target]
    logger.success(f"Loaded {dataset} — test: {X_test.shape}")
    return X_test, y_test


# ──────────────────────────────
# Evaluate
# ──────────────────────────────
def evaluate(model, X_test, y_test, name: str, params: dict = {}) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    results = {
        name: {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_pred_prob),
        }
    }

    print(f"\n{name} [Test]:")
    print(classification_report(y_test, y_pred))
    if params:
        print(f"Best params: {params}")

    return pd.DataFrame(results).T


# ──────────────────────────────
# Main
# ──────────────────────────────
@app.command()
def main():
    for dataset, (model_name, use_tuned) in BEST_MODELS.items():
        logger.info(f"\n{'=' * 50}\nDataset: {dataset.upper()}\n{'=' * 50}")

        X_test, y_test = load_test_data(dataset)

        if use_tuned:
            # load saved tuned model
            model, params = load_model_and_params(model_name, dataset)
            if model is None:
                logger.error(f"Tuned {model_name} not found for {dataset}")
                continue
            label = f"{model_name} [Tuned]"
        else:
            # use default hyperparameters
            logger.info(f"Using default {model_name} for {dataset}")
            train = pd.read_csv(PROCESSED_DATA_DIR / f"featured_train_{dataset}.csv")
            target = DATASETS[dataset]
            X_train = train.drop(columns=[target])
            y_train = train[target]
            model = XGBClassifier(
                n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1
            )
            model.fit(X_train, y_train)
            params = {}
            label = f"{model_name} [Default]"

        results = evaluate(model, X_test, y_test, label, params)
        logger.success(f"\n{dataset.upper()} Test Results:\n{results}")

        # save
        path = REPORTS_DIR / f"{dataset}_test_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results_to_json(results), f, indent=4)
        logger.success(f"Test results saved to {path}")


if __name__ == "__main__":
    app()
