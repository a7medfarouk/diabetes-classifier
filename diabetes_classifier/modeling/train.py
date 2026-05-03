from math import inf
from pathlib import Path
import warnings

from loguru import logger
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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
    "brfss": "Diabetes_binary",
    "prediction": "diabetes",
}


# ──────────────────────────────
# Load
# ──────────────────────────────
def load_data(dataset: str) -> tuple:
    logger.info(f"Loading {dataset} datasets...")
    train = pd.read_csv(PROCESSED_DATA_DIR / f"featured_train_{dataset}.csv")
    val = pd.read_csv(PROCESSED_DATA_DIR / f"featured_val_{dataset}.csv")

    target = DATASETS[dataset]
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_val = val.drop(columns=[target])
    y_val = val[target]

    logger.success(f"Loaded {dataset} — train: {X_train.shape}, val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


# ──────────────────────────────
# Define Models
# ──────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=50000, random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=50000, random_state=42
        ),
    }


def fine_tune_LR(X_train, y_train, X_val, y_val) -> tuple[LogisticRegression, dict]:
    def objective(trial: optuna.Trial) -> float:
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        C = trial.suggest_float("C", 1e-3, 100.0, log=True)

        params = {
            "solver": solver,
            "class_weight": class_weight,
            "C": C,
        }

        # if kernel == "poly":
        #     params["degree"] = trial.suggest_int("degree", 1, 3)
        #     params["gamma"] = trial.suggest_float("gamma", 1e-6, 1.0, log=True)
        # elif kernel == "rbf":
        #     params["gamma"] = trial.suggest_float("gamma", 1e-6, 1.0, log=True)

        model = LogisticRegression(**params, max_iter=50000)

        # Catch convergence warnings and penalize/prune the trial
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                model.fit(X_train, y_train)
            except ConvergenceWarning:
                # If it hits max_iter without converging, tell Optuna this is a bad path
                raise optuna.exceptions.TrialPruned()
            except Exception:
                raise optuna.exceptions.TrialPruned()

        # PR-AUC score metric
        val_scores = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_scores)

        return float(pr_auc)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.InMemoryStorage()

    study = optuna.create_study(direction="maximize", storage=storage)

    study.optimize(
        objective,
        n_trials=500,
        n_jobs=-1,
        show_progress_bar=True,
    )

    best_params = study.best_params

    # Train the final model with the optimal parameters
    best_estimator = LogisticRegression(**best_params, max_iter=50000)
    return best_estimator, best_params


def fine_tune_RF(X_train, y_train, X_val, y_val) -> tuple[RandomForestClassifier, dict]:
    def objective(trial: optuna.Trial) -> float:
        # n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        params = {
            # "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "class_weight": class_weight,
        }

        model = RandomForestClassifier(**params, n_estimators=100, n_jobs=-1)

        model.fit(X_train, y_train)

        # PR-AUC score metric
        val_scores = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_scores)

        return float(pr_auc)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.InMemoryStorage()

    study = optuna.create_study(direction="maximize", storage=storage)

    study.optimize(
        objective,
        n_trials=50,
        n_jobs=1,
        show_progress_bar=True,
    )

    best_params = study.best_params

    # Train the final model with the optimal parameters
    best_estimator = RandomForestClassifier(**best_params, n_estimators=500, n_jobs=-1)
    return best_estimator, best_params


# ──────────────────────────────
# Train and Evaluate
# ──────────────────────────────
def train_and_evaluate(
    models: dict, X_train, y_train, X_val, y_val, best_params: dict = {}
) -> pd.DataFrame:
    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_pred_prob = model.predict_proba(X_val)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred),
            "Recall": recall_score(y_val, y_pred),
            "F1": f1_score(y_val, y_pred),
            "ROC-AUC": roc_auc_score(y_val, y_pred_prob),
        }

        print(f"\n{name}:")
        print(classification_report(y_val, y_pred))
        if (params := best_params.get(name, None)) is not None:
            print(f"With parameters: {params}")

    return pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)


# ──────────────────────────────
# Fine-Tuned models
# ──────────────────────────────
def get_tuned_models(X_train, y_train, X_val, y_val) -> tuple[dict, dict]:
    best_params = {}
    models = {}

    # LR, best_params_LR = fine_tune_LR(X_train, y_train, X_val, y_val)
    # models["Logistic Regression"] = LR
    # best_params["Logistic Regression"] = best_params_LR

    RF, best_params_RF = fine_tune_RF(X_train, y_train, X_val, y_val)
    models["Random Forest"] = RF
    best_params["Random Forest"] = best_params_RF

    return models, best_params


# ──────────────────────────────
# Main
# ──────────────────────────────
@app.command()
def main(output_path: Path = MODELS_DIR):
    for dataset in DATASETS:
        logger.info(f"\n{'=' * 50}\nDataset: {dataset.upper()}\n{'=' * 50}")
        X_train, y_train, X_val, y_val = load_data(dataset)
        # results = train_and_evaluate(get_models(), X_train, y_train, X_val, y_val)
        # logger.success(f"\n{dataset.upper()} Results:\n{results}")
        models, best_params = get_tuned_models(X_train, y_train, X_val, y_val)
        tuned_results = train_and_evaluate(
            models, X_train, y_train, X_val, y_val, best_params
        )
        logger.success(f"\n{dataset.upper()} Results:\n{tuned_results}")


if __name__ == "__main__":
    app()
