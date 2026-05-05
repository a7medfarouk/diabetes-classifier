import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
import pandas as pd
from sklearn.neural_network import MLPClassifier
import typer
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

from diabetes_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

matplotlib.use("Agg")
mlflow.set_tracking_uri(uri=(Path.cwd() / "mlruns").as_uri())

app = typer.Typer()

DATASETS = {
    "brfss": "Diabetes_binary",
    "prediction": "diabetes",
}


# ──────────────────────────────
# Load Data
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
# PCA
# ──────────────────────────────
def apply_pca(X_train, X_val, variance_threshold: float = 0.95):
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    logger.info(
        f"PCA: {X_train.shape[1]} features -> {pca.n_components_} components ({variance_threshold * 100:.0f}% variance retained)"
    )
    return X_train_pca, X_val_pca, pca


# ──────────────────────────────
# Default Models
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
        "KNN": KNeighborsClassifier(n_neighbors=4, n_jobs=-1),
        "OCSVM": OneClassSVM(kernel="poly", max_iter=1000),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=50000, random_state=42
        ),
    }


# ──────────────────────────────
# Train and Evaluate
# ──────────────────────────────
def train_and_evaluate(
    models: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    best_params: dict = {},
    label: str = "",
) -> pd.DataFrame:
    results = {}
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.types.utils")
    mlflow.sklearn.autolog()
    mlflow.xgboost.autolog()

    # Initialize the MLflow client
    client = mlflow.MlflowClient()

    # Check if the experiment already exists
    exp = client.get_experiment_by_name("classifier-comparison")

    if exp is None:
        # Create a new experiment with specified name and tags
        experiment_id = client.create_experiment(
            name="classifier-comparison",
            tags={"topic": "classification-lab", "version": "v1"},
        )
    else:
        experiment_id = exp.experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id, run_name="model-comparison"
    ) as parent_run:
        for name, model in models.items():
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=name,
                parent_run_id=parent_run.info.run_id,
                nested=True,
            ):
                logger.info(f"Training {name} {label}...")
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                if hasattr(model, "predict_proba"):
                    y_pred_prob = model.predict_proba(X_val)[:, 1]
                else:
                    # For OCSVM
                    y_pred_prob = model.score_samples(X_val)

                results[name] = {
                    "Accuracy": accuracy_score(y_val, y_pred),
                    "Precision": precision_score(y_val, y_pred),
                    "Recall": recall_score(y_val, y_pred),
                    "F1": f1_score(y_val, y_pred),
                    "ROC-AUC": roc_auc_score(y_val, y_pred_prob),
                }

                print(f"\n{name} {label}:")
                print(classification_report(y_val, y_pred))
                if (params := best_params.get(name, None)) is not None:
                    print(f"Best params: {params}")

    mlflow.sklearn.autolog(disable=True)
    mlflow.xgboost.autolog(disable=True)
    return pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)


# ──────────────────────────────
# Save / Load
# ──────────────────────────────
def save_model_and_params(model, params: dict, name: str, dataset: str):
    path = MODELS_DIR / dataset
    path.mkdir(parents=True, exist_ok=True)
    filename = name.lower().replace(" ", "_")
    joblib.dump(model, path / f"{filename}.pkl")
    joblib.dump(params, path / f"{filename}_params.pkl")
    logger.success(f"Saved {name} model and params to {path}")


def load_model_and_params(name: str, dataset: str):
    path = MODELS_DIR / dataset
    filename = name.lower().replace(" ", "_")
    model_path = path / f"{filename}.pkl"
    params_path = path / f"{filename}_params.pkl"
    if model_path.exists() and params_path.exists():
        logger.info(f"Loading cached {name} for {dataset}")
        return joblib.load(model_path), joblib.load(params_path)
    return None, None


def save_results(results: dict, dataset: str):
    path = MODELS_DIR / dataset / "results.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for label, df in results.items():
            f.write(f"\n{'=' * 50}\n{label}\n{'=' * 50}\n")
            f.write(df.to_string())
            f.write("\n")
    logger.success(f"Results saved to {path}")


def results_to_json(results_df: pd.DataFrame) -> dict:
    return {
        model: {
            "Accuracy": round(row["Accuracy"], 4),
            "Precision": round(row["Precision"], 4),
            "Recall": round(row["Recall"], 4),
            "F1": round(row["F1"], 4),
            "ROC-AUC": round(row["ROC-AUC"], 4),
        }
        for model, row in results_df.iterrows()
    }


# ──────────────────────────────
# Fine Tuning
# ──────────────────────────────
def fine_tune_LR(
    X_train, y_train, X_val, y_val, dataset: str
) -> tuple[LogisticRegression, dict]:
    model, params = load_model_and_params("Logistic Regression", dataset)
    if model is not None:
        return model, params

    def objective(trial: optuna.Trial) -> float:
        p = {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
        m = LogisticRegression(**p, max_iter=50000)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                m.fit(X_train, y_train)
            except (ConvergenceWarning, Exception):
                raise optuna.exceptions.TrialPruned()
        return float(average_precision_score(y_val, m.predict_proba(X_val)[:, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500, n_jobs=-1, show_progress_bar=True)
    best_params = study.best_params
    best_model = LogisticRegression(**best_params, max_iter=50000)
    best_model.fit(X_train, y_train)
    save_model_and_params(best_model, best_params, "Logistic Regression", dataset)
    return best_model, best_params


def fine_tune_RF(
    X_train, y_train, X_val, y_val, dataset: str
) -> tuple[RandomForestClassifier, dict]:
    model, params = load_model_and_params("Random Forest", dataset)
    if model is not None:
        return model, params

    def objective(trial: optuna.Trial) -> float:
        p = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
        m = RandomForestClassifier(**p, n_estimators=100, n_jobs=-1)
        m.fit(X_train, y_train)
        return float(average_precision_score(y_val, m.predict_proba(X_val)[:, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=True)
    best_params = study.best_params
    best_model = RandomForestClassifier(**best_params, n_estimators=500, n_jobs=-1)
    best_model.fit(X_train, y_train)
    save_model_and_params(best_model, best_params, "Random Forest", dataset)
    return best_model, best_params


def fine_tune_XGB(
    X_train, y_train, X_val, y_val, dataset: str
) -> tuple[XGBClassifier, dict]:
    model, params = load_model_and_params("XGBoost", dataset)
    if model is not None:
        return model, params

    def objective(trial: optuna.Trial) -> float:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 4),
        }
        m = XGBClassifier(**p, random_state=42, eval_metric="logloss", n_jobs=-1)
        m.fit(X_train, y_train)
        return float(average_precision_score(y_val, m.predict_proba(X_val)[:, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=-1, show_progress_bar=True)
    best_params = study.best_params
    best_model = XGBClassifier(
        **best_params, random_state=42, eval_metric="logloss", n_jobs=-1
    )
    best_model.fit(X_train, y_train)
    save_model_and_params(best_model, best_params, "XGBoost", dataset)
    return best_model, best_params


def fine_tune_OCSVM(
    X_train, y_train, X_val, y_val, dataset: str
) -> tuple[OneClassSVM, dict]:
    model, params = load_model_and_params("OCSVM", dataset)
    if model is not None:
        return model, params

    def objective(trial: optuna.Trial) -> float:

        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        nu = trial.suggest_float("nu", 1e-3, 0.5, log=True)

        params = {
            "kernel": kernel,
            "nu": nu,
            "cache_size": 2000,
            "max_iter": 50000,
        }

        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 1, 3)
            params["gamma"] = trial.suggest_float("gamma", 1e-6, 1.0, log=True)
        elif kernel == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-6, 1.0, log=True)

        model = OneClassSVM(**params)

        # Catch convergence warnings and penalize/prune the trial
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                model.fit(X_train)
            except ConvergenceWarning:
                # If it hits max_iter without converging, tell Optuna this is a bad path
                raise optuna.exceptions.TrialPruned()
            except Exception:
                # Catch potential math/value errors from weird hyperparameter combos
                raise optuna.exceptions.TrialPruned()

        # PR-AUC score metric
        val_scores = -model.score_samples(X_val)
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
    best_model = OneClassSVM(**best_params, cache_size=2000, max_iter=-1)
    best_model.fit(X_train)

    save_model_and_params(best_model, best_params, "One-class SVM", dataset)
    return best_model, best_params


# ──────────────────────────────
# Get Tuned Models
# ──────────────────────────────
def get_tuned_models(X_train, y_train, X_val, y_val, dataset: str) -> tuple[dict, dict]:
    models, best_params = {}, {}

    models["Logistic Regression"], best_params["Logistic Regression"] = fine_tune_LR(
        X_train, y_train, X_val, y_val, dataset
    )
    models["Random Forest"], best_params["Random Forest"] = fine_tune_RF(
        X_train, y_train, X_val, y_val, dataset
    )
    models["XGBoost"], best_params["XGBoost"] = fine_tune_XGB(
        X_train, y_train, X_val, y_val, dataset
    )
    models["OCSVM"], best_params["OCSVM"] = fine_tune_OCSVM(
        X_train, y_train, X_val, y_val, dataset
    )
    # models["MLP"], best_params["MLP"] = fine_tune_MLP(
    #     X_train, y_train, X_val, y_val, dataset
    # )

    return models, best_params


# ──────────────────────────────
# Main
# ──────────────────────────────
@app.command()
def main(output_path: Path = MODELS_DIR):
    for dataset in DATASETS:
        logger.info(f"\n{'=' * 50}\nDataset: {dataset.upper()}\n{'=' * 50}")
        X_train, y_train, X_val, y_val = load_data(dataset)

        # Training with Default Hyper Parameters
        logger.info("-------------------- Default Hyperparameters --------------------")
        default_results = train_and_evaluate(
            get_models(), X_train, y_train, X_val, y_val, label="[Default]"
        )
        logger.success(f"\n{dataset.upper()} Default Results:\n{default_results}")

        # save default result in json for easy access later
        path = REPORTS_DIR / f"{dataset}_default_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump(results_to_json(default_results), file, indent=4)

        # 2. PCA then train with default hyper parameters
        logger.info("------------------PCA -----------------------")
        X_train_pca, X_val_pca, _ = apply_pca(X_train, X_val)
        pca_results = train_and_evaluate(
            get_models(), X_train_pca, y_train, X_val_pca, y_val, label="[PCA]"
        )
        logger.success(f"\n{dataset.upper()} PCA Results:\n{pca_results}")

        # save pca result in json for easy access later
        path = REPORTS_DIR / f"{dataset}_pca_metrics.json"
        with open(path, "w") as file:
            json.dump(results_to_json(pca_results), file, indent=4)

        # Tune the models then Train
        logger.info("---------------Tuning Hyperparameters -------------")
        tuned_models, best_params = get_tuned_models(
            X_train, y_train, X_val, y_val, dataset
        )
        tuned_results = train_and_evaluate(
            tuned_models, X_train, y_train, X_val, y_val, best_params, label="[Tuned]"
        )
        logger.success(f"\n{dataset.upper()} Tuned Results:\n{tuned_results}")

        # save tuned result in json for easy access later
        path = REPORTS_DIR / f"{dataset}_tuned_metrics.json"
        with open(path, "w") as file:
            json.dump(results_to_json(tuned_results), file, indent=4)

        # Save results
        save_results(
            {
                "Default": default_results,
                "PCA": pca_results,
                "Tuned": tuned_results,
            },
            dataset,
        )

        # Summary
        logger.info(f"\n{'=' * 50}\n{dataset.upper()} SUMMARY\n{'=' * 50}")
        logger.info(f"Default:\n{default_results}")
        logger.info(f"PCA:\n{pca_results}")
        logger.info(f"Tuned:\n{tuned_results}")


if __name__ == "__main__":
    app()
