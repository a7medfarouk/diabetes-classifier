import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from diabetes_classifier.config import PROCESSED_DATA_DIR

train = pd.read_csv(f"{PROCESSED_DATA_DIR}/featured_train_brfss.csv")
val = pd.read_csv(f"{PROCESSED_DATA_DIR}/featured_val_brfss.csv")

X_train = train.drop(columns=["Diabetes_binary"])
y_train = train["Diabetes_binary"]

X_val = val.drop(columns=["Diabetes_binary"])
y_val = val["Diabetes_binary"]

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=50000,
        random_state=42,
        n_jobs=-1,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
    ),
    # "SVM":
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1,
    ),
}


results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    # Metrics
    results[name] = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1": f1_score(y_val, y_pred),
        "ROC-AUC": roc_auc_score(y_val, y_pred_prob),
    }

    print(classification_report(y_val, y_pred))

results_df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)
print(results_df)
