from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import typer

from diabetes_classifier.config import PROCESSED_DATA_DIR

app = typer.Typer()

# ──────────────────────────────
# Load
# ──────────────────────────────

def load_training_sets() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading Training sets...")
    df_train_prediction = pd.read_csv(PROCESSED_DATA_DIR / "train_prediction_dataset.csv")
    df_train_brfss = pd.read_csv(PROCESSED_DATA_DIR / "train_brfss_dataset.csv")
    logger.success("Training sets loaded successfully.")
    return df_train_prediction, df_train_brfss

def load_validation_sets() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading Validation sets...")
    df_validation_prediction = pd.read_csv(PROCESSED_DATA_DIR / "validation_prediction_dataset.csv")
    df_validation_brfss = pd.read_csv(PROCESSED_DATA_DIR / "validation_brfss_dataset.csv")
    logger.success("Validation sets loaded successfully.")
    return df_validation_prediction, df_validation_brfss

def load_test_sets() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading Test sets...")
    df_test_prediction = pd.read_csv(PROCESSED_DATA_DIR / "test_prediction_dataset.csv")
    df_test_brfss = pd.read_csv(PROCESSED_DATA_DIR / "test_brfss_dataset.csv")
    logger.success("Test sets loaded successfully.")
    return df_test_prediction, df_test_brfss

# ──────────────────────────────
# Universal Cleaning (Pre-Split)
# ──────────────────────────────

def clean_brfss(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # drop duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} duplicate rows")
    
    # drop invalid Income values
    before = len(df)
    df = df[df["Income"].between(1, 8)]
    logger.info(f"Dropped {before - len(df)} rows with invalid Income values")
    
    # cap unrealistic BMI
    df["BMI"] = df["BMI"].clip(12, 70)
    
    
    return df.reset_index(drop=True)


def clean_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # drop duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} duplicate rows")
    
    # standardize smoking categories
    df["smoking_history"] = df["smoking_history"].replace({"not current": "former"})
    
    # drop invalid categorical values
    before = len(df)
    df = df[df["gender"] != "Other"]
    df = df[df["smoking_history"] != "ever"]
    logger.info(f"Dropped {before - len(df)} rows with invalid categorical values")
    
    # cap unrealistic values
    df["bmi"]         = df["bmi"].clip(12, 70)
    df["HbA1c_level"] = df["HbA1c_level"].clip(3.5, 15)
    
    return df.reset_index(drop=True)

# ──────────────────────────────
# Advanced Cleaning (Post-Split)
# ──────────────────────────────
# ──────────────────────────────
# Scaling
# ──────────────────────────────
def scale_prediction(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train, val, test = train.copy(), val.copy(), test.copy()

    # RobustScaler for HbA1c_level and blood_glucose_level
    robust_cols   = ["HbA1c_level", "blood_glucose_level"]
    robust_scaler = RobustScaler()
    train[robust_cols] = robust_scaler.fit_transform(train[robust_cols])
    val[robust_cols]   = robust_scaler.transform(val[robust_cols])
    test[robust_cols]  = robust_scaler.transform(test[robust_cols])

    # MinMaxScaler for age
    mm_scaler    = MinMaxScaler()
    train[["age"]] = mm_scaler.fit_transform(train[["age"]])
    val[["age"]]   = mm_scaler.transform(val[["age"]])
    test[["age"]]  = mm_scaler.transform(test[["age"]])

    # Log Transform + StandardScaler for bmi
    for df in [train, val, test]:
        df["bmi"] = np.log1p(df["bmi"])
    bmi_scaler     = StandardScaler()
    train[["bmi"]] = bmi_scaler.fit_transform(train[["bmi"]])
    val[["bmi"]]   = bmi_scaler.transform(val[["bmi"]])
    test[["bmi"]]  = bmi_scaler.transform(test[["bmi"]])

    scalers = {"robust": robust_scaler, "minmax": mm_scaler, "bmi": bmi_scaler}
    logger.info("Scaling applied to prediction dataset.")
    return train, val, test, scalers


def scale_brfss(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train, val, test = train.copy(), val.copy(), test.copy()

    # Log Transform + StandardScaler for BMI
    for df in [train, val, test]:
        df["BMI"] = np.log1p(df["BMI"])
    bmi_scaler      = StandardScaler()
    train[["BMI"]]  = bmi_scaler.fit_transform(train[["BMI"]])
    val[["BMI"]]    = bmi_scaler.transform(val[["BMI"]])
    test[["BMI"]]   = bmi_scaler.transform(test[["BMI"]])

    scalers = {"bmi": bmi_scaler}
    logger.info("Scaling applied to BRFSS dataset.")
    return train, val, test, scalers


# ──────────────────────────────
# Discretization
# ──────────────────────────────
def discretize_brfss(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, val, test = train.copy(), val.copy(), test.copy()

    bins   = [-1, 0, 13, 30]
    labels = ["Zero", "Moderate", "Severe"]

    for df in [train, val, test]:
        df["MentHlth_binned"] = pd.cut(df["MentHlth"], bins=bins, labels=labels)
        df["PhysHlth_binned"] = pd.cut(df["PhysHlth"], bins=bins, labels=labels)
        df.drop(columns=["MentHlth", "PhysHlth"], inplace=True)

    logger.info("Discretization applied to BRFSS dataset.")
    return train, val, test


# ──────────────────────────────
# Encoding
# ──────────────────────────────
def encode_brfss(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.get_dummies(train, columns=["MentHlth_binned", "PhysHlth_binned"], drop_first=True, dtype=int)
    
    # align val and test to train columns
    val  = pd.get_dummies(val,  columns=["MentHlth_binned", "PhysHlth_binned"], drop_first=True, dtype=int)
    test = pd.get_dummies(test, columns=["MentHlth_binned", "PhysHlth_binned"], drop_first=True, dtype=int)
    val,  _ = val.align(train,  join="right", axis=1, fill_value=0)
    test, _ = test.align(train, join="right", axis=1, fill_value=0)

    logger.info("Encoding applied to BRFSS dataset.")
    return train, val, test


def encode_prediction(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.get_dummies(train, columns=["gender", "smoking_history"], drop_first=True, dtype=int)

    # align val and test to train columns
    val  = pd.get_dummies(val,  columns=["gender", "smoking_history"], drop_first=True, dtype=int)
    test = pd.get_dummies(test, columns=["gender", "smoking_history"], drop_first=True, dtype=int)
    val,  _ = val.align(train,  join="right", axis=1, fill_value=0)
    test, _ = test.align(train, join="right", axis=1, fill_value=0)

    logger.info("Encoding applied to prediction dataset.")
    return train, val, test


# ──────────────────────────────
# Feature Interactions
# ──────────────────────────────
def feature_interactions_brfss(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for df in [train, val, test]:
        df["Cardio_Comorbidity_Score"] = df["HighBP"] + df["HighChol"] + df["Stroke"] + df["HeartDiseaseorAttack"]
        df.drop(columns=["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"], inplace=True)

        df["Lifestyle_Score"] = df["PhysActivity"] + df["Fruits"] + df["Veggies"]
        df.drop(columns=["PhysActivity", "Fruits", "Veggies"], inplace=True)

    logger.info("Feature interactions applied to BRFSS dataset.")
    return train, val, test


def feature_interactions_prediction(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for df in [train, val, test]:
        df["Cardio_Risk_Score"] = df["hypertension"] + df["heart_disease"]
        df.drop(columns=["hypertension", "heart_disease"], inplace=True)

    logger.info("Feature interactions applied to prediction dataset.")
    return train, val, test


# ──────────────────────────────
# Balancing (SMOTE — train only)
# ──────────────────────────────
def balance(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE applied. Train size: {len(X_train)} → {len(X_balanced)}")
    return X_balanced, y_balanced


# ──────────────────────────────
# Main
# ──────────────────────────────
@app.command()
def main(output_path: Path = PROCESSED_DATA_DIR):

    # 1. Load
    train_prediction, train_brfss = load_training_sets()
    val_prediction,   val_brfss   = load_validation_sets()
    test_prediction,  test_brfss  = load_test_sets()

    # 2. Scale
    train_prediction, val_prediction, test_prediction, _ = scale_prediction(train_prediction, val_prediction, test_prediction)
    train_brfss,      val_brfss,      test_brfss,      _ = scale_brfss(train_brfss, val_brfss, test_brfss)

    # 3. Discretize
    train_brfss, val_brfss, test_brfss = discretize_brfss(train_brfss, val_brfss, test_brfss)

    # 4. Encode
    train_brfss,      val_brfss,      test_brfss      = encode_brfss(train_brfss, val_brfss, test_brfss)
    train_prediction, val_prediction, test_prediction = encode_prediction(train_prediction, val_prediction, test_prediction)

    # 5. Feature Interactions
    train_brfss,      val_brfss,      test_brfss      = feature_interactions_brfss(train_brfss, val_brfss, test_brfss)
    train_prediction, val_prediction, test_prediction = feature_interactions_prediction(train_prediction, val_prediction, test_prediction)

    # 6. Separate features and target
    X_train_brfss      = train_brfss.drop(columns=["Diabetes_binary"])
    y_train_brfss      = train_brfss["Diabetes_binary"]
    X_train_prediction = train_prediction.drop(columns=["diabetes"])
    y_train_prediction = train_prediction["diabetes"]

    # 7. Balance train only
    X_train_brfss,      y_train_brfss      = balance(X_train_brfss,      y_train_brfss)
    X_train_prediction, y_train_prediction = balance(X_train_prediction, y_train_prediction)

    # 8. Save
    logger.info(f"Saving featured datasets to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    pd.concat([X_train_brfss, y_train_brfss], axis=1).to_csv(output_path / "featured_train_brfss.csv", index=False)
    val_brfss.to_csv(output_path  / "featured_val_brfss.csv",  index=False)
    test_brfss.to_csv(output_path / "featured_test_brfss.csv", index=False)

    pd.concat([X_train_prediction, y_train_prediction], axis=1).to_csv(output_path / "featured_train_prediction.csv", index=False)
    val_prediction.to_csv(output_path  / "featured_val_prediction.csv",  index=False)
    test_prediction.to_csv(output_path / "featured_test_prediction.csv", index=False)

    logger.success("Feature engineering complete.")


if __name__ == "__main__":
    app()
