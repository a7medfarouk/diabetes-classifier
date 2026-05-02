from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

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


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    return

if __name__ == "__main__":
    app()
