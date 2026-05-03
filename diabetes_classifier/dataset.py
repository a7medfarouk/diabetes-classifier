from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from diabetes_classifier.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from diabetes_classifier.features import clean_brfss, clean_prediction
from diabetes_classifier.validation import validate_merge_columns, validate_merge_counts

app = typer.Typer()

# ──────────────────────────────
# Load
# ──────────────────────────────


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading datasets...")
    df_prediction = pd.read_csv(RAW_DATA_DIR / "diabetes_prediction_dataset.csv")
    df_brfss2015 = pd.read_csv(
        RAW_DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
    )
    df_brfss2021 = pd.read_csv(
        RAW_DATA_DIR / "diabetes_binary_health_indicators_BRFSS2021.csv"
    )
    logger.success("Datasets loaded successfully.")
    return df_prediction, df_brfss2015, df_brfss2021


# ──────────────────────────────
# Merge
# ──────────────────────────────
def merge_brfss_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging BRFSS 2015 + 2021...")
    merged = pd.concat([df1, df2], ignore_index=True)
    validate_merge_counts(df1, df2, merged)
    logger.success(f"Merged BRFSS shape: {merged.shape}")
    return merged


# ──────────────────────────────
# Split
# ──────────────────────────────
def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.15,
    validation_size: float = 0.176,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    train, val = train_test_split(
        train,
        test_size=validation_size,
        random_state=random_state,
        stratify=train[target_col],
    )
    logger.info(f"Split -> train: {train.shape}, val: {val.shape}, test: {test.shape}")
    return train, val, test


# ──────────────────────────────
# Main
# ──────────────────────────────
@app.command()
def main(output_path: Path = INTERIM_DATA_DIR):

    # Load
    df_prediction, df_brfss2015, df_brfss2021 = load_datasets()

    # validate schema before merge
    validate_merge_columns(df_brfss2015, df_brfss2021)

    # Merge
    df_brfss_merged = merge_brfss_datasets(df_brfss2015, df_brfss2021)
    df_brfss_merged.to_csv(RAW_DATA_DIR / "diabetes_binary_health_indicators_BRFSS_merged.csv", index = False)
    logger.info(f"BRFSS Merged: {df_brfss_merged.shape}")
    
    
    # apply universal data cleaning by removing invalid values or categories and clipping unrealistic values.
    df_prediction = clean_prediction(df_prediction)
    df_brfss_merged = clean_brfss(df_brfss_merged)

    # Split
    logger.info("Splitting datasets...")
    train_brfss, val_brfss, test_brfss = split_data(
        df_brfss_merged, target_col="Diabetes_binary"
    )
    train_prediction, val_prediction, test_prediction = split_data(
        df_prediction, target_col="diabetes"
    )

    # Save
    logger.info(f"Saving processed datasets to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    train_brfss.to_csv(output_path / "train_brfss_dataset.csv", index=False)
    val_brfss.to_csv(output_path / "validation_brfss_dataset.csv", index=False)
    test_brfss.to_csv(output_path / "test_brfss_dataset.csv", index=False)

    train_prediction.to_csv(output_path / "train_prediction_dataset.csv", index=False)
    val_prediction.to_csv(
        output_path / "validation_prediction_dataset.csv", index=False
    )
    test_prediction.to_csv(output_path / "test_prediction_dataset.csv", index=False)

    logger.success("All datasets processed and saved successfully.")


if __name__ == "__main__":
    app()
