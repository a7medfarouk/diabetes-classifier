from pathlib import Path
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import typer

from diabetes_classifier.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from diabetes_classifier.validation import run_validation, validate_merge_counts

app = typer.Typer()

# ──────────────────────────────
# Load
# ──────────────────────────────
def load_datasets()-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    logger.info("Loading datasets...")
    df_diabetes_prediction = pd.read_csv(RAW_DATA_DIR / "diabetes_prediction_dataset.csv")
    df_diabetes_brfss2015 = pd.read_csv(RAW_DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv")
    df_diabetes_brfss2021 = pd.read_csv(RAW_DATA_DIR / "diabetes_binary_health_indicators_BRFSS2021.csv")
    logger.success("Datasets loaded successfully.")
    return df_diabetes_prediction, df_diabetes_brfss2015, df_diabetes_brfss2021

# ──────────────────────────────
# Clean
# ──────────────────────────────
def clean_brfss(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # fix invalid Income values
    df = df[df["Income"].between(1, 8)]

    return df


# ──────────────────────────────
# Merge
# ──────────────────────────────

def merge_brfss_datasets(df_brfss_2015: pd.DataFrame, df_brfss_2021: pd.DataFrame) -> pd.DataFrame:
    logger.info("Validating BRFSS schemas before merge...")
    assert list(df_brfss_2015.columns) == list(df_brfss_2021.columns), (
        f"Schema mismatch:\n"
        f"  2015 only: {set(df_brfss_2015.columns) - set(df_brfss_2021.columns)}\n"
        f"  2021 only: {set(df_brfss_2015.columns) - set(df_brfss_2021.columns)}"
    )
    logger.info("Merging BRFSS 2015 + 2021")
    merged = pd.concat([df_brfss_2015, df_brfss_2021], ignore_index = True)
    validate_merge_counts(df_brfss_2015, df_brfss_2021, merged)
    logger.success(f"Merged BRFSS shape: {merged.shape}")
    return merged


# ──────────────────────────────
# Split
# ──────────────────────────────
def split_data(
    df:pd.DataFrame,
    targer_col: str,
    test_size: float = 0.15,
    validation_size: float = 0.176,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df[targer_col]
    )
    
    train, val  = train_test_split(
        train, 
        test_size=validation_size, 
        random_state=random_state, 
        stratify=train[targer_col]
    )
    
    logger.info(f"Split -> train: {train.shape}, val: {val.shape}, test: {test.shape}")
    
    return train, val, test

@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR,
):
    df_prediction, df_brfss2015, df_brfss2021 = load_datasets()
    
    logger.info("Running validation on raw datasets...")
    
    result_prediction = run_validation(df_prediction, "diabetes_prediction")
    result_brfss2015 = run_validation(df_brfss2015,  "diabetes_brfss2015")
    result_brfss2021 = run_validation(df_brfss2021,  "diabetes_brfss2021")
    
        
    df_brfss2015 = clean_brfss(df_brfss2015)
    df_brfss2021 = clean_brfss(df_brfss2021)

    
    df_brfss_merged = merge_brfss_datasets(df_brfss2015, df_brfss2021)
    
    result_merged = run_validation(df_brfss_merged, "diabetes_brfss_merged")
    
    if not result_merged.success:
        logger.error("Merged BRFSS validation failed, check data docs for details.")
        raise ValueError("Merged dataset has quality issues, check validation results and data docs.")
    
    logger.info("Splitting Datasets...")
    train_brfss, val_brfss, test_brfss = split_data(df_brfss_merged, targer_col="Diabetes_binary")
    train_prediction, val_prediction, test_prediction = split_data(df_prediction, targer_col="diabetes")
    
    logger.info("Saving Processed Datasets to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    train_prediction.to_csv(output_path/"train_prediction_dataset.csv",  index=False)
    val_prediction.to_csv(output_path/"validation_prediction_dataset.csv",    index=False)
    test_prediction.to_csv(output_path/"test_prediction_dataset.csv",   index=False)

    train_brfss.to_csv(output_path/"train_brfss_dataset.csv", index=False)
    val_brfss.to_csv(output_path/"validation_brfss_dataset.csv",   index=False)
    test_brfss.to_csv(output_path/"test_brfss_dataset.csv",  index=False)
    
    logger.success("All datasets processed and saved successfully.")

if __name__ == "__main__":
    app()
