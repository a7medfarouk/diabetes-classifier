from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from diabetes_classifier.config import PROCESSED_DATA_DIR

app = typer.Typer()

def clean_brfss(df: pd.DataFrame):
    df = df.copy()
    
    before = len(df)
    
    df = df[df["Income"].between(1,8)]
    logger.info(f"Dropped {before - len(df)} rows with invalid Income values")
    
    
    df["BMI"] = df["BMI"].clip(12,70)
    
    return df.reset_index(drop=True)

def clean_prediction(df:pd.DataFrame):
    df = df.copy()
    
    before = len(df)
    
    df = df[df["gender"] != "Other"]
    df = df[df["smoking_history"] != "ever"]
    
    logger.info(f"Dropped {before - len(df)} rows with invalid categorical values")
    
    df["bmi"]         = df["bmi"].clip(12, 70)
    df["HbA1c_level"] = df["HbA1c_level"].clip(3.5, 15)
    
    return df.reset_index(drop=True)
    

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
