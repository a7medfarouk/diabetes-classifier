import great_expectations as gx
import pandas as pd
from loguru import logger

dataset_rules = {
    "diabetes_prediction": {
        "accuracy": {
            "age": (0, 80),
            "bmi": (10.0, 100.0),
            "HbA1c_level": (3.5, 20.0),
            "blood_glucose_level": (40, 500),
        },
        "unrealistic": {
            # ranges acquired from teh ranges used by this study: https://pmc.ncbi.nlm.nih.gov/articles/PMC8578343/
            # blood glucose and age are already inside realistic ranges
            "bmi": (12, 70),
            "HbA1c_level": (3.5, 15),
        },
        "completeness": [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level",
            "diabetes",
        ],
        "categorical": {
            "gender": ["Female", "Male"],
            "smoking_history": ["never", "No Info", "current", "former"],
            "hypertension": [0, 1],
            "heart_disease": [0, 1],
            "diabetes": [0, 1],
        },
    },
    "diabetes_brfss2015": {
        "accuracy": {
            "BMI": (10.0, 100.0),
            "MentHlth": (0, 30),
            "PhysHlth": (0, 30),
        },
        "unrealistic": {
            "BMI": (12, 70),
        },
        "completeness": [
            "Diabetes_binary",
            "HighBP",
            "HighChol",
            "CholCheck",
            "BMI",
            "Smoker",
            "Stroke",
            "HeartDiseaseorAttack",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
            "GenHlth",
            "MentHlth",
            "PhysHlth",
            "DiffWalk",
            "Sex",
            "Age",
            "Education",
            "Income",
        ],
        "categorical": {
            "Diabetes_binary": [0, 1],
            "HighBP": [0, 1],
            "HighChol": [0, 1],
            "CholCheck": [0, 1],
            "Smoker": [0, 1],
            "Stroke": [0, 1],
            "HeartDiseaseorAttack": [0, 1],
            "PhysActivity": [0, 1],
            "Fruits": [0, 1],
            "Veggies": [0, 1],
            "HvyAlcoholConsump": [0, 1],
            "AnyHealthcare": [0, 1],
            "NoDocbcCost": [0, 1],
            "DiffWalk": [0, 1],
            "Sex": [0, 1],
            "GenHlth": [1, 2, 3, 4, 5],
            "Age": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "Education": [1, 2, 3, 4, 5, 6],
            "Income": [1, 2, 3, 4, 5, 6, 7, 8],
        },
    },
    "diabetes_brfss2021": {
        "accuracy": {
            "BMI": (10.0, 100.0),
            "MentHlth": (0, 30),
            "PhysHlth": (0, 30),
        },
        "unrealistic": {
            "BMI": (12, 70),
        },
        "completeness": [
            "Diabetes_binary",
            "HighBP",
            "HighChol",
            "CholCheck",
            "BMI",
            "Smoker",
            "Stroke",
            "HeartDiseaseorAttack",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
            "GenHlth",
            "MentHlth",
            "PhysHlth",
            "DiffWalk",
            "Sex",
            "Age",
            "Education",
            "Income",
        ],
        "categorical": {
            "Diabetes_binary": [0, 1],
            "HighBP": [0, 1],
            "HighChol": [0, 1],
            "CholCheck": [0, 1],
            "Smoker": [0, 1],
            "Stroke": [0, 1],
            "HeartDiseaseorAttack": [0, 1],
            "PhysActivity": [0, 1],
            "Fruits": [0, 1],
            "Veggies": [0, 1],
            "HvyAlcoholConsump": [0, 1],
            "AnyHealthcare": [0, 1],
            "NoDocbcCost": [0, 1],
            "DiffWalk": [0, 1],
            "Sex": [0, 1],
            "GenHlth": [1, 2, 3, 4, 5],
            "Age": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "Education": [1, 2, 3, 4, 5, 6],
            "Income": [1, 2, 3, 4, 5, 6, 7, 8],
        },
    },
    "diabetes_brfss_merged": None,
}

dataset_rules["diabetes_brfss_merged"] = dataset_rules["diabetes_brfss2015"].copy()


# ──────────────────────────────
# Validation
# ──────────────────────────────
def run_validation(df: pd.DataFrame, dataset_name: str, context):

    data_source = context.data_sources.add_pandas(name=f"{dataset_name}_source")
    data_asset = data_source.add_dataframe_asset(name=f"{dataset_name}_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("my_batch")

    suite = context.suites.add(gx.ExpectationSuite(name=f"{dataset_name}_suite"))
    rules = dataset_rules.get(dataset_name, {})

    for col, (min_val, max_val) in rules.get("accuracy", {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=col, min_value=min_val, max_value=max_val
            )
        )

    for col, (min_val, max_val) in rules.get("unrealistic", {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=col, min_value=min_val, max_value=max_val
            )
        )

    for col in rules.get("completeness", []):
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    for col, allowed in rules.get("categorical", {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=col, value_set=allowed)
        )

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name=f"{dataset_name}_validation", data=batch_def, suite=suite
        )
    )

    results = validation_def.run(batch_parameters={"dataframe": df})

    if results.success:
        logger.success(f"Validation passed for {dataset_name}.")
    else:
        logger.warning(f"Validation failed for {dataset_name}, check data docs.")

    return results


def run_all_validations(datasets: dict, open_docs: bool = False):
    context = gx.get_context(mode="ephemeral")
    results = {}
    for dataset_name, df in datasets.items():
        results[dataset_name] = run_validation(df, dataset_name, context)
    context.build_data_docs()
    if open_docs:
        context.open_data_docs()
    return results


# ──────────────────────────────
# Merge checks
# ──────────────────────────────
def validate_merge_counts(df1: pd.DataFrame, df2: pd.DataFrame, merged: pd.DataFrame):
    assert len(merged) == len(df1) + len(df2), "Row count mismatch after merge"
    logger.success(f"Merge validated: {len(df1)} + {len(df2)} = {len(merged)} rows")


def validate_merge_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    assert set(df1.columns) == set(df2.columns), (
        f"Column mismatch: {set(df1.columns) - set(df2.columns)} vs {set(df2.columns) - set(df1.columns)}"
    )
    logger.success("Merge validated: Columns match between datasets")
