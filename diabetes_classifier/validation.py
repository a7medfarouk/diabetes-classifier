import pandas as pd
import great_expectations as gx


dataset_rules = {
    "diabetes_prediction": {
        "accuracy": {
            "age": (0, 120),
        },
        "completeness": [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"
        ],
        "categorical": {
            "gender": ["Female", "Male"],
            "smoking_history": ["never", "No Info", "current", "former", "not current"],
            "hypertension": [0, 1],
            "heart_disease": [0, 1],
            "diabetes": [0, 1],
        },
        "distribution": {
        }
    },
    "diabetes_brfss2015": {
        "accuracy": {
            "Age": (0, 120)
        },
        "completeness": [
            "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth",
            "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
        ],
        "categorical": {
            "Sex": [0, 1],
            "Smoker": [0, 1],
            "Diabetes_binary": [0, 1],
            "HighBP": [0, 1],
            "HighChol": [0, 1],
        },
        "distribution": {
        }
    },
    "diabetes_brfss2021": {
        "accuracy": {
            "Age": (0, 120)
        },
        "completeness": [
            "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth",
            "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
        ],
        "categorical": {
            "Sex": [0, 1],
            "Smoker": [0, 1],
            "Diabetes_binary": [0, 1],
            "HighBP": [0, 1],
        },
        "distribution": {
        }
    }
}


# ──────────────────────────────
# Validation function
# ──────────────────────────────

def run_validation(df: pd.DataFrame, dataset_name: str):
    context = gx.get_context(mode="ephemeral")
    
    data_source = context.data_sources.add_pandas(name=f"{dataset_name}_source")
    data_asset = data_source.add_dataframe_asset(name=f"{dataset_name}_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("my_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    
    # Create Expectation Suite
    suite = context.suites.add(gx.ExpectationSuite(name=f"{dataset_name}_suite"))
    rules = dataset_rules.get(dataset_name, {})

    # ACCURACY (value ranges)
    for col, (min_val, max_val) in rules.get("accuracy", {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=col, min_value=min_val, max_value=max_val
            )
        )

    # COMPLETENESS 
    for col in rules.get("completeness", []):
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # CATEGORICAL
    for col, allowed in rules.get("categorical", {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=col, value_set=allowed)
        )

    # DISTRIBUTION
    for col, (min_val, max_val) in rules.get("distribution", {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnMeanToBeBetween(column=col, min_value=min_val, max_value=max_val)
        )

    # Link batch to suite
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(name=f"{dataset_name}_validation", data=batch_def, suite=suite)
    )


    results = validation_def.run(batch_parameters={"dataframe": df})

    
    _print_report(results)
    
    context.build_data_docs()
    context.open_data_docs()
    return results


def _print_report(results):
    """Print a summary of the validation results."""
    success = results.success

    print("=" * 58)
    print("DATA VALIDATION REPORT  (Great Expectations v1.x)")
    print("=" * 58)
    print(f"Overall Result : {'PASSED' if success else 'FAILED'}")
    print("=" * 58)

    for exp_result in results.results:
        exp_type = exp_result.expectation_config.type
        col = exp_result.expectation_config.kwargs.get("column", "table-level")
        passed = exp_result.success
        status = "PASS" if passed else "FAIL"

        print(f"\n[{status}] {exp_type}")
        print(f"   Column : {col}")

        if not passed and exp_result.result:
            r = exp_result.result
            if r.get("unexpected_count"):
                print(f"   Issues : {r['unexpected_count']} unexpected values")
            if r.get("partial_unexpected_list"):
                print(f"   Sample : {r['partial_unexpected_list'][:3]}")

    print("\n" + "=" * 58)


# ──────────────────────────────
# Load DATA
# ──────────────────────────────
df_prediction = pd.read_csv('data/raw/diabetes_prediction_dataset.csv')
df_brfss2015 = pd.read_csv('data/raw/diabetes_binary_health_indicators_BRFSS2015.csv')
df_brfss2021 = pd.read_csv('data/raw/diabetes_binary_health_indicators_BRFSS2021.csv')

# ──────────────────────────────
# RUN VALIDATION
# ──────────────────────────────
run_validation(df_prediction, "diabetes_prediction")
run_validation(df_brfss2015, "diabetes_brfss2015")
run_validation(df_brfss2021, "diabetes_brfss2021")