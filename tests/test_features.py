import pandas as pd

from diabetes_classifier.features import (
    clean_brfss,
    clean_prediction,
    discretize_brfss,
    feature_interactions_prediction,
)


# ──────────────────────────────
# Test: Clean Prediction Dataset
# ──────────────────────────────
def test_clean_prediction():
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Other", "Male", "Male"],
            "smoking_history": ["never", "not current", "former", "ever", "never"],
            "bmi": [25.0, 10.0, 30.0, 40.0, 25.0],
            "HbA1c_level": [5.0, 6.0, 5.5, 20.0, 5.0],
        }
    )
    # Row 0 and Row 4 are duplicates.
    # Row 1 has "not current" (should become "former") and BMI of 10 (should clip to 12).
    # Row 2 has "Other" gender (should be dropped).
    # Row 3 has "ever" smoking (should be dropped) and HbA1c of 20 (should clip to 15).

    # 2. ACT
    cleaned_df = clean_prediction(df)

    # 3. ASSERT
    # Out of 5 rows, 1 duplicate is dropped, Row 2 is dropped, and Row 3 is dropped. We should have 2 rows left.
    assert len(cleaned_df) == 2

    # Check that "not current" was successfully renamed to "former"
    assert "not current" not in cleaned_df["smoking_history"].values
    assert "former" in cleaned_df["smoking_history"].values

    # Check that clipping worked perfectly (BMI minimum is 12, HbA1c max is 15)
    assert cleaned_df["bmi"].min() >= 12.0
    assert cleaned_df["HbA1c_level"].max() <= 15.0


# ──────────────────────────────
# Test: Clean BRFSS Dataset
# ──────────────────────────────
def test_clean_brfss():
    # 1. ARRANGE: Invalid income (9), out-of-bounds BMI (80)
    df = pd.DataFrame({"Income": [5, 9, 5], "BMI": [25.0, 25.0, 80.0]})
    # Row 0 and Row 2 will stay (Row 2 BMI gets clipped). Row 1 drops because Income=9.

    # 2. ACT
    cleaned_df = clean_brfss(df)

    # 3. ASSERT
    assert len(cleaned_df) == 2
    assert 9 not in cleaned_df["Income"].values
    assert cleaned_df["BMI"].max() <= 70.0


# ──────────────────────────────
# Test: Feature Interactions
# ──────────────────────────────
def test_feature_interactions_prediction():
    # 1. ARRANGE: Tiny datasets for train, val, and test
    df_train = pd.DataFrame({"hypertension": [1, 0], "heart_disease": [1, 0]})
    df_val = pd.DataFrame({"hypertension": [0], "heart_disease": [1]})
    df_test = pd.DataFrame({"hypertension": [1], "heart_disease": [0]})

    # 2. ACT
    train_res, val_res, test_res = feature_interactions_prediction(
        df_train, df_val, df_test
    )

    # 3. ASSERT
    # Did it do the math correctly? (1 + 1 = 2) and (0 + 0 = 0)
    assert list(train_res["Cardio_Risk_Score"]) == [2, 0]
    assert list(val_res["Cardio_Risk_Score"]) == [1]

    # Did it successfully drop the old columns?
    assert "hypertension" not in train_res.columns
    assert "heart_disease" not in train_res.columns


# ──────────────────────────────
# Test: Discretize (Binning)
# ──────────────────────────────
def test_discretize_brfss():
    # 1. ARRANGE: Test exactly how the boundaries are binned
    df = pd.DataFrame(
        {
            "MentHlth": [0, 10, 30],  # Should become: Zero, Moderate, Severe
            "PhysHlth": [0, 0, 0],  # Just a dummy column to pass the function
        }
    )

    # 2. ACT (We just pass the same dataframe 3 times for simplicity)
    train_res, _, _ = discretize_brfss(df, df.copy(), df.copy())

    # 3. ASSERT
    # Check if the labels match the expected bins
    assert list(train_res["MentHlth_binned"]) == ["Zero", "Moderate", "Severe"]

    # Check if the original column was dropped
    assert "MentHlth" not in train_res.columns
