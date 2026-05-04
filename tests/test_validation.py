from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Adjust this import based on your project structure
from diabetes_classifier.validation import (
    run_all_validations,
    validate_merge_columns,
    validate_merge_counts,
)


# ──────────────────────────────
# Test: Merge Counts
# ──────────────────────────────
def test_validate_merge_counts_success():
    # 1. ARRANGE: Perfect data
    df1 = pd.DataFrame({"A": [1, 2]})  # 2 rows
    df2 = pd.DataFrame({"A": [3, 4, 5]})  # 3 rows
    merged = pd.DataFrame({"A": [1, 2, 3, 4, 5]})  # 5 rows

    # 2 & 3. ACT & ASSERT: If this runs without crashing, the test passes!
    validate_merge_counts(df1, df2, merged)


def test_validate_merge_counts_failure():
    # 1. ARRANGE: Bad data (merged is missing a row)
    df1 = pd.DataFrame({"A": [1, 2]})
    df2 = pd.DataFrame({"A": [3, 4]})
    bad_merged = pd.DataFrame({"A": [1, 2, 3]})

    # 2 & 3. ACT & ASSERT: We EXPECT this to crash.
    # 'pytest.raises' tells Pytest: "If this code raises an AssertionError, the test PASSES."
    with pytest.raises(AssertionError, match="Row count mismatch after merge"):
        validate_merge_counts(df1, df2, bad_merged)


# ──────────────────────────────
# Test: Merge Columns
# ──────────────────────────────
def test_validate_merge_columns_success():
    # 1. ARRANGE: Columns match (even if order is different)
    df1 = pd.DataFrame({"Age": [25], "BMI": [22]})
    df2 = pd.DataFrame({"BMI": [30], "Age": [40]})

    # 2 & 3. ACT & ASSERT: Should not crash
    validate_merge_columns(df1, df2)


def test_validate_merge_columns_failure():
    # 1. ARRANGE: Columns do NOT match
    df1 = pd.DataFrame({"Age": [25], "BMI": [22]})
    df2 = pd.DataFrame({"Age": [40], "BloodPressure": [120]})  # Missing BMI, has BP

    # 2 & 3. ACT & ASSERT: We expect it to raise an error
    with pytest.raises(AssertionError):
        validate_merge_columns(df1, df2)

@patch("diabetes_classifier.validation.run_validation")
@patch("diabetes_classifier.validation.gx.get_context")
def test_run_all_validations(mock_get_context, mock_run_validation):
    # We create a fake 'context' object that great_expectations normally uses
    fake_context = MagicMock()
    mock_get_context.return_value = fake_context

    # A tiny fake dataset dictionary
    dummy_datasets = {
        "test_dataset": pd.DataFrame({"A": [1]}),
    }

    run_all_validations(dummy_datasets, open_docs=True)

    # Did we get the context in ephemeral mode?
    mock_get_context.assert_called_once_with(mode="ephemeral")

    # Did it try to run our dataset through the validation function?
    mock_run_validation.assert_called_once()

    # Did it build and open the docs like we asked?
    fake_context.build_data_docs.call_count == 1
    fake_context.open_data_docs.call_count == 1
