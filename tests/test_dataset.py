from unittest.mock import patch

import pandas as pd

from diabetes_classifier.dataset import load_datasets, merge_brfss_datasets, split_data


# ──────────────────────────────
# Test: Merge BRFSS Datasets
# ──────────────────────────────
def test_merge_brfss_datasets():
    #DUMMY VARS
    df1 = pd.DataFrame({"Age": [25, 30], "Diabetes_binary": [0, 1]})
    df2 = pd.DataFrame({"Age": [45, 50], "Diabetes_binary": [1, 0]})

    # We mock the validation function so we only test the merging logic in isolation
    with patch("diabetes_classifier.dataset.validate_merge_counts") as mock_validate:
        
        # 2. ACT: Call your merge function
        merged_df = merge_brfss_datasets(df1, df2)

        
        assert len(merged_df) == 4
        assert list(merged_df["Age"]) == [25, 30, 45, 50]
        

        mock_validate.assert_called_once()


# ──────────────────────────────
# Test: Split Data
# ──────────────────────────────
def test_split_data():
    #DUMMY VARS
    dummy_data = pd.DataFrame({
        "Feature_1": range(100),
        "target_col": [0]*50 + [1]*50  # 50 negative cases, 50 positive cases
    })

    # 2. ACT: Call your split function
    # Using specific sizes: 15% test, and 17.6% of the remainder for validation
    train, val, test = split_data(
        df=dummy_data, 
        target_col="target_col", 
        test_size=0.15, 
        validation_size=0.176,
        random_state=42
    )

    # 3. ASSERT: Verify the math
    # Total should equal 100
    assert len(train) + len(val) + len(test) == 100
    
    # 100 * 0.15 = 15 test rows
    assert len(test) == 15
    # The remaining is 85. 85 * 0.176 is ~15 val rows, leaving 70 for train.
    assert len(val) == 15
    assert len(train) == 70


# ──────────────────────────────
# Test: Load Datasets
# ──────────────────────────────
# We use @patch to intercept 'pd.read_csv' so it doesn't look for real files
@patch("diabetes_classifier.dataset.pd.read_csv")
def test_load_datasets(mock_read_csv):
    # 1. ARRANGE: Tell the mocked read_csv to just return a dummy DataFrame
    dummy_df = pd.DataFrame({"dummy_column": [1, 2, 3]})
    mock_read_csv.return_value = dummy_df


    df_pred, df_b2015, df_b2021 = load_datasets()

    assert mock_read_csv.call_count == 3
    
    assert df_pred.equals(dummy_df)
    assert df_b2015.equals(dummy_df)
    assert df_b2021.equals(dummy_df)