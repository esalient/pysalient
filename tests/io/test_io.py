import os  # For path manipulation

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import pytest

# Import the function to test
from pysalient.io import export_evaluation_results, load_evaluation_data

##########################
# Fixtures for Test Data #
##########################


@pytest.fixture(scope="module")  # Scope to module as data is read-only
def sample_data_dict():
    """Provides a dictionary with base data for tests."""
    return {
        "encounter_id": [1, 1, 1, 2, 2, 3],
        "event_timestamp": pd.to_datetime(
            [
                "2023-01-01 10:00:00",
                "2023-01-01 11:00:00",
                "2023-01-01 12:00:00",
                "2023-01-02 08:00:00",
                "2023-01-02 09:00:00",
                "2023-01-03 10:00:00",  # Add a third encounter
            ]
        ),
        "prediction_probability": [0.1, 0.2, 0.3, 0.8, 0.9, 0.5],
        "true_label": [0, 0, 1, 1, 1, 0],
        "model_identifier": [
            "Model_A",
            "Model_A",
            "Model_A",
            "Model_B",
            "Model_B",
            "Model_A",
        ],
        "task_identifier": ["sepsis", "sepsis", "sepsis", "aki", "aki", "sepsis"],
    }


@pytest.fixture(scope="module")
def sample_dataframe(sample_data_dict):
    """Provides a Pandas DataFrame based on sample_data_dict."""
    return pd.DataFrame(sample_data_dict)


@pytest.fixture(scope="module")
def sample_dataframe_with_nan():
    """Provides a Pandas DataFrame with NaN values in probability column for testing imputation."""
    import numpy as np
    data = {
        "encounter_id": [1, 1, 1, 2, 2, 3],
        "event_timestamp": pd.to_datetime([
            "2023-01-01 10:00:00",
            "2023-01-01 11:00:00", 
            "2023-01-01 12:00:00",
            "2023-01-02 08:00:00",
            "2023-01-02 09:00:00",
            "2023-01-03 10:00:00",
        ]),
        "prediction_probability": [0.1, np.nan, 0.3, 0.8, np.nan, 0.5],  # NaN values
        "true_label": [0, 0, 1, 1, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def valid_csv_path(tmp_path, sample_dataframe):
    """Creates a valid CSV file in a temporary directory."""
    path = tmp_path / "valid_data.csv"
    sample_dataframe.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def valid_parquet_path(tmp_path, sample_dataframe):
    """Creates a valid Parquet file in a temporary directory."""
    path = tmp_path / "valid_data.parquet"
    table = pa.Table.from_pandas(sample_dataframe, preserve_index=False)
    pq.write_table(table, path)
    return str(path)


@pytest.fixture
def csv_missing_proba_path(tmp_path, sample_dataframe):
    """Creates a CSV file missing the probability column."""
    path = tmp_path / "missing_proba.csv"
    df_missing = sample_dataframe.drop(columns=["prediction_probability"])
    df_missing.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_missing_agg_path(tmp_path, sample_dataframe):
    """Creates a CSV file missing the aggregation column."""
    path = tmp_path / "missing_agg.csv"
    df_missing = sample_dataframe.drop(columns=["encounter_id"])
    df_missing.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_different_delimiter_path(tmp_path, sample_dataframe):
    """Creates a CSV file with a semicolon delimiter."""
    path = tmp_path / "delimiter_data.csv"
    sample_dataframe.to_csv(path, index=False, sep=";")
    return str(path)


@pytest.fixture(scope="module")
def dataframe_with_float_timeseries(sample_data_dict):
    """Provides a DataFrame with a float timeseries column."""
    data = sample_data_dict.copy()
    # Replace timestamp with relative float time
    data["relative_time"] = [0.0, 1.0, 2.0, 22.0, 23.0, 48.0]
    del data["event_timestamp"]
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def dataframe_with_invalid_timeseries(sample_data_dict):
    """Provides a DataFrame with an invalid (string) timeseries column."""
    data = sample_data_dict.copy()
    # Replace timestamp with non-numeric/non-temporal string
    data["invalid_time"] = ["now", "later", "soon", "yesterday", "today", "tomorrow"]
    del data["event_timestamp"]
    return pd.DataFrame(data)


###############################
# Fixtures for Realistic Data #
###############################

# Define the base directory for tests and the data subdirectory
TESTS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TESTS_DIR, "data")


@pytest.fixture(scope="module")
def realistic_parquet_path():
    """Provides the path to a small, anonymized, realistic parquet file."""
    path = os.path.join(
        DATA_DIR, "anonymised_sample.parquet"
    )  # Use Australian English spelling
    if not os.path.exists(path):
        # Skip test if the anonymized data file hasn't been created yet
        pytest.skip(f"Realistic anonymized data file not found at: {path}")
    return path


@pytest.fixture(scope="module")
def realistic_table(realistic_parquet_path):
    """Provides a pre-loaded Arrow Table from the realistic data file."""
    # This fixture assumes the file exists due to the skip in realistic_parquet_path
    return pq.read_table(realistic_parquet_path)


########################
# Basic Test Arguments #
########################


@pytest.fixture
def base_load_args():
    """Provides base arguments for load_evaluation_data."""
    return {
        "y_proba_col": "prediction_probability",
        "y_label_col": "true_label",
        "aggregation_cols": "encounter_id",
        "timeseries_col": "event_timestamp",  # Renamed from timestamp_col
    }


################################
# Test Functions - Initial Set #
################################


def test_load_from_dataframe_success(sample_dataframe, base_load_args):
    """Tests successful loading from a Pandas DataFrame."""
    table = load_evaluation_data(source=sample_dataframe, **base_load_args)
    assert isinstance(table, pa.Table)
    assert table.num_rows == len(sample_dataframe)
    expected_cols = set(base_load_args.values())
    assert expected_cols.issubset(table.column_names)
    # Check specific data type (optional but good)
    assert pa.types.is_floating(table.schema.field(base_load_args["y_proba_col"]).type)
    assert pa.types.is_temporal(
        table.schema.field(base_load_args["timeseries_col"]).type
    )  # Updated col name


def test_load_from_csv_success(valid_csv_path, base_load_args, sample_dataframe):
    """Tests successful loading from a CSV file."""
    table = load_evaluation_data(source=valid_csv_path, **base_load_args)
    assert isinstance(table, pa.Table)
    assert table.num_rows == len(sample_dataframe)
    expected_cols = set(base_load_args.values())
    assert expected_cols.issubset(table.column_names)


def test_load_from_parquet_success(
    valid_parquet_path, base_load_args, sample_dataframe
):
    """Tests successful loading from a Parquet file."""
    table = load_evaluation_data(source=valid_parquet_path, **base_load_args)
    assert isinstance(table, pa.Table)
    assert table.num_rows == len(sample_dataframe)
    expected_cols = set(base_load_args.values())
    assert expected_cols.issubset(table.column_names)
    # Parquet preserves types better, so check is more reliable
    assert pa.types.is_float64(
        table.schema.field(base_load_args["y_proba_col"]).type
    )  # Example check
    assert pa.types.is_timestamp(
        table.schema.field(base_load_args["timeseries_col"]).type
    )  # Updated col name


def test_load_file_not_found(base_load_args):
    """Tests FileNotFoundError when the source file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_evaluation_data(source="non_existent_file.csv", **base_load_args)


def test_load_missing_required_column_csv(csv_missing_proba_path, base_load_args):
    """Tests ValueError when a required column is missing in the CSV."""
    with pytest.raises(
        ValueError, match="Missing required columns.*prediction_probability"
    ):
        load_evaluation_data(source=csv_missing_proba_path, **base_load_args)


def test_load_missing_aggregation_column_csv(csv_missing_agg_path, base_load_args):
    """Tests ValueError when the aggregation column is missing in the CSV."""
    with pytest.raises(ValueError, match="Missing required columns.*encounter_id"):
        load_evaluation_data(source=csv_missing_agg_path, **base_load_args)


def test_load_unknown_extension(tmp_path, base_load_args):
    """Tests TypeError when file extension is unknown and source_type not specified."""
    unknown_file = tmp_path / "data.unknown"
    unknown_file.touch()  # Create empty file
    with pytest.raises(TypeError, match="Cannot infer source type"):
        load_evaluation_data(source=str(unknown_file), **base_load_args)


def test_load_unsupported_source_type_arg(valid_csv_path, base_load_args):
    """Tests TypeError when an unsupported source_type argument is provided for a file."""
    with pytest.raises(
        TypeError, match="Unsupported source_type for file path: 'json'"
    ):
        load_evaluation_data(
            source=valid_csv_path, source_type="json", **base_load_args
        )


def test_load_unsupported_source_object(base_load_args):
    """Tests TypeError when the source object is not a str or DataFrame."""
    with pytest.raises(TypeError, match="Unsupported source type: <class 'int'>"):
        load_evaluation_data(source=123, **base_load_args)


def test_load_csv_with_options(
    csv_different_delimiter_path, base_load_args, sample_dataframe
):
    """Tests loading CSV with specific read options (delimiter)."""
    table = load_evaluation_data(
        source=csv_different_delimiter_path,
        read_options={"csv": {"parse_options": pv.ParseOptions(delimiter=";")}},
        **base_load_args,
    )
    assert isinstance(table, pa.Table)
    assert table.num_rows == len(sample_dataframe)
    expected_cols = set(base_load_args.values())
    assert expected_cols.issubset(table.column_names)


def test_load_with_optional_cols(valid_parquet_path, base_load_args, sample_dataframe):
    """Tests loading when optional model and task columns are specified."""
    args = {
        **base_load_args,
        "model_col": "model_identifier",
        "task_col": "task_identifier",
    }
    table = load_evaluation_data(source=valid_parquet_path, **args)
    assert isinstance(table, pa.Table)
    assert table.num_rows == len(sample_dataframe)
    expected_cols = set(args.values())
    assert expected_cols.issubset(table.column_names)
    assert "model_identifier" in table.column_names
    assert "task_identifier" in table.column_names


def test_load_missing_optional_col(valid_parquet_path, base_load_args):
    """Tests ValueError when a specified optional column is missing."""
    args = {
        **base_load_args,
        "model_col": "non_existent_model_col",  # This column doesn't exist
    }
    with pytest.raises(
        ValueError, match="Missing required columns.*non_existent_model_col"
    ):
        load_evaluation_data(source=valid_parquet_path, **args)


####################################
# Tests for Assign Task/Model Name #
####################################


def test_assign_task_name_success(sample_dataframe, base_load_args):
    """Tests successfully assigning a task name when task_col is not provided."""
    df_no_task = sample_dataframe.drop(columns=["task_identifier"], errors="ignore")
    args = {**base_load_args, "assign_task_name": "AssignedTask"}
    table = load_evaluation_data(source=df_no_task, **args)
    assert "task" in table.column_names
    assert table.column("task").to_pylist() == ["AssignedTask"] * len(df_no_task)


def test_assign_model_name_success(sample_dataframe, base_load_args):
    """Tests successfully assigning a model name when model_col is not provided."""
    df_no_model = sample_dataframe.drop(columns=["model_identifier"], errors="ignore")
    args = {**base_load_args, "assign_model_name": "AssignedModel"}
    table = load_evaluation_data(source=df_no_model, **args)
    assert "model" in table.column_names
    assert table.column("model").to_pylist() == ["AssignedModel"] * len(df_no_model)


def test_assign_both_names_success(sample_dataframe, base_load_args):
    """Tests successfully assigning both task and model names."""
    df_no_task_model = sample_dataframe.drop(
        columns=["task_identifier", "model_identifier"], errors="ignore"
    )
    args = {
        **base_load_args,
        "assign_task_name": "TaskX",
        "assign_model_name": "ModelY",
    }
    table = load_evaluation_data(source=df_no_task_model, **args)
    assert "task" in table.column_names
    assert "model" in table.column_names
    assert table.column("task").to_pylist() == ["TaskX"] * len(df_no_task_model)
    assert table.column("model").to_pylist() == ["ModelY"] * len(df_no_task_model)


def test_assign_task_name_conflict_error(sample_dataframe, base_load_args):
    """Tests ValueError when both task_col and assign_task_name are provided."""
    args = {
        **base_load_args,
        "task_col": "task_identifier",  # Exists in sample_dataframe
        "assign_task_name": "ConflictingTask",
    }
    with pytest.raises(
        ValueError, match="Cannot provide both `task_col` .* and `assign_task_name`"
    ):
        load_evaluation_data(source=sample_dataframe, **args)


def test_assign_model_name_conflict_error(sample_dataframe, base_load_args):
    """Tests ValueError when both model_col and assign_model_name are provided."""
    args = {
        **base_load_args,
        "model_col": "model_identifier",  # Exists in sample_dataframe
        "assign_model_name": "ConflictingModel",
    }
    with pytest.raises(
        ValueError, match="Cannot provide both `model_col` .* and `assign_model_name`"
    ):
        load_evaluation_data(source=sample_dataframe, **args)


def test_assign_task_name_column_exists_error(sample_dataframe, base_load_args):
    """Tests ValueError when assigning task name but a 'task' column already exists."""
    df_with_task_col = sample_dataframe.rename(columns={"task_identifier": "task"})
    args = {**base_load_args, "assign_task_name": "NewTask"}
    # Remove original task_col arg if present in base_load_args, just in case
    args.pop("task_col", None)
    with pytest.raises(
        ValueError, match="Cannot assign task name: column 'task' already exists"
    ):
        load_evaluation_data(source=df_with_task_col, **args)


def test_assign_model_name_column_exists_error(sample_dataframe, base_load_args):
    """Tests ValueError when assigning model name but a 'model' column already exists."""
    df_with_model_col = sample_dataframe.rename(columns={"model_identifier": "model"})
    args = {**base_load_args, "assign_model_name": "NewModel"}
    # Remove original model_col arg if present in base_load_args, just in case
    args.pop("model_col", None)
    with pytest.raises(
        ValueError, match="Cannot assign model name: column 'model' already exists"
    ):
        load_evaluation_data(source=df_with_model_col, **args)


#############################
# Tests with Realistic Data #
#############################


def test_load_from_realistic_parquet(realistic_parquet_path, base_load_args):
    """
    Tests loading from the anonymized, realistic Parquet file.
    Note: This test will be skipped if tests/data/anonymized_sample.parquet does not exist.
    """
    # IMPORTANT: Adjust base_load_args if column names differ in the realistic file!
    # Example:
    # realistic_args = base_load_args.copy()
    # realistic_args['y_proba_col'] = 'actual_probability_column_name_in_file'
    # realistic_args['y_label_col'] = 'actual_label_column_name_in_file'
    # realistic_args['aggregation_cols'] = 'actual_encounter_id_column_name_in_file'
    # realistic_args['timeseries_col'] = 'actual_timeseries_column_name_in_file' # Updated name
    # realistic_args['model_col'] = 'actual_model_id_column_name_in_file' # If using optional cols
    # realistic_args['task_col'] = 'actual_task_id_column_name_in_file'   # If using optional cols

    # Using base_load_args directly assumes column names match the synthetic data
    realistic_args = base_load_args

    table = load_evaluation_data(source=realistic_parquet_path, **realistic_args)
    assert isinstance(table, pa.Table)
    assert table.num_rows > 0  # Check it's not empty
    expected_cols = set(realistic_args.values())
    assert expected_cols.issubset(table.column_names)
    # Add more specific assertions based on expected structure/types in the real data
    # e.g., check timestamp resolution, presence of expected nulls etc.
    # Check that timeseries column is either temporal or floating, matching load_evaluation_data logic
    ts_type = table.schema.field(realistic_args["timeseries_col"]).type
    assert pa.types.is_temporal(ts_type) or pa.types.is_floating(ts_type)
    assert pa.types.is_floating(table.schema.field(realistic_args["y_proba_col"]).type)


###############################################
# Tests for Timeseries Column Type Validation #
###############################################


def test_load_from_dataframe_float_timeseries_success(
    dataframe_with_float_timeseries, base_load_args
):
    """Tests successful loading when timeseries_col is float."""
    args = base_load_args.copy()
    args["timeseries_col"] = "relative_time"  # Use the float column name

    table = load_evaluation_data(source=dataframe_with_float_timeseries, **args)
    assert isinstance(table, pa.Table)
    assert table.num_rows == len(dataframe_with_float_timeseries)
    # Check that the timeseries column is indeed float
    assert pa.types.is_floating(table.schema.field(args["timeseries_col"]).type)


def test_load_from_dataframe_invalid_timeseries_type(
    dataframe_with_invalid_timeseries, base_load_args
):
    """Tests ValueError when timeseries_col has an invalid type (string)."""
    args = base_load_args.copy()
    args["timeseries_col"] = "invalid_time"  # Use the invalid column name

    with pytest.raises(
        ValueError, match="must be a temporal .* or floating-point type"
    ):
        load_evaluation_data(source=dataframe_with_invalid_timeseries, **args)


# TODO: Add more tests for:
# - Multiple aggregation columns (list)
# - Edge cases like empty files/dataframes
# - More specific data type validation checks if added to main function
# - Errors during file reading (e.g., malformed CSV/Parquet) - might need specific fixtures


# Add a fixture for a sample PyArrow table for export tests
@pytest.fixture(scope="module")
def sample_arrow_table(sample_dataframe):
    """Provides a PyArrow Table based on the sample DataFrame."""
    return pa.Table.from_pandas(sample_dataframe, preserve_index=False)


#######################################
# Tests for export_evaluation_results #
#######################################


def test_export_to_dataframe_success(sample_arrow_table, sample_dataframe):
    """Tests exporting to a Pandas DataFrame."""
    df_exported = export_evaluation_results(
        results_table=sample_arrow_table, format="dataframe"
    )
    assert isinstance(df_exported, pd.DataFrame)
    # Use pandas testing utility for robust comparison (handles NaNs, types)
    pd.testing.assert_frame_equal(
        df_exported, sample_dataframe, check_dtype=False, check_like=True
    )


def test_export_to_csv_success(sample_arrow_table, sample_dataframe, tmp_path):
    """Tests exporting to a CSV file successfully."""
    output_csv_path = tmp_path / "output.csv"
    result = export_evaluation_results(
        results_table=sample_arrow_table,
        output_path=str(output_csv_path),
        format="csv",
    )
    assert result is None  # Function returns None on successful file write
    assert output_csv_path.exists()

    # Read back and verify content
    df_read = pd.read_csv(output_csv_path)
    # CSV read might change types (e.g., datetime), so compare carefully
    # Convert timestamp back if needed for comparison
    df_read["event_timestamp"] = pd.to_datetime(df_read["event_timestamp"])
    pd.testing.assert_frame_equal(
        df_read, sample_dataframe, check_dtype=False, check_like=True
    )


def test_export_to_csv_with_kwargs(sample_arrow_table, tmp_path):
    """Tests exporting to CSV with kwargs (e.g., no header)."""
    output_csv_path = tmp_path / "output_no_header.csv"
    export_evaluation_results(
        results_table=sample_arrow_table,
        output_path=str(output_csv_path),
        format="csv",
        write_options={"include_header": False},  # Pass WriteOptions via dict
        # Note: Direct kwargs to write_csv like 'sep' might need different handling
        # if they aren't part of WriteOptions in the function's implementation.
        # Based on the provided function, it expects WriteOptions in a dict.
    )
    assert output_csv_path.exists()

    # Read back and check header is missing
    with open(output_csv_path) as f:
        first_line = f.readline().strip()
    # Check first line data doesn't match header names
    assert "encounter_id" not in first_line
    # A simple check: the first data point should be present
    assert str(sample_arrow_table.column("encounter_id")[0].as_py()) in first_line


def test_export_to_parquet_success(sample_arrow_table, sample_dataframe, tmp_path):
    """Tests exporting to a Parquet file successfully."""
    output_parquet_path = tmp_path / "output.parquet"
    result = export_evaluation_results(
        results_table=sample_arrow_table,
        output_path=str(output_parquet_path),
        format="parquet",
    )
    assert result is None
    assert output_parquet_path.exists()

    # Read back and verify content
    table_read = pq.read_table(output_parquet_path)
    # Arrow tables comparison is more direct
    assert table_read.equals(sample_arrow_table)
    # Or convert back to pandas for comparison if preferred
    # df_read = table_read.to_pandas()
    # pd.testing.assert_frame_equal(df_read, sample_dataframe, check_like=True)


def test_export_to_parquet_with_kwargs(sample_arrow_table, tmp_path):
    """Tests exporting to Parquet with kwargs (e.g., compression)."""
    output_parquet_path = tmp_path / "output_compressed.parquet"
    export_evaluation_results(
        results_table=sample_arrow_table,
        output_path=str(output_parquet_path),
        format="parquet",
        compression="snappy",  # Example kwarg for pq.write_table
    )
    assert output_parquet_path.exists()

    # Optional: Check metadata or try reading back to ensure it's valid
    meta = pq.read_metadata(output_parquet_path)
    # Check if compression info is available and matches (might vary by pyarrow version)
    # This is a basic check; more robust checks might involve inspecting file internals
    # Check compression by iterating through row groups and column chunks
    found_snappy = False
    for i in range(meta.num_row_groups):
        rg_meta = meta.row_group(i)
        for j in range(rg_meta.num_columns):
            col_meta = rg_meta.column(j)
            # Check the compression attribute of the column chunk metadata
            if col_meta.compression and "SNAPPY" in col_meta.compression.upper():
                found_snappy = True
                break  # Exit inner loop once found
        if found_snappy:
            break  # Exit outer loop once found
    assert found_snappy, "SNAPPY compression not found in Parquet metadata"


def test_export_invalid_input_type(sample_dataframe):
    """Tests TypeError when results_table is not a pyarrow.Table."""
    with pytest.raises(TypeError, match="Expected results_table to be a pyarrow.Table"):
        export_evaluation_results(results_table=sample_dataframe, format="dataframe")


def test_export_invalid_format(sample_arrow_table):
    """Tests ValueError for an invalid format string."""
    with pytest.raises(ValueError, match="Invalid format 'invalid_fmt'"):
        export_evaluation_results(
            results_table=sample_arrow_table, format="invalid_fmt"
        )


def test_export_missing_path_for_csv(sample_arrow_table):
    """Tests ValueError if output_path is None for format='csv'."""
    with pytest.raises(
        ValueError, match="output_path must be provided for format 'csv'"
    ):
        export_evaluation_results(
            results_table=sample_arrow_table, format="csv", output_path=None
        )


def test_export_missing_path_for_parquet(sample_arrow_table):
    """Tests ValueError if output_path is None for format='parquet'."""
    with pytest.raises(
        ValueError, match="output_path must be provided for format 'parquet'"
    ):
        export_evaluation_results(
            results_table=sample_arrow_table, format="parquet", output_path=None
        )


##################################
# Tests for NaN Imputation      #
##################################


def test_load_evaluation_data_with_nan_imputation_default(sample_dataframe_with_nan):
    """Test that NaN values in probability column are imputed with default value (0.0)."""
    import numpy as np
    
    # Verify original data has NaN values
    assert sample_dataframe_with_nan['prediction_probability'].isna().sum() == 2
    
    # Load data with default imputation
    result_table = load_evaluation_data(
        source=sample_dataframe_with_nan,
        y_proba_col="prediction_probability",
        y_label_col="true_label",
        aggregation_cols="encounter_id",
        timeseries_col="event_timestamp",
        # y_proba_col_impute defaults to 0.0
    )
    
    # Check that no NaN values remain
    proba_array = result_table["prediction_probability"].to_numpy()
    assert not np.isnan(proba_array).any(), "NaN values should be imputed"
    
    # Check that NaN values were replaced with 0.0
    expected_values = [0.1, 0.0, 0.3, 0.8, 0.0, 0.5]  # NaN -> 0.0
    np.testing.assert_array_equal(proba_array, expected_values)


def test_load_evaluation_data_with_nan_imputation_custom(sample_dataframe_with_nan):
    """Test that NaN values in probability column are imputed with custom value."""
    import numpy as np
    
    # Load data with custom imputation value
    result_table = load_evaluation_data(
        source=sample_dataframe_with_nan,
        y_proba_col="prediction_probability",
        y_label_col="true_label",
        aggregation_cols="encounter_id",
        timeseries_col="event_timestamp",
        y_proba_col_impute=0.25,  # Custom impute value
    )
    
    # Check that no NaN values remain
    proba_array = result_table["prediction_probability"].to_numpy()
    assert not np.isnan(proba_array).any(), "NaN values should be imputed"
    
    # Check that NaN values were replaced with 0.25
    expected_values = [0.1, 0.25, 0.3, 0.8, 0.25, 0.5]  # NaN -> 0.25
    np.testing.assert_array_equal(proba_array, expected_values)


def test_load_evaluation_data_no_nan_no_change(sample_dataframe):
    """Test that data without NaN values is unchanged by imputation logic."""
    import numpy as np
    
    # Original data has no NaN values
    assert not sample_dataframe['prediction_probability'].isna().any()
    
    # Load data with imputation enabled
    result_table = load_evaluation_data(
        source=sample_dataframe,
        y_proba_col="prediction_probability",
        y_label_col="true_label",
        aggregation_cols="encounter_id",
        timeseries_col="event_timestamp",
        y_proba_col_impute=0.99,  # Should not affect anything
    )
    
    # Check that values are unchanged
    proba_array = result_table["prediction_probability"].to_numpy()
    expected_values = [0.1, 0.2, 0.3, 0.8, 0.9, 0.5]  # Original values
    np.testing.assert_array_equal(proba_array, expected_values)
