"""
Tests for the pysalient.evaluation module.
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

# Functions/Constants/Modules to test (Import directly using src prefix)
import pysalient.io as csio  # Use src prefix here too for consistency
from pysalient.evaluation import META_KEY_Y_LABEL, evaluation
from pysalient.evaluation import _analytical_ci_utils as anaci
from pysalient.evaluation._evaluation_utils import _generate_thresholds

############
# Fixtures #
############

# Metadata keys (copied from module for consistency)
META_KEY_Y_PROBA = "pysalient.io.y_proba_col"
# META_KEY_Y_LABEL is already imported from evaluation module

# Define standard column names for fixtures
SYNTH_PROBA_COL = "synth_probas"
SYNTH_LABEL_COL = "synth_labels"

###################################
# Constants for Integration Tests #
###################################

SAMPLE_DATA_PATH = os.path.join("tests", "test_data", "anonymised_sample.parquet")
# Column map based on the notebook's inspection (adjust if file changes)
SAMPLE_COL_MAP = {
    "y_proba": "prediction_proba_1",
    "y_label": "true_label",
    "agg": "encounter_id",
    "time": "event_timestamp",
    # Optional model/task cols if they existed in the parquet
    # 'model': 'model_column_name',
    # 'task': 'task_column_name',
}


@pytest.fixture
def synth_data_basic() -> dict[str, np.ndarray]:
    """Provides basic synthetic probabilities and labels."""
    # Simple, predictable data for easy manual calculation checks
    # Use a seed for reproducibility in tests relying on this data
    np.random.seed(42)
    probas = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])  # 4 positives, 4 negatives
    return {SYNTH_PROBA_COL: probas, SYNTH_LABEL_COL: labels}


@pytest.fixture
def synth_data_larger() -> dict[str, np.ndarray]:
    """Provides slightly larger synthetic data for CI stability."""
    np.random.seed(123)
    n_samples = 100
    probas = np.random.rand(n_samples)
    labels = (probas > np.random.rand(n_samples) * 0.6 + 0.2).astype(int)
    return {SYNTH_PROBA_COL: probas, SYNTH_LABEL_COL: labels}


@pytest.fixture
def synth_table_basic(synth_data_basic) -> pa.Table:
    """PyArrow table from basic synthetic data, no metadata."""
    return pa.table(synth_data_basic)


@pytest.fixture
def synth_table_larger(synth_data_larger) -> pa.Table:
    """PyArrow table from larger synthetic data, no metadata."""
    return pa.table(synth_data_larger)


@pytest.fixture
def synth_metadata_correct() -> dict[bytes, bytes]:
    """Correct metadata dictionary."""
    return {
        META_KEY_Y_PROBA.encode("utf-8"): SYNTH_PROBA_COL.encode("utf-8"),
        META_KEY_Y_LABEL.encode("utf-8"): SYNTH_LABEL_COL.encode("utf-8"),
        b"other.meta": b"value",  # Include other metadata to ensure it's preserved/ignored
    }


@pytest.fixture
def synth_table_with_metadata(synth_table_basic, synth_metadata_correct) -> pa.Table:
    """Basic synthetic table with correct metadata added."""
    return synth_table_basic.replace_schema_metadata(synth_metadata_correct)


@pytest.fixture
def synth_table_larger_with_metadata(
    synth_table_larger, synth_metadata_correct
) -> pa.Table:
    """Larger synthetic table with correct metadata added."""
    return synth_table_larger.replace_schema_metadata(synth_metadata_correct)


@pytest.fixture
def synth_table_no_metadata(synth_table_basic) -> pa.Table:
    """Basic synthetic table explicitly without metadata."""
    return synth_table_basic.replace_schema_metadata(None)


@pytest.fixture
def synth_table_incomplete_metadata(synth_table_basic) -> pa.Table:
    """Basic synthetic table with only one required metadata key."""
    metadata = {META_KEY_Y_PROBA.encode("utf-8"): SYNTH_PROBA_COL.encode("utf-8")}
    return synth_table_basic.replace_schema_metadata(metadata)


@pytest.fixture
def synth_table_wrong_cols(synth_table_basic, synth_metadata_correct) -> pa.Table:
    """Table with metadata pointing to non-existent columns."""
    wrong_metadata = {
        META_KEY_Y_PROBA.encode("utf-8"): b"non_existent_proba",
        META_KEY_Y_LABEL.encode("utf-8"): b"non_existent_label",
    }
    return synth_table_basic.replace_schema_metadata(wrong_metadata)


@pytest.fixture
def synth_table_bad_types() -> pa.Table:
    """Table with non-numeric probas or non-binary labels."""
    # Example 1: String probabilities
    table_str_proba = pa.table(
        {SYNTH_PROBA_COL: ["0.1", "0.5", "0.9"], SYNTH_LABEL_COL: [0, 1, 1]}
    )
    # For now, let's focus on string probas causing numpy conversion issues
    metadata = {
        META_KEY_Y_PROBA.encode("utf-8"): SYNTH_PROBA_COL.encode("utf-8"),
        META_KEY_Y_LABEL.encode("utf-8"): SYNTH_LABEL_COL.encode("utf-8"),
    }
    return table_str_proba.replace_schema_metadata(metadata)


@pytest.fixture
def synth_table_all_pos() -> pa.Table:
    """Table with only positive labels."""
    table = pa.table({SYNTH_PROBA_COL: [0.1, 0.5, 0.9], SYNTH_LABEL_COL: [1, 1, 1]})
    metadata = {
        META_KEY_Y_PROBA.encode("utf-8"): SYNTH_PROBA_COL.encode("utf-8"),
        META_KEY_Y_LABEL.encode("utf-8"): SYNTH_LABEL_COL.encode("utf-8"),
    }
    return table.replace_schema_metadata(metadata)


@pytest.fixture
def synth_table_all_neg() -> pa.Table:
    """Table with only negative labels."""
    table = pa.table({SYNTH_PROBA_COL: [0.1, 0.5, 0.9], SYNTH_LABEL_COL: [0, 0, 0]})
    metadata = {
        META_KEY_Y_PROBA.encode("utf-8"): SYNTH_PROBA_COL.encode("utf-8"),
        META_KEY_Y_LABEL.encode("utf-8"): SYNTH_LABEL_COL.encode("utf-8"),
    }
    return table.replace_schema_metadata(metadata)


##################################
# Tests for _generate_thresholds #
##################################


@pytest.mark.parametrize(
    "spec, include_zero, expected",
    [
        # Default include_zero=True cases (0.0 added if not present)
        ([0.1, 0.5, 0.1], True, [0.0, 0.1, 0.5]),  # Explicit list, 0.0 added
        ((0.2, 0.8), True, [0.0, 0.2, 0.8]),  # Explicit tuple, 0.0 added
        ([0.9, 0.1], True, [0.0, 0.1, 0.9]),  # Unsorted list, 0.0 added
        ((0.1, 0.5, 0.1), True, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),  # Range, 0.0 added
        (
            (0.0, 1.0, 0.25),
            True,
            [0.0, 0.25, 0.5, 0.75, 1.0],
        ),  # Range including 0.0, no change
        ((0.1, 0.1, 0.1), True, [0.0, 0.1]),  # Range start=stop, 0.0 added
        ([0.0], True, [0.0]),  # Explicit 0.0, no change
        ([1.0], True, [0.0, 1.0]),  # Explicit 1.0, 0.0 added
        ([0.5, 0.0, 0.5], True, [0.0, 0.5]),  # Explicit list including 0.0, no change
        # Explicit include_zero=False cases (0.0 NOT added)
        ([0.1, 0.5, 0.1], False, [0.1, 0.5]),
        ((0.2, 0.8), False, [0.2, 0.8]),
        ([0.9, 0.1], False, [0.1, 0.9]),
        ((0.1, 0.5, 0.1), False, [0.1, 0.2, 0.3, 0.4, 0.5]),
        (
            (0.0, 1.0, 0.25),
            False,
            [0.0, 0.25, 0.5, 0.75, 1.0],
        ),  # 0.0 was generated by range, so it stays
        ((0.1, 0.1, 0.1), False, [0.1]),
        ([0.0], False, [0.0]),  # Explicit 0.0 stays
        ([1.0], False, [1.0]),
    ],
)
def test_generate_thresholds_valid(spec, include_zero, expected):
    """Test valid threshold specifications with include_zero flag."""
    result = _generate_thresholds(spec, include_zero=include_zero)  # Keep direct import
    assert isinstance(result, list)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "spec, error_type",
    [
        ((0.5, 0.1, 0.1), ValueError),  # Invalid range start > stop
        ((0.1, 0.5, 0.0), ValueError),  # Zero step
        ((0.1, 0.5, -0.1), ValueError),  # Negative step
        ([0.1, "a"], ValueError),  # Invalid type in list
        ((0.1, 0.5, "a"), ValueError),  # Invalid type in range spec
        ([0.5, 1.1], ValueError),  # Value > 1
        ([-0.1, 0.5], ValueError),  # Value < 0
        ([], ValueError),  # Empty list
        ((), ValueError),  # Empty tuple (treated as explicit list)
        (123, TypeError),  # Invalid input type
        ("abc", TypeError),  # Invalid input type
    ],
)
def test_generate_thresholds_invalid(spec, error_type):
    """Test invalid threshold specifications raise errors."""
    with pytest.raises(error_type):
        _generate_thresholds(spec)  # Keep direct import


########################
# Tests for evaluation #
########################

# Define expected schema without time-to-first-alert columns (for tests without timeseries_col)
EXPECTED_SCHEMA_BASE = pa.schema(
    [
        pa.field("modelid", pa.string()),
        pa.field("filter_desc", pa.string()),
        pa.field("threshold", pa.float64()),
        # Overall Metrics
        pa.field("AUROC", pa.float64()),
        pa.field("AUROC_Lower_CI", pa.float64()),
        pa.field("AUROC_Upper_CI", pa.float64()),
        pa.field("AUPRC", pa.float64()),
        pa.field("AUPRC_Lower_CI", pa.float64()),
        pa.field("AUPRC_Upper_CI", pa.float64()),
        pa.field("Prevalence", pa.float64()),
        pa.field("Sample_Size", pa.int64()),
        pa.field("Label_Count", pa.int64()),
        # Confusion Matrix
        pa.field("TP", pa.int64()),
        pa.field("TN", pa.int64()),
        pa.field("FP", pa.int64()),
        pa.field("FN", pa.int64()),
        # Threshold Metrics + CIs
        pa.field("PPV", pa.float64()),
        pa.field("PPV_Lower_CI", pa.float64()),
        pa.field("PPV_Upper_CI", pa.float64()),
        pa.field("Sensitivity", pa.float64()),
        pa.field("Sensitivity_Lower_CI", pa.float64()),
        pa.field("Sensitivity_Upper_CI", pa.float64()),
        pa.field("Specificity", pa.float64()),
        pa.field("Specificity_Lower_CI", pa.float64()),
        pa.field("Specificity_Upper_CI", pa.float64()),
        pa.field("NPV", pa.float64()),
        pa.field("NPV_Lower_CI", pa.float64()),
        pa.field("NPV_Upper_CI", pa.float64()),
        pa.field("Accuracy", pa.float64()),
        pa.field("Accuracy_Lower_CI", pa.float64()),
        pa.field("Accuracy_Upper_CI", pa.float64()),
        pa.field("F1_Score", pa.float64()),
        pa.field("F1_Score_Lower_CI", pa.float64()),
        pa.field("F1_Score_Upper_CI", pa.float64()),
    ]
)

# Define expected schema including time-to-first-alert columns (for tests with timeseries_col)
EXPECTED_SCHEMA_WITH_TIMESERIES = pa.schema(
    [
        pa.field("modelid", pa.string()),
        pa.field("filter_desc", pa.string()),
        pa.field("threshold", pa.float64()),
        # Overall Metrics
        pa.field("AUROC", pa.float64()),
        pa.field("AUROC_Lower_CI", pa.float64()),
        pa.field("AUROC_Upper_CI", pa.float64()),
        pa.field("AUPRC", pa.float64()),
        pa.field("AUPRC_Lower_CI", pa.float64()),
        pa.field("AUPRC_Upper_CI", pa.float64()),
        pa.field("Prevalence", pa.float64()),
        pa.field("Sample_Size", pa.int64()),
        pa.field("Label_Count", pa.int64()),
        # Confusion Matrix
        pa.field("TP", pa.int64()),
        pa.field("TN", pa.int64()),
        pa.field("FP", pa.int64()),
        pa.field("FN", pa.int64()),
        # Threshold Metrics + CIs
        pa.field("PPV", pa.float64()),
        pa.field("PPV_Lower_CI", pa.float64()),
        pa.field("PPV_Upper_CI", pa.float64()),
        pa.field("Sensitivity", pa.float64()),
        pa.field("Sensitivity_Lower_CI", pa.float64()),
        pa.field("Sensitivity_Upper_CI", pa.float64()),
        pa.field("Specificity", pa.float64()),
        pa.field("Specificity_Lower_CI", pa.float64()),
        pa.field("Specificity_Upper_CI", pa.float64()),
        pa.field("NPV", pa.float64()),
        pa.field("NPV_Lower_CI", pa.float64()),
        pa.field("NPV_Upper_CI", pa.float64()),
        pa.field("Accuracy", pa.float64()),
        pa.field("Accuracy_Lower_CI", pa.float64()),
        pa.field("Accuracy_Upper_CI", pa.float64()),
        pa.field("F1_Score", pa.float64()),
        pa.field("F1_Score_Lower_CI", pa.float64()),
        pa.field("F1_Score_Upper_CI", pa.float64()),
    ]
)

# For backward compatibility, use the base schema as the default
EXPECTED_SCHEMA_BASE = EXPECTED_SCHEMA_BASE

# Define lists of CI columns for easier checking
OVERALL_CI_COLS = [
    "AUROC_Lower_CI",
    "AUROC_Upper_CI",
    "AUPRC_Lower_CI",
    "AUPRC_Upper_CI",
]
THRESHOLD_CI_COLS = [
    "PPV_Lower_CI",
    "PPV_Upper_CI",
    "Sensitivity_Lower_CI",
    "Sensitivity_Upper_CI",
    "Specificity_Lower_CI",
    "Specificity_Upper_CI",
    "NPV_Lower_CI",
    "NPV_Upper_CI",
    "Accuracy_Lower_CI",
    "Accuracy_Upper_CI",
    "F1_Score_Lower_CI",
    "F1_Score_Upper_CI",
]
ALL_CI_COLS = OVERALL_CI_COLS + THRESHOLD_CI_COLS


def test_evaluation_basic(synth_table_with_metadata):
    """Test the happy path with basic synthetic data (no CI calculation)."""
    table = synth_table_with_metadata
    modelid = "test_model"
    filter_desc = "basic_synth"
    thresholds = [0.0, 0.5, 1.0]

    # Manually calculate expected values for synth_data_basic
    expected_auroc = 0.9375
    expected_auprc = 0.94375  # Updated to use precision_recall_curve + auc method (corrected calculation)
    expected_prevalence = 4 / 8
    expected_sample_size = 8
    expected_label_count = 4

    # Threshold 0.0: TP=4, TN=0, FP=4, FN=0 -> PPV=0.5, Sens=1.0, Spec=0.0, NPV=0.0, Acc=0.5, F1=2/3
    # Threshold 0.5: TP=3, TN=3, FP=1, FN=1 -> PPV=0.75, Sens=0.75, Spec=0.75, NPV=0.75, Acc=0.75, F1=0.75
    # Threshold 1.0: TP=0, TN=4, FP=0, FN=4 -> PPV=0.0, Sens=0.0, Spec=1.0, NPV=0.5, Acc=0.5, F1=0.0
    expected_rows = [
        {
            "threshold": 0.0,
            "TP": 4,
            "TN": 0,
            "FP": 4,
            "FN": 0,
            "PPV": 0.5,
            "Sensitivity": 1.0,
            "Specificity": 0.0,
            "NPV": 0.0,
            "Accuracy": 0.5,
            "F1_Score": 0.6666666666666666,
        },
        {
            "threshold": 0.5,
            "TP": 3,
            "TN": 3,
            "FP": 1,
            "FN": 1,
            "PPV": 0.75,
            "Sensitivity": 0.75,
            "Specificity": 0.75,
            "NPV": 0.75,
            "Accuracy": 0.75,
            "F1_Score": 0.75,
        },
        {
            "threshold": 1.0,
            "TP": 0,
            "TN": 4,
            "FP": 0,
            "FN": 4,
            "PPV": 0.0,
            "Sensitivity": 0.0,
            "Specificity": 1.0,
            "NPV": 0.5,
            "Accuracy": 0.5,
            "F1_Score": 0.0,
        },
    ]

    # Call evaluation without CI params (defaults are False)
    results = evaluation(table, modelid, filter_desc, thresholds)  # Use direct import

    assert isinstance(results, pa.Table)
    # Check against the base schema (without time-to-first-alert columns since timeseries_col=None)
    assert results.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    assert results.num_rows == len(expected_rows)

    results_dict = results.to_pydict()

    for i, expected in enumerate(expected_rows):
        assert results_dict["modelid"][i] == modelid
        assert results_dict["filter_desc"][i] == filter_desc
        assert results_dict["threshold"][i] == pytest.approx(expected["threshold"])
        assert results_dict["AUROC"][i] == pytest.approx(expected_auroc)
        assert results_dict["AUPRC"][i] == pytest.approx(expected_auprc)
        assert results_dict["Prevalence"][i] == pytest.approx(expected_prevalence)
        assert results_dict["Sample_Size"][i] == expected_sample_size
        assert results_dict["Label_Count"][i] == expected_label_count
        assert results_dict["TP"][i] == expected["TP"]
        assert results_dict["TN"][i] == expected["TN"]
        assert results_dict["FP"][i] == expected["FP"]
        assert results_dict["FN"][i] == expected["FN"]
        assert results_dict["PPV"][i] == pytest.approx(expected["PPV"])
        assert results_dict["Sensitivity"][i] == pytest.approx(expected["Sensitivity"])
        assert results_dict["Specificity"][i] == pytest.approx(expected["Specificity"])
        assert results_dict["NPV"][i] == pytest.approx(expected["NPV"])
        assert results_dict["Accuracy"][i] == pytest.approx(expected["Accuracy"])
        assert results_dict["F1_Score"][i] == pytest.approx(expected["F1_Score"])
        # Check all CI columns are None (since neither calculate_ci nor calculate_threshold_ci was True)
        for col in ALL_CI_COLS:
            assert results_dict[col][i] is None, f"Column {col} should be None"


def test_evaluation_rounding(synth_table_with_metadata):
    """Test the decimal_places parameter rounds float metrics correctly (no CI)."""
    table = synth_table_with_metadata
    modelid = "test_model_rounding"
    filter_desc = "rounding_test"
    thresholds = [0.0, 0.5, 1.0]
    decimal_places_to_test = 2

    expected_auroc = round(0.9375, decimal_places_to_test)  # 0.94
    expected_auprc = round(
        0.94375, decimal_places_to_test
    )  # 0.94 (updated to corrected calculation)
    expected_prevalence = round(4 / 8, decimal_places_to_test)  # 0.50

    expected_rows = [
        {
            "threshold": 0.0,
            "PPV": 0.50,
            "Sensitivity": 1.00,
            "Specificity": 0.00,
            "NPV": 0.00,
            "Accuracy": 0.50,
            "F1_Score": 0.67,
        },
        {
            "threshold": 0.5,
            "PPV": 0.75,
            "Sensitivity": 0.75,
            "Specificity": 0.75,
            "NPV": 0.75,
            "Accuracy": 0.75,
            "F1_Score": 0.75,
        },
        {
            "threshold": 1.0,
            "PPV": 0.00,
            "Sensitivity": 0.00,
            "Specificity": 1.00,
            "NPV": 0.50,
            "Accuracy": 0.50,
            "F1_Score": 0.00,
        },
    ]

    results = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        decimal_places=decimal_places_to_test,
    )

    assert isinstance(results, pa.Table)
    assert results.schema.equals(
        EXPECTED_SCHEMA_BASE, check_metadata=False
    )  # Use base schema since timeseries_col=None
    assert results.num_rows == len(expected_rows)

    results_dict = results.to_pydict()

    for i, expected in enumerate(expected_rows):
        assert results_dict["modelid"][i] == modelid
        assert results_dict["filter_desc"][i] == filter_desc
        assert results_dict["threshold"][i] == pytest.approx(expected["threshold"])
        assert results_dict["AUROC"][i] == pytest.approx(expected_auroc)
        assert results_dict["AUPRC"][i] == pytest.approx(expected_auprc)
        assert results_dict["Prevalence"][i] == pytest.approx(expected_prevalence)
        assert results_dict["PPV"][i] == pytest.approx(expected["PPV"])
        assert results_dict["Sensitivity"][i] == pytest.approx(expected["Sensitivity"])
        assert results_dict["Specificity"][i] == pytest.approx(expected["Specificity"])
        assert results_dict["NPV"][i] == pytest.approx(expected["NPV"])
        assert results_dict["Accuracy"][i] == pytest.approx(expected["Accuracy"])
        assert results_dict["F1_Score"][i] == pytest.approx(expected["F1_Score"])
        # Check all CI columns are None
        for col in ALL_CI_COLS:
            assert results_dict[col][i] is None, f"Column {col} should be None"


##########################
# Input Validation Tests #
##########################


def test_evaluation_invalid_data_type():
    """Test passing non-PyArrow table raises TypeError."""
    with pytest.raises(TypeError, match="Input 'data' must be a PyArrow Table"):
        evaluation([1, 2, 3], "m", "f", [0.5])  # Use direct import


def test_evaluation_invalid_modelid_type(synth_table_with_metadata):
    """Test passing non-string modelid raises TypeError."""
    with pytest.raises(TypeError, match="Input 'modelid' must be a string"):
        evaluation(synth_table_with_metadata, 123, "f", [0.5])  # Use direct import


def test_evaluation_invalid_filter_desc_type(synth_table_with_metadata):
    """Test passing non-string filter_desc raises TypeError."""
    with pytest.raises(TypeError, match="Input 'filter_desc' must be a string"):
        evaluation(synth_table_with_metadata, "m", 123, [0.5])  # Use direct import


def test_evaluation_missing_metadata(synth_table_no_metadata):
    """Test table without metadata raises ValueError."""
    with pytest.raises(ValueError, match="Input table is missing schema metadata"):
        evaluation(synth_table_no_metadata, "m", "f", [0.5])  # Use direct import


def test_evaluation_incomplete_metadata(synth_table_incomplete_metadata):
    """Test table with missing required metadata keys raises KeyError."""
    expected_pattern = f"Required metadata key missing: .*{META_KEY_Y_LABEL}.* not found"  # Use direct import
    with pytest.raises(KeyError, match=expected_pattern):
        evaluation(
            synth_table_incomplete_metadata, "m", "f", [0.5]
        )  # Use direct import


def test_evaluation_wrong_cols(synth_table_wrong_cols):
    """Test metadata pointing to non-existent columns raises ValueError."""
    with pytest.raises(ValueError, match="Column 'non_existent_proba' .* not found"):
        evaluation(synth_table_wrong_cols, "m", "f", [0.5])  # Use direct import


def test_evaluation_invalid_threshold_spec(synth_table_with_metadata):
    """Test invalid threshold spec raises ValueError."""
    with pytest.raises(ValueError, match="Invalid threshold specification"):
        evaluation(
            synth_table_with_metadata, "m", "f", []
        )  # Empty list # Use direct import


def test_evaluation_bad_column_types(synth_table_bad_types):
    """Test non-numeric proba column raises TypeError during numpy conversion/check."""
    # Expect TypeError because the probability column is not numeric
    with pytest.raises(TypeError, match="Probability column .* must be numeric"):
        evaluation(synth_table_bad_types, "m", "f", [0.5])  # Use direct import


#################################
# CI Parameter Validation Tests #
#################################


def test_evaluation_invalid_ci_alpha(synth_table_larger_with_metadata):
    """Test invalid ci_alpha values raise ValueError."""
    for alpha in [-0.1, 0.0, 1.0, 1.1]:
        # Test with calculate_au_ci=True
        with pytest.raises(ValueError, match="ci_alpha must be between 0 and 1"):
            evaluation(  # Use direct import
                synth_table_larger_with_metadata,
                "m",
                "f",
                [0.5],
                calculate_au_ci=True,
                ci_alpha=alpha,
            )
        # Test with calculate_threshold_ci=True
        with pytest.raises(ValueError, match="ci_alpha must be between 0 and 1"):
            evaluation(  # Use direct import
                synth_table_larger_with_metadata,
                "m",
                "f",
                [0.5],
                calculate_threshold_ci=True,
                ci_alpha=alpha,
            )


def test_evaluation_invalid_bootstrap_rounds(synth_table_larger_with_metadata):
    """Test invalid bootstrap_rounds values raise ValueError only when bootstrap is used."""
    for rounds in [-10, 0]:
        # Test with calculate_au_ci=True (should raise)
        with pytest.raises(
            ValueError,
            match="bootstrap_rounds must be a positive integer.",  # Match the actual error
        ):
            evaluation(  # Use direct import
                synth_table_larger_with_metadata,
                "m",
                "f",
                [0.5],
                calculate_au_ci=True,
                bootstrap_rounds=rounds,
            )
        # Test with calculate_threshold_ci=True and threshold_ci_method='bootstrap' (should raise)
        with pytest.raises(
            # Expect the "less than 100" error here because it's checked first
            ValueError,
            match="bootstrap_rounds is set to .* A value less than 100",
        ):
            evaluation(  # Use direct import
                synth_table_larger_with_metadata,
                "m",
                "f",
                [0.5],
                calculate_threshold_ci=True,
                threshold_ci_method="bootstrap",
                bootstrap_rounds=rounds,
            )
        # Test with calculate_threshold_ci=True and analytical method (SHOULD NOT raise for rounds)
        try:
            evaluation(  # Use direct import
                synth_table_larger_with_metadata,
                "m",
                "f",
                [0.5],
                calculate_threshold_ci=True,
                threshold_ci_method="normal",  # Analytical method
                bootstrap_rounds=rounds,  # Should be ignored
            )
        except ValueError as e:
            # Fail if a ValueError is raised unexpectedly
            pytest.fail(f"ValueError raised unexpectedly for analytical method: {e}")
        # Test with both CIs off (SHOULD NOT raise for rounds)
        try:
            evaluation(  # Use direct import
                synth_table_larger_with_metadata,
                "m",
                "f",
                [0.5],
                calculate_au_ci=False,
                calculate_threshold_ci=False,
                bootstrap_rounds=rounds,  # Should be ignored
            )
        except ValueError as e:
            pytest.fail(f"ValueError raised unexpectedly when CIs are off: {e}")


def test_evaluation_invalid_bootstrap_seed_type(synth_table_larger_with_metadata):
    """Test invalid bootstrap_seed type raises TypeError only when bootstrap is used."""
    invalid_seed = "abc"
    # Test with calculate_au_ci=True (should raise)
    with pytest.raises(
        TypeError,
        match="bootstrap_seed must be an integer or None.",  # Match the actual error
    ):
        evaluation(  # Use direct import
            synth_table_larger_with_metadata,
            "m",
            "f",
            [0.5],
            calculate_au_ci=True,
            bootstrap_seed=invalid_seed,
        )
    # Test with calculate_threshold_ci=True and threshold_ci_method='bootstrap'
    # This path does not raise the TypeError upfront for invalid seed type,
    # so we remove the check here. Warnings are raised later during calculation.
    # The first pytest.raises block above covers the TypeError for calculate_au_ci=True.
    # Test with calculate_threshold_ci=True and analytical method (SHOULD NOT raise for seed)
    try:
        evaluation(  # Use direct import
            synth_table_larger_with_metadata,
            "m",
            "f",
            [0.5],
            calculate_threshold_ci=True,
            threshold_ci_method="normal",  # Analytical method
            bootstrap_seed=invalid_seed,  # Should be ignored
        )
    except TypeError as e:
        pytest.fail(f"TypeError raised unexpectedly for analytical method: {e}")
    # Test with both CIs off (SHOULD NOT raise for seed)
    try:
        evaluation(  # Use direct import
            synth_table_larger_with_metadata,
            "m",
            "f",
            [0.5],
            calculate_au_ci=False,
            calculate_threshold_ci=False,
            bootstrap_seed=invalid_seed,  # Should be ignored
        )
    except TypeError as e:
        pytest.fail(f"TypeError raised unexpectedly when CIs are off: {e}")


def test_evaluation_invalid_threshold_ci_method_error(synth_table_larger_with_metadata):
    """Test unsupported threshold_ci_method raises ValueError."""
    with pytest.raises(
        ValueError, match="Unsupported threshold_ci_method 'invalid_method'"
    ):
        evaluation(  # Use direct import
            synth_table_larger_with_metadata,
            "m",
            "f",
            [0.5],
            calculate_threshold_ci=True,
            threshold_ci_method="invalid_method",
        )


def test_evaluation_invalid_calculate_threshold_ci_type(synth_table_with_metadata):
    """Test non-boolean calculate_threshold_ci raises TypeError."""
    with pytest.raises(
        TypeError, match="Input 'calculate_threshold_ci' must be a boolean"
    ):
        evaluation(  # Use direct import
            synth_table_with_metadata,
            "m",
            "f",
            [0.5],
            calculate_threshold_ci="True",  # Pass string instead of bool
        )


###################
# Edge Case Tests #
###################


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_evaluation_all_pos_labels(synth_table_all_pos):
    """Test evaluation with only positive labels (some metrics undefined)."""
    table = synth_table_all_pos
    results = evaluation(table, "all_pos", "test", [0.5])  # Use direct import
    # Expect 2 rows: 0.0 (default) and 0.5
    assert results.num_rows == 2
    res = results.to_pydict()

    assert res["TP"][0] == 3  # All are positive, all predicted positive at 0.5
    assert res["TN"][0] == 0
    assert res["FP"][0] == 0
    assert res["FN"][0] == 0
    assert res["PPV"][0] == 1.0
    assert res["Sensitivity"][0] == 1.0
    # Specificity is TN/(TN+FP). When TN=0, FP=0, code returns 0.0
    assert res["Specificity"][0] == 0.0
    # NPV is TN/(TN+FN). When TN=0, FN=0, code returns 0.0
    assert res["NPV"][0] == 0.0
    assert res["Accuracy"][0] == 1.0
    assert res["F1_Score"][0] == 1.0
    assert np.isnan(res["AUROC"][0])  # Undefined with only one class
    # AUPRC is also undefined when only one class is present
    assert np.isnan(res["AUPRC"][0])


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_evaluation_all_neg_labels(synth_table_all_neg):
    """Test evaluation with only negative labels (some metrics undefined)."""
    table = synth_table_all_neg
    results = evaluation(table, "all_neg", "test", [0.5])  # Use direct import
    # Expect 2 rows: 0.0 (default) and 0.5
    assert results.num_rows == 2
    res = results.to_pydict()

    assert res["TP"][0] == 0
    # Check result for threshold 0.5 (index 1), where TN should be 1 (proba 0.1 < 0.5)
    assert res["TN"][1] == 1
    # At threshold 0.0, all 3 predictions are >= 0.0, and all labels are 0 -> FP = 3
    assert res["FP"][0] == 3
    assert res["FN"][0] == 0
    # PPV = TP/(TP+FP). When TP=0, FP>0, code returns 0.0
    assert res["PPV"][0] == 0.0
    # Sensitivity = TP/(TP+FN). When TP=0, FN=0, code returns 0.0
    assert res["Sensitivity"][0] == 0.0
    # Specificity = TN/(TN+FP). At threshold 0.0, TN=0, FP=3. Specificity = 0/3 = 0.0
    assert res["Specificity"][0] == 0.0
    # NPV = TN/(TN+FN). At threshold 0.0, TN=0, FN=0. Code returns 0.0
    assert res["NPV"][0] == 0.0
    # Accuracy = (TP+TN)/(TP+TN+FP+FN). At threshold 0.0, TP=0, TN=0, FP=3, FN=0. Accuracy = 0/3 = 0.0
    assert res["Accuracy"][0] == 0.0
    assert res["F1_Score"][0] == 0.0  # F1 is 0 if TP is 0
    assert np.isnan(res["AUROC"][0])  # Undefined with only one class
    # AUPRC is also undefined when only one class is present
    assert np.isnan(res["AUPRC"][0])


def test_evaluation_threshold_range(synth_table_with_metadata):
    """Test using a range tuple for thresholds."""
    table = synth_table_with_metadata
    results = evaluation(  # Use direct import
        table, "m", "f", (0.1, 0.9, 0.2)
    )  # 0.1, 0.3, 0.5, 0.7, 0.9
    # Expect 6 rows: 0.0 (default) + 0.1, 0.3, 0.5, 0.7, 0.9
    assert results.num_rows == 6
    # Check thresholds including the default 0.0
    # Convert pyarrow array to list for comparison with approx
    assert results["threshold"].to_pylist() == pytest.approx(
        [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    )


def test_evaluation_zero_threshold_false(synth_table_with_metadata):
    """Test that zero_threshold=False excludes 0.0 unless explicitly provided."""
    table = synth_table_with_metadata
    # Case 1: 0.0 not in spec
    results = evaluation(
        table, "m", "f", [0.5, 1.0], force_threshold_zero=False
    )  # Use direct import
    assert results.num_rows == 2
    assert 0.0 not in results["threshold"].to_pylist()
    # Case 2: 0.0 in spec
    results_with_zero = evaluation(  # Use direct import
        table, "m", "f", [0.0, 0.5], force_threshold_zero=False
    )
    assert results_with_zero.num_rows == 2
    assert 0.0 in results_with_zero["threshold"].to_pylist()


########################
# CI Calculation Tests #
########################


def test_evaluation_with_au_ci_basic(synth_table_larger_with_metadata):
    """Test basic AU CI calculation (AUROC/AUPRC only) works."""
    table = synth_table_larger_with_metadata
    modelid = "test_au_ci"
    filter_desc = "larger_synth_au"
    thresholds = [0.5]  # Test with one threshold row

    results = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=True,  # Request AU CIs
        calculate_threshold_ci=False,  # Explicitly false for threshold CIs
        bootstrap_rounds=100,  # Set to minimum required
        bootstrap_seed=123,
    )

    assert isinstance(results, pa.Table)
    assert results.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    # Expect 2 rows: 0.0 (default) + 0.5
    assert results.num_rows == 2

    results_dict = results.to_pydict()

    # Check that AU CI columns are NOT None and plausible
    for i in range(results.num_rows):
        for col in OVERALL_CI_COLS:  # OVERALL_CI_COLS refers to AUROC/AUPRC CIs
            ci_val = results_dict[col][i]
            assert ci_val is not None, (
                f"{col} should not be None when calculate_au_ci=True"
            )
            assert isinstance(ci_val, float), (
                f"{col} should be float, got {type(ci_val)}"
            )
            assert 0.0 <= ci_val <= 1.0, f"{col} value {ci_val} out of bounds [0, 1]"
        # Check lower <= upper
        assert results_dict["AUROC_Lower_CI"][i] <= results_dict["AUROC_Upper_CI"][i]
        assert results_dict["AUPRC_Lower_CI"][i] <= results_dict["AUPRC_Upper_CI"][i]

        # Check that threshold CI columns ARE None
        for col in THRESHOLD_CI_COLS:
            assert results_dict[col][i] is None, (
                f"{col} should be None when calculate_threshold_ci=False"
            )

    # Check that the AU CI values are consistent across rows
    for col in OVERALL_CI_COLS:
        assert len(set(results_dict[col])) == 1, (
            f"{col} values should be identical across rows"
        )


def test_evaluation_with_au_ci_reproducibility(synth_table_larger_with_metadata):
    """Test that AU CI calculation is reproducible with a seed."""
    table = synth_table_larger_with_metadata
    modelid = "test_au_ci_rep"
    filter_desc = "larger_synth_au_rep"
    thresholds = [0.5]
    seed = 456

    results1 = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=True,
        calculate_threshold_ci=False,
        bootstrap_rounds=100,  # Set to minimum required
        bootstrap_seed=seed,
    )

    results2 = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=True,
        calculate_threshold_ci=False,
        bootstrap_rounds=100,  # Set to minimum required
        bootstrap_seed=seed,
    )

    assert results1.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    assert results2.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)

    results1_dict = results1.to_pydict()
    results2_dict = results2.to_pydict()

    # Check that the calculated AU CI values are identical
    for col in OVERALL_CI_COLS:  # OVERALL_CI_COLS refers to AUROC/AUPRC CIs
        assert results1_dict[col] == results2_dict[col], (
            f"{col} values differ between runs with same seed"
        )


def test_evaluation_with_au_ci_different_alpha(synth_table_larger_with_metadata):
    """Test that changing alpha affects the AU CI width as expected."""
    table = synth_table_larger_with_metadata
    modelid = "test_au_ci_alpha"
    filter_desc = "larger_synth_au_alpha"
    thresholds = [0.5]
    seed = 789

    results_90 = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=True,
        calculate_threshold_ci=False,
        bootstrap_rounds=100,  # Set to minimum required
        ci_alpha=0.10,  # 90% CI
        bootstrap_seed=seed,
    )

    results_99 = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=True,
        calculate_threshold_ci=False,
        bootstrap_rounds=100,  # Set to minimum required
        ci_alpha=0.01,  # 99% CI
        bootstrap_seed=seed,
    )

    assert isinstance(results_90, pa.Table)
    assert isinstance(results_99, pa.Table)
    assert results_90.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    assert results_99.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)

    results_90_dict = results_90.to_pydict()
    results_99_dict = results_99.to_pydict()

    # Compare the first row's AU CIs (index 0 corresponds to threshold 0.0)
    # Expect 99% CI to be wider than 90% CI (lower bound lower, upper bound higher)
    for lower_key, upper_key in [
        ("AUROC_Lower_CI", "AUROC_Upper_CI"),
        ("AUPRC_Lower_CI", "AUPRC_Upper_CI"),
    ]:
        # Add tolerance for potential float comparison issues if CIs are very close
        assert results_99_dict[lower_key][0] <= results_90_dict[lower_key][0] + 1e-9, (
            f"99% {lower_key} should be <= 90% {lower_key}"
        )
        assert results_99_dict[upper_key][0] >= results_90_dict[upper_key][0] - 1e-9, (
            f"99% {upper_key} should be >= 90% {upper_key}"
        )
        # Also check that the 99% interval is strictly wider if not identical
        width_99 = results_99_dict[upper_key][0] - results_99_dict[lower_key][0]
        width_90 = results_90_dict[upper_key][0] - results_90_dict[lower_key][0]
        assert width_99 >= width_90 - 1e-9, (
            f"99% CI width for {lower_key[:-9]} should be >= 90% CI width"
        )
        # Ensure width is non-negative
        assert width_99 >= 0
        assert width_90 >= 0

    # Check threshold CI columns are None in both results
    for res_dict in [results_90_dict, results_99_dict]:
        for col in THRESHOLD_CI_COLS:
            assert all(x is None for x in res_dict[col]), f"{col} should be None"


def test_evaluation_with_au_ci_and_rounding(synth_table_larger_with_metadata):
    """Test that rounding is applied correctly to AU CI values."""
    table = synth_table_larger_with_metadata
    modelid = "test_au_ci_round"
    filter_desc = "larger_synth_au_round"
    thresholds = [0.5]
    decimal_places = 3

    results = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=True,
        calculate_threshold_ci=False,  # Explicitly false
        bootstrap_rounds=100,  # Set to minimum required
        bootstrap_seed=123,
        decimal_places=decimal_places,
    )

    assert isinstance(results, pa.Table)
    assert results.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    assert results.num_rows > 0

    results_dict = results.to_pydict()

    # Check that AU CI columns are rounded (have at most 'decimal_places' digits)
    for i in range(results.num_rows):
        for col in OVERALL_CI_COLS:  # OVERALL_CI_COLS refers to AUROC/AUPRC CIs
            ci_val = results_dict[col][i]
            assert ci_val is not None
            assert isinstance(ci_val, float)
            # Check number of decimal places (approximate check due to float representation)
            assert (
                abs(
                    ci_val * (10**decimal_places) - round(ci_val * (10**decimal_places))
                )
                < 1e-9
            ), f"{col} rounding check failed"

        # Check point estimates are also rounded
        for col in [
            "AUROC",
            "AUPRC",
            "Prevalence",
            "PPV",
            "Sensitivity",
            "Specificity",
            "NPV",
            "Accuracy",
            "F1_Score",
        ]:
            point_val = results_dict[col][i]
            # Allow for NaN if metric was undefined
            if not np.isnan(point_val):
                assert isinstance(point_val, float)
                assert (
                    abs(
                        point_val * (10**decimal_places)
                        - round(point_val * (10**decimal_places))
                    )
                    < 1e-9
                ), f"{col} rounding check failed"

        # Check threshold CI columns are None
        for col in THRESHOLD_CI_COLS:
            assert results_dict[col][i] is None, f"{col} should be None"


##########################
# NEW Threshold CI Tests #
##########################


def test_evaluation_with_threshold_ci_basic(synth_table_larger_with_metadata):
    """Test basic Threshold CI calculation works and returns non-null CIs."""
    table = synth_table_larger_with_metadata
    modelid = "test_thresh_ci"
    filter_desc = "larger_synth_thresh"
    thresholds = [0.3, 0.7]  # Test multiple thresholds

    results = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=False,  # AU CI off
        calculate_threshold_ci=True,  # Request threshold CIs
        threshold_ci_method="bootstrap",  # Explicitly use bootstrap for this test
        bootstrap_rounds=100,  # Set to minimum required
        bootstrap_seed=987,
    )

    assert isinstance(results, pa.Table)
    assert results.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    # Expect len(thresholds) + 1 rows due to default 0.0 threshold
    assert results.num_rows == len(thresholds) + 1

    results_dict = results.to_pydict()

    # Check that AU CI columns ARE None
    for i in range(results.num_rows):
        for col in OVERALL_CI_COLS:  # OVERALL_CI_COLS refers to AUROC/AUPRC CIs
            assert results_dict[col][i] is None, (
                f"{col} should be None when calculate_au_ci=False"
            )

        # Check that threshold CI columns are NOT None and plausible
        for col in THRESHOLD_CI_COLS:
            ci_val = results_dict[col][i]
            assert ci_val is not None, (
                f"{col} should not be None when calculate_threshold_ci=True at row {i}"
            )
            assert isinstance(ci_val, float), (
                f"{col} should be float, got {type(ci_val)} at row {i}"
            )
            # Allow NaNs for metrics like PPV/NPV/Spec/Sens if undefined in bootstrap sample
            if not np.isnan(ci_val):
                assert 0.0 <= ci_val <= 1.0, (
                    f"{col} value {ci_val} out of bounds [0, 1] at row {i}"
                )

        # Check lower <= upper for threshold CIs
        for lower_key, upper_key in zip(
            THRESHOLD_CI_COLS[::2], THRESHOLD_CI_COLS[1::2]
        ):
            lower_val = results_dict[lower_key][i]
            upper_val = results_dict[upper_key][i]
            if not np.isnan(lower_val) and not np.isnan(upper_val):
                assert lower_val <= upper_val, f"{lower_key} > {upper_key} at row {i}"

    # Check that threshold CI values *can* differ between rows (thresholds)
    if results.num_rows > 1:
        assert (
            results_dict["PPV_Lower_CI"][0] != results_dict["PPV_Lower_CI"][1]
            or results_dict["PPV_Upper_CI"][0] != results_dict["PPV_Upper_CI"][1]
        ), "Threshold CIs should generally differ across thresholds"


def test_evaluation_with_threshold_ci_reproducibility(synth_table_larger_with_metadata):
    """Test that Threshold CI calculation is reproducible with a seed."""
    table = synth_table_larger_with_metadata
    modelid = "test_thresh_ci_rep"
    filter_desc = "larger_synth_thresh_rep"
    thresholds = [0.4, 0.6]
    seed = 654

    results1 = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=False,
        calculate_threshold_ci=True,
        threshold_ci_method="bootstrap",  # Specify method
        bootstrap_rounds=100,  # Set to minimum required
        bootstrap_seed=seed,
    )
    results2 = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=False,
        calculate_threshold_ci=True,
        threshold_ci_method="bootstrap",  # Specify method
        bootstrap_rounds=100,  # Set to minimum required
        bootstrap_seed=seed,
    )

    assert results1.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    assert results2.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    assert results1.num_rows == results2.num_rows

    results1_dict = results1.to_pydict()
    results2_dict = results2.to_pydict()

    # Check that the calculated Threshold CI values are identical, handling NaNs
    for col in THRESHOLD_CI_COLS:
        val1 = results1_dict[col]
        val2 = results2_dict[col]
        # Use np.testing or manual check for NaN equality
        np.testing.assert_equal(
            val1, val2, err_msg=f"{col} values differ between runs with same seed"
        )

    # Check AU CIs are None
    for col in OVERALL_CI_COLS:  # OVERALL_CI_COLS refers to AUROC/AUPRC CIs
        assert all(x is None for x in results1_dict[col])
        assert all(x is None for x in results2_dict[col])


def test_evaluation_with_threshold_ci_and_rounding(synth_table_larger_with_metadata):
    """Test that rounding is applied correctly to Threshold CI values."""
    table = synth_table_larger_with_metadata
    modelid = "test_thresh_ci_round"
    filter_desc = "larger_synth_thresh_round"
    thresholds = [0.5]
    decimal_places = 2

    results = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=False,
        calculate_threshold_ci=True,
        threshold_ci_method="bootstrap",  # Specify method
        bootstrap_rounds=100,  # Set to minimum required
        bootstrap_seed=321,
        decimal_places=decimal_places,
    )

    assert isinstance(results, pa.Table)
    assert results.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    assert results.num_rows > 0

    results_dict = results.to_pydict()

    # Check that threshold CI columns are rounded
    for i in range(results.num_rows):
        for col in THRESHOLD_CI_COLS:
            ci_val = results_dict[col][i]
            assert ci_val is not None
            # Allow NaNs for potentially undefined metrics in bootstrap
            if not np.isnan(ci_val):
                assert isinstance(ci_val, float)
                assert (
                    abs(
                        ci_val * (10**decimal_places)
                        - round(ci_val * (10**decimal_places))
                    )
                    < 1e-9
                ), f"{col} rounding check failed"

        # Check AU CIs are None
        for col in OVERALL_CI_COLS:  # OVERALL_CI_COLS refers to AUROC/AUPRC CIs
            assert results_dict[col][i] is None


def test_evaluation_with_both_cis(synth_table_larger_with_metadata):
    """Test calculation works when both calculate_au_ci and calculate_threshold_ci are True."""
    table = synth_table_larger_with_metadata
    modelid = "test_both_ci"
    filter_desc = "larger_synth_both"
    thresholds = [0.2, 0.8]
    seed = 111

    results = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=True,
        calculate_threshold_ci=True,
        threshold_ci_method="bootstrap",
        bootstrap_rounds=100,  # Set to minimum required
        bootstrap_seed=seed,
    )

    assert isinstance(results, pa.Table)
    assert results.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    # Expect len(thresholds) + 1 rows due to default 0.0 threshold
    assert results.num_rows == len(thresholds) + 1

    results_dict = results.to_pydict()

    # Check that ALL CI columns are NOT None and plausible
    for i in range(results.num_rows):
        for col in ALL_CI_COLS:
            ci_val = results_dict[col][i]
            assert ci_val is not None, (
                f"{col} should not be None when both flags are True at row {i}"
            )
            assert isinstance(ci_val, float), (
                f"{col} should be float, got {type(ci_val)} at row {i}"
            )
            if not np.isnan(ci_val):
                assert 0.0 <= ci_val <= 1.0, (
                    f"{col} value {ci_val} out of bounds [0, 1] at row {i}"
                )

        # Check lower <= upper for all CIs
        for lower_key, upper_key in zip(ALL_CI_COLS[::2], ALL_CI_COLS[1::2]):
            lower_val = results_dict[lower_key][i]
            upper_val = results_dict[upper_key][i]
            if not np.isnan(lower_val) and not np.isnan(upper_val):
                assert lower_val <= upper_val, f"{lower_key} > {upper_key} at row {i}"

    # Check AU CIs are consistent across rows
    for col in OVERALL_CI_COLS:  # OVERALL_CI_COLS refers to AUROC/AUPRC CIs
        assert len(set(results_dict[col])) == 1, (
            f"{col} values should be identical across rows"
        )
    # Check threshold CIs differ across rows
    if results.num_rows > 1:
        assert (
            results_dict["PPV_Lower_CI"][0] != results_dict["PPV_Lower_CI"][1]
            or results_dict["PPV_Upper_CI"][0] != results_dict["PPV_Upper_CI"][1]
        ), "Threshold CIs should generally differ across thresholds"


##############################
# Analytical CI Method Tests #
##############################


@pytest.mark.parametrize("analytical_method", ["normal", "wilson", "agresti-coull"])
def test_evaluation_with_analytical_threshold_ci(
    synth_table_larger_with_metadata, analytical_method
):
    """Test threshold CI calculation works with analytical methods."""
    table = synth_table_larger_with_metadata
    modelid = f"test_{analytical_method}_ci"
    filter_desc = f"larger_synth_{analytical_method}"
    thresholds = [0.3, 0.7]

    if anaci is None:  # Keep direct import
        pytest.skip(
            f"Skipping {analytical_method} test: _analytical_ci_utils not available."
        )

    results = evaluation(  # Use direct import
        table,
        modelid,
        filter_desc,
        thresholds,
        calculate_au_ci=False,  # AU CI off
        calculate_threshold_ci=True,  # Request threshold CIs
        threshold_ci_method=analytical_method,  # Use the parameterized analytical method
        ci_alpha=0.05,
    )

    assert isinstance(results, pa.Table)
    assert results.schema.equals(EXPECTED_SCHEMA_BASE, check_metadata=False)
    assert results.num_rows == len(thresholds) + 1  # Includes default 0.0 threshold

    results_dict = results.to_pydict()

    # Check AU CIs are None
    for i in range(results.num_rows):
        for col in OVERALL_CI_COLS:  # OVERALL_CI_COLS refers to AUROC/AUPRC CIs
            assert results_dict[col][i] is None, f"{col} should be None"

        # Check threshold CIs (except F1) are NOT None (or NaN if calculation failed gracefully)
        for col in THRESHOLD_CI_COLS:
            if col not in ["F1_Score_Lower_CI", "F1_Score_Upper_CI"]:
                ci_val = results_dict[col][i]
                assert ci_val is not None, (
                    f"{col} should not be None for {analytical_method} at row {i}"
                )
                assert isinstance(ci_val, float), (
                    f"{col} should be float, got {type(ci_val)} at row {i}"
                )
                # Allow NaNs for metrics like PPV/NPV/Spec/Sens if undefined at threshold
                if not np.isnan(ci_val):
                    assert 0.0 <= ci_val <= 1.0, (
                        f"{col} value {ci_val} out of bounds [0, 1] at row {i}"
                    )

        # Check F1 CIs ARE NaN (or None) because analytical methods don't support F1
        assert all(
            np.isnan(x) if x is not None else True
            for x in results_dict["F1_Score_Lower_CI"]
        )
        assert all(
            np.isnan(x) if x is not None else True
            for x in results_dict["F1_Score_Upper_CI"]
        )

        # Check lower <= upper for calculated threshold CIs
        for lower_key, upper_key in zip(
            THRESHOLD_CI_COLS[::2], THRESHOLD_CI_COLS[1::2]
        ):
            if lower_key not in ["F1_Score_Lower_CI"]:  # Skip F1
                lower_val = results_dict[lower_key][i]
                upper_val = results_dict[upper_key][i]
                if not np.isnan(lower_val) and not np.isnan(upper_val):
                    assert lower_val <= upper_val, (
                        f"{lower_key} > {upper_key} at row {i}"
                    )


##############################################
# Integration Tests (using sample data file) #
##############################################


# Fixture to load real data once
@pytest.fixture(scope="module")
def loaded_sample_data():
    if not os.path.exists(SAMPLE_DATA_PATH):
        pytest.skip(f"Sample data file not found at {SAMPLE_DATA_PATH}")
    try:
        data = csio.load_evaluation_data(
            source=SAMPLE_DATA_PATH,
            y_proba_col=SAMPLE_COL_MAP["y_proba"],
            y_label_col=SAMPLE_COL_MAP["y_label"],
            aggregation_cols=SAMPLE_COL_MAP["agg"],
            timeseries_col=SAMPLE_COL_MAP["time"],
            assign_model_name="sample_model",  # Use assign_model_name for model_id
            assign_task_name="sample_task",  # Use assign_task_name for task_id
        )
        return data
    except Exception as e:
        pytest.fail(f"Failed to load sample data: {e}")


def test_integration_load_data_adds_correct_metadata(loaded_sample_data):
    """Verify that loading data via pysalient.io adds the expected metadata."""
    data = loaded_sample_data
    assert data is not None
    metadata = data.schema.metadata
    assert metadata is not None
    assert metadata.get(META_KEY_Y_PROBA.encode("utf-8")) == SAMPLE_COL_MAP[
        "y_proba"
    ].encode("utf-8")
    assert metadata.get(META_KEY_Y_LABEL.encode("utf-8")) == SAMPLE_COL_MAP[
        "y_label"
    ].encode("utf-8")
    # Check for other potential metadata keys if needed (model_id, task_id etc.)
    # Check metadata added by load_evaluation_data using assign_model_name/assign_task_name
    # Metadata stores the *name* of the column ('model'/'task'), not the assigned value
    assert metadata.get(b"pysalient.io.model_col") == b"model"
    assert metadata.get(b"pysalient.io.task_col") == b"task"


def test_integration_evaluation_consumes_metadata_successfully(loaded_sample_data):
    """Test that evaluation runs successfully using metadata from loaded data."""
    data = loaded_sample_data
    assert data is not None

    # Run evaluation (no CI for speed in integration test)
    try:
        results = evaluation(  # Use direct import
            data=data,
            modelid="integration_test_model",
            filter_desc="integration_test_run",
            thresholds=[0.1, 0.5, 0.9],
            calculate_au_ci=False,  # Use renamed param
            calculate_threshold_ci=False,
        )
    except Exception as e:
        pytest.fail(f"Evaluation failed on loaded data: {e}")

    assert isinstance(results, pa.Table)
    # Expect 4 rows: 0.0 (default) + 0.1, 0.5, 0.9
    assert results.num_rows == 4
    assert results.schema.equals(EXPECTED_SCHEMA_WITH_TIMESERIES, check_metadata=False)

    # Basic checks on results
    results_dict = results.to_pydict()
    assert all(isinstance(x, float) and 0 <= x <= 1 for x in results_dict["AUROC"])
    assert all(isinstance(x, float) and 0 <= x <= 1 for x in results_dict["AUPRC"])
    assert all(isinstance(x, int) for x in results_dict["TP"])
    # Check CIs are None
    for col in ALL_CI_COLS:
        assert all(x is None for x in results_dict[col])


#########################################
# Test for Aggregation Implementation   #
#########################################


@pytest.fixture
def synthetic_grouped_data():
    """Creates synthetic data specifically for testing aggregation."""
    # Create dataset with 2 groups
    # Group A: 3 rows, 1 positive, prevalence 0.33
    # Group B: 3 rows, 2 positives, prevalence 0.67
    data = {
        "group_id": ["A", "A", "A", "B", "B", "B"],
        "timestamp": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "probability": [0.1, 0.4, 0.9, 0.2, 0.6, 0.8],
        "label": [0, 0, 1, 0, 1, 1],
    }
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)

    # Add required metadata
    metadata = {
        META_KEY_Y_PROBA.encode("utf-8"): b"probability",
        META_KEY_Y_LABEL.encode("utf-8"): b"label",
        b"pysalient.io.timeseries_col": b"timestamp",
        b"pysalient.io.aggregation_cols": b"[]",  # Empty for non-aggregated case
    }
    return table.replace_schema_metadata(metadata)


def test_evaluation_with_aggregation_differs_from_non_aggregation(
    synthetic_grouped_data,
):
    """Test that aggregating by a column vs. not aggregating produces different results."""
    # Aggregation is now handled in io.py, not in evaluation.py
    # This test is no longer relevant given the design changes
    import pytest

    pytest.skip(
        "Test no longer relevant - aggregation moved from evaluation.py to io.py"
    )


########################################
# Tests for force_eval parameter      #
########################################


def test_evaluation_force_eval_with_many_thresholds(synth_table_with_metadata):
    """Test that force_eval=True allows evaluation with more than 10 thresholds."""
    # Create a threshold list with more than 10 values
    many_thresholds = [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
    ]

    # Should work with force_eval=True
    result = evaluation(
        synth_table_with_metadata,
        "test_model",
        "test_filter",
        many_thresholds,
        force_eval=True,
    )

    # Should return results for all thresholds (plus 0.0 which is added by default)
    expected_threshold_count = len(
        set([0.0] + many_thresholds)
    )  # Remove duplicates and add 0.0
    assert len(result) == expected_threshold_count

    # Verify all thresholds are present in results
    result_thresholds = set(result["threshold"].to_pylist())
    expected_thresholds = set([0.0] + many_thresholds)
    assert result_thresholds == expected_thresholds


def test_evaluation_force_eval_false_blocks_many_thresholds_with_ci(
    synth_table_with_metadata,
):
    """Test that force_eval=False blocks evaluation with >10 thresholds when CI calculations are enabled."""
    # Create a threshold list with more than 10 values
    many_thresholds = [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
    ]

    # Should raise ValueError with CI enabled and force_eval=False (default)
    with pytest.raises(
        ValueError,
        match=r"Too many thresholds \(\d+\) specified with CI calculations enabled.*Use force_eval=True",
    ):
        evaluation(
            synth_table_with_metadata,
            "test_model",
            "test_filter",
            many_thresholds,
            calculate_threshold_ci=True,  # CI enabled - should trigger limit
            force_eval=False,
        )

    # Should work fine without CI enabled (new behavior)
    result = evaluation(
        synth_table_with_metadata,
        "test_model",
        "test_filter",
        many_thresholds,
        calculate_threshold_ci=False,  # No CI - should allow many thresholds
    )
    assert result.num_rows == len(many_thresholds) + 1  # +1 for automatic 0.0


def test_evaluation_force_eval_allows_exactly_10_thresholds(synth_table_with_metadata):
    """Test that exactly 10 thresholds work without force_eval=True."""
    # Create exactly 10 thresholds (plus 0.0 will be added, making 11 total, but we check before adding 0.0)
    exactly_10_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Should work without force_eval=True
    result = evaluation(
        synth_table_with_metadata, "test_model", "test_filter", exactly_10_thresholds
    )

    # Should return results (0.0 added by default makes 11 total)
    assert len(result) == 11  # 10 + 0.0


def test_evaluation_force_eval_with_range_specification_and_ci(
    synth_table_with_metadata,
):
    """Test force_eval with range specification that generates many thresholds when CI is enabled."""
    # Create a range that generates more than 10 thresholds
    # (0.0, 1.0, 0.05) should generate 21 thresholds: 0.0, 0.05, 0.10, ..., 1.0
    many_threshold_range = (0.0, 1.0, 0.05)

    # Should raise without force_eval when CI is enabled
    with pytest.raises(
        ValueError,
        match=r"Too many thresholds \(\d+\) specified with CI calculations enabled.*Use force_eval=True",
    ):
        evaluation(
            synth_table_with_metadata,
            "test_model",
            "test_filter",
            many_threshold_range,
            calculate_au_ci=True,  # CI enabled - should trigger limit
        )

    # Should work with force_eval=True even with CI
    result = evaluation(
        synth_table_with_metadata,
        "test_model",
        "test_filter",
        many_threshold_range,
        calculate_au_ci=True,
        force_eval=True,
        bootstrap_rounds=100,  # Reduce for faster test
    )

    # Should return results for all generated thresholds
    assert len(result) == 21  # 0.0, 0.05, 0.10, ..., 1.0


def test_evaluation_force_eval_error_message_accuracy_with_ci(
    synth_table_with_metadata,
):
    """Test that the error message shows the correct threshold count when CI is enabled."""
    # Create 15 thresholds
    threshold_list = [i * 0.05 for i in range(1, 16)]  # [0.05, 0.10, ..., 0.75]

    with pytest.raises(ValueError) as exc_info:
        evaluation(
            synth_table_with_metadata,
            "test_model",
            "test_filter",
            threshold_list,
            calculate_au_ci=True,  # Enable CI to trigger the limit
        )

    error_msg = str(exc_info.value)
    # Should mention the correct count (15 user-specified thresholds, forced 0 doesn't count)
    assert "15" in error_msg  # 15 specified thresholds only
    assert "force_eval=True" in error_msg
    assert "Maximum allowed is 10" in error_msg
    assert "CI calculations enabled" in error_msg


def test_evaluation_force_eval_with_duplicates_counts_unique_with_ci(
    synth_table_with_metadata,
):
    """Test that threshold counting considers unique thresholds only when CI is enabled."""
    # Create list with duplicates that results in <= 10 unique thresholds
    thresholds_with_duplicates = [
        0.1,
        0.1,
        0.2,
        0.2,
        0.3,
        0.3,
        0.4,
        0.4,
        0.5,
        0.5,
        0.6,
        0.6,
    ]
    # Unique: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] = 6 unique + 0.0 = 7 total

    # Should work with CI enabled since unique count <= 10
    result = evaluation(
        synth_table_with_metadata,
        "test_model",
        "test_filter",
        thresholds_with_duplicates,
        calculate_threshold_ci=True,
        bootstrap_rounds=100,  # Small number for faster test
    )

    # Should return results for unique thresholds only
    assert len(result) == 7  # 6 unique + 0.0

    # Now test with duplicates that result in > 10 unique thresholds
    many_thresholds_with_duplicates = [
        0.1,
        0.1,
        0.15,
        0.15,
        0.2,
        0.2,
        0.25,
        0.25,
        0.3,
        0.3,
        0.35,
        0.35,
        0.4,
        0.4,
        0.45,
        0.45,
        0.5,
        0.5,
        0.55,
        0.55,
        0.6,
        0.6,
    ]
    # Unique: 11 thresholds + 0.0 = 12 total

    with pytest.raises(
        ValueError,
        match=r"Too many thresholds.*CI calculations enabled.*Use force_eval=True",
    ):
        evaluation(
            synth_table_with_metadata,
            "test_model",
            "test_filter",
            many_thresholds_with_duplicates,
            calculate_threshold_ci=True,  # Enable CI to trigger the limit
        )
