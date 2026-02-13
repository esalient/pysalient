"""
Tests for the pysalient.visualisation module.
"""

import numpy as np
import pyarrow as pa
import pytest
from pandas.io.formats.style import Styler

# Module to test
import pysalient.visualisation as viz

############
# Fixtures #
############


@pytest.fixture
def sample_eval_table() -> pa.Table:
    """Provides a sample PyArrow Table similar to evaluation results."""
    data = {
        "threshold": [0.1, 0.30000000000000004, 0.5, 0.7000000000000001, 0.9],
        "AUROC": [0.987654] * 5,
        "AUPRC": [0.912345] * 5,
        "Prevalence": [0.55555] * 5,
        "ppv": [0.6, 0.7512, 0.8888, 0.92345, 0.95],
        "sensitivity": [1.0, 0.95678, 0.85, 0.7654, 0.6],
        "tp": [10, 9, 8, 7, 6],
        "Sample Size": [20] * 5,
        "other_col": ["a", "b", "c", "d", "e"],  # Non-numeric column
    }
    return pa.table(data)


@pytest.fixture
def table_missing_cols() -> pa.Table:
    """Provides a table missing some default float columns."""
    data = {
        "threshold": [0.1, 0.5, 0.9],
        "AUROC": [0.98] * 3,
        # Missing AUPRC, Prevalence
        "ppv": [0.6, 0.8, 0.95],
        "sensitivity": [1.0, 0.85, 0.6],
        "tp": [10, 8, 6],
        "Sample Size": [20] * 3,
    }
    return pa.table(data)


#########
# Tests #
#########


def test_returns_styler(sample_eval_table):
    """Test that the function returns a Pandas Styler object."""
    result = viz.format_evaluation_table(sample_eval_table)
    assert isinstance(result, Styler)


def test_default_formatting(sample_eval_table):
    """Test default formatting (3 decimal places) is applied to default columns."""
    styler = viz.format_evaluation_table(sample_eval_table)  # Default decimal_places=3
    html_output = styler.to_html()

    # Check default float columns are formatted to 3 decimal places
    assert ">0.100<" in html_output  # threshold[0]
    assert ">0.988<" in html_output  # AUROC[0] (0.987654 rounded)
    assert ">0.912<" in html_output  # AUPRC[0] (0.912345 rounded)
    assert ">0.556<" in html_output  # Prevalence[0] (0.55555 rounded)
    assert ">0.600<" in html_output  # ppv[0]
    assert ">1.000<" in html_output  # sensitivity[0]

    # Check a non-float column is NOT formatted with decimals (exact value check)
    assert ">10<" in html_output  # tp[0]
    assert ">20<" in html_output  # Sample Size[0]
    assert ">a<" in html_output  # other_col[0]

    # Check a value that requires rounding up
    assert ">0.765<" in html_output  # sensitivity[3] (0.7654 rounded)
    # Check a value that requires rounding down
    assert ">0.751<" in html_output  # ppv[1] (0.7512 rounded)


def test_custom_decimal_places(sample_eval_table):
    """Test specifying decimal_places works correctly."""
    styler = viz.format_evaluation_table(sample_eval_table, decimal_places=2)
    html_output = styler.to_html()

    # Check default float columns are formatted to 2 decimal places
    assert ">0.10<" in html_output  # threshold[0]
    assert ">0.99<" in html_output  # AUROC[0] (0.987654 rounded)
    assert ">0.91<" in html_output  # AUPRC[0] (0.912345 rounded)
    assert ">0.56<" in html_output  # Prevalence[0] (0.55555 rounded)
    assert ">0.60<" in html_output  # ppv[0]
    assert ">1.00<" in html_output  # sensitivity[0]

    # Check a non-float column is NOT formatted
    assert ">10<" in html_output  # tp[0]

    # Check rounding
    assert ">0.77<" in html_output  # sensitivity[3] (0.7654 rounded)
    assert ">0.75<" in html_output  # ppv[1] (0.7512 rounded)


def test_no_rounding(sample_eval_table):
    """Test decimal_places=None results in no formatting."""
    styler = viz.format_evaluation_table(sample_eval_table, decimal_places=None)
    # Check that the internal display funcs dict is empty or doesn't contain formatters
    assert not styler._display_funcs


def test_custom_float_columns(sample_eval_table):
    """Test specifying float_columns works correctly."""
    custom_cols = ["AUROC", "ppv", "non_existent_col"]
    styler = viz.format_evaluation_table(
        sample_eval_table, decimal_places=4, float_columns=custom_cols
    )
    html_output = styler.to_html()

    # Check included columns ARE formatted to 4 decimal places
    assert ">0.9877<" in html_output  # AUROC[0] (0.987654 rounded)
    assert ">0.6000<" in html_output  # ppv[0]
    assert ">0.7512<" in html_output  # ppv[1]

    # Check default columns NOT included ARE NOT formatted (check for original or default pandas format)
    # Note: Pandas default float format might vary, so check for non-4-decimal format.
    # Checking for the exact unformatted value might be too strict if pandas applies some default.
    # Let's check they are *not* formatted to 4 decimals.
    assert ">0.1000<" not in html_output  # threshold[0] should not be 4dp
    assert (
        ">0.9123<" in html_output or ">0.912345<" in html_output
    )  # AUPRC[0] - check original or pandas default, not 4dp
    assert ">0.5556<" not in html_output  # Prevalence[0] should not be 4dp
    assert ">1.0000<" not in html_output  # sensitivity[0] should not be 4dp

    # Check non-float column is NOT formatted
    assert ">10<" in html_output  # tp[0]


def test_handles_missing_columns(table_missing_cols):
    """Test default formatting handles missing columns gracefully."""
    styler = viz.format_evaluation_table(
        table_missing_cols, decimal_places=3
    )  # Default 3dp
    html_output = styler.to_html()

    # Check existing default columns ARE formatted to 3 decimal places
    assert ">0.100<" in html_output  # threshold[0]
    assert ">0.980<" in html_output  # AUROC[0]
    assert ">0.600<" in html_output  # ppv[0]
    assert ">1.000<" in html_output  # sensitivity[0]
    assert ">0.850<" in html_output  # sensitivity[1]

    # Check non-float column is NOT formatted
    assert ">10<" in html_output  # tp[0]

    # Check missing default columns are simply not present in the output table headers
    assert "<th>AUPRC</th>" not in html_output
    assert "<th>Prevalence</th>" not in html_output


def test_input_type_error():
    """Test passing a non-PyArrow table raises TypeError."""
    with pytest.raises(TypeError, match="Input 'table' must be a PyArrow Table."):
        viz.format_evaluation_table([1, 2, 3])  # Pass a list instead of table


def test_invalid_decimal_places_error(sample_eval_table):
    """Test invalid decimal_places raises ValueError."""
    with pytest.raises(
        ValueError, match="'decimal_places' must be a non-negative integer or None."
    ):
        viz.format_evaluation_table(sample_eval_table, decimal_places=-1)
    with pytest.raises(
        ValueError, match="'decimal_places' must be a non-negative integer or None."
    ):
        viz.format_evaluation_table(sample_eval_table, decimal_places=1.5)


def test_invalid_float_columns_error(sample_eval_table):
    """Test invalid float_columns raises TypeError."""
    with pytest.raises(
        TypeError, match="'float_columns' must be a list of strings or None."
    ):
        viz.format_evaluation_table(
            sample_eval_table, float_columns="AUROC"
        )  # Pass string
    with pytest.raises(
        TypeError, match="'float_columns' must be a list of strings or None."
    ):
        viz.format_evaluation_table(
            sample_eval_table, float_columns=[1, 2]
        )  # Pass list of ints


# --- Tests for Plotting Functions ---


# Fixture to skip tests if Altair is not available
needs_altair = pytest.mark.skipif(
    not viz._ALTAIR_AVAILABLE, reason="altair not installed"
)


# Metadata keys for test fixtures
_META_KEY_Y_PROBA = "pysalient.io.y_proba_col"
_META_KEY_Y_LABEL = "pysalient.io.y_label_col"


@pytest.fixture
def sample_eval_table_with_curves() -> pa.Table:
    """Provides a sample evaluation result table with ROC/PR curve data."""
    # Import evaluation to create proper table with curve data
    from pysalient.evaluation import evaluation

    probas = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 0, 1, 1, 1]

    table = pa.table({"y_proba": probas, "y_label": labels})
    metadata = {
        _META_KEY_Y_PROBA.encode("utf-8"): b"y_proba",
        _META_KEY_Y_LABEL.encode("utf-8"): b"y_label",
    }
    table = table.replace_schema_metadata(metadata)

    # Run evaluation with curve export
    result = evaluation(
        table,
        "test_model",
        "test_filter",
        [0.5],
        export_roc_curve_data=True,
    )
    return result


@pytest.fixture
def sample_eval_table_without_curves() -> pa.Table:
    """Provides a sample evaluation result table WITHOUT curve data."""
    from pysalient.evaluation import evaluation

    probas = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 0, 1, 1, 1]

    table = pa.table({"y_proba": probas, "y_label": labels})
    metadata = {
        _META_KEY_Y_PROBA.encode("utf-8"): b"y_proba",
        _META_KEY_Y_LABEL.encode("utf-8"): b"y_label",
    }
    table = table.replace_schema_metadata(metadata)

    # Run evaluation WITHOUT curve export
    result = evaluation(
        table,
        "test_model",
        "test_filter",
        [0.5],
        export_roc_curve_data=False,
    )
    return result


@needs_altair
def test_plot_roc_curve_runs(sample_eval_table_with_curves):
    """Test that plot_roc_curve runs without error."""
    chart = viz.plot_roc_curve(sample_eval_table_with_curves)
    # Check it returns something (Altair Chart object)
    assert chart is not None
    # Check it has expected properties
    assert hasattr(chart, "to_dict")  # Altair charts have to_dict method


@needs_altair
def test_plot_roc_curve_with_threshold(sample_eval_table_with_curves):
    """Test plot_roc_curve with threshold highlighting."""
    chart = viz.plot_roc_curve(
        sample_eval_table_with_curves,
        threshold=0.5,
    )
    assert chart is not None
    # The chart should contain multiple layers when threshold is specified
    chart_dict = chart.to_dict()
    assert "layer" in chart_dict or "mark" in chart_dict


@needs_altair
def test_plot_roc_curve_custom_dimensions(sample_eval_table_with_curves):
    """Test plot_roc_curve with custom width and height."""
    chart = viz.plot_roc_curve(
        sample_eval_table_with_curves,
        width=600,
        height=500,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert chart_dict.get("width") == 600
    assert chart_dict.get("height") == 500


@needs_altair
def test_plot_roc_curve_missing_curve_data_error(
    sample_eval_table_without_curves,
):
    """Test plot_roc_curve raises error when curve data is missing."""
    with pytest.raises(ValueError, match="ROC curve data columns not found"):
        viz.plot_roc_curve(sample_eval_table_without_curves)


@needs_altair
def test_plot_pr_curve_runs(sample_eval_table_with_curves):
    """Test that plot_precision_recall_curve runs without error."""
    chart = viz.plot_precision_recall_curve(sample_eval_table_with_curves)
    assert chart is not None
    assert hasattr(chart, "to_dict")


@needs_altair
def test_plot_pr_curve_with_threshold(sample_eval_table_with_curves):
    """Test plot_precision_recall_curve with threshold highlighting."""
    chart = viz.plot_precision_recall_curve(
        sample_eval_table_with_curves,
        threshold=0.5,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert "layer" in chart_dict or "mark" in chart_dict


@needs_altair
def test_plot_pr_curve_custom_dimensions(sample_eval_table_with_curves):
    """Test plot_precision_recall_curve with custom width and height."""
    chart = viz.plot_precision_recall_curve(
        sample_eval_table_with_curves,
        width=500,
        height=400,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert chart_dict.get("width") == 500
    assert chart_dict.get("height") == 400


@needs_altair
def test_plot_pr_curve_missing_curve_data_error(
    sample_eval_table_without_curves,
):
    """Test plot_precision_recall_curve raises error when curve data is missing."""
    with pytest.raises(ValueError, match="PR curve data columns not found"):
        viz.plot_precision_recall_curve(sample_eval_table_without_curves)


def test_altair_import_error_without_altair():
    """Test that Altair functions raise ImportError when altair is not available."""
    # This test checks the error message when altair is not installed
    # We can't easily test this if altair IS installed, so we'll skip if it is
    if viz._ALTAIR_AVAILABLE:
        pytest.skip("Altair is available, cannot test ImportError path")

    # Create a dummy table (won't actually be used since import check happens first)
    dummy_table = pa.table({"x": [1, 2, 3]})

    with pytest.raises(ImportError, match="altair is required"):
        viz.plot_roc_curve(dummy_table)

    with pytest.raises(ImportError, match="altair is required"):
        viz.plot_precision_recall_curve(dummy_table)
