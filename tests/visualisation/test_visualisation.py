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


# Minimal data for plotting tests
@pytest.fixture
def sample_plot_data():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.3, 0.8, 0.6, 0.2, 0.9, 0.4, 0.7])
    return y_true, y_score


# Fixture to potentially skip tests if plotting dependencies are missing
# Note: This requires matplotlib and sklearn to be installed in the test environment
# If they are optional, these tests might need to be skipped conditionally.
# We proceed assuming they are available for testing purposes.
needs_matplotlib = pytest.mark.skipif(
    not viz._MATPLOTLIB_AVAILABLE, reason="matplotlib not installed"
)
needs_sklearn = pytest.mark.skipif(
    not viz._SKLEARN_METRICS_AVAILABLE, reason="scikit-learn not installed"
)


@needs_matplotlib
@needs_sklearn
def test_plot_roc_curve_runs(sample_plot_data):
    """Test that plot_roc_curve runs without error and returns Axes."""
    y_true, y_score = sample_plot_data
    ax = viz.plot_roc_curve(y_true, y_score, model_name="TestModel")
    assert isinstance(ax, viz.Axes)
    # Basic check if title is set (more detailed checks are complex)
    assert ax.get_title() == "Receiver Operating Characteristic (ROC) Curve"
    # Clean up the plot figure
    plt = viz.plt  # Get the imported plt
    if plt:
        plt.close(ax.figure)


@needs_matplotlib
@needs_sklearn
def test_plot_roc_curve_with_existing_ax(sample_plot_data):
    """Test plot_roc_curve works with a pre-existing Axes object."""
    y_true, y_score = sample_plot_data
    plt = viz.plt
    if not plt:  # Skip if plt is None due to import error guard
        pytest.skip("matplotlib not available")

    fig, ax_existing = plt.subplots()
    ax_returned = viz.plot_roc_curve(y_true, y_score, ax=ax_existing)
    assert ax_returned is ax_existing  # Should return the same axes
    assert len(ax_existing.lines) > 0  # Check if something was plotted
    plt.close(fig)


@needs_matplotlib
@needs_sklearn
def test_plot_pr_curve_runs(sample_plot_data):
    """Test that plot_precision_recall_curve runs without error and returns Axes."""
    y_true, y_score = sample_plot_data
    ax = viz.plot_precision_recall_curve(y_true, y_score, model_name="TestModelPR")
    assert isinstance(ax, viz.Axes)
    assert ax.get_title() == "Precision-Recall Curve"
    # Clean up the plot figure
    plt = viz.plt
    if plt:
        plt.close(ax.figure)


@needs_matplotlib
@needs_sklearn
def test_plot_pr_curve_with_existing_ax(sample_plot_data):
    """Test plot_precision_recall_curve works with a pre-existing Axes object."""
    y_true, y_score = sample_plot_data
    plt = viz.plt
    if not plt:  # Skip if plt is None due to import error guard
        pytest.skip("matplotlib not available")

    fig, ax_existing = plt.subplots()
    ax_returned = viz.plot_precision_recall_curve(y_true, y_score, ax=ax_existing)
    assert ax_returned is ax_existing
    assert len(ax_existing.lines) > 0
    plt.close(fig)


def test_order_by_valid_string():
    table = pa.table({"threshold": [0.9, 0.1], "AUROC": [0.8, 0.7]})
    html = viz.format_evaluation_table(table, order_by="threshold").to_html()
    assert html.find(">0.100<") < html.find(">0.900<")


def test_order_by_valid_list():
    table = pa.table(
        {
            "modelid": ["b", "a", "a"],
            "threshold": [0.7, 0.9, 0.1],
            "AUROC": [0.7, 0.9, 0.1],
        }
    )
    html = viz.format_evaluation_table(
        table, order_by=["modelid", "threshold"], decimal_places=1
    ).to_html()
    assert html.find(">a<") < html.find(">b<")
    assert html.find(">0.1<") < html.find(">0.9<")


def test_order_by_missing_column_raises(sample_eval_table):
    with pytest.raises(RuntimeError, match="Failed to sort"):
        viz.format_evaluation_table(sample_eval_table, order_by="missing_col")


def test_order_by_invalid_type_raises(sample_eval_table):
    with pytest.raises(RuntimeError, match="Failed to sort"):
        viz.format_evaluation_table(sample_eval_table, order_by=123)


def test_ci_column_true_creates_separate_columns():
    table = pa.table(
        {
            "threshold": [0.5],
            "AUROC": [0.85],
            "AUROC_Lower_CI": [0.82],
            "AUROC_Upper_CI": [0.88],
        }
    )
    html = viz.format_evaluation_table(table, ci_column=True).to_html()
    assert "AUROC CI" in html
    assert "AUROC_Lower_CI" not in html
    assert "AUROC_Upper_CI" not in html


def test_ci_column_false_inlines_ci_text():
    table = pa.table(
        {
            "threshold": [0.5],
            "AUROC": [0.85],
            "AUROC_Lower_CI": [0.82],
            "AUROC_Upper_CI": [0.88],
        }
    )
    html = viz.format_evaluation_table(table, ci_column=False).to_html()
    assert "AUROC CI" not in html
    assert "0.850 [0.820 - 0.880]" in html


def test_float_columns_validation_and_filtering(sample_eval_table):
    with pytest.raises(TypeError, match="float_columns"):
        viz.format_evaluation_table(sample_eval_table, float_columns=[1, "AUROC"])

    html = viz.format_evaluation_table(
        sample_eval_table, float_columns=["AUROC", "missing"], decimal_places=4
    ).to_html()
    assert ">0.9877<" in html
    assert ">0.1000<" not in html


@needs_matplotlib
@needs_sklearn
def test_plot_roc_curve_without_axes(sample_plot_data):
    y_true, y_score = sample_plot_data
    ax = viz.plot_roc_curve(y_true, y_score)
    assert isinstance(ax, viz.Axes)
    viz.plt.close(ax.figure)


@needs_matplotlib
@needs_sklearn
def test_plot_roc_curve_label_with_model_name(sample_plot_data):
    y_true, y_score = sample_plot_data
    ax = viz.plot_roc_curve(y_true, y_score, model_name="ModelX")
    labels = [line.get_label() for line in ax.lines]
    assert any("ModelX (AUC =" in label for label in labels)
    viz.plt.close(ax.figure)


@needs_matplotlib
@needs_sklearn
def test_plot_roc_curve_label_without_model_name(sample_plot_data):
    y_true, y_score = sample_plot_data
    ax = viz.plot_roc_curve(y_true, y_score)
    labels = [line.get_label() for line in ax.lines]
    assert any("ROC curve (AUC =" in label for label in labels)
    viz.plt.close(ax.figure)


@needs_matplotlib
@needs_sklearn
def test_plot_roc_curve_bad_input_shapes():
    with pytest.raises(ValueError):
        viz.plot_roc_curve(np.array([0, 1]), np.array([0.1]))


@needs_matplotlib
@needs_sklearn
def test_precision_recall_curve_variants(sample_plot_data):
    y_true, y_score = sample_plot_data
    ax = viz.plot_precision_recall_curve(y_true, y_score, model_name="ModelPR")
    labels = [line.get_label() for line in ax.lines]
    assert any("ModelPR (PR Curve)" == label for label in labels)
    viz.plt.close(ax.figure)

    with pytest.raises(ValueError):
        viz.plot_precision_recall_curve(np.array([0, 1]), np.array([0.1]))


def test_plot_import_error(monkeypatch):
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._MATPLOTLIB_AVAILABLE", False
    )
    with pytest.raises(ImportError, match="matplotlib is required"):
        viz.plot_roc_curve([0, 1], [0.1, 0.9])


def test_plot_pr_import_error_when_matplotlib_unavailable(monkeypatch):
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._MATPLOTLIB_AVAILABLE", False
    )
    with pytest.raises(ImportError, match="matplotlib is required"):
        viz.plot_precision_recall_curve([0, 1], [0.1, 0.9])


class _FakeAxes:
    def __init__(self):
        self.lines = []
        self._title = ""

    def plot(self, *args, **kwargs):
        self.lines.append({"args": args, "kwargs": kwargs})

    def set_xlabel(self, *_args, **_kwargs):
        return None

    def set_ylabel(self, *_args, **_kwargs):
        return None

    def set_title(self, title, **_kwargs):
        self._title = title

    def set_xlim(self, *_args, **_kwargs):
        return None

    def set_ylim(self, *_args, **_kwargs):
        return None

    def legend(self, *_args, **_kwargs):
        return None

    def grid(self, *_args, **_kwargs):
        return None

    def set_aspect(self, *_args, **_kwargs):
        return None


def test_plot_roc_curve_executes_without_matplotlib(monkeypatch):
    fake_ax = _FakeAxes()
    monkeypatch.setattr("pysalient.visualisation.visualisation._MATPLOTLIB_AVAILABLE", True)
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._SKLEARN_METRICS_AVAILABLE", True
    )
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation.roc_curve",
        lambda y_true, y_score: (np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5, 0.1])),
    )
    monkeypatch.setattr("pysalient.visualisation.visualisation.auc", lambda x, y: 0.8)

    ax = viz.plot_roc_curve([0, 1], [0.1, 0.9], model_name="M", ax=fake_ax)
    assert ax is fake_ax
    assert len(fake_ax.lines) == 2
    assert fake_ax._title == "Receiver Operating Characteristic (ROC) Curve"


def test_plot_roc_curve_executes_without_model_name(monkeypatch):
    fake_ax = _FakeAxes()
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._MATPLOTLIB_AVAILABLE", True
    )
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._SKLEARN_METRICS_AVAILABLE", True
    )
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation.roc_curve",
        lambda y_true, y_score: (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.5, 0.1]),
        ),
    )
    monkeypatch.setattr("pysalient.visualisation.visualisation.auc", lambda x, y: 0.8)

    ax = viz.plot_roc_curve([0, 1], [0.1, 0.9], ax=fake_ax)
    assert ax is fake_ax
    assert len(fake_ax.lines) == 2


def test_plot_precision_recall_executes_without_matplotlib(monkeypatch):
    fake_ax = _FakeAxes()
    monkeypatch.setattr("pysalient.visualisation.visualisation._MATPLOTLIB_AVAILABLE", True)
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._SKLEARN_METRICS_AVAILABLE", True
    )
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation.precision_recall_curve",
        lambda y_true, y_score: (np.array([1.0, 0.5]), np.array([0.0, 1.0]), np.array([0.3])),
    )

    ax = viz.plot_precision_recall_curve([0, 1], [0.1, 0.9], model_name="M", ax=fake_ax)
    assert ax is fake_ax
    assert len(fake_ax.lines) == 1
    assert fake_ax._title == "Precision-Recall Curve"


def test_plot_roc_curve_import_error_when_sklearn_unavailable(monkeypatch):
    monkeypatch.setattr("pysalient.visualisation.visualisation._MATPLOTLIB_AVAILABLE", True)
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._SKLEARN_METRICS_AVAILABLE", False
    )
    monkeypatch.setattr("pysalient.visualisation.visualisation.roc_curve", None)
    monkeypatch.setattr("pysalient.visualisation.visualisation.auc", None)
    with pytest.raises(ImportError, match="scikit-learn is required for ROC"):
        viz.plot_roc_curve([0, 1], [0.1, 0.9], ax=_FakeAxes())


def test_plot_pr_curve_import_error_when_sklearn_unavailable(monkeypatch):
    monkeypatch.setattr("pysalient.visualisation.visualisation._MATPLOTLIB_AVAILABLE", True)
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._SKLEARN_METRICS_AVAILABLE", False
    )
    monkeypatch.setattr("pysalient.visualisation.visualisation.precision_recall_curve", None)
    with pytest.raises(ImportError, match="scikit-learn is required for PR"):
        viz.plot_precision_recall_curve([0, 1], [0.1, 0.9], ax=_FakeAxes())


def test_plot_roc_curve_runtime_error_when_no_ax_and_no_plt(monkeypatch):
    monkeypatch.setattr("pysalient.visualisation.visualisation._MATPLOTLIB_AVAILABLE", True)
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._SKLEARN_METRICS_AVAILABLE", True
    )
    monkeypatch.setattr("pysalient.visualisation.visualisation.plt", None)
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation.roc_curve",
        lambda y_true, y_score: (np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5, 0.1])),
    )
    monkeypatch.setattr("pysalient.visualisation.visualisation.auc", lambda x, y: 0.8)
    with pytest.raises(RuntimeError, match="could not be created"):
        viz.plot_roc_curve([0, 1], [0.1, 0.9], ax=None)


def test_plot_pr_curve_runtime_error_when_no_ax_and_no_plt(monkeypatch):
    monkeypatch.setattr("pysalient.visualisation.visualisation._MATPLOTLIB_AVAILABLE", True)
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation._SKLEARN_METRICS_AVAILABLE", True
    )
    monkeypatch.setattr("pysalient.visualisation.visualisation.plt", None)
    monkeypatch.setattr(
        "pysalient.visualisation.visualisation.precision_recall_curve",
        lambda y_true, y_score: (np.array([1.0, 0.5]), np.array([0.0, 1.0]), np.array([0.3])),
    )
    with pytest.raises(RuntimeError, match="could not be created"):
        viz.plot_precision_recall_curve([0, 1], [0.1, 0.9], ax=None)

