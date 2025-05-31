"""
Tests for the _bootstrap_utils module.
"""

import numpy as np
import pytest

# Assume sklearn is available for testing purposes, or mock if necessary
try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    # Define dummy functions if sklearn not present, tests using them will be skipped
    def roc_auc_score(y_true, y_pred):
        return 0.5

    def average_precision_score(y_true, y_pred):
        return 0.5


# Import the function to test - catch potential import errors
try:
    # Import from the new internal location
    from pysalient.evaluation._bootstrap_utils import calculate_bootstrap_ci
except ImportError:
    pytest.skip(
        "Skipping bootstrap tests: _internal._bootstrap_utils not found",
        allow_module_level=True,
    )

#############
# Test Data #
#############
SEED = 123
N_SAMPLES = 100
N_ROUNDS = 200  # Use fewer rounds for faster testing


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Generates reproducible sample data."""
    rng = np.random.RandomState(SEED)
    y_true = rng.randint(0, 2, size=N_SAMPLES)
    # Generate scores somewhat correlated with labels
    y_pred = y_true * 0.6 + rng.rand(N_SAMPLES) * 0.4
    return y_true, y_pred


#############################
# Basic Functionality Tests #
#############################


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="requires scikit-learn")
def test_basic_ci_calculation_auroc(sample_data):
    """Test basic CI calculation for AUROC."""
    y_true, y_pred = sample_data
    lower, upper = calculate_bootstrap_ci(
        y_true, y_pred, roc_auc_score, n_rounds=N_ROUNDS, alpha=0.05, seed=SEED
    )
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert lower <= upper
    # Check if bounds seem reasonable (very loose check)
    actual_auroc = roc_auc_score(y_true, y_pred)
    assert lower < actual_auroc + 0.1  # Allow some margin
    assert upper > actual_auroc - 0.1  # Allow some margin


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="requires scikit-learn")
def test_basic_ci_calculation_auprc(sample_data):
    """Test basic CI calculation for AUPRC."""
    y_true, y_pred = sample_data
    lower, upper = calculate_bootstrap_ci(
        y_true,
        y_pred,
        average_precision_score,
        n_rounds=N_ROUNDS,
        alpha=0.05,
        seed=SEED,
    )
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert lower <= upper


def test_reproducibility_with_seed(sample_data):
    """Test that the same seed yields the same results."""
    y_true, y_pred = sample_data
    metric = (
        roc_auc_score if SKLEARN_AVAILABLE else lambda yt, yp: np.mean(yt)
    )  # Use dummy if needed

    ci1_lower, ci1_upper = calculate_bootstrap_ci(
        y_true, y_pred, metric, n_rounds=N_ROUNDS, alpha=0.05, seed=SEED
    )
    ci2_lower, ci2_upper = calculate_bootstrap_ci(
        y_true, y_pred, metric, n_rounds=N_ROUNDS, alpha=0.05, seed=SEED
    )

    assert ci1_lower == ci2_lower
    assert ci1_upper == ci2_upper


def test_different_alpha_values(sample_data):
    """Test that different alpha values produce different interval widths."""
    y_true, y_pred = sample_data
    metric = roc_auc_score if SKLEARN_AVAILABLE else lambda yt, yp: np.mean(yt)

    ci95_lower, ci95_upper = calculate_bootstrap_ci(
        y_true, y_pred, metric, n_rounds=N_ROUNDS, alpha=0.05, seed=SEED
    )
    ci99_lower, ci99_upper = calculate_bootstrap_ci(
        y_true, y_pred, metric, n_rounds=N_ROUNDS, alpha=0.01, seed=SEED
    )  # 99% CI

    # 99% CI should be wider or equal to 95% CI
    assert ci99_lower <= ci95_lower
    assert ci99_upper >= ci95_upper
    # The intervals should generally be different, but allow for edge cases
    # where they might be the same if data/sampling leads to it.
    # The primary check is that 99% CI is wider or equal.


##########################
# Input Validation Tests #
##########################


def test_invalid_alpha(sample_data):
    """Test error handling for invalid alpha values."""
    y_true, y_pred = sample_data
    metric = roc_auc_score if SKLEARN_AVAILABLE else lambda yt, yp: np.mean(yt)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        calculate_bootstrap_ci(y_true, y_pred, metric, alpha=0)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        calculate_bootstrap_ci(y_true, y_pred, metric, alpha=1)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        calculate_bootstrap_ci(y_true, y_pred, metric, alpha=1.1)


def test_invalid_n_rounds(sample_data):
    """Test error handling for invalid n_rounds."""
    y_true, y_pred = sample_data
    metric = roc_auc_score if SKLEARN_AVAILABLE else lambda yt, yp: np.mean(yt)
    with pytest.raises(ValueError, match="n_rounds must be a positive integer"):
        calculate_bootstrap_ci(y_true, y_pred, metric, n_rounds=0)
    with pytest.raises(ValueError, match="n_rounds must be a positive integer"):
        calculate_bootstrap_ci(y_true, y_pred, metric, n_rounds=-10)
    with pytest.raises(ValueError, match="n_rounds must be a positive integer"):
        calculate_bootstrap_ci(y_true, y_pred, metric, n_rounds=10.5)  # type check


def test_invalid_seed_type(sample_data):
    """Test error handling for invalid seed type."""
    y_true, y_pred = sample_data
    metric = roc_auc_score if SKLEARN_AVAILABLE else lambda yt, yp: np.mean(yt)
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        calculate_bootstrap_ci(y_true, y_pred, metric, seed="abc")


def test_invalid_input_types(sample_data):
    """Test error handling for non-numpy array inputs."""
    y_true, y_pred = sample_data
    metric = roc_auc_score if SKLEARN_AVAILABLE else lambda yt, yp: np.mean(yt)
    with pytest.raises(TypeError, match="y_true and y_pred must be NumPy arrays"):
        calculate_bootstrap_ci(list(y_true), y_pred, metric)
    with pytest.raises(TypeError, match="y_true and y_pred must be NumPy arrays"):
        calculate_bootstrap_ci(y_true, list(y_pred), metric)


def test_mismatched_lengths(sample_data):
    """Test error handling for input arrays with different lengths."""
    y_true, _ = sample_data
    y_pred_short = y_true[: N_SAMPLES - 1]
    metric = roc_auc_score if SKLEARN_AVAILABLE else lambda yt, yp: np.mean(yt)
    with pytest.raises(ValueError, match="must have the same length"):
        calculate_bootstrap_ci(y_true, y_pred_short, metric)


def test_empty_arrays(sample_data):
    """Test error handling for empty input arrays."""
    y_true_empty = np.array([])
    y_pred_empty = np.array([])
    metric = roc_auc_score if SKLEARN_AVAILABLE else lambda yt, yp: np.mean(yt)
    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        calculate_bootstrap_ci(y_true_empty, y_pred_empty, metric)


def test_invalid_metric_func_type(sample_data):
    """Test error handling for non-callable metric_func."""
    y_true, y_pred = sample_data
    with pytest.raises(TypeError, match="metric_func must be a callable function"):
        calculate_bootstrap_ci(y_true, y_pred, "not_a_function")


def test_metric_func_returns_non_scalar(sample_data):
    """Test error handling when metric_func doesn't return a scalar."""
    y_true, y_pred = sample_data

    def non_scalar_metric(yt, yp):
        return np.array([np.mean(yt), np.mean(yp)])  # Returns array

    with pytest.raises(
        ValueError, match="metric_func must return a single numeric value"
    ):
        calculate_bootstrap_ci(y_true, y_pred, non_scalar_metric)


####################
#  Edge Case Tests #
####################


def test_metric_func_fails_sometimes(sample_data):
    """Test behavior when metric_func fails during some bootstrap rounds."""
    y_true, y_pred = sample_data
    call_count = 0

    def failing_metric(yt, yp):
        nonlocal call_count
        call_count += 1
        if call_count % 4 == 0:  # Fail every 4th call (more than 20%)
            raise ValueError("Simulated metric failure")
        base_metric = roc_auc_score if SKLEARN_AVAILABLE else lambda yt, yp: np.mean(yt)
        return base_metric(yt, yp)

    # Expect a RuntimeWarning about failed rounds
    with pytest.warns(RuntimeWarning, match="of bootstrap rounds failed"):
        lower, upper = calculate_bootstrap_ci(
            y_true, y_pred, failing_metric, n_rounds=N_ROUNDS, alpha=0.05, seed=SEED
        )
    # Check that results are still produced
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower <= upper


def test_metric_func_fails_always(sample_data):
    """Test behavior when metric_func fails every time."""
    y_true, y_pred = sample_data

    def always_failing_metric(yt, yp):
        raise ValueError("Always fails")

    # Expect a ValueError because the initial sanity check should fail
    with pytest.raises(
        ValueError, match="metric_func failed basic check: Always fails"
    ):
        calculate_bootstrap_ci(
            y_true,
            y_pred,
            always_failing_metric,
            n_rounds=N_ROUNDS,
            alpha=0.05,
            seed=SEED,
        )
