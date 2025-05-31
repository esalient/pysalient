"""
Tests for the _analytical_ci_utils module.
"""

import numpy as np
import pytest
from scipy.stats import norm

# Module to test - catch potential import errors
try:
    # Import from the new internal location
    from pysalient.evaluation import _analytical_ci_utils as anaci
except ImportError:
    pytest.skip(
        "Skipping analytical CI tests: _internal._analytical_ci_utils not found",
        allow_module_level=True,
    )

#############
# CONSTANTS #
#############
ALPHA_95 = 0.05
Z_95 = norm.ppf(1 - ALPHA_95 / 2)  # 1.96


########################
# Test Data & Fixtures #
########################
@pytest.mark.parametrize(
    "successes, total_trials, alpha, expected_lower, expected_upper",
    [
        # Basic cases (alpha=0.05) - Recalculated expected values
        (50, 100, ALPHA_95, 0.4020, 0.5980),  # p=0.5
        (10, 100, ALPHA_95, 0.0412, 0.1588),  # p=0.1
        (90, 100, ALPHA_95, 0.8412, 0.9588),  # p=0.9
        (5, 10, ALPHA_95, 0.1901, 0.8099),  # Smaller n - Adjusted
        (1, 10, ALPHA_95, 0.0000, 0.2860),  # p near 0, clipped lower - Adjusted Upper
        (9, 10, ALPHA_95, 0.7140, 1.0000),  # p near 1, clipped upper - Adjusted Lower
        (0, 10, ALPHA_95, 0.0000, 0.0000),  # Zero successes
        (10, 10, ALPHA_95, 1.0000, 1.0000),  # All successes
        # Different alpha (e.g., 99% CI, alpha=0.01, z=2.576)
        (50, 100, 0.01, 0.3712, 0.6288),
        # Edge case: zero trials
        (0, 0, ALPHA_95, np.nan, np.nan),
    ],
    ids=[
        "p=0.5,n=100",
        "p=0.1,n=100",
        "p=0.9,n=100",
        "p=0.5,n=10",
        "p=0.1,n=10",
        "p=0.9,n=10",
        "p=0.0,n=10",
        "p=1.0,n=10",
        "p=0.5,n=100,alpha=0.01",
        "n=0",
    ],
)
def test_calculate_normal_approx_ci(
    successes, total_trials, alpha, expected_lower, expected_upper
):
    """Test the normal approximation CI calculation."""
    lower, upper = anaci.calculate_normal_approx_ci(successes, total_trials, alpha)

    if np.isnan(expected_lower):
        assert np.isnan(lower)
        assert np.isnan(upper)
    else:
        assert lower == pytest.approx(expected_lower, abs=1e-4)
        assert upper == pytest.approx(expected_upper, abs=1e-4)
    assert lower <= upper or (np.isnan(lower) and np.isnan(upper))


@pytest.mark.parametrize(
    "successes, total_trials, alpha, expected_lower, expected_upper",
    [
        # Basic cases (alpha=0.05) - Recalculated expected values
        (50, 100, ALPHA_95, 0.4039, 0.5961),  # p=0.5
        (10, 100, ALPHA_95, 0.0552, 0.1744),  # p=0.1 - Adjusted
        (90, 100, ALPHA_95, 0.8256, 0.9448),  # p=0.9 - Adjusted
        (5, 10, ALPHA_95, 0.2366, 0.7634),  # Smaller n
        (1, 10, ALPHA_95, 0.0179, 0.4042),  # p near 0 - Adjusted Upper
        (9, 10, ALPHA_95, 0.5958, 0.9821),  # p near 1 - Adjusted Lower
        (0, 10, ALPHA_95, 0.0000, 0.2775),  # Zero successes - Adjusted Upper
        (10, 10, ALPHA_95, 0.7225, 1.0000),  # All successes - Adjusted Lower
        # Different alpha (e.g., 99% CI, alpha=0.01)
        (50, 100, 0.01, 0.3753, 0.6247),  # Adjusted
        # Edge case: zero trials
        (0, 0, ALPHA_95, np.nan, np.nan),
    ],
    ids=[
        "p=0.5,n=100",
        "p=0.1,n=100",
        "p=0.9,n=100",
        "p=0.5,n=10",
        "p=0.1,n=10",
        "p=0.9,n=10",
        "p=0.0,n=10",
        "p=1.0,n=10",
        "p=0.5,n=100,alpha=0.01",
        "n=0",
    ],
)
def test_calculate_wilson_score_ci(
    successes, total_trials, alpha, expected_lower, expected_upper
):
    """Test the Wilson score interval CI calculation."""
    lower, upper = anaci.calculate_wilson_score_ci(successes, total_trials, alpha)

    if np.isnan(expected_lower):
        assert np.isnan(lower)
        assert np.isnan(upper)
    else:
        assert lower == pytest.approx(expected_lower, abs=1e-4)
        assert upper == pytest.approx(expected_upper, abs=1e-4)
    assert lower <= upper or (np.isnan(lower) and np.isnan(upper))


@pytest.mark.parametrize(
    "successes, total_trials, alpha, expected_lower, expected_upper",
    [
        # Basic cases (alpha=0.05) - Recalculated expected values
        (50, 100, ALPHA_95, 0.4038, 0.5962),  # p=0.5
        (10, 100, ALPHA_95, 0.0535, 0.1761),  # p=0.1 - Adjusted
        (90, 100, ALPHA_95, 0.8239, 0.9465),  # p=0.9 - Adjusted
        (5, 10, ALPHA_95, 0.2366, 0.7634),  # Smaller n
        (1, 10, ALPHA_95, 0.0000, 0.4259),  # p near 0 - Adjusted Lower (clipped)
        (9, 10, ALPHA_95, 0.5741, 1.0000),  # p near 1 - Adjusted Upper (clipped)
        (0, 10, ALPHA_95, 0.0000, 0.3209),  # Zero successes - Adjusted Upper
        (10, 10, ALPHA_95, 0.6791, 1.0000),  # All successes - Adjusted Lower
        # Different alpha (e.g., 99% CI, alpha=0.01)
        (50, 100, 0.01, 0.3753, 0.6247),  # Adjusted
        # Edge case: zero trials
        (0, 0, ALPHA_95, np.nan, np.nan),
    ],
    ids=[
        "p=0.5,n=100",
        "p=0.1,n=100",
        "p=0.9,n=100",
        "p=0.5,n=10",
        "p=0.1,n=10",
        "p=0.9,n=10",
        "p=0.0,n=10",
        "p=1.0,n=10",
        "p=0.5,n=100,alpha=0.01",
        "n=0",
    ],
)
def test_calculate_agresti_coull_ci(
    successes, total_trials, alpha, expected_lower, expected_upper
):
    """Test the Agresti-Coull interval CI calculation."""
    lower, upper = anaci.calculate_agresti_coull_ci(successes, total_trials, alpha)

    if np.isnan(expected_lower):
        assert np.isnan(lower)
        assert np.isnan(upper)
    else:
        assert lower == pytest.approx(expected_lower, abs=1e-4)
        assert upper == pytest.approx(expected_upper, abs=1e-4)
    assert lower <= upper or (np.isnan(lower) and np.isnan(upper))


##########################
# Input Validation Tests #
##########################
@pytest.mark.parametrize(
    "successes, total_trials, alpha, error_type, match_str",
    [
        ("50", 100, ALPHA_95, TypeError, "successes must be an integer"),
        (50, "100", ALPHA_95, TypeError, "total_trials must be an integer"),
        (50, 100, "0.05", TypeError, "alpha must be a float"),
        (50, 100, 1.1, ValueError, "alpha must be between 0 and 1"),
        (50, 100, 0.0, ValueError, "alpha must be between 0 and 1"),
        (50, -10, ALPHA_95, ValueError, "total_trials cannot be negative"),
        (-5, 100, ALPHA_95, ValueError, "successes cannot be negative"),
        (
            110,
            100,
            ALPHA_95,
            ValueError,
            "successes cannot be greater than total_trials",
        ),
    ],
)
def test_ci_input_validation(successes, total_trials, alpha, error_type, match_str):
    """Test input validation for CI helper functions."""
    with pytest.raises(error_type, match=match_str):
        anaci.calculate_normal_approx_ci(successes, total_trials, alpha)
    with pytest.raises(error_type, match=match_str):
        anaci.calculate_wilson_score_ci(successes, total_trials, alpha)
    with pytest.raises(error_type, match=match_str):
        anaci.calculate_agresti_coull_ci(successes, total_trials, alpha)
