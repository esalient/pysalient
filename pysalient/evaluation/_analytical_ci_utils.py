"""
Internal utility functions for calculating analytical confidence intervals for proportions.
"""

import numpy as np
from scipy.stats import norm

# Define a small epsilon to prevent division by zero or log(0) issues
EPSILON = 1e-9


def _validate_proportion_ci_inputs(successes: int, total_trials: int, alpha: float):
    """Validates inputs common to proportion CI functions."""
    if not isinstance(successes, int | np.integer):
        raise TypeError("successes must be an integer.")
    if not isinstance(total_trials, int | np.integer):
        raise TypeError("total_trials must be an integer.")
    if not isinstance(alpha, float | np.floating):
        raise TypeError("alpha must be a float.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive).")
    if total_trials < 0:
        raise ValueError("total_trials cannot be negative.")
    if successes < 0:
        raise ValueError("successes cannot be negative.")
    if successes > total_trials:
        raise ValueError("successes cannot be greater than total_trials.")


def calculate_normal_approx_ci(
    successes: int, total_trials: int, alpha: float
) -> tuple[float, float]:
    """
    Calculates the confidence interval for a proportion using the normal approximation
    (Wald interval).

    Args:
        successes: Number of successful outcomes.
        total_trials: Total number of trials.
        alpha: Significance level (e.g., 0.05 for a 95% CI).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
        Returns (np.nan, np.nan) if total_trials is 0.
    """
    _validate_proportion_ci_inputs(successes, total_trials, alpha)

    if total_trials == 0:
        return (np.nan, np.nan)

    p_hat = successes / total_trials
    z = norm.ppf(1 - alpha / 2)
    se = np.sqrt(p_hat * (1 - p_hat) / total_trials)

    lower = p_hat - z * se
    upper = p_hat + z * se

    # Clip results to [0, 1]
    lower_clipped = np.clip(lower, 0.0, 1.0)
    upper_clipped = np.clip(upper, 0.0, 1.0)

    return float(lower_clipped), float(upper_clipped)


def calculate_wilson_score_ci(
    successes: int, total_trials: int, alpha: float
) -> tuple[float, float]:
    """
    Calculates the confidence interval for a proportion using the Wilson score interval.
    This method provides better coverage than the normal approximation, especially
    for small n or p close to 0 or 1.

    Args:
        successes: Number of successful outcomes.
        total_trials: Total number of trials.
        alpha: Significance level (e.g., 0.05 for a 95% CI).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
        Returns (np.nan, np.nan) if total_trials is 0.
    """
    _validate_proportion_ci_inputs(successes, total_trials, alpha)

    if total_trials == 0:
        return (np.nan, np.nan)

    p_hat = successes / total_trials
    z = norm.ppf(1 - alpha / 2)
    z_squared = z**2

    denominator = 1 + z_squared / total_trials
    center_adjustment = z_squared / (2 * total_trials)

    adjusted_p = p_hat + center_adjustment

    interval_width_term = z * np.sqrt(
        (p_hat * (1 - p_hat) / total_trials) + (z_squared / (4 * total_trials**2))
    )

    lower = (adjusted_p - interval_width_term) / denominator
    upper = (adjusted_p + interval_width_term) / denominator

    # Clip results to [0, 1]
    lower_clipped = np.clip(lower, 0.0, 1.0)
    upper_clipped = np.clip(upper, 0.0, 1.0)

    return float(lower_clipped), float(upper_clipped)


def calculate_agresti_coull_ci(
    successes: int, total_trials: int, alpha: float
) -> tuple[float, float]:
    """
    Calculates the confidence interval for a proportion using the Agresti-Coull interval.
    This method adjusts the number of successes and trials before applying the
    normal approximation formula, improving performance for small sample sizes.

    Args:
        successes: Number of successful outcomes.
        total_trials: Total number of trials.
        alpha: Significance level (e.g., 0.05 for a 95% CI).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
        Returns (np.nan, np.nan) if total_trials is 0.
    """
    _validate_proportion_ci_inputs(successes, total_trials, alpha)

    if total_trials == 0:
        return (np.nan, np.nan)

    z = norm.ppf(1 - alpha / 2)
    z_squared = z**2

    # Adjusted number of successes and trials
    n_tilde = total_trials + z_squared
    p_tilde = (successes + z_squared / 2) / n_tilde

    # Use normal approximation formula with adjusted values
    se_tilde = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)

    lower = p_tilde - z * se_tilde
    upper = p_tilde + z * se_tilde

    # Clip results to [0, 1]
    lower_clipped = np.clip(lower, 0.0, 1.0)
    upper_clipped = np.clip(upper, 0.0, 1.0)

    return float(lower_clipped), float(upper_clipped)
