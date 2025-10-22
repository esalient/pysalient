"""
Internal utility functions for bootstrapping calculations.
"""

import logging
from collections.abc import Callable

import numpy as np

# Configure logging - Get logger, but don't configure basicConfig here
# Configuration should happen at the application level
logger = logging.getLogger(__name__)

# Attempt to import scikit-learn metrics for type hinting/checking if needed
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    # Allow import, functions using it will handle the missing dependency
    roc_auc_score = None
    average_precision_score = None


def calculate_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_rounds: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
    verbosity: int = 0,  # Add verbosity parameter
) -> tuple[float, float]:
    """
    Calculates bootstrap confidence intervals for a given metric function.

    Performs non-parametric bootstrapping by resampling the input arrays
    with replacement `n_rounds` times, calculating the metric on each sample,
    and determining the confidence interval from the distribution of results.

    Args:
        y_true: NumPy array of true binary labels (0 or 1).
        y_pred: NumPy array of predicted probabilities or scores.
        metric_func: A callable function that accepts y_true and y_pred
                     (in that order) and returns a single float metric value
                     (e.g., sklearn.metrics.roc_auc_score).
        n_rounds: The number of bootstrap samples to draw. Defaults to 1000.
        alpha: The significance level for the confidence interval.
               Defaults to 0.05 for a 95% CI (1 - alpha).
               Must be in the range (0, 1).
        seed: Optional random seed for reproducibility of bootstrap sampling.
        verbosity: Controls logging level: <= -1 (INFO+), 0 (WARN+), >= 1 (ERROR only).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.

    Raises:
        ValueError: If alpha is not in the range (0, 1), or if input arrays
                    are empty or have mismatched lengths.
        TypeError: If inputs are not NumPy arrays or metric_func is not callable.
        Exception: Propagates exceptions raised by the metric_func during calculation.
    """
    ####################
    # Input Validation #
    ####################
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be NumPy arrays.")
    if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
        raise ValueError("Input arrays cannot be empty.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Input arrays y_true and y_pred must have the same length.")
    if not callable(metric_func):
        raise TypeError("metric_func must be a callable function.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive).")
    if not isinstance(n_rounds, int) or n_rounds <= 0:
        raise ValueError("n_rounds must be a positive integer.")
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be an integer or None.")

    # Check if metric_func works on a small sample: basic check
    # This helps catch immediate issues like incompatible function signatures
    try:
        sample_indices = np.random.choice(
            np.arange(min(5, y_true.shape[0])),
            size=min(5, y_true.shape[0]),
            replace=True,
        )
        check_output = metric_func(y_true[sample_indices], y_pred[sample_indices])
        if not isinstance(check_output, float | int | np.number):
            raise ValueError("metric_func must return a single numeric value.")
    except Exception as e:
        raise ValueError(f"metric_func failed basic check: {e}") from e
    #################
    # Bootstrapping #
    #################
    rng = np.random.RandomState(seed)
    bootstrap_replicates = np.zeros(shape=n_rounds, dtype=float)
    sample_indices = np.arange(y_true.shape[0])

    for i in range(n_rounds):
        bootstrap_idx = rng.choice(
            sample_indices, size=sample_indices.shape[0], replace=True
        )

        # Log bootstrap sample class distribution
        bootstrap_labels = y_true[bootstrap_idx]
        pos_count = np.sum(bootstrap_labels == 1)
        neg_count = np.sum(bootstrap_labels == 0)
        # Conditional logging based on verbosity
        if verbosity <= -2:  # Use DEBUG level for very detailed info
            logger.debug(
                f"Bootstrap round {i + 1}: Sample distribution - "
                f"Positives: {pos_count} ({pos_count / len(bootstrap_labels):.1%}), "
                f"Negatives: {neg_count} ({neg_count / len(bootstrap_labels):.1%})"
            )

        # Handle potential errors during metric calculation within the loop
        metric_result = np.nan  # Default to NaN
        try:
            metric_result = metric_func(y_true[bootstrap_idx], y_pred[bootstrap_idx])
            # Log metric result details based on verbosity
            if np.isnan(metric_result):
                # Log NaN results at INFO level if verbosity allows
                if verbosity <= -1:
                    logger.info(
                        f"Bootstrap round {i + 1}: Metric func '{metric_func.__name__}' "
                        f"returned NaN with {pos_count} positives and {neg_count} negatives"
                    )
            else:
                # Log successful results at DEBUG level if verbosity allows
                if verbosity <= -2:
                    logger.debug(
                        f"Bootstrap round {i + 1}: Metric func '{metric_func.__name__}' "
                        f"returned {metric_result:.3f}"
                    )

        except Exception as e:
            # Log warnings only if verbosity allows (<= 0)
            if verbosity <= 0:
                logger.warning(
                    f"Bootstrap round {i + 1}: Metric func '{metric_func.__name__}' failed "
                    f"with error: {type(e).__name__} - {e}. "
                    f"Sample had {pos_count} positives and {neg_count} negatives. "
                    "Storing NaN.",
                    exc_info=(
                        verbosity <= -1
                    ),  # Include stack trace only at INFO level or lower
                )
            metric_result = np.nan

        bootstrap_replicates[i] = metric_result

    # Filter out potential NaNs if metric calculation failed in some rounds
    valid_replicates = bootstrap_replicates[~np.isnan(bootstrap_replicates)]
    failed_rounds = n_rounds - len(valid_replicates)
    failure_rate = failed_rounds / n_rounds

    # Log detailed statistics about valid/invalid results (INFO level)
    if verbosity <= -1:
        logger.info(
            f"Bootstrap Results Summary for {metric_func.__name__}:\n"
            f"- Total Rounds: {n_rounds}\n"
            f"- Failed Rounds: {failed_rounds} ({failure_rate:.1%})\n"
            f"- Valid Results: {len(valid_replicates)}\n"
            f"- Valid Range: [{np.min(valid_replicates):.3f}, {np.max(valid_replicates):.3f}]"
            if len(valid_replicates) > 0
            else "- No valid results"
        )

    if failure_rate > 0.2:  # Warn if >20% of rounds failed
        # Use logger.warning if verbosity allows (<= 0)
        if verbosity <= 0:
            logger.warning(
                f"{failure_rate:.1%} of bootstrap rounds failed metric calculation for {metric_func.__name__}. "
                f"Only {len(valid_replicates)} valid results used for CI estimation."
            )
        # Keep the original warnings.warn as well for broader visibility if not suppressed
        if verbosity <= 0:  # Only show Python warning if logger warning is also shown
            import warnings

            warnings.warn(
                f"{failure_rate:.1%} of bootstrap rounds failed metric calculation for {metric_func.__name__}. "
                f"Only {len(valid_replicates)} valid results for CI estimation.",
                RuntimeWarning,
            )

    if len(valid_replicates) == 0:
        # If no valid replicates, CI cannot be calculated (ERROR level)
        # Errors are always logged regardless of verbosity >= 1
        logger.error(
            f"Metric func '{metric_func.__name__}': All bootstrap rounds failed calculation; "
            "cannot compute CI. This may indicate an issue with the metric calculation "
            "or the bootstrap sample characteristics."
        )
        return (np.nan, np.nan)

    # Log mean and standard deviation of valid replicates (INFO level)
    if verbosity <= -1:
        logger.info(
            f"Bootstrap Replicates Distribution for {metric_func.__name__}:\n"
            f"- Mean: {np.mean(valid_replicates):.3f}\n"
            f"- Standard Deviation: {np.std(valid_replicates):.3f}"
        )

    #####################################################
    # Calculate Confidence Interval (Percentile Method) #
    #####################################################
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1 - alpha / 2.0) * 100

    # Use np.percentile for calculation
    lower_ci = np.percentile(valid_replicates, lower_percentile)
    upper_ci = np.percentile(valid_replicates, upper_percentile)

    # Log calculated CI bounds (INFO level)
    if verbosity <= -1:
        logger.info(
            f"Calculated CI for {metric_func.__name__} (alpha={alpha}): [{lower_ci:.3f}, {upper_ci:.3f}]"
        )

    # Clip results to handle potential floating point inaccuracies near 0 or 1
    lower_ci_clipped = np.clip(lower_ci, 0.0, 1.0)
    upper_ci_clipped = np.clip(upper_ci, 0.0, 1.0)

    return float(lower_ci_clipped), float(upper_ci_clipped)
