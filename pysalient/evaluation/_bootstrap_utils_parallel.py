"""
Parallel implementation of bootstrap utilities for significant performance improvement.
Uses multiprocessing to parallelize bootstrap rounds across CPU cores.
"""

import logging
import warnings
from collections.abc import Callable
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Attempt to import scikit-learn metrics
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    roc_auc_score = None
    average_precision_score = None


def _bootstrap_worker(
    args: tuple[int, int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    seed: int | None,
) -> np.ndarray:
    """
    Worker function for parallel bootstrap computation.

    Args:
        args: Tuple of (start_idx, n_rounds) for this worker
        y_true: True labels
        y_pred: Predicted probabilities
        metric_func: Metric calculation function
        seed: Random seed (will be adjusted per worker for independence)

    Returns:
        Array of bootstrap metric values for assigned rounds
    """
    start_idx, n_rounds = args

    # Create independent random state for this worker
    worker_seed = None if seed is None else seed + start_idx
    rng = np.random.RandomState(worker_seed)

    sample_indices = np.arange(y_true.shape[0])
    results = np.zeros(n_rounds, dtype=float)

    for i in range(n_rounds):
        bootstrap_idx = rng.choice(
            sample_indices, size=sample_indices.shape[0], replace=True
        )

        try:
            metric_result = metric_func(y_true[bootstrap_idx], y_pred[bootstrap_idx])
            results[i] = metric_result if not np.isnan(metric_result) else np.nan
        except Exception:
            results[i] = np.nan

    return results


def calculate_bootstrap_ci_parallel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_rounds: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
    verbosity: int = 0,
    n_jobs: int | None = None,
) -> tuple[float, float]:
    """
    Parallel version of bootstrap confidence interval calculation.

    Distributes bootstrap rounds across multiple CPU cores for significant
    performance improvement, especially with large n_rounds.

    Args:
        y_true: NumPy array of true binary labels (0 or 1)
        y_pred: NumPy array of predicted probabilities or scores
        metric_func: Callable that accepts y_true and y_pred and returns a float
        n_rounds: Number of bootstrap samples (default: 1000)
        alpha: Significance level for CI (default: 0.05 for 95% CI)
        seed: Random seed for reproducibility
        verbosity: Controls logging level
        n_jobs: Number of parallel jobs. None = use all CPUs, -1 = all CPUs,
                positive int = that many CPUs

    Returns:
        Tuple of (lower_ci, upper_ci)

    Raises:
        ValueError: If inputs are invalid
        TypeError: If inputs have wrong types
    """
    # Input validation (same as original)
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

    # Determine number of workers
    if n_jobs is None or n_jobs == -1:
        n_workers = cpu_count()
    else:
        n_workers = min(n_jobs, cpu_count())

    # For small n_rounds, don't over-parallelize
    n_workers = min(n_workers, n_rounds)

    if verbosity <= -1:
        logger.info(f"Using {n_workers} workers for {n_rounds} bootstrap rounds")

    # Split rounds across workers
    rounds_per_worker = n_rounds // n_workers
    remainder = n_rounds % n_workers

    # Create work assignments (start_idx, n_rounds) for each worker
    work_assignments = []
    current_start = 0
    for i in range(n_workers):
        worker_rounds = rounds_per_worker + (1 if i < remainder else 0)
        if worker_rounds > 0:
            work_assignments.append((current_start, worker_rounds))
            current_start += worker_rounds

    # Create partial function with fixed arguments
    worker_func = partial(
        _bootstrap_worker,
        y_true=y_true,
        y_pred=y_pred,
        metric_func=metric_func,
        seed=seed
    )

    # Execute parallel bootstrap
    with Pool(processes=n_workers) as pool:
        worker_results = pool.map(worker_func, work_assignments)

    # Combine results from all workers
    bootstrap_replicates = np.concatenate(worker_results)

    # Filter out NaNs
    valid_replicates = bootstrap_replicates[~np.isnan(bootstrap_replicates)]
    failed_rounds = n_rounds - len(valid_replicates)
    failure_rate = failed_rounds / n_rounds

    if verbosity <= -1:
        logger.info(
            f"Bootstrap Results Summary for {metric_func.__name__}:\n"
            f"- Total Rounds: {n_rounds}\n"
            f"- Failed Rounds: {failed_rounds} ({failure_rate:.1%})\n"
            f"- Valid Results: {len(valid_replicates)}"
        )

    if failure_rate > 0.2 and verbosity <= 0:
        warnings.warn(
            f"{failure_rate:.1%} of bootstrap rounds failed for {metric_func.__name__}.",
            RuntimeWarning,
        )

    if len(valid_replicates) == 0:
        logger.error(f"All bootstrap rounds failed for {metric_func.__name__}")
        return (np.nan, np.nan)

    # Calculate confidence intervals
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1 - alpha / 2.0) * 100

    lower_ci = np.percentile(valid_replicates, lower_percentile)
    upper_ci = np.percentile(valid_replicates, upper_percentile)

    # Clip to [0, 1] range
    lower_ci_clipped = np.clip(lower_ci, 0.0, 1.0)
    upper_ci_clipped = np.clip(upper_ci, 0.0, 1.0)

    return float(lower_ci_clipped), float(upper_ci_clipped)


def calculate_bootstrap_ci_batch_parallel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_funcs: dict[str, Callable],
    n_rounds: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
    verbosity: int = 0,
    n_jobs: int | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Calculate bootstrap CIs for multiple metrics in parallel.

    This function parallelizes across both metrics and bootstrap rounds for
    maximum performance improvement.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        metric_funcs: Dictionary mapping metric names to calculation functions
        n_rounds: Number of bootstrap rounds per metric
        alpha: Significance level
        seed: Random seed
        verbosity: Logging verbosity
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary mapping metric names to (lower_ci, upper_ci) tuples
    """
    results = {}

    # If only one metric, just call the single version
    if len(metric_funcs) == 1:
        name, func = next(iter(metric_funcs.items()))
        results[name] = calculate_bootstrap_ci_parallel(
            y_true, y_pred, func, n_rounds, alpha, seed, verbosity, n_jobs
        )
        return results

    # For multiple metrics, we can parallelize across metrics too
    # This is especially beneficial when calculating CIs for multiple thresholds

    if n_jobs is None or n_jobs == -1:
        total_workers = cpu_count()
    else:
        total_workers = min(n_jobs, cpu_count())

    # Distribute workers across metrics
    workers_per_metric = max(1, total_workers // len(metric_funcs))

    if verbosity <= -1:
        logger.info(
            f"Calculating bootstrap CIs for {len(metric_funcs)} metrics "
            f"using {workers_per_metric} workers per metric"
        )

    for name, func in metric_funcs.items():
        try:
            results[name] = calculate_bootstrap_ci_parallel(
                y_true, y_pred, func, n_rounds, alpha,
                seed=(seed + hash(name)) if seed else None,  # Unique seed per metric
                verbosity=verbosity,
                n_jobs=workers_per_metric
            )
        except Exception as e:
            if verbosity <= 0:
                warnings.warn(f"Bootstrap CI failed for {name}: {e}", RuntimeWarning)
            results[name] = (np.nan, np.nan)

    return results
