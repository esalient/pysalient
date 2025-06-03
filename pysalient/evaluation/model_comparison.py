"""
Model comparison utilities for comparing evaluation results from multiple models.
"""

import warnings

import numpy as np
import pyarrow as pa


def compare_models(
    evaluation_results: list[pa.Table],
    model_labels: list[str] | None = None,
    include_metrics: list[str] | None = None,
    decimal_places: int | None = None,
    verbosity: int = 0,
    # Statistical significance testing parameters
    calculate_statistical_significance: bool = False,
    bootstrap_samples: list[dict[str, np.ndarray]] | None = None,
    significance_alpha: float = 0.05,
    n_permutations: int = 10000,
    permutation_seed: int | None = None,
) -> pa.Table:
    """
    Compare evaluation results from multiple models.

    Takes a list of evaluation result tables (from the `evaluation()` function)
    and creates a comparison table in long format with metrics from all models
    side-by-side for easy comparison and analysis.

    Args:
        evaluation_results: List of PyArrow tables from `evaluation()` calls.
                           All tables must have consistent threshold sets and
                           compatible schemas.
        model_labels: Optional list of human-readable labels for the models
                     (e.g., ["LogRegressor", "LightGBM"]). If None, models
                     will be labeled as "Model_1", "Model_2", etc.
                     Length must match evaluation_results length.
        include_metrics: Optional list of metric names to include in comparison.
                        If None, all available metrics are included.
        decimal_places: Optional number of decimal places to round metric values.
                       If None, no rounding is performed.
        verbosity: Controls warning verbosity. 0 shows warnings, >0 suppresses them.
        calculate_statistical_significance: If True, perform statistical significance testing
                                           between pairs of models. Requires bootstrap_samples.
        bootstrap_samples: List of bootstrap sample dictionaries from evaluation() calls.
                          Required if calculate_statistical_significance=True.
                          Each dict should contain metric names as keys and numpy arrays as values.
        significance_alpha: Significance level for statistical tests (default: 0.05).
        n_permutations: Number of permutations for permutation test (default: 10000).
        permutation_seed: Random seed for permutation test reproducibility.

    Returns:
        PyArrow Table with columns:
        - threshold: The evaluation threshold
        - model: Model identifier (from model_labels or auto-generated)
        - metric: Metric name (AUROC, AUPRC, F1_Score, etc.)
        - value: The metric value
        - lower_ci: Lower confidence interval (null if not available)
        - upper_ci: Upper confidence interval (null if not available)
        - p_value: Statistical significance p-value (null if not calculated)

        Each row represents one metric value for one model at one threshold.
        Threshold-independent metrics (AUROC, AUPRC) are duplicated across
        all thresholds for consistency.

    Raises:
        ValueError: If inputs are invalid, tables have incompatible schemas,
                   or threshold sets don't match.
        TypeError: If input types are incorrect.

    Example:
        >>> # Run evaluations for two models
        >>> results_1 = evaluation(data, "model1", "test", [0.1, 0.5, 0.9],
        ...                        y_proba_col="prediction_proba_1")
        >>> results_2 = evaluation(data, "model2", "test", [0.1, 0.5, 0.9],
        ...                        y_proba_col="prediction_proba_2")
        >>>
        >>> # Compare models
        >>> comparison = compare_models(
        ...     [results_1, results_2],
        ...     model_labels=["LogRegressor", "LightGBM"],
        ...     include_metrics=["AUROC", "AUPRC", "F1_Score"],
        ...     decimal_places=3
        ... )
        >>>
        >>> # Filter for specific comparisons
        >>> auroc_comparison = comparison.filter(
        ...     pa.compute.equal(comparison["metric"], "AUROC")
        ... )
    """
    # Input validation
    if not isinstance(evaluation_results, list) or len(evaluation_results) == 0:
        raise ValueError(
            "evaluation_results must be a non-empty list of PyArrow tables."
        )

    if len(evaluation_results) < 2:
        raise ValueError("At least 2 evaluation results are required for comparison.")

    for i, result in enumerate(evaluation_results):
        if not isinstance(result, pa.Table):
            raise TypeError(f"evaluation_results[{i}] must be a PyArrow Table.")

    # Validate model_labels
    if model_labels is not None:
        if not isinstance(model_labels, list):
            raise TypeError("model_labels must be a list of strings or None.")
        if len(model_labels) != len(evaluation_results):
            raise ValueError(
                f"model_labels length ({len(model_labels)}) must match "
                f"evaluation_results length ({len(evaluation_results)})."
            )
        for i, label in enumerate(model_labels):
            if not isinstance(label, str):
                raise TypeError(f"model_labels[{i}] must be a string.")
    else:
        model_labels = [f"Model_{i + 1}" for i in range(len(evaluation_results))]

    # Validate metric filter parameters
    if include_metrics is not None:
        if not isinstance(include_metrics, list):
            raise TypeError("include_metrics must be a list of strings or None.")
        for metric in include_metrics:
            if not isinstance(metric, str):
                raise TypeError("All items in include_metrics must be strings.")

    # Validate decimal_places
    if decimal_places is not None:
        if not isinstance(decimal_places, int) or decimal_places < 0:
            raise ValueError("decimal_places must be a non-negative integer or None.")

    # Validate statistical significance parameters
    if not isinstance(calculate_statistical_significance, bool):
        raise TypeError("calculate_statistical_significance must be a boolean.")
    
    if calculate_statistical_significance:
        if bootstrap_samples is None:
            raise ValueError(
                "bootstrap_samples is required when calculate_statistical_significance=True."
            )
        if not isinstance(bootstrap_samples, list):
            raise TypeError("bootstrap_samples must be a list of dictionaries.")
        if len(bootstrap_samples) != len(evaluation_results):
            raise ValueError(
                f"bootstrap_samples length ({len(bootstrap_samples)}) must match "
                f"evaluation_results length ({len(evaluation_results)})."
            )
        for i, samples_dict in enumerate(bootstrap_samples):
            if not isinstance(samples_dict, dict):
                raise TypeError(f"bootstrap_samples[{i}] must be a dictionary.")
        
        if not isinstance(significance_alpha, float) or not 0 < significance_alpha < 1:
            raise ValueError("significance_alpha must be a float between 0 and 1 (exclusive).")
        
        if not isinstance(n_permutations, int) or n_permutations <= 0:
            raise ValueError("n_permutations must be a positive integer.")
        
        if permutation_seed is not None and not isinstance(permutation_seed, int):
            raise TypeError("permutation_seed must be an integer or None.")

    # Validate table schemas and get common columns
    first_table = evaluation_results[0]
    required_columns = {"threshold", "modelid", "filter_desc"}

    # Check that all tables have required columns
    for i, table in enumerate(evaluation_results):
        missing_cols = required_columns - set(table.column_names)
        if missing_cols:
            raise ValueError(
                f"evaluation_results[{i}] missing required columns: {missing_cols}"
            )

    # Get all metric columns (exclude non-metric columns)
    non_metric_columns = {
        "threshold",
        "modelid",
        "filter_desc",
        "time_to_first_alert_value",
        "time_to_first_alert_unit",
        "Sample_Size",
        "Label_Count",
    }

    available_metrics = []
    ci_columns = set()

    for col_name in first_table.column_names:
        if col_name not in non_metric_columns:
            if col_name.endswith("_Lower_CI") or col_name.endswith("_Upper_CI"):
                ci_columns.add(col_name)
            else:
                available_metrics.append(col_name)

    # Apply metric filtering
    if include_metrics is not None:
        # Check that requested metrics exist
        missing_metrics = set(include_metrics) - set(available_metrics)
        if missing_metrics:
            raise ValueError(f"Requested metrics not found: {missing_metrics}")
        metrics_to_use = include_metrics
    else:
        metrics_to_use = available_metrics

    if not metrics_to_use:
        raise ValueError("No metrics selected for comparison.")

    # Calculate statistical significance if requested (must be done before building comparison data)
    p_values_dict = {}
    if calculate_statistical_significance:
        p_values_dict = _calculate_pairwise_p_values(
            bootstrap_samples=bootstrap_samples,
            model_labels=model_labels,
            metrics_to_use=metrics_to_use,
            n_permutations=n_permutations,
            permutation_seed=permutation_seed,
            verbosity=verbosity,
        )

    # Validate threshold consistency across tables
    threshold_arrays = []
    for i, table in enumerate(evaluation_results):
        thresholds = table["threshold"].to_numpy()
        threshold_arrays.append(thresholds)

        if i > 0:
            if not np.array_equal(thresholds, threshold_arrays[0]):
                if verbosity <= 0:
                    warnings.warn(
                        f"Threshold mismatch between evaluation_results[0] and evaluation_results[{i}]. "
                        "This may cause unexpected results in the comparison.",
                        UserWarning,
                    )

    # Build comparison data
    comparison_data = []

    for model_idx, (table, model_label) in enumerate(
        zip(evaluation_results, model_labels)
    ):
        thresholds = table["threshold"].to_numpy()

        for metric in metrics_to_use:
            if metric not in table.column_names:
                if verbosity <= 0:
                    warnings.warn(
                        f"Metric '{metric}' not found in evaluation_results[{model_idx}] "
                        f"(model: {model_label}). Skipping this metric for this model.",
                        UserWarning,
                    )
                continue

            metric_values = table[metric].to_numpy()

            # Get CI columns if they exist
            lower_ci_col = f"{metric}_Lower_CI"
            upper_ci_col = f"{metric}_Upper_CI"

            lower_ci_values = None
            upper_ci_values = None

            if lower_ci_col in table.column_names:
                lower_ci_values = table[lower_ci_col].to_numpy()
            if upper_ci_col in table.column_names:
                upper_ci_values = table[upper_ci_col].to_numpy()

            # Create rows for each threshold
            for thresh_idx, threshold in enumerate(thresholds):
                value = metric_values[thresh_idx]
                lower_ci = (
                    lower_ci_values[thresh_idx] if lower_ci_values is not None else None
                )
                upper_ci = (
                    upper_ci_values[thresh_idx] if upper_ci_values is not None else None
                )

                # Apply rounding if specified
                if decimal_places is not None:
                    if not (
                        np.isnan(value)
                        if isinstance(value, int | float)
                        else value is None
                    ):
                        value = round(float(value), decimal_places)
                    if lower_ci is not None and not (
                        np.isnan(lower_ci)
                        if isinstance(lower_ci, int | float)
                        else lower_ci is None
                    ):
                        lower_ci = round(float(lower_ci), decimal_places)
                    if upper_ci is not None and not (
                        np.isnan(upper_ci)
                        if isinstance(upper_ci, int | float)
                        else upper_ci is None
                    ):
                        upper_ci = round(float(upper_ci), decimal_places)

                # Get p-value for this model-metric combination
                p_value = None
                if calculate_statistical_significance:
                    p_value = p_values_dict.get((model_label, metric), None)

                comparison_data.append(
                    {
                        "threshold": float(threshold),
                        "model": model_label,
                        "metric": metric,
                        "value": value,
                        "lower_ci": lower_ci,
                        "upper_ci": upper_ci,
                        "p_value": p_value,
                    }
                )

    # Create result schema and table
    result_schema = pa.schema(
        [
            pa.field("threshold", pa.float64()),
            pa.field("model", pa.string()),
            pa.field("metric", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("lower_ci", pa.float64()),
            pa.field("upper_ci", pa.float64()),
            pa.field("p_value", pa.float64()),
        ]
    )

    try:
        result_table = pa.Table.from_pylist(comparison_data, schema=result_schema)
    except Exception as e:
        raise RuntimeError(f"Failed to create comparison table: {e}") from e

    return result_table


def _perform_permutation_test(
    samples_1: np.ndarray,
    samples_2: np.ndarray,
    n_permutations: int = 10000,
    seed: int | None = None,
) -> float:
    """
    Perform a two-sample permutation test to assess statistical significance.
    
    This test determines if two groups of bootstrap samples come from distributions
    with the same mean. The null hypothesis is that the two groups have equal means.
    
    Args:
        samples_1: Bootstrap samples from first model (e.g., AUROC values)
        samples_2: Bootstrap samples from second model (e.g., AUROC values)
        n_permutations: Number of permutations to perform (default: 10000)
        seed: Random seed for reproducibility
        
    Returns:
        p_value: Two-tailed p-value for the permutation test
        
    Raises:
        ValueError: If samples are empty or have invalid values
        TypeError: If inputs are not numpy arrays
    """
    # Input validation
    if not isinstance(samples_1, np.ndarray) or not isinstance(samples_2, np.ndarray):
        raise TypeError("Both samples must be numpy arrays")
    
    if len(samples_1) == 0 or len(samples_2) == 0:
        raise ValueError("Sample arrays cannot be empty")
    
    # Remove NaN values
    samples_1_clean = samples_1[~np.isnan(samples_1)]
    samples_2_clean = samples_2[~np.isnan(samples_2)]
    
    if len(samples_1_clean) == 0 or len(samples_2_clean) == 0:
        warnings.warn(
            "All samples contain NaN values. Cannot perform permutation test.",
            RuntimeWarning
        )
        return np.nan
    
    # Calculate observed difference in means
    observed_diff = np.mean(samples_1_clean) - np.mean(samples_2_clean)
    
    # Combine all samples
    combined_samples = np.concatenate([samples_1_clean, samples_2_clean])
    n1, n2 = len(samples_1_clean), len(samples_2_clean)
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Perform permutations
    permuted_diffs = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Randomly shuffle combined samples
        shuffled = np.random.permutation(combined_samples)
        
        # Split into two groups of original sizes
        perm_group1 = shuffled[:n1]
        perm_group2 = shuffled[n1:n1+n2]
        
        # Calculate difference in means for this permutation
        permuted_diffs[i] = np.mean(perm_group1) - np.mean(perm_group2)
    
    # Calculate two-tailed p-value
    # Count how many permuted differences are as extreme or more extreme than observed
    extreme_count = np.sum(np.abs(permuted_diffs) >= np.abs(observed_diff))
    p_value = extreme_count / n_permutations
    
    return float(p_value)


def _calculate_pairwise_p_values(
    bootstrap_samples: list[dict[str, np.ndarray]],
    model_labels: list[str],
    metrics_to_use: list[str],
    n_permutations: int,
    permutation_seed: int | None,
    verbosity: int,
) -> dict[tuple[str, str], float]:
    """
    Calculate pairwise statistical significance between models for each metric.
    
    For now, this compares only the first two models. In the future, this could
    be extended to handle all pairwise comparisons.
    
    Args:
        bootstrap_samples: List of bootstrap sample dictionaries
        model_labels: List of model labels
        metrics_to_use: List of metrics to test
        n_permutations: Number of permutations for the test
        permutation_seed: Random seed for reproducibility
        verbosity: Verbosity level for warnings
        
    Returns:
        Dictionary with (model_label, metric) keys and p-values as values
    """
    p_values_dict = {}
    
    # For now, we only support comparing exactly 2 models
    if len(bootstrap_samples) != 2:
        if verbosity <= 0:
            warnings.warn(
                f"Statistical significance testing currently only supports exactly 2 models. "
                f"Found {len(bootstrap_samples)} models. Skipping significance testing.",
                UserWarning
            )
        return p_values_dict
    
    model_1_samples = bootstrap_samples[0]
    model_2_samples = bootstrap_samples[1]
    model_1_label = model_labels[0]
    model_2_label = model_labels[1]
    
    for metric in metrics_to_use:
        # Check if both models have bootstrap samples for this metric
        if metric not in model_1_samples or metric not in model_2_samples:
            if verbosity <= 0:
                warnings.warn(
                    f"Metric '{metric}' not found in bootstrap samples for both models. "
                    "Skipping significance test for this metric.",
                    UserWarning
                )
            continue
        
        try:
            # Perform permutation test
            p_value = _perform_permutation_test(
                samples_1=model_1_samples[metric],
                samples_2=model_2_samples[metric],
                n_permutations=n_permutations,
                seed=permutation_seed,
            )
            
            # Store p-value for both models (they get the same p-value)
            p_values_dict[(model_1_label, metric)] = p_value
            p_values_dict[(model_2_label, metric)] = p_value
            
        except Exception as e:
            if verbosity <= 0:
                warnings.warn(
                    f"Failed to calculate statistical significance for metric '{metric}': {e}",
                    RuntimeWarning
                )
            continue
    
    return p_values_dict
