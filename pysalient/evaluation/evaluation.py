"""
Evaluation module for calculating performance metrics.
"""

import json
import warnings

import numpy as np
import pyarrow as pa

# Attempt to import scikit-learn metrics, raising an informative error if not found.
try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
    )
except ImportError:
    # Allow the module to be imported, but the function will fail if sklearn is needed.
    roc_auc_score = None
    average_precision_score = None
    accuracy_score = None
    f1_score = None
    # Handle verbosity for this warning
    # if verbosity <= -1: # This check needs to be inside the function where verbosity is available
    #     logging.info("scikit-learn not found...") # Example using logging
    # if verbosity <= 0: # This check needs to be inside the function where verbosity is available
    #     warnings.warn(
    #         "scikit-learn not found. The 'evaluation' function requires scikit-learn "
    #         "for some metric calculations. Please install it (`pip install scikit-learn`).",
    #         ImportWarning,
    #     )
    SKLEARN_AVAILABLE = False  # Define based on import success
else:
    SKLEARN_AVAILABLE = True

# Import internal helpers using absolute paths
from ._evaluation_process import _process_single_evaluation
from ._evaluation_utils import _generate_thresholds

# Import CI utilities using absolute paths
try:
    from ._bootstrap_utils import calculate_bootstrap_ci
except ImportError:
    calculate_bootstrap_ci = None

try:
    from . import _analytical_ci_utils as anaci
except ImportError:
    anaci = None

# Check if at least one CI method is available if requested later
_ci_utils_available = calculate_bootstrap_ci is not None or anaci is not None


# Define standard metadata keys (using strings for broader compatibility, bytes are also fine)
META_KEY_Y_PROBA = "pysalient.io.y_proba_col"
META_KEY_Y_LABEL = "pysalient.io.y_label_col"
META_KEY_AGGREGATION_COLS = "pysalient.io.aggregation_cols"
META_KEY_TIMESERIES_COL = "pysalient.io.timeseries_col"


# This is the public API function
def evaluation(
    data: pa.Table,
    modelid: str,
    filter_desc: str,
    thresholds: list[float] | tuple[float, ...] | tuple[float, float, float],
    time_to_event_cols: dict[str, str] | None = None,  # NEW: Clinical event columns for time-to-event metrics
    aggregation_func: str = "median",  # NEW: Aggregation function for time-to-event metrics
    time_unit: str = "hours",  # NEW: Time unit for column naming in time-to-event metrics
    time_to_event_fillna: float | None = None,  # NEW: Fill NaN values in time-to-event metrics
    force_threshold_zero: bool = True,
    decimal_places: int | None = None,
    verbosity: int = 0,  # Add verbosity parameter
    force_eval: bool = False,  # Force evaluation even if more than 10 thresholds
    #################
    # CI Parameters #
    #################
    calculate_au_ci: bool = False,  # Calculate Area Under Curve CI (AUROC/AUPRC) using bootstrap
    calculate_threshold_ci: bool = False,  # Calculate threshold-specific metric CIs (PPV, Sens, Spec, etc.)
    threshold_ci_method: str = "bootstrap",  # Method for threshold CIs ('bootstrap', 'normal', 'wilson', 'agresti-coull')
    ci_alpha: float = 0.05,  # Significance level for ALL CIs
    bootstrap_rounds: int = 1000,  # Used if calculate_au_ci=True OR (calculate_threshold_ci=True and threshold_ci_method='bootstrap')
    bootstrap_seed: (
        int | None
    ) = None,  # Used if calculate_au_ci=True OR (calculate_threshold_ci=True and threshold_ci_method='bootstrap')
) -> pa.Table:
    """
    Performs evaluation on prediction data across multiple thresholds.

    Calculates performance metrics across multiple thresholds on the input data.
    This function assumes the input `data` table contains event-level predictions
    or has already been aggregated upstream (e.g., by `pysalient.io.reader`)
    if group-level evaluation is desired. It no longer performs aggregation internally.

    Reads required column names (probability, label) from the input table's
    schema metadata (keys `pysalient.io.y_proba_col` and `pysalient.io.y_label_col`).

    Calculates overall metrics (AUROC, AUPRC, Prevalence, Sample_Size, Label_Count)
    once using the provided data. Optionally calculates confidence intervals (CI)
    for Area Under Curve metrics (AUROC, AUPRC) using the bootstrap method if
    `calculate_au_ci` is True.

    Calculates threshold-specific metrics (TP, TN, FP, FN, PPV, Sensitivity,
    Specificity, NPV, Accuracy, F1-Score) for each specified threshold using the
    provided data. Optionally calculates CIs for these threshold-specific metrics
    using the method specified by `threshold_ci_method` if `calculate_threshold_ci`
    is True.


    Uses scikit-learn for AUROC/AUPRC/Accuracy/F1-Score and NumPy/internal helpers
    for other calculations. Optionally rounds float metrics.

    Args:
        data: Input PyArrow Table, loaded via pysalient.io. Must contain
              columns specified in its schema metadata under keys like
              META_KEY_Y_PROBA and META_KEY_Y_LABEL.
        modelid: An identifier string for the model being evaluated (user-provided).
        filter_desc: A string describing any filtering applied to the data
                     (e.g., 'all_sites_test', 'cohort_A', user-provided).
        thresholds: Specifies the thresholds for evaluation. Can be:
            - A list or tuple of specific float values (e.g., [0.1, 0.25, 0.5]).
            - A tuple of three floats (start, stop, step) to generate a range
              (e.g., (0.1, 0.9, 0.05) generates 0.1, 0.15, ..., 0.9).
              Note: np.linspace is now used for range generation.
        time_to_event_cols: Optional dictionary mapping metric base names to clinical event column names.
                           Key: Base name for output metrics (e.g., 'bc' for blood culture metrics).
                           Value: Column name containing clinical event timestamps.
                           Generates 3 metrics per key: {aggregation_func}_{time_unit}_from_first_alert_to_{key},
                           count_first_alerts_before_{key}, count_first_alerts_after_or_at_{key}.
                           Only calculated if aggregation metadata is present. Defaults to None.
        aggregation_func: Aggregation function for time-to-event metrics across encounters.
                         Supported: 'median', 'mean', 'min', 'max', 'std', 'var' (any NumPy function).
                         Defaults to 'median' for compatibility with reference implementation.
        time_unit: Unit label for time-to-event column names. This is purely for column naming/metadata
                  and does not affect the actual calculations. Use a descriptive name that matches
                  what your timestamp differences represent (e.g., 'hours', 'minutes', 'days').
                  Defaults to 'hours'.
        time_to_event_fillna: Fill value for NaN time-to-event metrics. If provided, replaces NaN values
                             in time-to-event hours columns with this value. Use 0.0 for zero imputation.
                             If None (default), NaN values are preserved. Only affects time-to-event metrics.
        force_threshold_zero: If True (default), forces the 0.0 threshold to always be included
                              in the evaluation thresholds. If False, 0.0 is only included
                              if specified in the `thresholds` input or generated by a range.
        decimal_places: If provided as an integer, rounds the calculated float
                        metrics (AUROC, AUPRC, Prevalence, PPV, Sensitivity, Specificity,
                        NPV, Accuracy, F1-Score, CIs)
                        to the specified number of decimal places. Defaults to None (no rounding).
        verbosity: Controls the verbosity of logging and warnings.
                   - `<= -1`: Show INFO, WARNING, and ERROR level messages.
                   - `== 0`: Show WARNING and ERROR level messages (default).
                   - `>= 1`: Show only ERROR level messages (suppress warnings).
        force_eval: If True, bypasses the threshold count check and forces evaluation
                   even if more than 10 thresholds are specified. If False (default),
                   raises a ValueError if more than 10 thresholds would be evaluated.
        calculate_au_ci: If True, calculate confidence intervals for Area Under Curve
                         metrics (AUROC, AUPRC) using the bootstrap method. Defaults to False.
        calculate_threshold_ci: If True, calculate confidence intervals for threshold-specific
                                metrics (PPV, Sensitivity, Specificity, NPV, Accuracy, F1_Score).
                                Defaults to False. The method used is determined by `threshold_ci_method`.
        threshold_ci_method: Method for threshold-specific CI calculation. Supported methods:
                             'bootstrap': Uses non-parametric bootstrap resampling. (Requires _bootstrap_utils).
                             'normal': Uses the normal approximation (Wald interval). (Requires _analytical_ci_utils).
                             'wilson': Uses the Wilson score interval. (Requires _analytical_ci_utils).
                             'agresti-coull': Uses the Agresti-Coull interval. (Requires _analytical_ci_utils).
                             Defaults to 'bootstrap'.
                             Note: Analytical methods ('normal', 'wilson', 'agresti-coull') do not support F1-Score CI calculation;
                             NaNs will be returned for F1 CIs if these methods are selected.
        ci_alpha: Significance level for all confidence intervals (e.g., 0.05 for 95% CI).
                  Defaults to 0.05. Used if `calculate_au_ci` or `calculate_threshold_ci` is True.
        bootstrap_rounds: Number of bootstrap rounds to perform. Defaults to 1000.
                          Used if `calculate_au_ci=True` OR (`calculate_threshold_ci=True` and `threshold_ci_method='bootstrap'`).
        bootstrap_seed: Optional random seed for reproducible bootstrap sampling.
                        Defaults to None. Used if `calculate_au_ci=True` OR (`calculate_threshold_ci=True` and `threshold_ci_method='bootstrap'`).

    Returns:
        A new PyArrow Table containing the evaluation results, with one row per
        threshold. The metrics reflect performance on the input data.
        Base columns include: 'modelid', 'filter_desc', 'threshold', 'AUROC',
        'AUROC_Lower_CI', 'AUROC_Upper_CI', 'AUPRC', 'AUPRC_Lower_CI',
        'AUPRC_Upper_CI', 'Prevalence', 'Sample_Size', 'Label_Count', 'TP', 'TN',
        'FP', 'FN', 'PPV', 'PPV_Lower_CI', 'PPV_Upper_CI', 'Sensitivity',
        'Sensitivity_Lower_CI', 'Sensitivity_Upper_CI', 'Specificity',
        'Specificity_Lower_CI', 'Specificity_Upper_CI', 'NPV', 'NPV_Lower_CI',
        'NPV_Upper_CI', 'Accuracy', 'Accuracy_Lower_CI', 'Accuracy_Upper_CI',
        'F1_Score', 'F1_Score_Lower_CI', 'F1_Score_Upper_CI'.

        Additional dynamic columns are added for each key in `time_to_event_cols`:
        '{aggregation_func}_{time_unit}_from_first_alert_to_{key}' (float64),
        'count_first_alerts_before_{key}' (int64),
        'count_first_alerts_after_or_at_{key}' (int64).
        Note: 'Sample_Size' reflects the number of rows (events or pre-aggregated groups)
        in the input data, and 'Label_Count' reflects the number of positive labels
        in the input data.
        Time-to-event columns are only added if `time_to_event_cols` is provided and
        aggregation metadata is present.
        AU CI columns contain nulls if `calculate_au_ci` is False.
        Threshold CI columns contain nulls if `calculate_threshold_ci` is False,
        if the method is unsupported for a specific metric (e.g., F1 with analytical methods),
        or if calculation fails.

    Raises:
        ValueError: If required metadata keys are missing, column names from
                    metadata don't exist in the table, thresholds are invalid,
                    threshold specification format is incorrect, data types
                    are unsuitable, or CI parameters are invalid.
        TypeError: If input data is not a PyArrow Table or columns have wrong types,
                   or if aggregation results in non-numeric types where numeric are expected.
        ImportError: If scikit-learn is required but not installed, or if required
                     CI utility modules (_bootstrap_utils or _analytical_ci_utils)
                     cannot be imported when the corresponding `threshold_ci_method` is requested.
        KeyError: If required metadata keys (e.g., META_KEY_Y_PROBA) are missing.
        RuntimeError: If metric calculation or CI calculation fails unexpectedly.
    """
    ######################################
    # 1. Validate Inputs & Read Metadata #
    ######################################
    if not isinstance(data, pa.Table):
        raise TypeError("Input 'data' must be a PyArrow Table.")
    if not isinstance(modelid, str):
        raise TypeError("Input 'modelid' must be a string.")
    if not isinstance(filter_desc, str):
        raise TypeError("Input 'filter_desc' must be a string.")
    if decimal_places is not None:
        if not isinstance(decimal_places, int) or decimal_places < 0:
            raise ValueError(
                "Input 'decimal_places' must be a non-negative integer or None."
            )
    # Add type check for calculate_threshold_ci
    if not isinstance(calculate_threshold_ci, bool):
        raise TypeError("Input 'calculate_threshold_ci' must be a boolean.")

    # Validate new time-to-event parameters
    if time_to_event_cols is not None:
        if not isinstance(time_to_event_cols, dict):
            raise TypeError("Input 'time_to_event_cols' must be a dictionary or None.")
        if not time_to_event_cols:  # Check for empty dict
            raise ValueError("Input 'time_to_event_cols' cannot be an empty dictionary.")
        for key, value in time_to_event_cols.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("All keys and values in 'time_to_event_cols' must be strings.")
            if not key or not value:  # Check for empty strings
                raise ValueError("All keys and values in 'time_to_event_cols' must be non-empty strings.")

    if not isinstance(aggregation_func, str):
        raise TypeError("Input 'aggregation_func' must be a string.")
    if not aggregation_func:
        raise ValueError("Input 'aggregation_func' cannot be an empty string.")

    # Validate time_to_event_fillna parameter
    if time_to_event_fillna is not None:
        if not isinstance(time_to_event_fillna, (int, float)):
            raise TypeError("Input 'time_to_event_fillna' must be a number or None.")

    # Validate that aggregation_func is a valid NumPy function
    if not hasattr(np, aggregation_func) or not callable(getattr(np, aggregation_func)):
        raise ValueError(f"Input 'aggregation_func' ('{aggregation_func}') is not a valid NumPy aggregation function.")

    ###########################################
    # Validate CI Parameters and Dependencies #
    ###########################################
    au_ci_requested = calculate_au_ci
    threshold_ci_requested = calculate_threshold_ci
    supported_threshold_ci_methods = ["bootstrap", "normal", "wilson", "agresti-coull"]
    analytical_methods = ["normal", "wilson", "agresti-coull"]

    # Validate alpha if any CI is requested
    if au_ci_requested or threshold_ci_requested:
        if not 0 < ci_alpha < 1:
            raise ValueError("ci_alpha must be between 0 and 1 (exclusive).")

    # Validate AU CI (Bootstrap only)
    if au_ci_requested:
        if calculate_bootstrap_ci is None:
            raise ImportError(
                "AU CI calculation requested (calculate_au_ci=True) but internal _bootstrap_utils module not found or failed to import."
            )
        # Validate bootstrap parameters
        if not isinstance(bootstrap_rounds, int) or bootstrap_rounds <= 0:
            raise ValueError("bootstrap_rounds must be a positive integer.")
        if bootstrap_seed is not None and not isinstance(bootstrap_seed, int):
            raise TypeError("bootstrap_seed must be an integer or None.")

    # Add validation and warning for bootstrap_rounds if CI calculation is requested
    if calculate_au_ci or (
        calculate_threshold_ci and threshold_ci_method == "bootstrap"
    ):
        if bootstrap_rounds < 100:
            raise ValueError(
                f"bootstrap_rounds is set to {bootstrap_rounds}. A value less than 100 "
                "is insufficient for reliable confidence interval estimates. "
                "Please increase bootstrap_rounds to at least 100."
            )
        # Wrap warning in verbosity check
        if verbosity <= 0:
            # Consider using logging.info here if a logger is configured
            pass  # For now, no INFO equivalent for this warning
            warnings.warn(
                f"bootstrap_rounds is set to {bootstrap_rounds}. A value less than 500 "
                "may lead to less reliable confidence interval estimates. Consider increasing "
                "bootstrap_rounds to 1000 or more for more stable results.",
                UserWarning,
            )

    # Validate Threshold CI
    if threshold_ci_requested:
        # Validate method selection
        if threshold_ci_method not in supported_threshold_ci_methods:
            raise ValueError(
                f"Unsupported threshold_ci_method '{threshold_ci_method}'. Supported methods are: {supported_threshold_ci_methods}"
            )

        # Validate dependencies and parameters based on method
        if threshold_ci_method == "bootstrap":
            if calculate_bootstrap_ci is None:
                raise ImportError(
                    "Bootstrap CI requested for thresholds (threshold_ci_method='bootstrap') but internal _bootstrap_utils module not found or failed to import."
                )
            # The bootstrap_rounds and seed validation is now handled above
        elif threshold_ci_method in analytical_methods:
            if anaci is None:
                raise ImportError(
                    f"Analytical CI method '{threshold_ci_method}' requested but internal _analytical_ci_utils module not found or failed to import."
                )

    # Validate specific sklearn dependencies if threshold metrics/CIs are needed
    sklearn_threshold_metrics_needed = []
    if calculate_threshold_ci:  # Need accuracy/f1 for CI calculation
        sklearn_threshold_metrics_needed.extend(
            [
                ("Accuracy", accuracy_score),
                ("F1-Score", f1_score),
            ]
        )
    else:  # Need accuracy/f1 only for point estimates if threshold CI is off
        sklearn_threshold_metrics_needed.extend(
            [
                ("Accuracy", accuracy_score),
                ("F1-Score", f1_score),
            ]
        )

    missing_sklearn_threshold_metrics = [
        name for name, func in sklearn_threshold_metrics_needed if func is None
    ]
    if missing_sklearn_threshold_metrics:
        # Raise error if needed for CI, otherwise warn for point estimates
        if calculate_threshold_ci:
            raise ImportError(
                f"scikit-learn is required for {', '.join(missing_sklearn_threshold_metrics)} threshold CI calculation(s) but not found. "
                "Please install it (`pip install scikit-learn`)."
            )
        else:
            # Wrap warning in verbosity check
            if verbosity <= 0:
                # Consider using logging.info here if a logger is configured
                pass  # For now, no INFO equivalent for this warning
                warnings.warn(
                    f"scikit-learn not found, {', '.join(missing_sklearn_threshold_metrics)} point estimates will not be calculated. "
                    "Install scikit-learn for these metrics.",
                    ImportWarning,
                )

    metadata = data.schema.metadata
    if not metadata:
        raise ValueError("Input table is missing schema metadata.")

    try:
        # Use .get() with a default of None, then check
        y_proba_col_bytes = metadata.get(META_KEY_Y_PROBA.encode("utf-8"))
        y_label_col_bytes = metadata.get(META_KEY_Y_LABEL.encode("utf-8"))
        aggregation_cols_bytes = metadata.get(META_KEY_AGGREGATION_COLS.encode("utf-8"))

        if y_proba_col_bytes is None:
            raise KeyError(f"Metadata key '{META_KEY_Y_PROBA}' not found.")
        if y_label_col_bytes is None:
            raise KeyError(f"Metadata key '{META_KEY_Y_LABEL}' not found.")

        y_proba_col = y_proba_col_bytes.decode("utf-8")
        y_label_col = y_label_col_bytes.decode("utf-8")

        # aggregation_cols is optional - only needed for time-to-event calculations
        aggregation_cols = None
        if aggregation_cols_bytes is not None:
            # Parse JSON-stored aggregation_cols - it might be a list or single string
            aggregation_cols_json = aggregation_cols_bytes.decode("utf-8")
            aggregation_cols_list = json.loads(aggregation_cols_json)

            # Handle case where it's an empty list (when perform_aggregation=False)
            if aggregation_cols_list:
                # For time-to-event calculations, we need the first aggregation column
                # (assuming single column for encounter grouping)
                aggregation_cols = aggregation_cols_list[0] if isinstance(aggregation_cols_list, list) else aggregation_cols_list
            else:
                # Empty list means no aggregation column available
                aggregation_cols = None


    except KeyError as e:
        raise KeyError(f"Required metadata key missing: {e}") from e
    except Exception as e:
        raise ValueError(f"Error reading metadata: {e}") from e

    if y_proba_col not in data.column_names:
        raise ValueError(f"Column '{y_proba_col}' (from metadata) not found in table.")
    if y_label_col not in data.column_names:
        raise ValueError(f"Column '{y_label_col}' (from metadata) not found in table.")

    # Validate time-to-event columns if provided
    time_to_event_enabled = False
    if time_to_event_cols is not None:
        if aggregation_cols is None:
            if verbosity <= 0:
                warnings.warn(
                    "time_to_event_cols provided but aggregation metadata not found. "
                    "Time-to-event calculations will be skipped.",
                    UserWarning
                )
        else:
            # Check that all specified clinical event columns exist
            for event_key, event_col in time_to_event_cols.items():
                if event_col not in data.column_names:
                    raise ValueError(f"Time-to-event column '{event_col}' (for key '{event_key}') not found in table.")
            
            # Validate that time_to_event_cols contain timestamp columns
            for event_key, event_col in time_to_event_cols.items():
                col_type = data[event_col].type
                if not pa.types.is_temporal(col_type):
                    warnings.warn(
                        f"time_to_event_cols['{event_key}'] = '{event_col}' is not a timestamp column "
                        f"(found {col_type}). Expected timestamp column for time calculations. "
                        f"This may cause calculation failures.",
                        UserWarning
                    )

            # Check that aggregation column exists
            if aggregation_cols not in data.column_names:
                raise ValueError(f"Aggregation column '{aggregation_cols}' (from metadata) not found in table.")

            time_to_event_enabled = True
    ################################
    # 2. Parse/Generate Thresholds #
    ################################
    try:
        # First generate thresholds without forcing 0 to count user-specified thresholds only
        user_threshold_list = _generate_thresholds(
            thresholds, include_zero=False
        )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid threshold specification: {e}") from e

    # Check threshold count and enforce limit if force_eval is False
    # Only count user-specified thresholds, not the forced 0 threshold
    if len(user_threshold_list) > 10 and not force_eval:
        raise ValueError(
            f"Too many thresholds ({len(user_threshold_list)}) specified. "
            f"Maximum allowed is 10 thresholds to prevent excessive computation. "
            f"Use force_eval=True to bypass this check and evaluate all {len(user_threshold_list)} thresholds."
        )

    # Now generate the final threshold list including forced 0 if needed
    try:
        threshold_list = _generate_thresholds(
            thresholds, include_zero=force_threshold_zero
        )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid threshold specification: {e}") from e

    #############################################################
    # 3. Extract Data & Handle Timeseries                       #
    #############################################################
    # --- Extract probabilities and labels directly from input data ---
    try:
        probas_to_eval = data[y_proba_col].to_numpy()
        labels_to_eval = data[y_label_col].to_numpy()
    except Exception as e:
        raise TypeError(
            f"Failed to convert columns '{y_proba_col}' or '{y_label_col}' to NumPy arrays: {e}"
        ) from e

    # Basic validation on numpy arrays
    if not np.issubdtype(probas_to_eval.dtype, np.number):
        raise TypeError(
            f"Probability column '{y_proba_col}' must be numeric (found {probas_to_eval.dtype})."
        )
    if not np.issubdtype(labels_to_eval.dtype, np.integer) and not np.issubdtype(
        labels_to_eval.dtype, np.bool_
    ):
        raise TypeError(
            f"Label column '{y_label_col}' must be integer or boolean (found {labels_to_eval.dtype})."
        )

    # Ensure labels are binary 0/1
    unique_labels = np.unique(labels_to_eval)
    if not np.all(np.isin(unique_labels, [0, 1])):
        if np.isnan(labels_to_eval).any():
            raise ValueError(
                f"Label column '{y_label_col}' contains NaN values. Please handle NaNs before evaluation."
            )
        # Wrap warning in verbosity check
        if verbosity <= 0:
            # Consider using logging.info here if a logger is configured
            pass  # For now, no INFO equivalent for this warning
            warnings.warn(
                f"Label column '{y_label_col}' contains values other than 0 and 1. Treating non-zero as 1.",
                UserWarning,
            )
        labels_to_eval = (labels_to_eval != 0).astype(np.int8)  # Coerce to binary 0/1
    elif not np.issubdtype(labels_to_eval.dtype, np.integer):
        # Ensure integer type even if already binary
        labels_to_eval = labels_to_eval.astype(np.int8)


    #############################################################
    # 4. Perform Evaluation using processed probas and labels   #
    #############################################################

    # Check sklearn dependencies needed for overall metrics (AUROC/AUPRC)
    sklearn_overall_metrics_needed = [
        ("AUROC", roc_auc_score),
        ("AUPRC", average_precision_score),
    ]
    missing_sklearn_overall = [
        name for name, func in sklearn_overall_metrics_needed if func is None
    ]
    if missing_sklearn_overall:
        raise ImportError(
            f"scikit-learn is required for {', '.join(missing_sklearn_overall)} calculation(s) but not found. "
            "Please install it (`pip install scikit-learn`)."
        )

    # --- MODIFIED: Call helper function ONCE with prepared data ---
    results_list = []  # Initialize results_list here
    try:
        # Ensure probas_to_eval and labels_to_eval are not None before calling
        if probas_to_eval is None or labels_to_eval is None:
            raise RuntimeError(
                "Internal error: probabilities or labels were not correctly prepared before evaluation call."
            )

        results_list = _process_single_evaluation(
            probas=probas_to_eval,
            labels=labels_to_eval,
            modelid=modelid,
            filter_desc=filter_desc,
            threshold_list=threshold_list,
            timeseries_array=None,  # No timeseries data
            timeseries_pa_type=None,  # No timeseries type
            time_unit=time_unit,  # Pass time unit for column naming
            aggregation_keys=None,
            aggregation_cols=aggregation_cols,  # Pass aggregation column name
            time_to_event_cols=time_to_event_cols,  # Pass time-to-event columns
            time_to_event_enabled=time_to_event_enabled,  # Pass whether enabled
            aggregation_func=aggregation_func,  # Pass aggregation function
            time_to_event_fillna=time_to_event_fillna,  # Pass fillna value
            data=data,  # Pass full data table for time-to-event calculations
            decimal_places=decimal_places,
            calculate_au_ci=calculate_au_ci,
            calculate_threshold_ci=calculate_threshold_ci,
            threshold_ci_method=threshold_ci_method,
            ci_alpha=ci_alpha,
            bootstrap_rounds=bootstrap_rounds,
            bootstrap_seed=bootstrap_seed,
            verbosity=verbosity,  # Pass verbosity down
        )
    except Exception as e:
        # Catch potential errors during the main evaluation process
        raise RuntimeError(
            f"Error during metric calculation in _process_single_evaluation: {e}"
        ) from e

    ####################################
    # 5. Convert List to PyArrow Table #
    ####################################
    # Define base schema for type safety and consistency
    schema_fields = [
        pa.field("modelid", pa.string()),
        pa.field("filter_desc", pa.string()),
        pa.field("threshold", pa.float64()),
    ]


    # Add overall metrics
    schema_fields.extend([
        # Overall Metrics
        pa.field("AUROC", pa.float64()),
        pa.field("AUROC_Lower_CI", pa.float64()),
        pa.field("AUROC_Upper_CI", pa.float64()),
        pa.field("AUPRC", pa.float64()),
        pa.field("AUPRC_Lower_CI", pa.float64()),
        pa.field("AUPRC_Upper_CI", pa.float64()),
        pa.field("Prevalence", pa.float64()),
        pa.field("Sample_Size", pa.int64()),  # Represents events or groups
        pa.field("Label_Count", pa.int64()),  # Represents positive events or groups
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
    ])

    # Add dynamic time-to-event columns if enabled
    if time_to_event_enabled and time_to_event_cols is not None:
        for event_key in time_to_event_cols.keys():
            # Add 3 columns per event key
            schema_fields.extend([
                pa.field(f"{aggregation_func}_{time_unit}_from_first_alert_to_{event_key}", pa.float64()),
                pa.field(f"count_first_alerts_before_{event_key}", pa.int64()),
                pa.field(f"count_first_alerts_after_or_at_{event_key}", pa.int64()),
            ])

    result_schema = pa.schema(schema_fields)

    try:
        # Create the result table from the pylist
        if not results_list:  # Handle case where evaluation might have returned empty
            # Wrap warning in verbosity check
            if verbosity <= 0:
                # Consider using logging.info here if a logger is configured
                pass  # For now, no INFO equivalent for this warning
                warnings.warn(
                    "Evaluation resulted in an empty list. Returning empty table.",
                    UserWarning,
                )
            result_table = pa.Table.from_pylist([], schema=result_schema)
        else:
            result_table = pa.Table.from_pylist(results_list, schema=result_schema)
    except Exception as e:
        # More specific error message for table creation
        raise RuntimeError(
            f"Failed to create result PyArrow Table from evaluation list: {e}\nResults List Sample: {results_list[:5]}"
        ) from e

    return result_table
