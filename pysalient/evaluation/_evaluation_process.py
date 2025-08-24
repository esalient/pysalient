import warnings
from typing import Any

import numpy as np
import pyarrow as pa

# Attempt to import scikit-learn metrics, raising an informative error if not found.
try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
        precision_recall_curve,
        auc,
    )
except ImportError:
    # Allow the module to be imported, but the function will fail if sklearn is needed.
    roc_auc_score = None
    average_precision_score = None
    accuracy_score = None
    f1_score = None
    precision_recall_curve = None
    auc = None
    warnings.warn(
        "scikit-learn not found. The 'evaluation' function requires scikit-learn "
        "for some metric calculations. Please install it (`pip install scikit-learn`).",
        ImportWarning,
    )
    SKLEARN_AVAILABLE = False  # Define based on import success
else:
    SKLEARN_AVAILABLE = True

# Import internal CI utilities using relative paths
try:
    # Use relative import within the evaluation module
    from ._bootstrap_utils import calculate_bootstrap_ci
except ImportError:
    calculate_bootstrap_ci = None  # Will be checked later if needed

try:
    # Use relative import within the evaluation module
    from . import _analytical_ci_utils as anaci
except ImportError:
    anaci = None  # Will be checked later if needed

# Check if at least one CI method is available if requested later
_ci_utils_available = calculate_bootstrap_ci is not None or anaci is not None


# Define standard metadata keys (using strings for broader compatibility, bytes are also fine)
META_KEY_Y_PROBA = "pysalient.io.y_proba_col"
META_KEY_Y_LABEL = "pysalient.io.y_label_col"
META_KEY_TIMESERIES_COL = "pysalient.io.timeseries_col"


def _process_single_evaluation(
    probas: np.ndarray,
    labels: np.ndarray,
    modelid: str,
    filter_desc: str,
    threshold_list: list[float],
    timeseries_array: np.ndarray | None,  # Add timeseries array parameter
    timeseries_pa_type: pa.DataType | None,  # Add original PyArrow type
    time_unit: str | None,  # Add time unit parameter
    aggregation_keys: (
        tuple | None
    ) = None,  # Kept for potential future internal use, but unused by current `evaluation` logic
    aggregation_cols: (
        str | None
    ) = None,  # Column name for aggregation (encounter ID)
    time_to_event_cols: dict[str, str] | None = None,  # NEW: Clinical event columns
    time_to_event_enabled: bool = False,  # NEW: Whether time-to-event is enabled
    aggregation_func: str = "median",  # NEW: Aggregation function for time-to-event metrics
    time_to_event_fillna: float | None = None,  # NEW: Fill NaN values in time-to-event metrics
    data: pa.Table | None = None,  # NEW: Full data table for time-to-event calculations
    decimal_places: int | None = None,
    calculate_au_ci: bool = False,
    calculate_threshold_ci: bool = False,
    threshold_ci_method: str = "bootstrap",
    ci_alpha: float = 0.05,
    bootstrap_rounds: int = 1000,
    bootstrap_seed: int | None = None,
    verbosity: int = 0,  # Add verbosity parameter
) -> list[dict[str, Any]]:  # noqa: C901
    # Define the analytical methods for confidence interval calculations
    analytical_methods = ["normal", "wilson", "agresti-coull"]
    """
    Process evaluation metrics for a single set of probabilities and labels.

    This is a helper function used by the main evaluation function after any
    potential aggregation has occurred. Includes calculation for "Time to First Alert"
    for each threshold, based on when the model's predicted probability first exceeds
    that threshold.

    Note: analytical_methods variable is defined within this function's scope and used by
    the inner threshold_ci calculation logic.

    Args:
        probas: NumPy array of predicted probabilities (event-level or aggregated)
        labels: NumPy array of true labels (binary 0/1, event-level or aggregated)
        modelid: Model identifier
        filter_desc: Description of any filtering applied
        threshold_list: List of thresholds to evaluate
        timeseries_array: NumPy array containing time information (indices, floats, or timestamps)
                          corresponding to labels/probas, or None if not provided.
        timeseries_pa_type: Original PyArrow DataType of the timeseries column, or None.
        time_unit: Time unit for calculation and column naming. Supports standard time units
                   ('second', 'minute', 'hour', 'day', 'week') and common abbreviations.
                   Time differences are calculated in seconds then converted to this unit.
        aggregation_keys: Tuple of values that identify this group (if aggregated) - NOW LARGELY UNUSED
        aggregation_cols: Column name for encounter aggregation (required for time-to-event calculations)
        time_to_event_cols: Dictionary mapping metric keys to clinical event column names for time-to-event metrics
        time_to_event_enabled: Whether time-to-event calculations should be performed
        aggregation_func: NumPy aggregation function name for time-to-event metrics across encounters
        data: Full PyArrow table containing all data (required for time-to-event calculations)
        decimal_places: Number of decimal places for rounding
        calculate_au_ci: Whether to calculate CIs for area under curve metrics
        calculate_threshold_ci: Whether to calculate CIs for threshold metrics
        threshold_ci_method: Method for threshold CIs
        ci_alpha: Significance level for CIs
        bootstrap_rounds: Number of bootstrap rounds
        bootstrap_seed: Random seed for reproducibility
        verbosity: Controls logging level in called functions (e.g., bootstrap).

    Returns:
        List of dictionaries containing evaluation results for each threshold, including
        time_to_first_alert_value and time_to_first_alert_unit.

    Raises:
        ValueError: If timeseries data is integer or float type but `time_unit` is missing.
        TypeError: If timeseries data type is not integer, float, or temporal.
    """
    results = []

    # Handle non-existent or empty labels
    if len(labels) == 0:
        warnings.warn(
            "Empty data for evaluation. Returning empty results.",
            UserWarning,
        )
        return results

    unique_labels = np.unique(labels)
    
    # Extract data arrays for time-to-event calculations if enabled
    encounter_ids_array = None
    alert_timestamps_array = None
    clinical_event_arrays = {}
    
    if time_to_event_enabled and data is not None and time_to_event_cols is not None:
        try:
            # Extract encounter IDs
            encounter_ids_array = data[aggregation_cols].to_numpy()
            
            # Extract alert timestamps (timeseries column)
            if timeseries_array is not None:
                alert_timestamps_array = timeseries_array
            else:
                # Try to get timeseries column from metadata
                metadata = data.schema.metadata
                if metadata:
                    timeseries_col_bytes = metadata.get(META_KEY_TIMESERIES_COL.encode("utf-8"))
                    if timeseries_col_bytes is not None:
                        timeseries_col_from_metadata = timeseries_col_bytes.decode("utf-8")
                        if timeseries_col_from_metadata in data.column_names:
                            timeseries_column = data[timeseries_col_from_metadata]
                            alert_timestamps_array = timeseries_column.to_numpy()
                            timeseries_pa_type = timeseries_column.type
                        else:
                            raise ValueError(f"Timeseries column '{timeseries_col_from_metadata}' from metadata not found in table")
                    else:
                        raise ValueError("timeseries_array is None and no timeseries column found in metadata")
                else:
                    raise ValueError("timeseries_array is None and no metadata available")
            
            # Extract clinical event timestamp arrays
            for event_key, event_col in time_to_event_cols.items():
                clinical_event_arrays[event_key] = data[event_col].to_numpy()
                
        except Exception as e:
            warnings.warn(
                f"Failed to extract arrays for time-to-event calculations: {e}. "
                "Time-to-event metrics will be skipped.",
                RuntimeWarning
            )
            time_to_event_enabled = False

    # Initialize overall metrics

    # Calculate overall metrics (AUROC/AUPRC)
    try:
        # Calculate overall metrics only if labels contain both classes
        if len(unique_labels) > 1:
            auroc = roc_auc_score(labels, probas)
            # Use same AUPRC calculation as current_process.py for exact match
            precision, recall, thresholds = precision_recall_curve(labels, probas)
            auprc = auc(recall, precision)
        else:
            auroc = np.nan
            auprc = np.nan  # Or prevalence if preferred for single class
            warnings.warn(
                "Only one class present in labels. AUROC and AUPRC are undefined.",
                UserWarning,
            )
    except Exception as e:
        warnings.warn(
            f"Error calculating overall metrics with scikit-learn: {e}",
            RuntimeWarning,
        )
        auroc = np.nan
        auprc = np.nan

    sample_size = len(
        labels
    )  # This is now the number of events OR the number of groups post-aggregation
    label_count = int(
        np.sum(labels)
    )  # Ensure integer type for table schema. Represents positive events or positive groups post-aggregation.
    prevalence = label_count / sample_size if sample_size > 0 else 0.0

    # Calculate Overall Confidence Intervals (Bootstrap only)
    auroc_lower_ci, auroc_upper_ci = None, None
    auprc_lower_ci, auprc_upper_ci = None, None

    # Overall CI only uses bootstrap method for now
    if calculate_au_ci and calculate_bootstrap_ci is not None:
        # Only calculate overall CIs if metrics themselves are defined
        if not np.isnan(auroc) and not np.isnan(auprc):
            try:
                # Calculate CI for AUROC
                auroc_lower_ci, auroc_upper_ci = calculate_bootstrap_ci(
                    y_true=labels,
                    y_pred=probas,
                    metric_func=roc_auc_score,  # Pass the function itself
                    n_rounds=bootstrap_rounds,
                    alpha=ci_alpha,
                    seed=bootstrap_seed,
                    verbosity=verbosity,  # Pass verbosity
                )

                # Calculate CI for AUPRC using same method as point estimate
                def _auprc_for_bootstrap(y_true_boot, y_pred_boot):
                    precision_boot, recall_boot, _ = precision_recall_curve(y_true_boot, y_pred_boot)
                    return auc(recall_boot, precision_boot)
                
                auprc_lower_ci, auprc_upper_ci = calculate_bootstrap_ci(
                    y_true=labels,
                    y_pred=probas,
                    metric_func=_auprc_for_bootstrap,  # Use same calculation method as point estimate
                    n_rounds=bootstrap_rounds,
                    alpha=ci_alpha,
                    seed=bootstrap_seed,  # Use same seed for consistency if set
                    verbosity=verbosity,  # Pass verbosity
                )

            except Exception as e:
                warnings.warn(
                    f"Overall confidence interval calculation failed: {e}",
                    RuntimeWarning,
                )
                # Ensure CIs remain None if calculation fails
                auroc_lower_ci, auroc_upper_ci = None, None
                auprc_lower_ci, auprc_upper_ci = None, None
        else:
            warnings.warn(
                "Skipping overall CI calculation because AUROC or AUPRC is undefined.",
                UserWarning,
            )

    # Apply Rounding to Overall Metrics
    if decimal_places is not None:
        # Helper function for safe rounding (handles None and NaN)
        def _safe_round(value, dp):
            if value is None or np.isnan(value):
                return value
            return round(value, dp)

        auroc = _safe_round(auroc, decimal_places)
        auprc = _safe_round(auprc, decimal_places)
        prevalence = _safe_round(prevalence, decimal_places)
        # Round overall CIs if they were calculated
        auroc_lower_ci = _safe_round(auroc_lower_ci, decimal_places)
        auroc_upper_ci = _safe_round(auroc_upper_ci, decimal_places)
        auprc_lower_ci = _safe_round(auprc_lower_ci, decimal_places)
        auprc_upper_ci = _safe_round(auprc_upper_ci, decimal_places)

    # Set up local versions of accuracy/f1 score, handling potential missing sklearn
    _local_accuracy_score = (
        accuracy_score if accuracy_score is not None else lambda yt, yp: np.nan
    )
    _local_f1_score = (
        f1_score if f1_score is not None else lambda yt, yp, zero_division=0.0: np.nan
    )

    # Loop Through Thresholds
    for threshold in threshold_list:
        # Initialize Threshold CI variables
        ppv_lower_ci, ppv_upper_ci = None, None
        sensitivity_lower_ci, sensitivity_upper_ci = None, None
        specificity_lower_ci, specificity_upper_ci = None, None
        npv_lower_ci, npv_upper_ci = None, None
        accuracy_lower_ci, accuracy_upper_ci = None, None
        f1_lower_ci, f1_upper_ci = None, None

        # a. Classify predictions (for point estimates)
        predicted_labels = (probas >= threshold).astype(np.int8)

        # b. Calculate TP, TN, FP, FN (for point estimates)
        tp = int(np.sum((predicted_labels == 1) & (labels == 1)))
        tn = int(np.sum((predicted_labels == 0) & (labels == 0)))
        fp = int(np.sum((predicted_labels == 1) & (labels == 0)))
        fn = int(np.sum((predicted_labels == 0) & (labels == 1)))

        # c. Calculate Point Estimates (PPV, Sensitivity, etc.)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sensitivity = (
            tp / (tp + fn) if (tp + fn) > 0 else 0.0
        )  # Also known as Recall or True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        # Use locally checked/defined functions for accuracy/f1
        accuracy = _local_accuracy_score(labels, predicted_labels)
        f1 = _local_f1_score(labels, predicted_labels, zero_division=0.0)

        # d. Calculate Threshold-Specific Confidence Intervals (Optional)
        if calculate_threshold_ci:
            if threshold_ci_method == "bootstrap":
                if calculate_bootstrap_ci is not None:
                    ######################################################################
                    # Define Metric Calculation Functions (with threshold) for Bootstrap #
                    ######################################################################
                    def _calculate_ppv_boot(y_true_boot, y_pred_boot):
                        pred_labels_boot = (y_pred_boot >= threshold).astype(np.int8)
                        tp_boot = np.sum((pred_labels_boot == 1) & (y_true_boot == 1))
                        fp_boot = np.sum((pred_labels_boot == 1) & (y_true_boot == 0))
                        denominator = tp_boot + fp_boot
                        return tp_boot / denominator if denominator > 0 else np.nan

                    def _calculate_sensitivity_boot(y_true_boot, y_pred_boot):
                        pred_labels_boot = (y_pred_boot >= threshold).astype(np.int8)
                        tp_boot = np.sum((pred_labels_boot == 1) & (y_true_boot == 1))
                        fn_boot = np.sum((pred_labels_boot == 0) & (y_true_boot == 1))
                        denominator = tp_boot + fn_boot
                        return tp_boot / denominator if denominator > 0 else np.nan

                    def _calculate_specificity_boot(y_true_boot, y_pred_boot):
                        pred_labels_boot = (y_pred_boot >= threshold).astype(np.int8)
                        tn_boot = np.sum((pred_labels_boot == 0) & (y_true_boot == 0))
                        fp_boot = np.sum((pred_labels_boot == 1) & (y_true_boot == 0))
                        denominator = tn_boot + fp_boot
                        return tn_boot / denominator if denominator > 0 else np.nan

                    def _calculate_npv_boot(y_true_boot, y_pred_boot):
                        pred_labels_boot = (y_pred_boot >= threshold).astype(np.int8)
                        tn_boot = np.sum((pred_labels_boot == 0) & (y_true_boot == 0))
                        fn_boot = np.sum((pred_labels_boot == 0) & (y_true_boot == 1))
                        denominator = tn_boot + fn_boot
                        return tn_boot / denominator if denominator > 0 else np.nan

                    def _calculate_accuracy_boot(y_true_boot, y_pred_boot):
                        pred_labels_boot = (y_pred_boot >= threshold).astype(np.int8)
                        return _local_accuracy_score(
                            y_true_boot, pred_labels_boot
                        )  # Use checked version

                    def _calculate_f1_boot(y_true_boot, y_pred_boot):
                        pred_labels_boot = (y_pred_boot >= threshold).astype(np.int8)
                        return _local_f1_score(
                            y_true_boot, pred_labels_boot, zero_division=0.0
                        )  # Use checked version

                    ###############################################
                    # Call calculate_bootstrap_ci for each metric #
                    ###############################################
                    metric_calculators_bootstrap = {
                        "PPV": (_calculate_ppv_boot, "ppv_lower_ci", "ppv_upper_ci"),
                        "Sensitivity": (
                            _calculate_sensitivity_boot,
                            "sensitivity_lower_ci",
                            "sensitivity_upper_ci",
                        ),
                        "Specificity": (
                            _calculate_specificity_boot,
                            "specificity_lower_ci",
                            "specificity_upper_ci",
                        ),
                        "NPV": (_calculate_npv_boot, "npv_lower_ci", "npv_upper_ci"),
                        "Accuracy": (
                            _calculate_accuracy_boot,
                            "accuracy_lower_ci",
                            "accuracy_upper_ci",
                        ),
                        "F1_Score": (_calculate_f1_boot, "f1_lower_ci", "f1_upper_ci"),
                    }
                    ci_results = {}
                    for metric_name, (
                        calc_func,
                        lower_key,
                        upper_key,
                    ) in metric_calculators_bootstrap.items():
                        try:
                            lower, upper = calculate_bootstrap_ci(
                                y_true=labels,
                                y_pred=probas,
                                metric_func=calc_func,
                                n_rounds=bootstrap_rounds,
                                alpha=ci_alpha,
                                seed=bootstrap_seed,
                                verbosity=verbosity,  # Pass verbosity
                            )
                            ci_results[lower_key] = lower
                            ci_results[upper_key] = upper
                        except Exception as e:
                            warnings.warn(
                                f"Bootstrap CI calculation failed for {metric_name} at threshold {threshold}: {e}",
                                RuntimeWarning,
                            )
                            ci_results[lower_key] = None
                            ci_results[upper_key] = None

                    ppv_lower_ci, ppv_upper_ci = ci_results.get(
                        "ppv_lower_ci"
                    ), ci_results.get("ppv_upper_ci")
                    sensitivity_lower_ci, sensitivity_upper_ci = ci_results.get(
                        "sensitivity_lower_ci"
                    ), ci_results.get("sensitivity_upper_ci")
                    specificity_lower_ci, specificity_upper_ci = ci_results.get(
                        "specificity_lower_ci"
                    ), ci_results.get("specificity_upper_ci")
                    npv_lower_ci, npv_upper_ci = ci_results.get(
                        "npv_lower_ci"
                    ), ci_results.get("npv_upper_ci")
                    accuracy_lower_ci, accuracy_upper_ci = ci_results.get(
                        "accuracy_lower_ci"
                    ), ci_results.get("accuracy_upper_ci")
                    f1_lower_ci, f1_upper_ci = ci_results.get(
                        "f1_lower_ci"
                    ), ci_results.get("f1_upper_ci")
                else:
                    # Bootstrap requested but module not available
                    warnings.warn(
                        "Bootstrap CI requested but module not available. Skipping threshold CIs.",
                        ImportWarning,
                    )

            elif threshold_ci_method in analytical_methods:
                if anaci is not None:  # Assuming anaci module is available
                    #################################
                    # Select Analytical CI Function #
                    #################################
                    if threshold_ci_method == "normal":
                        analytical_ci_func = anaci.calculate_normal_approx_ci
                    elif threshold_ci_method == "wilson":
                        analytical_ci_func = anaci.calculate_wilson_score_ci
                    elif threshold_ci_method == "agresti-coull":
                        analytical_ci_func = anaci.calculate_agresti_coull_ci
                    else:
                        analytical_ci_func = None  # Should not happen

                    if analytical_ci_func:
                        ##################
                        # Sensitivity CI #
                        ##################
                        try:
                            total_pos = tp + fn
                            sensitivity_lower_ci, sensitivity_upper_ci = (
                                analytical_ci_func(tp, total_pos, ci_alpha)
                                if total_pos > 0
                                else (np.nan, np.nan)
                            )
                        except Exception as e:
                            warnings.warn(
                                f"Sensitivity CI calculation ({threshold_ci_method}) failed at threshold {threshold}: {e}",
                                RuntimeWarning,
                            )
                            sensitivity_lower_ci, sensitivity_upper_ci = None, None
                        ##################
                        # Specificity CI #
                        ##################
                        try:
                            total_neg = tn + fp
                            specificity_lower_ci, specificity_upper_ci = (
                                analytical_ci_func(tn, total_neg, ci_alpha)
                                if total_neg > 0
                                else (np.nan, np.nan)
                            )
                        except Exception as e:
                            warnings.warn(
                                f"Specificity CI calculation ({threshold_ci_method}) failed at threshold {threshold}: {e}",
                                RuntimeWarning,
                            )
                            specificity_lower_ci, specificity_upper_ci = None, None
                        ##########
                        # PPV CI #
                        ##########
                        try:
                            total_pred_pos = tp + fp
                            ppv_lower_ci, ppv_upper_ci = (
                                analytical_ci_func(tp, total_pred_pos, ci_alpha)
                                if total_pred_pos > 0
                                else (np.nan, np.nan)
                            )
                        except Exception as e:
                            warnings.warn(
                                f"PPV CI calculation ({threshold_ci_method}) failed at threshold {threshold}: {e}",
                                RuntimeWarning,
                            )
                            ppv_lower_ci, ppv_upper_ci = None, None
                        ##########
                        # NPV CI #
                        ##########
                        try:
                            total_pred_neg = tn + fn
                            npv_lower_ci, npv_upper_ci = (
                                analytical_ci_func(tn, total_pred_neg, ci_alpha)
                                if total_pred_neg > 0
                                else (np.nan, np.nan)
                            )
                        except Exception as e:
                            warnings.warn(
                                f"NPV CI calculation ({threshold_ci_method}) failed at threshold {threshold}: {e}",
                                RuntimeWarning,
                            )
                            npv_lower_ci, npv_upper_ci = None, None
                        ###############
                        # Accuracy CI #
                        ###############
                        try:
                            total_samples = tp + tn + fp + fn
                            accuracy_lower_ci, accuracy_upper_ci = (
                                analytical_ci_func(tp + tn, total_samples, ci_alpha)
                                if total_samples > 0
                                else (np.nan, np.nan)
                            )
                        except Exception as e:
                            warnings.warn(
                                f"Accuracy CI calculation ({threshold_ci_method}) failed at threshold {threshold}: {e}",
                                RuntimeWarning,
                            )
                            accuracy_lower_ci, accuracy_upper_ci = None, None

                        # F1-Score CI not supported by these analytical methods
                        f1_lower_ci, f1_upper_ci = np.nan, np.nan

                else:
                    # Analytical method requested but module not available
                    warnings.warn(
                        f"Analytical CI method '{threshold_ci_method}' requested but module not available. Skipping threshold CIs.",
                        ImportWarning,
                    )

            else:  # Unknown ci_method (should not happen due to earlier validation)
                warnings.warn(
                    f"Unknown threshold_ci_method '{threshold_ci_method}' encountered during threshold CI calculation. Skipping.",
                    UserWarning,
                )

        # e. Apply Rounding (after all calculations for the threshold)
        if decimal_places is not None:
            # Helper function for safe rounding (handles None and NaN)
            # Defined earlier, no need to redefine

            # Round point estimates
            ppv = _safe_round(ppv, decimal_places)
            sensitivity = _safe_round(sensitivity, decimal_places)
            specificity = _safe_round(specificity, decimal_places)
            npv = _safe_round(npv, decimal_places)
            accuracy = _safe_round(accuracy, decimal_places)
            f1 = _safe_round(f1, decimal_places)

            # Round threshold CIs
            ppv_lower_ci = _safe_round(ppv_lower_ci, decimal_places)
            ppv_upper_ci = _safe_round(ppv_upper_ci, decimal_places)
            sensitivity_lower_ci = _safe_round(sensitivity_lower_ci, decimal_places)
            sensitivity_upper_ci = _safe_round(sensitivity_upper_ci, decimal_places)
            specificity_lower_ci = _safe_round(specificity_lower_ci, decimal_places)
            specificity_upper_ci = _safe_round(specificity_upper_ci, decimal_places)
            npv_lower_ci = _safe_round(npv_lower_ci, decimal_places)
            npv_upper_ci = _safe_round(npv_upper_ci, decimal_places)
            accuracy_lower_ci = _safe_round(accuracy_lower_ci, decimal_places)
            accuracy_upper_ci = _safe_round(accuracy_upper_ci, decimal_places)
            f1_lower_ci = _safe_round(f1_lower_ci, decimal_places)
            f1_upper_ci = _safe_round(f1_upper_ci, decimal_places)

            # Round threshold value as well
            threshold = _safe_round(threshold, decimal_places)

        # Calculate time-to-event metrics for this threshold if enabled
        time_to_event_metrics = {}
        if time_to_event_enabled and time_to_event_cols is not None:
            # Filter to true positives at this threshold (ARCHITECTURE.md line 226)
            tp_mask = (probas >= threshold) & (labels == 1)
            tp_count = np.sum(tp_mask)
            
            if tp_count > 0:  # Only proceed if there are true positives
                for event_key, event_col in time_to_event_cols.items():
                    try:
                        # Get true positive data for this event
                        tp_encounter_ids = encounter_ids_array[tp_mask]
                        tp_alert_timestamps = alert_timestamps_array[tp_mask]
                        tp_event_timestamps = clinical_event_arrays[event_key][tp_mask]
                        
                        # Calculate time differences - always in seconds first for precision
                        # Then convert to requested time_unit
                        if pa.types.is_temporal(timeseries_pa_type):
                            # Both are temporal - calculate difference in seconds (preserves precision)
                            time_diffs_seconds = (tp_event_timestamps - tp_alert_timestamps).astype('timedelta64[s]').astype(float)
                        else:
                            # For numeric timestamps, assume they are already in the correct unit
                            time_diffs_seconds = (tp_event_timestamps - tp_alert_timestamps).astype(float)
                        
                        # Convert from seconds to requested time unit using a conversion factor
                        # This avoids multiple if/else statements and maintains precision
                        time_unit_conversions = {
                            'second': 1.0, 'seconds': 1.0, 'sec': 1.0, 'secs': 1.0, 's': 1.0,
                            'minute': 60.0, 'minutes': 60.0, 'min': 60.0, 'mins': 60.0, 'm': 60.0,
                            'hour': 3600.0, 'hours': 3600.0, 'hr': 3600.0, 'hrs': 3600.0, 'h': 3600.0,
                            'day': 86400.0, 'days': 86400.0, 'd': 86400.0,
                            'week': 604800.0, 'weeks': 604800.0, 'w': 604800.0,
                        }
                        
                        # Get conversion factor (default to hours for backward compatibility)
                        time_unit_key = time_unit.lower() if time_unit else 'hour'
                        conversion_factor = time_unit_conversions.get(time_unit_key, 3600.0)  # Default to hours
                        
                        # Convert to requested unit
                        time_diffs = time_diffs_seconds / conversion_factor
                        
                        # Group by encounter and take max time per encounter (following notebook logic)
                        unique_encounters = np.unique(tp_encounter_ids)
                        encounter_max_times = []
                        
                        for enc_id in unique_encounters:
                            enc_mask = tp_encounter_ids == enc_id
                            enc_time_diffs = time_diffs[enc_mask]
                            # Use nanmax to handle NaN values like Pandas .max() does
                            max_time = np.nanmax(enc_time_diffs)
                            encounter_max_times.append(max_time)
                        
                        encounter_max_times = np.array(encounter_max_times)
                        
                        # Apply NaN-aware aggregation function across encounters to match Pandas behavior
                        if len(encounter_max_times) > 0:
                            # Use NaN-aware version of aggregation function to match Pandas behavior
                            if aggregation_func == 'median':
                                agg_time = np.nanmedian(encounter_max_times)
                            elif aggregation_func == 'mean':
                                agg_time = np.nanmean(encounter_max_times)
                            elif aggregation_func == 'min':
                                agg_time = np.nanmin(encounter_max_times)
                            elif aggregation_func == 'max':
                                agg_time = np.nanmax(encounter_max_times)
                            elif aggregation_func == 'std':
                                agg_time = np.nanstd(encounter_max_times)
                            elif aggregation_func == 'var':
                                agg_time = np.nanvar(encounter_max_times)
                            else:
                                # For other functions, try nan version first, fallback to regular
                                nan_func_name = f'nan{aggregation_func}'
                                if hasattr(np, nan_func_name):
                                    nan_agg_func = getattr(np, nan_func_name)
                                    agg_time = nan_agg_func(encounter_max_times)
                                else:
                                    # Fallback: filter out NaN values manually then use regular function
                                    valid_times = encounter_max_times[~np.isnan(encounter_max_times)]
                                    if len(valid_times) > 0:
                                        agg_func = getattr(np, aggregation_func)
                                        agg_time = agg_func(valid_times)
                                    else:
                                        agg_time = np.nan
                            # Count encounters properly, excluding NaN values 
                            valid_times = encounter_max_times[~np.isnan(encounter_max_times)]
                            count_before = np.sum(valid_times > 0)
                            count_after_or_at = np.sum(valid_times <= 0)
                            
                            # Calculate aggregation for only alerts after or at event time
                            after_or_at_times = encounter_max_times[encounter_max_times <= 0]
                            if len(after_or_at_times) > 0:
                                # Apply same NaN-aware logic for after/at times
                                if aggregation_func == 'median':
                                    agg_time_after_or_at = np.nanmedian(after_or_at_times)
                                elif aggregation_func == 'mean':
                                    agg_time_after_or_at = np.nanmean(after_or_at_times)
                                elif aggregation_func == 'min':
                                    agg_time_after_or_at = np.nanmin(after_or_at_times)
                                elif aggregation_func == 'max':
                                    agg_time_after_or_at = np.nanmax(after_or_at_times)
                                elif aggregation_func == 'std':
                                    agg_time_after_or_at = np.nanstd(after_or_at_times)
                                elif aggregation_func == 'var':
                                    agg_time_after_or_at = np.nanvar(after_or_at_times)
                                else:
                                    # For other functions, try nan version first, fallback to regular
                                    nan_func_name = f'nan{aggregation_func}'
                                    if hasattr(np, nan_func_name):
                                        nan_agg_func = getattr(np, nan_func_name)
                                        agg_time_after_or_at = nan_agg_func(after_or_at_times)
                                    else:
                                        # Fallback: filter out NaN values manually then use regular function
                                        valid_times = after_or_at_times[~np.isnan(after_or_at_times)]
                                        if len(valid_times) > 0:
                                            agg_func = getattr(np, aggregation_func)
                                            agg_time_after_or_at = agg_func(valid_times)
                                        else:
                                            agg_time_after_or_at = np.nan
                            else:
                                agg_time_after_or_at = np.nan
                        else:
                            agg_time = np.nan
                            count_before = 0
                            count_after_or_at = 0
                            agg_time_after_or_at = np.nan
                        
                        # Store results
                        time_to_event_metrics[f"{aggregation_func}_{time_unit}_from_first_alert_to_{event_key}"] = agg_time
                        time_to_event_metrics[f"count_first_alerts_before_{event_key}"] = count_before
                        time_to_event_metrics[f"count_first_alerts_after_or_at_{event_key}"] = count_after_or_at
                        
                    except Exception as e:
                        # Log warning and set NaN/0 values for this event
                        warnings.warn(
                            f"Time-to-event calculation failed for event '{event_key}' at threshold {threshold}: {e}",
                            RuntimeWarning
                        )
                        time_to_event_metrics[f"{aggregation_func}_{time_unit}_from_first_alert_to_{event_key}"] = np.nan
                        time_to_event_metrics[f"count_first_alerts_before_{event_key}"] = 0
                        time_to_event_metrics[f"count_first_alerts_after_or_at_{event_key}"] = 0
            else:
                # No true positives at this threshold - set all metrics to NaN/0
                for event_key in time_to_event_cols.keys():
                    time_to_event_metrics[f"{aggregation_func}_{time_unit}_from_first_alert_to_{event_key}"] = np.nan
                    time_to_event_metrics[f"count_first_alerts_before_{event_key}"] = 0
                    time_to_event_metrics[f"count_first_alerts_after_or_at_{event_key}"] = 0

        # Calculate Time to First Alert for this threshold
        time_to_first_alert_value: float | None = None
        time_to_first_alert_unit: str | None = None

        if timeseries_array is not None and timeseries_pa_type is not None:
            if np.any(
                predicted_labels == 1
            ):  # Check if there's at least one predicted alert
                first_alert_idx = np.argmax(predicted_labels == 1)

                if pa.types.is_integer(timeseries_pa_type) or pa.types.is_floating(
                    timeseries_pa_type
                ):
                    if time_unit is None:
                        # This should ideally be caught earlier in `evaluation`, but double-check
                        raise ValueError(
                            "time_unit is required when timeseries_col contains integers or floats."
                        )
                    time_to_first_alert_value = float(
                        timeseries_array[first_alert_idx]
                    )  # Ensure float conversion
                    time_to_first_alert_unit = time_unit
                elif pa.types.is_temporal(timeseries_pa_type):
                    try:
                        # Ensure we have valid timestamps before calculation
                        first_alert_timestamp = timeseries_array[first_alert_idx]
                        start_timestamp = timeseries_array[0]

                        # Check for NaT (Not a Time) which can occur with invalid data
                        if np.isnat(first_alert_timestamp) or np.isnat(start_timestamp):
                            warnings.warn(
                                "Encountered NaT (Not a Time) in timeseries data when calculating time to first alert. Setting metric to None.",
                                UserWarning,
                            )
                            # Keep values as None
                        else:
                            # Calculate timedelta
                            delta = first_alert_timestamp - start_timestamp
                            # Convert timedelta64 to seconds (float)
                            time_to_first_alert_value = delta.astype(
                                "timedelta64[s]"
                            ).astype(float)
                            time_to_first_alert_unit = "seconds"

                    except Exception as e:
                        warnings.warn(
                            f"Error calculating time difference for temporal timeseries: {e}. Setting metric to None.",
                            RuntimeWarning,
                        )
                        # Keep values as None in case of calculation error

                else:
                    # This should have been caught in `evaluation`
                    raise TypeError(
                        f"Unsupported timeseries data type for time_to_first_alert: {timeseries_pa_type}. Expected integer, float, or temporal."
                    )
            # else: No predicted alert at this threshold, metrics remain None

        # f. Create row data dictionary
        row_data = {
            "modelid": modelid,
            "filter_desc": filter_desc,
            "threshold": threshold,
            # Overall metrics (repeated for each threshold row)
            "AUROC": auroc,
            "AUROC_Lower_CI": auroc_lower_ci,
            "AUROC_Upper_CI": auroc_upper_ci,
            "AUPRC": auprc,
            "AUPRC_Lower_CI": auprc_lower_ci,
            "AUPRC_Upper_CI": auprc_upper_ci,
            "Prevalence": prevalence,
            "Sample_Size": sample_size,  # Note: Number of events or groups
            "Label_Count": label_count,  # Note: Number of positive events or groups
            # Confusion matrix components
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            # Threshold-specific metrics and their CIs
            "PPV": ppv,
            "PPV_Lower_CI": ppv_lower_ci,
            "PPV_Upper_CI": ppv_upper_ci,
            "Sensitivity": sensitivity,
            "Sensitivity_Lower_CI": sensitivity_lower_ci,
            "Sensitivity_Upper_CI": sensitivity_upper_ci,
            "Specificity": specificity,
            "Specificity_Lower_CI": specificity_lower_ci,
            "Specificity_Upper_CI": specificity_upper_ci,
            "NPV": npv,
            "NPV_Lower_CI": npv_lower_ci,
            "NPV_Upper_CI": npv_upper_ci,
            "Accuracy": accuracy,
            "Accuracy_Lower_CI": accuracy_lower_ci,
            "Accuracy_Upper_CI": accuracy_upper_ci,
            "F1_Score": f1,
            "F1_Score_Lower_CI": f1_lower_ci,
            "F1_Score_Upper_CI": f1_upper_ci,
        }
        
        # Add time-to-first-alert fields only if timeseries data was provided
        if timeseries_array is not None:
            row_data.update({
                "time_to_first_alert_value": time_to_first_alert_value,
                "time_to_first_alert_unit": time_to_first_alert_unit,
            })
        
        # Add time-to-event metrics if they were calculated
        if time_to_event_metrics:
            # DO NOT round time-to-event metrics here - preserve full precision for visualization
            # The visualization layer will handle formatting according to decimal_places
            final_time_to_event_metrics = time_to_event_metrics
            
            # Apply fillna to time-to-event metrics if requested
            if time_to_event_fillna is not None:
                fillna_time_to_event_metrics = {}
                for key, value in final_time_to_event_metrics.items():
                    if ("_from_first_alert_to_" in key) and (value is None or (isinstance(value, float) and np.isnan(value))):
                        # This is a time metric with NaN/None - fill it
                        fillna_time_to_event_metrics[key] = time_to_event_fillna
                    else:
                        # Keep original value
                        fillna_time_to_event_metrics[key] = value
                final_time_to_event_metrics = fillna_time_to_event_metrics
            
            row_data.update(final_time_to_event_metrics)

        # Append row data
        results.append(row_data)

    return results
