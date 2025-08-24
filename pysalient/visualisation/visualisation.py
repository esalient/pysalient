"""
Visualisation and display helpers
"""

from typing import Any

import pandas as pd  # Add pandas import
import pyarrow as pa
from pandas.io.formats.style import Styler

# Define columns typically needing float formatting in evaluation results
# Include both common cases (lowercase often used in fixtures/input, uppercase is output standard)
DEFAULT_FLOAT_COLS = [
    "threshold",
    "Threshold",
    "AUROC",
    "auroc",
    "AUPRC",
    "auprc",
    "Prevalence",
    "prevalence",
    "PPV",
    "ppv",
    "Sensitivity",
    "sensitivity",
    "Specificity",
    "specificity",
    "NPV",
    "npv",
    "Accuracy",
    "accuracy",
    "F1_Score",
    "f1_score",
    "F1",
    "f1",  # Common variations for F1
    # CI columns (less likely to vary, but keep consistent uppercase)
    "AUROC_Lower_CI",
    "AUROC_Upper_CI",
    "AUPRC_Lower_CI",
    "AUPRC_Upper_CI",
    "PPV_Lower_CI",
    "PPV_Upper_CI",
    "Sensitivity_Lower_CI",
    "Sensitivity_Upper_CI",
    "Specificity_Lower_CI",
    "Specificity_Upper_CI",
    "NPV_Lower_CI",
    "NPV_Upper_CI",
    "Accuracy_Lower_CI",
    "Accuracy_Upper_CI",
    "F1_Score_Lower_CI",
    "F1_Score_Upper_CI",
]


def format_evaluation_table(
    table: pa.Table,
    decimal_places: int | None = 3,
    float_columns: list[str] | None = None,
    ci_column: bool = True,  # Added parameter
    order_by: str | list[str] | None = "threshold",  # Added parameter
) -> Styler:
    """
    Converts an evaluation result PyArrow Table to a styled Pandas DataFrame
    with formatted float columns and potentially combined CI columns.

    Args:
        table: The input PyArrow Table (typically the output of evaluation).
        decimal_places: Number of decimal places to round float columns to.
                        If None, no rounding format is applied. Defaults to 3.
        float_columns: Optional list of column names to apply float formatting to.
                       If None, defaults to common float metrics columns defined
                       in DEFAULT_FLOAT_COLS, excluding original CI columns and
                       metric columns if ci_column is False.
        ci_column: If True (default), display confidence intervals in a separate
                   '{Metric} CI' column. If False, integrate the CI into the
                   main metric column as a string like 'value [lower - upper]'.
        order_by: Column name(s) to sort the DataFrame by before styling. 
                  Can be a single column name (str) or list of column names (list[str]).
                  If None, no sorting is applied. Defaults to "threshold" to sort by threshold values.

    Returns:
        A pandas Styler object ready for display in environments like Jupyter.

    Raises:
        TypeError: If the input 'table' is not a PyArrow Table, or if 'order_by' is not a string/list.
        ValueError: If 'decimal_places' is invalid, or if 'order_by' columns don't exist in the table.
        RuntimeError: If DataFrame conversion or sorting fails.
    """
    if not isinstance(table, pa.Table):
        raise TypeError("Input 'table' must be a PyArrow Table.")

    try:
        df = table.to_pandas()
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert PyArrow Table to Pandas DataFrame: {e}"
        ) from e

    # Apply sorting if order_by is specified
    if order_by is not None:
        try:
            if isinstance(order_by, str):
                # Single column
                if order_by in df.columns:
                    df = df.sort_values(by=order_by).reset_index(drop=True)
                else:
                    raise ValueError(f"Column '{order_by}' not found in table. Available columns: {df.columns.tolist()}")
            elif isinstance(order_by, list):
                # Multiple columns
                missing_cols = [col for col in order_by if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Columns {missing_cols} not found in table. Available columns: {df.columns.tolist()}")
                df = df.sort_values(by=order_by).reset_index(drop=True)
            else:
                raise TypeError("'order_by' must be a string, list of strings, or None.")
        except Exception as e:
            raise RuntimeError(f"Failed to sort DataFrame by '{order_by}': {e}") from e

    if decimal_places is None:
        # Return unformatted Styler if no rounding needed
        return df.style

    # Validate decimal_places
    if not isinstance(decimal_places, int) or decimal_places < 0:
        raise ValueError("'decimal_places' must be a non-negative integer or None.")

    format_str = f"{{:,.{decimal_places}f}}"  # e.g., '{:,.3f}'

    # Metrics for which to handle CI columns
    metrics_with_ci = [
        "AUROC",
        "AUPRC",
        "PPV",
        "Sensitivity",
        "Specificity",
        "NPV",
        "Accuracy",
        "F1_Score",
    ]
    cols_to_drop = []
    original_metric_cols_found = (
        set()
    )  # Keep track of original metrics found that have CIs
    metric_cols_modified_for_ci = set()  # Track metrics modified when ci_column=False

    # Helper to format CI values or metric values
    def format_value(value):
        if pd.isnull(value):
            return "nan"  # Or potentially "" or some other placeholder
        try:
            return format_str.format(value)
        except (ValueError, TypeError):
            # Handle cases where the value might already be non-numeric
            if isinstance(value, str):
                return value  # Return string as is if formatting fails
            return "nan"  # Fallback for other errors

    # --- Conditional CI Handling ---
    if ci_column:
        # Create separate CI columns (original logic)
        for metric in metrics_with_ci:
            lower_ci_col = f"{metric}_Lower_CI"
            upper_ci_col = f"{metric}_Upper_CI"
            new_ci_col = f"{metric} CI"

            if (
                metric in df.columns
                and lower_ci_col in df.columns
                and upper_ci_col in df.columns
            ):
                # Create the combined CI string column
                df[new_ci_col] = df.apply(
                    lambda row: f"[{format_value(row[lower_ci_col])} - {format_value(row[upper_ci_col])}]",
                    axis=1,
                )
                # Mark original CI columns for removal
                cols_to_drop.extend([lower_ci_col, upper_ci_col])
                # Mark the original metric column as found (for potential formatting)
                original_metric_cols_found.add(metric)
    else:
        # Integrate CI into the main metric column
        for metric in metrics_with_ci:
            lower_ci_col = f"{metric}_Lower_CI"
            upper_ci_col = f"{metric}_Upper_CI"

            if (
                metric in df.columns
                and lower_ci_col in df.columns
                and upper_ci_col in df.columns
            ):
                # Apply formatting row-wise to the original metric column
                def format_metric_with_ci(row):
                    metric_val = row[metric]
                    lower_ci = row[lower_ci_col]
                    upper_ci = row[upper_ci_col]

                    formatted_metric = format_value(metric_val)

                    # Only add CI if both bounds are present
                    if pd.notnull(lower_ci) and pd.notnull(upper_ci):
                        formatted_lower = format_value(lower_ci)
                        formatted_upper = format_value(upper_ci)
                        return f"{formatted_metric} [{formatted_lower} - {formatted_upper}]"
                    else:
                        # Return only the formatted metric if CIs are missing
                        return formatted_metric

                df[metric] = df.apply(format_metric_with_ci, axis=1)

                # Mark original CI columns for removal
                cols_to_drop.extend([lower_ci_col, upper_ci_col])
                # Mark this metric column as modified (won't be float formatted)
                metric_cols_modified_for_ci.add(metric)
                # Also mark as found
                original_metric_cols_found.add(
                    metric
                )  # Still needed for column ordering logic potentially

    # Drop the original CI columns
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # --- Determine final columns to apply float formatting ---
    if float_columns is None:
        # Use default columns, filtering based on existence and ci_column setting
        potential_cols = [col for col in DEFAULT_FLOAT_COLS if col in df.columns]
        # Start with potential default float columns
        cols_to_format_set = set(potential_cols)
        
        # Dynamically add time-to-event columns (they have specific naming pattern)
        for col in df.columns:
            if ("_from_first_alert_to_" in col and 
                not col.startswith("count_")):  # Exclude count columns (integers)
                cols_to_format_set.add(col)
        
        # Add original metrics found, ONLY if ci_column is True (otherwise they are strings)
        if ci_column:
            cols_to_format_set.update(original_metric_cols_found)
        # Remove columns that were dropped
        cols_to_format_set -= set(cols_to_drop)
        # Remove metric columns that were modified into strings
        cols_to_format_set -= metric_cols_modified_for_ci
        # Remove any columns that look like the *new* CI columns (shouldn't be needed but safe)
        cols_to_format_set -= {
            f"{m} CI" for m in metrics_with_ci if f"{m} CI" in df.columns
        }

    else:
        # Use user-provided columns, filtering based on existence and ci_column setting
        if not isinstance(float_columns, list) or not all(
            isinstance(c, str) for c in float_columns
        ):
            raise TypeError("'float_columns' must be a list of strings or None.")

        # Start with user list filtered by existence
        cols_to_format_set = {col for col in float_columns if col in df.columns}
        # Remove columns that were dropped
        cols_to_format_set -= set(cols_to_drop)
        # Remove metric columns that were modified into strings
        cols_to_format_set -= metric_cols_modified_for_ci
        # Ensure original metrics found AND requested by user are formatted ONLY if ci_column is True
        # (This logic is implicitly handled by removing modified cols above)

    final_cols_to_format = list(cols_to_format_set)
    format_dict = {col: format_str for col in final_cols_to_format if col in df.columns}

    # --- Apply formatting and reorder columns ---
    if (
        format_dict or not ci_column
    ):  # Reorder even if no formatting, if ci_column changed structure
        # Ensure column order is preserved, placing new CI cols next to metrics if ci_column=True
        final_column_order = []
        processed_metrics = set()
        # Get original column names that are still present in the DataFrame
        original_cols_in_df = [col for col in table.column_names if col in df.columns]

        for col in original_cols_in_df:
            is_metric_with_ci = (
                col in metrics_with_ci and col in original_metric_cols_found
            )

            if is_metric_with_ci and ci_column:
                # If ci_column is True, add metric and its separate CI column
                if col not in processed_metrics:
                    final_column_order.append(col)
                    ci_col_name = f"{col} CI"
                    if (
                        ci_col_name in df.columns
                    ):  # Check if CI col was actually created
                        final_column_order.append(ci_col_name)
                    processed_metrics.add(col)
            elif is_metric_with_ci and not ci_column:
                # If ci_column is False, just add the modified metric column
                if col not in processed_metrics:
                    final_column_order.append(col)
                    processed_metrics.add(col)
            elif col not in processed_metrics:  # Add other non-metric/non-CI columns
                # Avoid adding original _Lower_CI/_Upper_CI cols if they somehow survived drop
                if not any(
                    col.endswith(suffix) for suffix in ["_Lower_CI", "_Upper_CI"]
                ):
                    final_column_order.append(col)

        # Add any columns present in the df but not captured above
        # (e.g., columns not in original table, or CI columns if logic missed them)
        ordered_cols_set = set(final_column_order)
        remaining_cols = [col for col in df.columns if col not in ordered_cols_set]
        # Prioritize putting remaining known metrics + their CIs if ci_column=True
        remaining_ordered = []
        processed_remaining = set()
        if ci_column:
            for metric in metrics_with_ci:
                if metric in remaining_cols and metric not in processed_remaining:
                    remaining_ordered.append(metric)
                    processed_remaining.add(metric)
                    ci_col_name = f"{metric} CI"
                    if ci_col_name in remaining_cols:
                        remaining_ordered.append(ci_col_name)
                        processed_remaining.add(ci_col_name)

        # Add any truly remaining columns
        remaining_ordered.extend(
            [col for col in remaining_cols if col not in processed_remaining]
        )
        final_column_order.extend(remaining_ordered)

        # Ensure all df columns are accounted for and exist, prevent key errors
        final_column_order = [col for col in final_column_order if col in df.columns]
        # Ensure uniqueness (though logic should prevent duplicates)
        final_column_order = list(dict.fromkeys(final_column_order))

        # Reindex DataFrame and apply formatting
        try:
            # Apply formatting only if format_dict is not empty
            if format_dict:
                return df[final_column_order].style.format(format_dict)
            else:
                # If no formatting needed but reordering happened (e.g., ci_column=False)
                return df[final_column_order].style
        except KeyError as e:
            # Provide more context if reordering causes issues
            print(
                f"KeyError during styling. Final columns attempted: {final_column_order}"
            )
            print(f"DataFrame columns available: {df.columns.tolist()}")
            print(f"Format dictionary keys: {list(format_dict.keys())}")
            raise e
        except Exception as e:  # Catch other potential errors during styling/reindexing
            print(f"Error during styling/reindexing: {e}")
            print(f"Final column order: {final_column_order}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            raise e

    else:
        # Return unformatted styler if no columns matched or format_dict is empty
        # and no reordering was triggered by ci_column=False
        return df.style


# --- Plotting Functions ---

# Guard imports for plotting libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    Axes = None  # type: ignore # Need to define Axes type hint even if unavailable
    _MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    _SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    roc_curve = None
    precision_recall_curve = None
    auc = None
    _SKLEARN_METRICS_AVAILABLE = False


def plot_roc_curve(
    y_true: Any,
    y_score: Any,
    model_name: str | None = None,
    ax: Any | None = None,
    **kwargs,
):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true: True binary labels.
        y_score: Target scores, can either be probability estimates of the positive
                 class or confidence values.
        model_name: Optional name of the model for the plot label.
        ax: Optional Matplotlib Axes object to plot on. If None, a new figure
            and axes are created.
        **kwargs: Additional keyword arguments passed to `ax.plot()`.

    Returns:
        matplotlib.axes.Axes: The Axes object with the ROC curve plotted.

    Raises:
        ImportError: If matplotlib or scikit-learn is not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Please install it.")
    if not _SKLEARN_METRICS_AVAILABLE or roc_curve is None or auc is None:
        raise ImportError(
            "scikit-learn is required for ROC calculation. Please install it."
        )

    # Ensure input arrays are numpy arrays for sklearn compatibility
    # (Assuming they might be passed as lists or pandas Series)
    try:
        import numpy as np

        y_true_np = np.asarray(y_true)
        y_score_np = np.asarray(y_score)
    except ImportError:
        # numpy should be a core dependency anyway, but handle just in case
        raise ImportError("numpy is required for data processing before plotting.")
    except Exception as e:
        raise TypeError(f"Could not convert y_true or y_score to NumPy arrays: {e}")

    if ax is None and plt is not None:
        fig, ax = plt.subplots(figsize=(8, 8))  # Create figure and axes if not provided
    elif ax is None:
        # This case should not happen if _MATPLOTLIB_AVAILABLE is True, but safety check
        raise RuntimeError(
            "Matplotlib Axes object not provided and could not be created."
        )

    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true_np, y_score_np)
    roc_auc = auc(fpr, tpr)

    # Construct the label
    plot_label = f"AUC = {roc_auc:0.2f}"
    if model_name:
        plot_label = f"{model_name} (AUC = {roc_auc:0.2f})"
    else:
        plot_label = f"ROC curve (AUC = {roc_auc:0.2f})"

    # Plot the ROC curve
    ax.plot(fpr, tpr, label=plot_label, **kwargs)

    # Plot the chance line
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance (AUC = 0.5)")

    # Set labels, title, limits, grid, legend, aspect ratio
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])  # Slightly above 1 for visibility
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    return ax


def plot_precision_recall_curve(
    y_true: Any,
    y_score: Any,
    model_name: str | None = None,
    ax: Any | None = None,
    **kwargs,
):
    """
    Plots the Precision-Recall (PR) curve.

    Args:
        y_true: True binary labels.
        y_score: Target scores, can either be probability estimates of the positive
                 class or confidence values.
        model_name: Optional name of the model for the plot label.
        ax: Optional Matplotlib Axes object to plot on. If None, a new figure
            and axes are created.
        **kwargs: Additional keyword arguments passed to `ax.plot()`.

    Returns:
        matplotlib.axes.Axes: The Axes object with the PR curve plotted.

    Raises:
        ImportError: If matplotlib or scikit-learn is not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Please install it.")
    if not _SKLEARN_METRICS_AVAILABLE or precision_recall_curve is None:
        raise ImportError(
            "scikit-learn is required for PR curve calculation. Please install it."
        )

    # Ensure input arrays are numpy arrays for sklearn compatibility
    try:
        import numpy as np

        y_true_np = np.asarray(y_true)
        y_score_np = np.asarray(y_score)
    except ImportError:
        raise ImportError("numpy is required for data processing before plotting.")
    except Exception as e:
        raise TypeError(f"Could not convert y_true or y_score to NumPy arrays: {e}")

    if ax is None and plt is not None:
        fig, ax = plt.subplots(figsize=(8, 8))  # Create figure and axes if not provided
    elif ax is None:
        raise RuntimeError(
            "Matplotlib Axes object not provided and could not be created."
        )

    # Calculate Precision-Recall curve points
    precision, recall, _ = precision_recall_curve(y_true_np, y_score_np)

    # Construct the label
    plot_label = "PR Curve"
    if model_name:
        plot_label = f"{model_name} (PR Curve)"

    # Plot the Precision-Recall curve
    # Note: Plotting recall (x) vs precision (y)
    ax.plot(recall, precision, label=plot_label, **kwargs)

    # Set labels, title, limits, grid, legend
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])  # Slightly above 1 for visibility
    ax.legend(loc="best")  # Often lower-left or best for PR curves
    ax.grid(True)
    # Aspect ratio is usually not set to 'equal' for PR curves

    return ax
