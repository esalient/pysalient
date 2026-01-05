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
                    raise ValueError(
                        f"Column '{order_by}' not found in table. Available columns: {df.columns.tolist()}"
                    )
            elif isinstance(order_by, list):
                # Multiple columns
                missing_cols = [col for col in order_by if col not in df.columns]
                if missing_cols:
                    raise ValueError(
                        f"Columns {missing_cols} not found in table. Available columns: {df.columns.tolist()}"
                    )
                df = df.sort_values(by=order_by).reset_index(drop=True)
            else:
                raise TypeError(
                    "'order_by' must be a string, list of strings, or None."
                )
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
            if "_from_first_alert_to_" in col and not col.startswith(
                "count_"
            ):  # Exclude count columns (integers)
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

# Guard import for Altair
try:
    import altair as alt

    _ALTAIR_AVAILABLE = True
except ImportError:
    alt = None
    _ALTAIR_AVAILABLE = False


def plot_roc_curve(
    evaluation_result: pa.Table,
    threshold: float | None = None,
    width: int = 400,
    height: int = 400,
) -> Any:
    """
    Plots the ROC curve using Altair from evaluation results.

    This function consumes the output of `pysalient.evaluation.evaluation()`
    when called with `export_roc_curve_data=True`, and creates an interactive
    Altair chart displaying the ROC curve.

    Args:
        evaluation_result: PyArrow Table from evaluation() with export_roc_curve_data=True.
                          Must contain 'ROC_FPR', 'ROC_TPR', 'ROC_Thresholds', and 'AUROC' columns.
        threshold: Optional threshold value to highlight on the curve. When provided,
                   displays a point marker at the corresponding (FPR, TPR) position,
                   an annotation with the threshold value, and dashed lines to both axes.
        width: Chart width in pixels. Defaults to 400.
        height: Chart height in pixels. Defaults to 400.

    Returns:
        altair.Chart: The ROC curve chart object, ready for display in Jupyter
                      or export to various formats.

    Raises:
        ImportError: If Altair is not installed.
        ValueError: If ROC curve data columns are not found in evaluation_result.

    Example:
        >>> import pysalient.evaluation as ev
        >>> import pysalient.visualisation as viz
        >>> result = ev.evaluation(data, "model1", "test", [0.3, 0.5], export_roc_curve_data=True)
        >>> chart = viz.plot_roc_curve(result, threshold=0.5)
        >>> chart  # Display in Jupyter
    """
    if not _ALTAIR_AVAILABLE:
        raise ImportError(
            "altair is required for this function. "
            "Please install it with: pip install altair>=5.0"
        )

    # Validate required columns exist
    required_cols = ["ROC_FPR", "ROC_TPR", "ROC_Thresholds", "AUROC"]
    missing_cols = [
        col for col in required_cols if col not in evaluation_result.column_names
    ]
    if missing_cols:
        raise ValueError(
            f"ROC curve data columns not found: {missing_cols}. "
            "Call evaluation() with export_roc_curve_data=True to include curve data."
        )

    # Extract first row (curve data is same for all thresholds)
    df = evaluation_result.to_pandas()
    fpr = df["ROC_FPR"].iloc[0]
    tpr = df["ROC_TPR"].iloc[0]
    thresholds = df["ROC_Thresholds"].iloc[0]
    auroc = df["AUROC"].iloc[0]

    # Handle None/NaN curve data
    if fpr is None or tpr is None:
        raise ValueError(
            "ROC curve data is None. This may occur when only one class is present in labels."
        )

    # Create curve dataframe
    import numpy as np

    curve_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds})

    # Base ROC curve
    roc_line = (
        alt.Chart(curve_df)
        .mark_line(color="steelblue", strokeWidth=2)
        .encode(
            x=alt.X(
                "FPR:Q",
                title="False Positive Rate (1 - Specificity)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "TPR:Q",
                title="True Positive Rate (Sensitivity)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("FPR:Q", format=".3f", title="FPR"),
                alt.Tooltip("TPR:Q", format=".3f", title="TPR"),
                alt.Tooltip("Threshold:Q", format=".3f", title="Threshold"),
            ],
        )
    )

    # Diagonal reference line (chance line)
    diagonal_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    diagonal = (
        alt.Chart(diagonal_df)
        .mark_line(strokeDash=[5, 5], color="grey", strokeWidth=1)
        .encode(x=alt.X("x:Q"), y=alt.Y("y:Q"))
    )

    # Title with AUROC
    title = f"ROC Curve (AUROC = {auroc:.3f})"

    # Combine base chart
    chart = (roc_line + diagonal).properties(width=width, height=height, title=title)

    # Add threshold highlighting if specified
    if threshold is not None:
        # Find closest threshold point
        threshold_array = np.array(thresholds)
        idx = np.abs(threshold_array - threshold).argmin()
        point_fpr = fpr[idx]
        point_tpr = tpr[idx]
        actual_threshold = thresholds[idx]

        point_df = pd.DataFrame(
            {"FPR": [point_fpr], "TPR": [point_tpr], "Threshold": [actual_threshold]}
        )

        # Point marker
        point = (
            alt.Chart(point_df)
            .mark_circle(size=100, color="red", filled=True)
            .encode(
                x="FPR:Q",
                y="TPR:Q",
                tooltip=[
                    alt.Tooltip("FPR:Q", format=".3f", title="FPR"),
                    alt.Tooltip("TPR:Q", format=".3f", title="TPR"),
                    alt.Tooltip("Threshold:Q", format=".3f", title="Threshold"),
                ],
            )
        )

        # Annotation text
        text = (
            alt.Chart(point_df)
            .mark_text(
                align="left", dx=10, dy=-10, fontSize=12, color="red", fontWeight="bold"
            )
            .encode(x="FPR:Q", y="TPR:Q", text=alt.value(f"t={actual_threshold:.3f}"))
        )

        # Horizontal line to y-axis
        h_line_df = pd.DataFrame({"x": [0, point_fpr], "y": [point_tpr, point_tpr]})
        h_line = (
            alt.Chart(h_line_df)
            .mark_line(strokeDash=[2, 2], color="red", opacity=0.5, strokeWidth=1)
            .encode(x="x:Q", y="y:Q")
        )

        # Vertical line to x-axis
        v_line_df = pd.DataFrame({"x": [point_fpr, point_fpr], "y": [0, point_tpr]})
        v_line = (
            alt.Chart(v_line_df)
            .mark_line(strokeDash=[2, 2], color="red", opacity=0.5, strokeWidth=1)
            .encode(x="x:Q", y="y:Q")
        )

        chart = chart + point + text + h_line + v_line

    return chart


def plot_precision_recall_curve(
    evaluation_result: pa.Table,
    threshold: float | None = None,
    width: int = 400,
    height: int = 400,
) -> Any:
    """
    Plots the Precision-Recall curve using Altair from evaluation results.

    This function consumes the output of `pysalient.evaluation.evaluation()`
    when called with `export_roc_curve_data=True`, and creates an interactive
    Altair chart displaying the Precision-Recall curve.

    Args:
        evaluation_result: PyArrow Table from evaluation() with export_roc_curve_data=True.
                          Must contain 'PR_Precision', 'PR_Recall', 'PR_Thresholds',
                          and 'AUPRC' columns.
        threshold: Optional threshold value to highlight on the curve. When provided,
                   displays a point marker at the corresponding (Recall, Precision) position,
                   an annotation with the threshold value, and dashed lines to both axes.
        width: Chart width in pixels. Defaults to 400.
        height: Chart height in pixels. Defaults to 400.

    Returns:
        altair.Chart: The Precision-Recall curve chart object, ready for display
                      in Jupyter or export to various formats.

    Raises:
        ImportError: If Altair is not installed.
        ValueError: If PR curve data columns are not found in evaluation_result.

    Example:
        >>> import pysalient.evaluation as ev
        >>> import pysalient.visualisation as viz
        >>> result = ev.evaluation(data, "model1", "test", [0.3, 0.5], export_roc_curve_data=True)
        >>> chart = viz.plot_precision_recall_curve(result, threshold=0.5)
        >>> chart  # Display in Jupyter
    """
    if not _ALTAIR_AVAILABLE:
        raise ImportError(
            "altair is required for this function. "
            "Please install it with: pip install altair>=5.0"
        )

    # Validate required columns exist
    required_cols = ["PR_Precision", "PR_Recall", "PR_Thresholds", "AUPRC"]
    missing_cols = [
        col for col in required_cols if col not in evaluation_result.column_names
    ]
    if missing_cols:
        raise ValueError(
            f"PR curve data columns not found: {missing_cols}. "
            "Call evaluation() with export_roc_curve_data=True to include curve data."
        )

    # Extract first row (curve data is same for all thresholds)
    df = evaluation_result.to_pandas()
    precision = df["PR_Precision"].iloc[0]
    recall = df["PR_Recall"].iloc[0]
    pr_thresholds = df["PR_Thresholds"].iloc[0]
    auprc = df["AUPRC"].iloc[0]

    # Handle None/NaN curve data
    if precision is None or recall is None:
        raise ValueError(
            "PR curve data is None. This may occur when only one class is present in labels."
        )

    # Note: sklearn's precision_recall_curve returns thresholds with length n-1
    # where n is the length of precision/recall. The last precision/recall point
    # corresponds to threshold=1.0 (all predictions negative).
    # We need to handle this length mismatch.
    import numpy as np

    # Extend thresholds to match precision/recall length
    # The last point is at threshold approaching infinity (or 1.0 for normalized scores)
    if len(pr_thresholds) < len(precision):
        extended_thresholds = list(pr_thresholds) + [1.0] * (
            len(precision) - len(pr_thresholds)
        )
    else:
        extended_thresholds = pr_thresholds

    # Create curve dataframe
    curve_df = pd.DataFrame(
        {"Recall": recall, "Precision": precision, "Threshold": extended_thresholds}
    )

    # Base PR curve
    pr_line = (
        alt.Chart(curve_df)
        .mark_line(color="darkorange", strokeWidth=2)
        .encode(
            x=alt.X(
                "Recall:Q", title="Recall (Sensitivity)", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y("Precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("Recall:Q", format=".3f", title="Recall"),
                alt.Tooltip("Precision:Q", format=".3f", title="Precision"),
                alt.Tooltip("Threshold:Q", format=".3f", title="Threshold"),
            ],
        )
    )

    # Title with AUPRC
    title = f"Precision-Recall Curve (AUPRC = {auprc:.3f})"

    # Base chart (no diagonal for PR curves)
    chart = pr_line.properties(width=width, height=height, title=title)

    # Add threshold highlighting if specified
    if threshold is not None:
        # Find closest threshold point (use original thresholds for matching)
        threshold_array = np.array(pr_thresholds)
        idx = np.abs(threshold_array - threshold).argmin()

        # Ensure we don't go out of bounds
        idx = min(idx, len(recall) - 1)

        point_recall = recall[idx]
        point_precision = precision[idx]
        actual_threshold = pr_thresholds[idx] if idx < len(pr_thresholds) else 1.0

        point_df = pd.DataFrame(
            {
                "Recall": [point_recall],
                "Precision": [point_precision],
                "Threshold": [actual_threshold],
            }
        )

        # Point marker
        point = (
            alt.Chart(point_df)
            .mark_circle(size=100, color="red", filled=True)
            .encode(
                x="Recall:Q",
                y="Precision:Q",
                tooltip=[
                    alt.Tooltip("Recall:Q", format=".3f", title="Recall"),
                    alt.Tooltip("Precision:Q", format=".3f", title="Precision"),
                    alt.Tooltip("Threshold:Q", format=".3f", title="Threshold"),
                ],
            )
        )

        # Annotation text
        text = (
            alt.Chart(point_df)
            .mark_text(
                align="left", dx=10, dy=-10, fontSize=12, color="red", fontWeight="bold"
            )
            .encode(
                x="Recall:Q",
                y="Precision:Q",
                text=alt.value(f"t={actual_threshold:.3f}"),
            )
        )

        # Horizontal line to y-axis
        h_line_df = pd.DataFrame(
            {"x": [0, point_recall], "y": [point_precision, point_precision]}
        )
        h_line = (
            alt.Chart(h_line_df)
            .mark_line(strokeDash=[2, 2], color="red", opacity=0.5, strokeWidth=1)
            .encode(x="x:Q", y="y:Q")
        )

        # Vertical line to x-axis
        v_line_df = pd.DataFrame(
            {"x": [point_recall, point_recall], "y": [0, point_precision]}
        )
        v_line = (
            alt.Chart(v_line_df)
            .mark_line(strokeDash=[2, 2], color="red", opacity=0.5, strokeWidth=1)
            .encode(x="x:Q", y="y:Q")
        )

        chart = chart + point + text + h_line + v_line

    return chart
