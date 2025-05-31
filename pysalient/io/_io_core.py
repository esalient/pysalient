"""
Internal core IO operations: loading, name handling, aggregation.
"""

import os
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq


def _load_data_to_pyarrow(
    source: str | pd.DataFrame,
    resolved_source_type: str | None,
    read_options: dict[str, Any],
) -> tuple[pa.Table, str]:
    """
    Loads data from various sources into a PyArrow Table.

    Handles detection of source type if not provided and uses appropriate
    PyArrow readers.

    Args:
        source: Path to a CSV/Parquet file or a Pandas DataFrame.
        resolved_source_type: Explicit source type ('csv', 'parquet', 'pandas') or None.
        read_options: Dictionary containing specific read options for PyArrow readers.

    Returns:
        A tuple containing:
            - The loaded PyArrow Table.
            - The resolved source type string ('csv', 'parquet', 'pandas').

    Raises:
        FileNotFoundError: If the source path does not exist.
        ValueError: If source type is invalid or conversion fails.
        TypeError: If the source type is unsupported or cannot be inferred.
    """
    table: pa.Table | None = None

    if isinstance(source, pd.DataFrame):
        if resolved_source_type is None:
            resolved_source_type = "pandas"
        elif resolved_source_type != "pandas":
            raise ValueError(
                f"Source is a DataFrame, but source_type is '{resolved_source_type}'"
            )
        try:
            table = pa.Table.from_pandas(source, preserve_index=False)
        except Exception as e:
            raise ValueError(
                f"Failed to convert Pandas DataFrame to PyArrow Table: {e}"
            ) from e

    elif isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source file not found: {source}")

        if resolved_source_type is None:
            _, ext = os.path.splitext(source)
            ext = ext.lower()
            if ext == ".csv":
                resolved_source_type = "csv"
            elif ext == ".parquet":
                resolved_source_type = "parquet"
            else:
                raise TypeError(
                    f"Cannot infer source type from file extension: {ext}. Please specify source_type."
                )

        if resolved_source_type == "csv":
            try:
                csv_opts = read_options.get("csv", {})
                table = pv.read_csv(source, **csv_opts)
            except Exception as e:
                raise ValueError(
                    f"Failed to read CSV file '{source}' with PyArrow: {e}"
                ) from e
        elif resolved_source_type == "parquet":
            try:
                pq_opts = read_options.get("parquet", {})
                table = pq.read_table(source, **pq_opts)
            except Exception as e:
                raise ValueError(
                    f"Failed to read Parquet file '{source}' with PyArrow: {e}"
                ) from e
        else:
            raise TypeError(
                f"Unsupported source_type for file path: '{resolved_source_type}'"
            )

    else:
        raise TypeError(
            f"Unsupported source type: {type(source)}. Must be a file path (str) or Pandas DataFrame."
        )

    if table is None:
        # This should ideally not be reached if logic above is correct, but acts as a safeguard
        raise ValueError("Failed to load data into PyArrow Table.")

    # We need to ensure resolved_source_type is not None here before returning
    if resolved_source_type is None:
        raise ValueError(
            "Internal error: resolved_source_type was not determined."
        )  # Should not happen

    return table, resolved_source_type


def _handle_assigned_names(
    table: pa.Table,
    task_col: str | None,
    assign_task_name: str | None,
    model_col: str | None,
    assign_model_name: str | None,
) -> tuple[pa.Table, str | None, str | None]:
    """
    Handles assigning constant task/model names if provided and checks for conflicts.

    Adds 'task' or 'model' columns if `assign_task_name` or `assign_model_name`
    are provided and the corresponding `task_col` or `model_col` are None.

    Args:
        table: The input PyArrow Table.
        task_col: Original task column name from input args.
        assign_task_name: Constant task name to assign.
        model_col: Original model column name from input args.
        assign_model_name: Constant model name to assign.

    Returns:
        A tuple containing:
            - The potentially modified PyArrow Table (with added columns).
            - The final task column name (original or newly added 'task').
            - The final model column name (original or newly added 'model').

    Raises:
        ValueError: If conflicting arguments are provided (e.g., both `task_col`
                    and `assign_task_name`) or if trying to assign a name when
                    a column named 'task'/'model' already exists.
    """
    added_task_col_name = "task"
    added_model_col_name = "model"
    final_task_col = task_col
    final_model_col = model_col

    if task_col is not None and assign_task_name is not None:
        raise ValueError(
            f"Cannot provide both `task_col` ('{task_col}') and `assign_task_name` ('{assign_task_name}')."
        )
    if model_col is not None and assign_model_name is not None:
        raise ValueError(
            f"Cannot provide both `model_col` ('{model_col}') and `assign_model_name` ('{assign_model_name}')."
        )

    # Assign task name if requested and task_col not provided
    if task_col is None and assign_task_name is not None:
        if added_task_col_name in table.column_names:
            raise ValueError(
                f"Cannot assign task name: column '{added_task_col_name}' already exists in the data."
            )
        task_array = pa.array([assign_task_name] * table.num_rows, type=pa.string())
        table = table.append_column(
            pa.field(added_task_col_name, pa.string()), task_array
        )
        final_task_col = (
            added_task_col_name  # Use the newly added column for subsequent checks
        )

    # Assign model name if requested and model_col not provided
    if model_col is None and assign_model_name is not None:
        if added_model_col_name in table.column_names:
            raise ValueError(
                f"Cannot assign model name: column '{added_model_col_name}' already exists in the data."
            )
        model_array = pa.array([assign_model_name] * table.num_rows, type=pa.string())
        table = table.append_column(
            pa.field(added_model_col_name, pa.string()), model_array
        )
        final_model_col = (
            added_model_col_name  # Use the newly added column for subsequent checks
        )

    return table, final_task_col, final_model_col


def _perform_aggregation(
    table: pa.Table,
    aggregation_cols: str | list[str],
    y_proba_col: str,
    y_label_col: str,
    proba_agg_func: str = "mean",
    label_agg_func: str = "max",
) -> pa.Table:
    """
    Performs aggregation on the PyArrow Table using Pandas.

    Groups by `aggregation_cols` and applies specified aggregation functions
    to probability and label columns. Keeps other columns by taking the 'first'
    value within each group.

    Args:
        table: The input PyArrow Table.
        aggregation_cols: Column name(s) to group by.
        y_proba_col: Name of the probability column.
        y_label_col: Name of the label column.
        proba_agg_func: Aggregation function for the probability column ('mean', 'max', 'min', etc.).
        label_agg_func: Aggregation function for the label column ('max', 'first', 'mean', 'min', etc.).

    Returns:
        The aggregated PyArrow Table.

    Raises:
        ValueError: If aggregation columns are not found or aggregation fails.
    """
    # Ensure aggregation_cols is a list
    if isinstance(aggregation_cols, str):
        aggregation_cols_list = [aggregation_cols]
    else:
        # Ensure it's a list and make a copy to avoid modifying original
        aggregation_cols_list = list(aggregation_cols)

    # Check if aggregation columns actually exist before trying to group
    missing_agg_cols = set(aggregation_cols_list) - set(table.column_names)
    if missing_agg_cols:
        raise ValueError(f"Aggregation columns not found in data: {missing_agg_cols}")

    # Convert to pandas for easier aggregation
    df = table.to_pandas()

    # Define aggregation specification
    agg_spec = {}
    agg_spec[y_proba_col] = proba_agg_func
    agg_spec[y_label_col] = label_agg_func

    # Add other columns to keep, taking their first value within each group
    other_cols_to_keep = [
        col
        for col in df.columns
        if col not in aggregation_cols_list and col not in agg_spec
    ]
    agg_spec.update({col: "first" for col in other_cols_to_keep})

    # Perform the aggregation
    try:
        # Using as_index=False keeps aggregation_cols_list as columns
        aggregated_df = df.groupby(aggregation_cols_list, as_index=False).agg(agg_spec)
        # Ensure original column order where possible, prioritizing key columns
        original_order = [c for c in table.column_names if c in aggregated_df.columns]
        other_new_cols = [c for c in aggregated_df.columns if c not in original_order]
        aggregated_df = aggregated_df[original_order + other_new_cols]

    except Exception as e:
        raise ValueError(f"Pandas aggregation failed: {e}") from e

    # Convert back to PyArrow Table, replacing the original table
    try:
        # Preserve schema where possible, though types might change slightly (e.g. int->float for mean)
        # PyArrow handles schema inference here. Metadata will be re-applied later.
        aggregated_table = pa.Table.from_pandas(aggregated_df, preserve_index=False)
    except Exception as e:
        raise ValueError(
            f"Failed to convert aggregated Pandas DataFrame back to PyArrow Table: {e}"
        ) from e

    return aggregated_table
