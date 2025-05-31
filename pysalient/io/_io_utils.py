"""
Internal utilities for IO operations, like validation and metadata handling.
"""

import json

import pyarrow as pa


def _validate_columns(
    table: pa.Table,
    y_proba_col: str,
    y_label_col: str,
    timeseries_col: str,
    aggregation_cols: str | list[str] | None,
    model_col: str | None,
    task_col: str | None,
) -> tuple[list[str], dict[str, str]]:
    """
    Validates the existence and basic types of required and optional columns.

    Args:
        table: The PyArrow Table to validate.
        y_proba_col: Name of the probability column.
        y_label_col: Name of the label column.
        timeseries_col: Name of the timeseries column.
        aggregation_cols: Name(s) of the aggregation column(s).
        model_col: Name of the model column (if provided).
        task_col: Name of the task column (if provided).

    Returns:
        A tuple containing:
            - agg_cols_list: A list of aggregation column names (empty if None).
            - optional_cols_map: A dictionary mapping internal names ('model_col', 'task_col')
                                 to the actual column names used.

    Raises:
        ValueError: If required columns are missing or have fundamentally incorrect types.
    """
    required_cols: set[str] = {
        y_proba_col,
        y_label_col,
        timeseries_col,
    }
    agg_cols_list: list[str] = []

    if isinstance(aggregation_cols, str):
        required_cols.add(aggregation_cols)
        agg_cols_list = [aggregation_cols]
    elif isinstance(aggregation_cols, list):
        required_cols.update(aggregation_cols)
        agg_cols_list = list(aggregation_cols)
    # If aggregation_cols is None, agg_cols_list remains empty

    optional_cols_map: dict[str, str] = {}
    if model_col:
        required_cols.add(model_col)
        optional_cols_map["model_col"] = model_col
    if task_col:
        required_cols.add(task_col)
        optional_cols_map["task_col"] = task_col

    missing_cols = required_cols - set(table.column_names)
    if missing_cols:
        raise ValueError(f"Missing required columns in the loaded data: {missing_cols}")

    # --- Type Validations ---
    # Validate timeseries column type
    ts_field = table.schema.field(timeseries_col)
    if not (pa.types.is_temporal(ts_field.type) or pa.types.is_floating(ts_field.type)):
        raise ValueError(
            f"Column '{timeseries_col}' must be a temporal (date/time/timestamp) "
            f"or floating-point type, but found {ts_field.type}."
        )

    # Validate probability column type
    proba_field = table.schema.field(y_proba_col)
    if not pa.types.is_floating(proba_field.type):
        raise ValueError(
            f"Probability column '{y_proba_col}' must be a floating-point type, "
            f"but found {proba_field.type}."
        )

    # Validate label column type (allow numeric or boolean)
    label_field = table.schema.field(y_label_col)
    if not (
        pa.types.is_integer(label_field.type)
        or pa.types.is_floating(label_field.type)
        or pa.types.is_boolean(label_field.type)
    ):
        raise ValueError(
            f"Label column '{y_label_col}' must be a numeric (integer/float) or boolean type, "
            f"but found {label_field.type}."
        )

    # Optional: Validate model/task columns are string-like if they exist
    if model_col and not pa.types.is_string(table.schema.field(model_col).type):
        # Allow dictionary type for potential categorical mapping? For now, strict string.
        if not pa.types.is_dictionary(table.schema.field(model_col).type):
            raise ValueError(
                f"Model column '{model_col}' must be a string or dictionary type, but found {table.schema.field(model_col).type}."
            )
    if task_col and not pa.types.is_string(table.schema.field(task_col).type):
        if not pa.types.is_dictionary(table.schema.field(task_col).type):
            raise ValueError(
                f"Task column '{task_col}' must be a string or dictionary type, but found {table.schema.field(task_col).type}."
            )

    return agg_cols_list, optional_cols_map


def _attach_metadata(
    table: pa.Table,
    y_proba_col: str,
    y_label_col: str,
    timeseries_col: str,
    agg_cols_list: list[str],
    optional_cols_map: dict[str, str],
) -> pa.Table:
    """
    Attaches standardized metadata keys to the table schema.

    Args:
        table: The PyArrow Table.
        y_proba_col: Name of the probability column.
        y_label_col: Name of the label column.
        timeseries_col: Name of the timeseries column.
        agg_cols_list: List of aggregation column names.
        optional_cols_map: Dictionary mapping internal names ('model_col', 'task_col')
                           to actual column names.

    Returns:
        The PyArrow Table with updated schema metadata.
    """
    metadata = {
        b"pysalient.io.y_proba_col": y_proba_col.encode("utf-8"),
        b"pysalient.io.y_label_col": y_label_col.encode("utf-8"),
        b"pysalient.io.timeseries_col": timeseries_col.encode("utf-8"),
        # Store aggregation cols as a JSON list string for consistency
        b"pysalient.io.aggregation_cols": json.dumps(agg_cols_list).encode("utf-8"),
    }
    if "model_col" in optional_cols_map:
        metadata[b"pysalient.io.model_col"] = optional_cols_map["model_col"].encode(
            "utf-8"
        )
    if "task_col" in optional_cols_map:
        metadata[b"pysalient.io.task_col"] = optional_cols_map["task_col"].encode(
            "utf-8"
        )

    # Attach the metadata to the table's schema
    # Preserve existing metadata if any
    existing_metadata = table.schema.metadata or {}
    existing_metadata.update(metadata)  # Add/overwrite our keys
    table_with_metadata = table.replace_schema_metadata(existing_metadata)
    return table_with_metadata
