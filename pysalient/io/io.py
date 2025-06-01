# import os
from typing import Any

import pandas as pd  # Still needed for type hint and potential aggregation
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from pandas.io.formats.style import Styler  # For type hinting formatted results

# Import internal helpers using relative imports
from ._io_core import (
    _handle_assigned_names,
    _load_data_to_pyarrow,
    _perform_aggregation,
)
from ._io_utils import _attach_metadata, _validate_columns


def load_evaluation_data(
    source: str | pd.DataFrame,
    y_proba_col: str,
    y_label_col: str,
    aggregation_cols: str | list[str],
    timeseries_col: str,
    model_col: str | None = None,
    task_col: str | None = None,
    assign_task_name: str | None = None,
    assign_model_name: str | None = None,
    y_proba_col_impute: float = 0.0,
    source_type: (
        str | None
    ) = None,  # Explicitly 'csv', 'parquet', 'pandas' if needed, otherwise inferred
    read_options: (
        dict[str, Any] | None
    ) = None,  # e.g., {'csv': {'delimiter': ','}, 'parquet': {}}
    **kwargs: Any,
) -> pa.Table:
    """
    Loads and validates evaluation data from various sources into a PyArrow Table.

    Can optionally add 'task' and 'model' columns if they don't exist in the source
    by providing constant names via `assign_task_name` and `assign_model_name`.

    Args:
        source: Path to a CSV/Parquet file or a Pandas DataFrame.
        y_proba_col: Name of the column containing predicted probabilities.
        y_label_col: Name of the column containing the true labels.
        aggregation_cols: Name of the column(s) used for grouping/aggregation
                          (e.g., 'encounter_id', ['person_id', 'admission_id']).
        timeseries_col: Name of the column containing the event time (datetime/timestamp) or relative time (float).
        model_col: Optional name of the column *already existing* in the source data
                   that identifies different models (for multi-model comparison).
                   Takes precedence over `assign_model_name`. Cannot be used if `assign_model_name` is set.
        task_col: Optional name of the column *already existing* in the source data
                  that identifies different prediction tasks (for multi-task evaluation).
                  Takes precedence over `assign_task_name`. Cannot be used if `assign_task_name` is set.
        assign_task_name: Optional name to assign to all rows for the task.
                          If provided and `task_col` is None, a new column named 'task'
                          will be added to the table with this constant value. Raises ValueError
                          if a column named 'task' already exists or if `task_col` is also provided.
        assign_model_name: Optional name to assign to all rows for the model.
                           If provided and `model_col` is None, a new column named 'model'
                           will be added to the table with this constant value. Raises ValueError
                           if a column named 'model' already exists or if `model_col` is also provided.
        y_proba_col_impute: Value to use for imputing NaN values in the y_proba_col.
                           Defaults to 0.0. This ensures that NaN prediction probabilities
                           don't cause downstream evaluation failures.
        source_type: Optional hint for the source type ('csv', 'parquet', 'pandas').
                     If None, attempts to infer from the source path extension or type.
        read_options: Optional dictionary containing specific read options for
                      pyarrow readers (e.g., CSV delimiter). Keys should match
                      source_type ('csv', 'parquet').
        **kwargs: Additional keyword arguments including:
            perform_aggregation: bool, default=False
                If True and aggregation_cols is provided, aggregates the data by the specified columns.
                If False, the aggregation columns are recorded in metadata but no aggregation is performed.
            proba_agg_func: str, default='mean'
                The aggregation function to use for the probability column when aggregating.
                Common values: 'mean', 'max', 'min'.
            label_agg_func: str, default='max'
                The aggregation function to use for the label column when aggregating.
                Default is 'max' to indicate any positive case in a group makes the group positive.
                Other common values: 'first', 'mean', 'min'.

    Returns:
        A validated PyArrow Table containing the required columns (potentially including
        added 'task' and/or 'model' columns). If aggregation is performed (perform_aggregation=True),
        the resulting table will have one row per aggregation group.

    Raises:
        FileNotFoundError: If the source path does not exist.
        ValueError: If required columns are missing, types are incorrect, source type is invalid,
                    conflicting arguments are provided (e.g., both `task_col` and `assign_task_name`),
                    or if trying to assign a task/model name when a column named 'task'/'model'
                    already exists.
        TypeError: If the source type is unsupported or cannot be inferred.
    """
    table: pa.Table = None
    # resolved_source_type = source_type # Handled internally now

    if read_options is None:
        read_options = {}

    #####################################
    # 1. Detect Source Type & Load Data #
    #####################################
    # Use the internal helper function to load data
    try:
        table, resolved_source_type = _load_data_to_pyarrow(
            source=source,
            resolved_source_type=source_type,  # Pass original hint
            read_options=read_options,
        )
    except (FileNotFoundError, ValueError, TypeError) as e:
        # Re-raise errors from the loading function
        raise e
    # 'table' and 'resolved_source_type' are now populated

    # 2. Handle assigned task/model names and perform conflict checks #
    ###################################################################
    try:
        table, task_col, model_col = _handle_assigned_names(
            table=table,
            task_col=task_col,
            assign_task_name=assign_task_name,
            model_col=model_col,
            assign_model_name=assign_model_name,
        )
    except ValueError as e:
        # Re-raise errors from the helper
        raise e
    # 'table', 'task_col', 'model_col' are now potentially updated

    #################################
    # 2.3 Handle NaN Imputation     #
    #################################
    # Impute NaN values in the probability column if present
    if y_proba_col in table.column_names:
        proba_column = table[y_proba_col]
        # Check if column has any null values using PyArrow compute
        import pyarrow.compute as pc
        
        null_count = pc.sum(pc.is_null(proba_column)).as_py()
        if null_count > 0:
            # Impute NaN/null values with the specified impute value
            imputed_column = pc.fill_null(proba_column, y_proba_col_impute)
            
            # Create new table with imputed column
            schema = table.schema
            columns = []
            for i, field in enumerate(schema):
                if field.name == y_proba_col:
                    columns.append(imputed_column)
                else:
                    columns.append(table.column(i))
            
            table = pa.table(columns, schema=schema)

    ################################
    # 2.5 Perform Aggregation (if requested) #
    ################################
    perform_aggregation = kwargs.get("perform_aggregation", False)
    if aggregation_cols is not None and perform_aggregation:
        try:
            table = _perform_aggregation(
                table=table,
                aggregation_cols=aggregation_cols,
                y_proba_col=y_proba_col,
                y_label_col=y_label_col,
                proba_agg_func=kwargs.get("proba_agg_func", "mean"),
                label_agg_func=kwargs.get("label_agg_func", "max"),
            )
        except ValueError as e:
            # Re-raise errors from the helper
            raise e
        # 'table' is now the aggregated table

    # 3. Validate Columns (Now includes potentially added task/model columns) #
    ###########################################################################
    try:
        # Note: We pass the *original* aggregation_cols parameter here,
        # as _validate_columns handles converting str to list internally if needed.
        agg_cols_list, optional_cols_map = _validate_columns(
            table=table,
            y_proba_col=y_proba_col,
            y_label_col=y_label_col,
            timeseries_col=timeseries_col,
            aggregation_cols=aggregation_cols,  # Pass original spec
            model_col=model_col,  # Pass potentially updated name
            task_col=task_col,  # Pass potentially updated name
        )
    except ValueError as e:
        # Re-raise validation errors
        raise e
    # agg_cols_list and optional_cols_map are now populated

    ##########################
    # 4. Add Schema Metadata #
    ##########################
    try:
        table = _attach_metadata(
            table=table,
            y_proba_col=y_proba_col,
            y_label_col=y_label_col,
            timeseries_col=timeseries_col,
            agg_cols_list=agg_cols_list,
            optional_cols_map=optional_cols_map,
        )
    except Exception as e:
        # Catch potential errors during metadata attachment
        raise RuntimeError(f"Failed to attach metadata to table: {e}") from e

    return table


def export_evaluation_results(
    results_table: pa.Table,
    output_path: str | None = None,
    format: str = "dataframe",
    **kwargs: Any,
) -> pd.DataFrame | None:
    """
    Exports raw evaluation results (PyArrow Table) to different formats.

    Prioritizes native PyArrow writers for CSV and Parquet formats.

    Args:
        results_table: The PyArrow Table containing the evaluation results.
        output_path: The file path to write to. Required for 'csv' and 'parquet' formats.
                     Ignored for 'dataframe' format.
        format: The desired output format. Options: 'csv', 'parquet', 'dataframe'.
                Defaults to 'dataframe'.
        **kwargs: Additional keyword arguments to pass to the underlying write functions.
                  For 'csv': pyarrow.csv.WriteOptions (e.g., include_header=True).
                  For 'parquet': pyarrow.parquet.write_table options (e.g., compression='snappy').
                  For 'dataframe': pandas.DataFrame constructor options (rarely needed here).

    Returns:
        If format is 'dataframe', returns a Pandas DataFrame.
        If format is 'csv' or 'parquet', writes to the specified output_path and returns None.

    Raises:
        ValueError: If `results_table` is not a PyArrow Table.
        ValueError: If `format` is not one of 'csv', 'parquet', 'dataframe'.
        ValueError: If `output_path` is None when `format` is 'csv' or 'parquet'.
        TypeError: If `results_table` is not a PyArrow Table.
        Any exceptions raised by the underlying PyArrow or Pandas write functions.
    """
    if not isinstance(results_table, pa.Table):
        raise TypeError(
            f"Expected results_table to be a pyarrow.Table, got {type(results_table).__name__}"
        )

    valid_formats = ["dataframe", "csv", "parquet"]
    if format not in valid_formats:
        raise ValueError(f"Invalid format '{format}'. Must be one of {valid_formats}")

    if format in ["csv", "parquet"] and output_path is None:
        raise ValueError(f"output_path must be provided for format '{format}'")

    if format == "dataframe":
        # Convert to Pandas DataFrame using any relevant kwargs
        # Note: kwargs here might be intended for pandas constructor, use carefully
        return results_table.to_pandas(**kwargs)
    elif format == "csv":
        # Use PyArrow's CSV writer
        # Separate WriteOptions if provided
        write_options_dict = kwargs.pop("write_options", {})
        write_options = pv.WriteOptions(**write_options_dict)
        pv.write_csv(results_table, output_path, write_options=write_options, **kwargs)
        return None
    elif format == "parquet":
        # Use PyArrow's Parquet writer
        pq.write_table(results_table, output_path, **kwargs)
        return None
    # Should be unreachable due to format validation, but added for safety
    else:
        raise ValueError(
            f"Internal error: Unhandled format '{format}'"
        )  # Should not happen


def export_formatted_results(
    styler: Styler,
    output_path: str | None = None,
    format: str = "dataframe",
    **kwargs: Any,
) -> pd.DataFrame | None:
    """
    Exports formatted evaluation results (Pandas Styler) to different formats.

    Uses Pandas writers as PyArrow cannot handle Styler formatting directly.
    Note that Styler formatting (e.g., colors, conditional formatting) might
    not be preserved in CSV or Parquet formats, but the underlying data is exported.

    Args:
        styler: The Pandas Styler object containing the formatted evaluation results.
        output_path: The file path to write to. Required for 'csv' and 'parquet' formats.
                     Ignored for 'dataframe' format.
        format: The desired output format. Options: 'csv', 'parquet', 'dataframe'.
                Defaults to 'dataframe'.
        **kwargs: Additional keyword arguments to pass to the underlying Pandas write functions.
                  For 'csv': pandas.DataFrame.to_csv options (e.g., index=False, sep=',').
                  For 'parquet': pandas.DataFrame.to_parquet options (e.g., engine='pyarrow', compression='snappy').

    Returns:
        If format is 'dataframe', returns the underlying Pandas DataFrame from the Styler.
        If format is 'csv' or 'parquet', writes to the specified output_path and returns None.

    Raises:
        ValueError: If `styler` is not a Pandas Styler object.
        ValueError: If `format` is not one of 'csv', 'parquet', 'dataframe'.
        ValueError: If `output_path` is None when `format` is 'csv' or 'parquet'.
        TypeError: If `styler` is not a Pandas Styler object.
        Any exceptions raised by the underlying Pandas write functions.
    """
    if not isinstance(styler, Styler):
        raise TypeError(
            f"Expected styler to be a pandas.io.formats.style.Styler, got {type(styler).__name__}"
        )

    valid_formats = ["dataframe", "csv", "parquet"]
    if format not in valid_formats:
        raise ValueError(f"Invalid format '{format}'. Must be one of {valid_formats}")

    if format in ["csv", "parquet"] and output_path is None:
        raise ValueError(f"output_path must be provided for format '{format}'")

    # Get the underlying data
    df = styler.data

    if format == "dataframe":
        return df
    elif format == "csv":
        # Use Pandas' to_csv
        df.to_csv(output_path, **kwargs)
        return None
    elif format == "parquet":
        # Use Pandas' to_parquet
        df.to_parquet(output_path, **kwargs)
        return None
    # Should be unreachable due to format validation, but added for safety
    else:
        raise ValueError(
            f"Internal error: Unhandled format '{format}'"
        )  # Should not happen
