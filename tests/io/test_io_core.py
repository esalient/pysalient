import pandas as pd
import pyarrow as pa
import pytest
from pandas.io.formats.style import Styler

from pysalient.io._io_core import _load_data_to_pyarrow, _perform_aggregation
from pysalient.io.io import export_evaluation_results, export_formatted_results


def test_load_data_to_pyarrow_dataframe_with_conflicting_source_type():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="source_type is 'csv'"):
        _load_data_to_pyarrow(df, "csv", {})


def test_load_data_to_pyarrow_unknown_extension(tmp_path):
    source = tmp_path / "file.xyz"
    source.write_text("x")
    with pytest.raises(TypeError, match="Cannot infer source type"):
        _load_data_to_pyarrow(str(source), None, {})


def test_load_data_to_pyarrow_read_failure_csv(monkeypatch, tmp_path):
    source = tmp_path / "file.csv"
    source.write_text("a\n1\n")

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("pysalient.io._io_core.pv.read_csv", _raise)
    with pytest.raises(ValueError, match="Failed to read CSV file"):
        _load_data_to_pyarrow(str(source), "csv", {})


def test_perform_aggregation_missing_columns():
    table = pa.table({"x": [1, 2], "y": [0.1, 0.2], "label": [0, 1]})
    with pytest.raises(ValueError, match="Aggregation columns not found"):
        _perform_aggregation(table, "missing", "y", "label")


def test_perform_aggregation_pandas_failure(monkeypatch):
    table = pa.table({"id": [1, 1], "y": [0.1, 0.2], "label": [0, 1]})

    def _raise(*args, **kwargs):
        raise RuntimeError("agg failed")

    monkeypatch.setattr("pandas.core.groupby.generic.DataFrameGroupBy.agg", _raise)
    with pytest.raises(ValueError, match="Pandas aggregation failed"):
        _perform_aggregation(table, "id", "y", "label")


def test_export_evaluation_results_invalid_format():
    table = pa.table({"a": [1]})
    with pytest.raises(ValueError, match="Invalid format"):
        export_evaluation_results(table, format="invalid")


def test_export_evaluation_results_missing_output_path():
    table = pa.table({"a": [1]})
    with pytest.raises(ValueError, match="output_path must be provided"):
        export_evaluation_results(table, format="csv")


def test_export_formatted_results_invalid_styler_type():
    with pytest.raises(TypeError, match="Expected styler"):
        export_formatted_results(123)


def test_export_formatted_results_all_format_branches(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    styler = Styler(df)

    returned_df = export_formatted_results(styler, format="dataframe")
    assert returned_df.equals(df)

    csv_path = tmp_path / "out.csv"
    parquet_path = tmp_path / "out.parquet"
    assert export_formatted_results(styler, output_path=str(csv_path), format="csv") is None
    assert csv_path.exists()
    assert (
        export_formatted_results(styler, output_path=str(parquet_path), format="parquet")
        is None
    )
    assert parquet_path.exists()
