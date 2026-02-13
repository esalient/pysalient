"""Tests for pysalient.io._io_utils helpers."""

import json

import pandas as pd
import pyarrow as pa
import pytest

from pysalient.io._io_utils import _attach_metadata, _validate_columns


def _base_table() -> pa.Table:
    return pa.table(
        {
            "encounter_id": ["e1", "e1", "e2"],
            "event_timestamp": pa.array(
                pd.to_datetime(
                    [
                        "2024-01-01T00:00:00",
                        "2024-01-01T01:00:00",
                        "2024-01-01T02:00:00",
                    ]
                ),
                type=pa.timestamp("ns"),
            ),
            "prediction_probability": [0.1, 0.9, 0.3],
            "true_label": [0, 1, 0],
            "model_identifier": ["m1", "m1", "m2"],
            "task_identifier": ["t1", "t1", "t2"],
        }
    )


def test_validate_columns_success_with_all_inputs():
    table = _base_table()
    agg_cols_list, optional_cols_map = _validate_columns(
        table=table,
        y_proba_col="prediction_probability",
        y_label_col="true_label",
        timeseries_col="event_timestamp",
        aggregation_cols=["encounter_id"],
        model_col="model_identifier",
        task_col="task_identifier",
    )
    assert agg_cols_list == ["encounter_id"]
    assert optional_cols_map == {
        "model_col": "model_identifier",
        "task_col": "task_identifier",
    }


def test_validate_columns_supports_float_timeseries_and_none_aggregation():
    table = pa.table(
        {
            "time_float": [0.0, 1.0, 2.0],
            "prediction_probability": [0.3, 0.7, 0.8],
            "true_label": [0, 1, 1],
        }
    )
    agg_cols_list, optional_cols_map = _validate_columns(
        table=table,
        y_proba_col="prediction_probability",
        y_label_col="true_label",
        timeseries_col="time_float",
        aggregation_cols=None,
        model_col=None,
        task_col=None,
    )
    assert agg_cols_list == []
    assert optional_cols_map == {}


def test_validate_columns_missing_required_column_raises():
    table = _base_table().drop(["prediction_probability"])
    with pytest.raises(ValueError, match="Missing required columns"):
        _validate_columns(
            table=table,
            y_proba_col="prediction_probability",
            y_label_col="true_label",
            timeseries_col="event_timestamp",
            aggregation_cols="encounter_id",
            model_col=None,
            task_col=None,
        )


def test_validate_columns_invalid_timeseries_type_raises():
    table = pa.table(
        {
            "encounter_id": ["e1", "e2"],
            "event_timestamp": ["now", "later"],
            "prediction_probability": [0.3, 0.7],
            "true_label": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="must be a temporal .* or floating-point"):
        _validate_columns(
            table=table,
            y_proba_col="prediction_probability",
            y_label_col="true_label",
            timeseries_col="event_timestamp",
            aggregation_cols="encounter_id",
            model_col=None,
            task_col=None,
        )


def test_validate_columns_invalid_probability_type_raises():
    table = pa.table(
        {
                "encounter_id": ["e1", "e2"],
                "event_timestamp": pa.array(
                    pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T01:00:00"]),
                    type=pa.timestamp("ns"),
                ),
            "prediction_probability": [1, 0],
            "true_label": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="must be a floating-point type"):
        _validate_columns(
            table=table,
            y_proba_col="prediction_probability",
            y_label_col="true_label",
            timeseries_col="event_timestamp",
            aggregation_cols="encounter_id",
            model_col=None,
            task_col=None,
        )


def test_validate_columns_invalid_label_type_raises():
    table = pa.table(
        {
                "encounter_id": ["e1", "e2"],
                "event_timestamp": pa.array(
                    pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T01:00:00"]),
                    type=pa.timestamp("ns"),
                ),
            "prediction_probability": [0.4, 0.6],
            "true_label": ["n", "y"],
        }
    )
    with pytest.raises(ValueError, match="must be a numeric .* or boolean"):
        _validate_columns(
            table=table,
            y_proba_col="prediction_probability",
            y_label_col="true_label",
            timeseries_col="event_timestamp",
            aggregation_cols="encounter_id",
            model_col=None,
            task_col=None,
        )


def test_validate_columns_invalid_model_column_type_raises():
    table = pa.table(
        {
                "encounter_id": ["e1", "e2"],
                "event_timestamp": pa.array(
                    pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T01:00:00"]),
                    type=pa.timestamp("ns"),
                ),
            "prediction_probability": [0.4, 0.6],
            "true_label": [0, 1],
            "model_identifier": [1.2, 3.4],
        }
    )
    with pytest.raises(ValueError, match="Model column .* must be a string"):
        _validate_columns(
            table=table,
            y_proba_col="prediction_probability",
            y_label_col="true_label",
            timeseries_col="event_timestamp",
            aggregation_cols="encounter_id",
            model_col="model_identifier",
            task_col=None,
        )


def test_validate_columns_invalid_task_column_type_raises():
    table = pa.table(
        {
                "encounter_id": ["e1", "e2"],
                "event_timestamp": pa.array(
                    pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T01:00:00"]),
                    type=pa.timestamp("ns"),
                ),
            "prediction_probability": [0.4, 0.6],
            "true_label": [0, 1],
            "task_identifier": [1.2, 3.4],
        }
    )
    with pytest.raises(ValueError, match="Task column .* must be a string"):
        _validate_columns(
            table=table,
            y_proba_col="prediction_probability",
            y_label_col="true_label",
            timeseries_col="event_timestamp",
            aggregation_cols="encounter_id",
            model_col=None,
            task_col="task_identifier",
        )


def test_attach_metadata_adds_expected_keys_and_preserves_existing():
    table = _base_table().replace_schema_metadata({b"existing": b"keepme"})
    updated = _attach_metadata(
        table=table,
        y_proba_col="prediction_probability",
        y_label_col="true_label",
        timeseries_col="event_timestamp",
        agg_cols_list=["encounter_id"],
        optional_cols_map={
            "model_col": "model_identifier",
            "task_col": "task_identifier",
        },
    )
    metadata = updated.schema.metadata
    assert metadata is not None
    assert metadata[b"existing"] == b"keepme"
    assert metadata[b"pysalient.io.y_proba_col"] == b"prediction_probability"
    assert metadata[b"pysalient.io.y_label_col"] == b"true_label"
    assert metadata[b"pysalient.io.timeseries_col"] == b"event_timestamp"
    assert json.loads(metadata[b"pysalient.io.aggregation_cols"].decode("utf-8")) == [
        "encounter_id"
    ]
    assert metadata[b"pysalient.io.model_col"] == b"model_identifier"
    assert metadata[b"pysalient.io.task_col"] == b"task_identifier"
