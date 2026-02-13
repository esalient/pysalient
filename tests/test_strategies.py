"""Tests for Hypothesis strategies backed by Pandera schemas."""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402

from tests.schemas import (  # noqa: E402
    evaluation_input_schema,
    evaluation_results_schema,
    io_csv_input_schema,
    io_parquet_input_schema,
    time_to_event_input_schema,
    visualisation_input_schema,
)
from tests.strategies import (  # noqa: E402
    encounter_grouped_strategy,
    evaluation_data_strategy,
    evaluation_results_strategy,
    io_sample_strategy,
    temporal_data_strategy,
)

STRATEGY_SETTINGS = settings(
    max_examples=12,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)


@STRATEGY_SETTINGS
@given(data=evaluation_data_strategy())
def test_evaluation_data_strategy_matches_schema(data):
    validated = evaluation_input_schema.validate(data)
    assert len(validated) >= 2


@STRATEGY_SETTINGS
@given(data=temporal_data_strategy())
def test_temporal_data_strategy_produces_datetime_events(data):
    assert str(data["event_timestamp"].dtype).startswith("datetime64")
    converted = data.copy()
    converted["culture_event"] = 0
    converted["suspected_infection"] = 1
    validated = time_to_event_input_schema.validate(converted)
    assert len(validated) == len(data)


@STRATEGY_SETTINGS
@given(data=encounter_grouped_strategy())
def test_encounter_grouped_strategy_has_multiple_rows_per_encounter(data):
    counts = data.groupby("encounter_id").size()
    assert counts.min() >= 2
    validated = evaluation_input_schema.validate(
        data.assign(event_timestamp=data["event_timestamp"].astype("int64") / 1e9)
    )
    assert len(validated) == len(data)


@STRATEGY_SETTINGS
@given(data=io_sample_strategy())
def test_io_sample_strategy_matches_io_schemas(data):
    validated_csv = io_csv_input_schema.validate(data)
    validated_parquet = io_parquet_input_schema.validate(data)
    assert len(validated_csv) == len(validated_parquet)


@STRATEGY_SETTINGS
@given(data=evaluation_results_strategy())
def test_evaluation_results_strategy_matches_result_schema(data):
    validated = evaluation_results_schema.validate(data)
    visualisation_view = validated.rename(columns={"Sample_Size": "Sample Size"})
    visualisation_input_schema.validate(visualisation_view)
    assert len(validated.columns) >= 30
