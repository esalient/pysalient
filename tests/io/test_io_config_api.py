"""
Tests for config-based IO API compatibility.
"""

import pandas as pd
import pyarrow as pa

from pysalient.io import load_evaluation_data
from pysalient.io._models import ColumnConfig, LoadConfig


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [1, 1, 2],
            "event_timestamp": pd.to_datetime(
                ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-02 08:00:00"]
            ),
            "prediction_probability": [0.1, 0.2, 0.8],
            "true_label": [0, 1, 1],
        }
    )


def test_load_evaluation_data_accepts_config_and_matches_legacy():
    """Config API should produce equivalent table output to legacy arguments."""
    df = _sample_df()
    legacy = load_evaluation_data(
        source=df,
        y_proba_col="prediction_probability",
        y_label_col="true_label",
        aggregation_cols="encounter_id",
        timeseries_col="event_timestamp",
    )

    config = LoadConfig(
        source=df,
        columns=ColumnConfig(
            y_proba_col="prediction_probability",
            y_label_col="true_label",
            aggregation_cols="encounter_id",
            timeseries_col="event_timestamp",
        ),
    )
    with_config = load_evaluation_data(config)

    assert isinstance(with_config, pa.Table)
    assert with_config.to_pydict() == legacy.to_pydict()
    assert with_config.schema.metadata == legacy.schema.metadata


def test_load_evaluation_data_config_read_options_passed_through():
    """Config read_options should be accepted and used."""
    df = _sample_df()
    config = LoadConfig(
        source=df,
        columns=ColumnConfig(
            y_proba_col="prediction_probability",
            y_label_col="true_label",
            aggregation_cols="encounter_id",
            timeseries_col="event_timestamp",
        ),
        read_options={"csv": {}},
    )
    table = load_evaluation_data(config)
    assert isinstance(table, pa.Table)
