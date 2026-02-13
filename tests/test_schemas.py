"""Tests for shared Pandera schemas."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from tests.schemas import (
    evaluation_input_schema,
    evaluation_results_schema,
    io_csv_input_schema,
    io_parquet_input_schema,
    time_to_event_input_schema,
    visualisation_input_schema,
)


def _valid_evaluation_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "modelid": ["model-a"],
            "filter_desc": ["all"],
            "threshold": [0.5],
            "AUROC": [0.8],
            "AUROC_Lower_CI": [0.7],
            "AUROC_Upper_CI": [0.9],
            "AUPRC": [0.7],
            "AUPRC_Lower_CI": [0.6],
            "AUPRC_Upper_CI": [0.8],
            "Prevalence": [0.4],
            "Sample_Size": [100],
            "Label_Count": [40],
            "TP": [30],
            "TN": [50],
            "FP": [10],
            "FN": [10],
            "PPV": [0.75],
            "PPV_Lower_CI": [0.65],
            "PPV_Upper_CI": [0.85],
            "Sensitivity": [0.75],
            "Sensitivity_Lower_CI": [0.65],
            "Sensitivity_Upper_CI": [0.85],
            "Specificity": [0.8333],
            "Specificity_Lower_CI": [0.73],
            "Specificity_Upper_CI": [0.93],
            "NPV": [0.8333],
            "NPV_Lower_CI": [0.73],
            "NPV_Upper_CI": [0.93],
            "Accuracy": [0.8],
            "Accuracy_Lower_CI": [0.7],
            "Accuracy_Upper_CI": [0.9],
            "F1_Score": [0.75],
            "F1_Score_Lower_CI": [0.65],
            "F1_Score_Upper_CI": [0.85],
        }
    )


def test_evaluation_input_schema_accepts_valid_data() -> None:
    df = pd.DataFrame(
        {
            "encounter_id": ["enc-1", "enc-2"],
            "event_timestamp": [1.0, 2.0],
            "y_proba": [0.1, 0.8],
            "y_label": [0, 1],
        }
    )
    validated = evaluation_input_schema.validate(df)
    assert len(validated) == 2


def test_evaluation_input_schema_rejects_invalid_probability() -> None:
    df = pd.DataFrame(
        {
            "encounter_id": ["enc-1"],
            "event_timestamp": [1.0],
            "y_proba": [1.2],
            "y_label": [1],
        }
    )
    with pytest.raises(SchemaError):
        evaluation_input_schema.validate(df)


def test_evaluation_results_schema_accepts_valid_data() -> None:
    validated = evaluation_results_schema.validate(_valid_evaluation_results_frame())
    assert validated.loc[0, "TP"] == 30


def test_evaluation_results_schema_rejects_negative_confusion_count() -> None:
    df = _valid_evaluation_results_frame()
    df.loc[0, "TP"] = -1
    with pytest.raises(SchemaError):
        evaluation_results_schema.validate(df)


def test_io_csv_schema_accepts_valid_data() -> None:
    df = pd.DataFrame(
        {
            "encounter_id": ["enc-1", "enc-2"],
            "prediction_probability": [0.2, 0.7],
            "true_label": [0, 1],
        }
    )
    validated = io_csv_input_schema.validate(df)
    assert list(validated["true_label"]) == [0, 1]


def test_io_csv_schema_rejects_null_encounter() -> None:
    df = pd.DataFrame(
        {
            "encounter_id": [None],
            "prediction_probability": [0.2],
            "true_label": [0],
        }
    )
    with pytest.raises(SchemaError):
        io_csv_input_schema.validate(df)


def test_io_parquet_schema_accepts_valid_data() -> None:
    df = pd.DataFrame(
        {
            "encounter_id": ["enc-1"],
            "event_timestamp": [pd.Timestamp("2024-01-01T12:00:00")],
            "prediction_probability": [0.8],
            "true_label": [1],
        }
    )
    validated = io_parquet_input_schema.validate(df)
    assert str(validated["event_timestamp"].dtype).startswith("datetime64")


def test_io_parquet_schema_rejects_invalid_timestamp_type() -> None:
    df = pd.DataFrame(
        {
            "encounter_id": ["enc-1"],
            "event_timestamp": ["not-a-datetime"],
            "prediction_probability": [0.8],
            "true_label": [1],
        }
    )
    with pytest.raises(SchemaError):
        io_parquet_input_schema.validate(df)


def test_time_to_event_schema_accepts_valid_data() -> None:
    df = pd.DataFrame(
        {
            "encounter_id": ["enc-1", "enc-2"],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "y_proba": [0.4, 0.6],
            "y_label": [0, 1],
            "culture_event": [0, 1],
            "suspected_infection": [1, 0],
        }
    )
    validated = time_to_event_input_schema.validate(df)
    assert len(validated.columns) == 6


def test_time_to_event_schema_rejects_non_binary_event_flag() -> None:
    df = pd.DataFrame(
        {
            "encounter_id": ["enc-1"],
            "event_timestamp": [pd.Timestamp("2024-01-01")],
            "y_proba": [0.4],
            "y_label": [0],
            "culture_event": [2],
            "suspected_infection": [1],
        }
    )
    with pytest.raises(SchemaError):
        time_to_event_input_schema.validate(df)


def test_visualisation_schema_accepts_valid_data() -> None:
    df = pd.DataFrame(
        {
            "threshold": [0.25, 0.5],
            "AUROC": [0.7, 0.72],
            "AUPRC": [0.4, 0.45],
            "Prevalence": [0.2, 0.2],
            "PPV": [0.5, 0.52],
            "Sensitivity": [0.6, 0.61],
            "Specificity": [0.8, 0.82],
            "NPV": [0.85, 0.86],
            "Accuracy": [0.74, 0.75],
            "F1_Score": [0.55, 0.56],
            "Sample Size": [200, 200],
        }
    )
    validated = visualisation_input_schema.validate(df)
    assert validated["Sample Size"].min() == 200


def test_visualisation_schema_rejects_negative_sample_size() -> None:
    df = pd.DataFrame(
        {
            "threshold": [0.25],
            "AUROC": [0.7],
            "AUPRC": [0.4],
            "Prevalence": [0.2],
            "PPV": [0.5],
            "Sensitivity": [0.6],
            "Specificity": [0.8],
            "NPV": [0.85],
            "Accuracy": [0.74],
            "F1_Score": [0.55],
            "Sample Size": [-1],
        }
    )
    with pytest.raises(SchemaError):
        visualisation_input_schema.validate(df)
