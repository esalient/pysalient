"""
Pandera schemas for pysalient test data validation.

This module defines schemas using pandera for validating test data structures,
replacing the need for static parquet files in tests.
"""

import pandera as pa
import pandas as pd


# Use DataFrameSchema for better compatibility
evaluation_data_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "event_timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "culture_event": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "suspected_infection": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "true_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "prediction_proba_1": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "prediction_proba_2": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True, required=False),
    },
    coerce=True,
    strict=False
)


# Minimal schema for basic testing
minimal_evaluation_data_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "event_timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "true_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "prediction_probability": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
    },
    coerce=True,
    strict=True
)
