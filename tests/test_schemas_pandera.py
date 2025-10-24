"""
Tests for pandera schema validation.

This module tests the pandera schemas defined in tests/schemas.py and demonstrates
how to use pandera for data validation instead of static parquet files.
"""

import os

import pandas as pd
import pandera as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.schemas import evaluation_data_schema, minimal_evaluation_data_schema


class TestEvaluationDataSchema:
    """Tests for the EvaluationDataSchema."""

    def test_schema_validates_sample_data(self):
        """Test that the schema validates against the existing sample parquet file."""
        # Load the existing sample data
        sample_path = os.path.join("tests", "test_data", "anonymised_sample.parquet")
        if not os.path.exists(sample_path):
            pytest.skip(f"Sample data not found at {sample_path}")

        table = pq.read_table(sample_path)
        df = table.to_pandas()

        # Validate using pandera
        validated_df = evaluation_data_schema.validate(df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == len(df)

    def test_schema_rejects_invalid_probability(self):
        """Test that schema rejects probabilities outside [0, 1] range."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "culture_event": [1.0, 0.0],
            "suspected_infection": [1.0, 0.0],
            "true_label": [1, 0],
            "prediction_proba_1": [1.5, 0.5],  # Invalid: > 1.0
            "prediction_proba_2": [0.5, 0.5],
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_data_schema.validate(invalid_df)

    def test_schema_rejects_invalid_label(self):
        """Test that schema rejects labels not in {0, 1}."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "culture_event": [1.0, 0.0],
            "suspected_infection": [1.0, 0.0],
            "true_label": [1, 2],  # Invalid: 2 not in {0, 1}
            "prediction_proba_1": [0.5, 0.5],
            "prediction_proba_2": [0.5, 0.5],
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_data_schema.validate(invalid_df)

    def test_schema_accepts_missing_optional_columns(self):
        """Test that schema accepts data without optional columns."""
        minimal_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "true_label": [1, 0],
            "prediction_proba_1": [0.8, 0.3],
        })

        # Should validate successfully with strict=False
        validated_df = evaluation_data_schema.validate(minimal_df, lazy=True)
        assert validated_df is not None


class TestMinimalEvaluationDataSchema:
    """Tests for the MinimalEvaluationDataSchema."""

    def test_minimal_schema_validates_basic_data(self):
        """Test that minimal schema validates basic required columns."""
        df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2", "enc_3"],
            "event_timestamp": [1.0, 2.0, 3.0],
            "true_label": [1, 0, 1],
            "prediction_probability": [0.8, 0.3, 0.9],
        })

        validated_df = minimal_evaluation_data_schema.validate(df)
        assert validated_df is not None
        assert len(validated_df) == 3

    def test_minimal_schema_rejects_extra_columns(self):
        """Test that minimal schema with strict=True rejects extra columns."""
        df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "true_label": [1, 0],
            "prediction_probability": [0.8, 0.3],
            "extra_column": ["a", "b"],  # Extra column not in schema
        })

        with pytest.raises(pa.errors.SchemaError):
            minimal_evaluation_data_schema.validate(df)


class TestHypothesisStrategies:
    """Tests using hypothesis for property-based testing."""

    @given(
        num_rows=st.integers(min_value=1, max_value=20),
        encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
        timestamps=st.lists(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
        labels=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=20),
        probas=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
    )
    @settings(max_examples=5, deadline=5000)
    def test_minimal_schema_with_hypothesis_generated_data(self, num_rows, encounter_ids, timestamps, labels, probas):
        """Test that randomly generated data validates against minimal schema."""
        # Truncate lists to num_rows
        df = pd.DataFrame({
            "encounter_id": encounter_ids[:num_rows],
            "event_timestamp": timestamps[:num_rows],
            "true_label": labels[:num_rows],
            "prediction_probability": probas[:num_rows],
        })

        # Hypothesis generates data that should always validate
        try:
            validated_df = minimal_evaluation_data_schema.validate(df)
            assert validated_df is not None
            assert len(validated_df) == len(df)
        except pa.errors.SchemaError as e:
            # If validation fails, it's a bug in our schema or strategy
            pytest.fail(f"Schema validation failed on hypothesis-generated data: {e}")

    @given(
        num_rows=st.integers(min_value=1, max_value=20),
        encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
        timestamps=st.lists(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
        culture_events=st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=20),
        suspected_infections=st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=20),
        labels=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=20),
        probas_1=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
        probas_2=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
    )
    @settings(max_examples=5, deadline=5000)
    def test_full_schema_with_hypothesis_generated_data(self, num_rows, encounter_ids, timestamps, culture_events, suspected_infections, labels, probas_1, probas_2):
        """Test that randomly generated data validates against full schema."""
        # Truncate lists to num_rows
        df = pd.DataFrame({
            "encounter_id": encounter_ids[:num_rows],
            "event_timestamp": timestamps[:num_rows],
            "culture_event": culture_events[:num_rows],
            "suspected_infection": suspected_infections[:num_rows],
            "true_label": labels[:num_rows],
            "prediction_proba_1": probas_1[:num_rows],
            "prediction_proba_2": probas_2[:num_rows],
        })

        try:
            validated_df = evaluation_data_schema.validate(df)
            assert validated_df is not None
            assert len(validated_df) == len(df)
            # Verify all probabilities are in valid range
            assert (validated_df["prediction_proba_1"] >= 0.0).all()
            assert (validated_df["prediction_proba_1"] <= 1.0).all()
            assert (validated_df["prediction_proba_2"] >= 0.0).all()
            assert (validated_df["prediction_proba_2"] <= 1.0).all()
            # Verify labels are binary
            assert validated_df["true_label"].isin([0, 1]).all()
        except pa.errors.SchemaError as e:
            pytest.fail(f"Schema validation failed on hypothesis-generated data: {e}")


class TestDataInvariants:
    """Property-based tests for data invariants."""

    @given(
        probabilities=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100
        ),
        labels=st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_probability_label_relationship(self, probabilities, labels):
        """Test invariants about probability and label relationships."""
        # Ensure lists are same length
        min_len = min(len(probabilities), len(labels))
        probabilities = probabilities[:min_len]
        labels = labels[:min_len]

        df = pd.DataFrame({
            "encounter_id": [f"enc_{i}" for i in range(len(probabilities))],
            "event_timestamp": [float(i) for i in range(len(probabilities))],
            "true_label": labels,
            "prediction_probability": probabilities,
        })

        validated_df = minimal_evaluation_data_schema.validate(df)

        # Test invariant: all probabilities are in [0, 1]
        assert (validated_df["prediction_probability"] >= 0.0).all()
        assert (validated_df["prediction_probability"] <= 1.0).all()

        # Test invariant: all labels are in {0, 1}
        assert validated_df["true_label"].isin([0, 1]).all()

        # Test invariant: number of rows is preserved
        assert len(validated_df) == len(probabilities)
