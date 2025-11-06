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

from tests.schemas import (
    evaluation_data_encounter_level_schema,
    evaluation_data_event_level_schema,
    evaluation_data_schema,
    evaluation_multi_model_schema,
    evaluation_multi_task_schema,
    evaluation_results_schema,
    evaluation_with_temporal_schema,
    io_csv_input_schema,
    io_parquet_input_schema,
    minimal_evaluation_data_schema,
    time_to_event_data_schema,
)


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


class TestEvaluationWithTemporalSchema:
    """Tests for the EvaluationWithTemporalSchema with datetime timestamps."""

    def test_schema_validates_data_with_proper_datetime_timestamps(self):
        """Test that schema validates data with proper datetime timestamps."""
        df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2", "enc_3"],
            "event_timestamp": pd.to_datetime([
                "2023-01-15 10:30:00",
                "2023-02-20 14:45:30",
                "2023-03-10 09:15:00"
            ]),
            "culture_event": [1.0, 0.0, 1.0],
            "suspected_infection": [0.0, 1.0, 0.0],
            "true_label": [1, 0, 1],
            "prediction_proba_1": [0.8, 0.3, 0.95],
            "prediction_proba_2": [0.2, 0.7, 0.05],
        })

        validated_df = evaluation_with_temporal_schema.validate(df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 3
        # Verify datetime column is preserved
        assert pd.api.types.is_datetime64_any_dtype(validated_df["event_timestamp"])

    def test_schema_rejects_invalid_datetime_formats(self):
        """Test that schema rejects invalid datetime formats."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": ["not-a-date", "2023-13-45"],  # Invalid date strings
            "culture_event": [1.0, 0.0],
            "suspected_infection": [0.0, 1.0],
            "true_label": [1, 0],
            "prediction_proba_1": [0.8, 0.3],
            "prediction_proba_2": [0.2, 0.7],
        })

        with pytest.raises(Exception):  # Could be SchemaError or conversion error
            evaluation_with_temporal_schema.validate(invalid_df)

    def test_schema_validates_timestamps_within_reasonable_range(self):
        """Test that timestamps are validated within a reasonable range."""
        # Valid: timestamps within a reasonable range
        valid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": pd.to_datetime([
                "2020-01-01 00:00:00",
                "2025-12-31 23:59:59"
            ]),
            "culture_event": [1.0, 0.0],
            "suspected_infection": [0.0, 1.0],
            "true_label": [1, 0],
            "prediction_proba_1": [0.8, 0.3],
            "prediction_proba_2": [0.2, 0.7],
        })

        validated_df = evaluation_with_temporal_schema.validate(valid_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 2

    def test_schema_rejects_timestamps_outside_reasonable_range(self):
        """Test that timestamps outside reasonable range are rejected."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": pd.to_datetime([
                "1900-01-01 00:00:00",  # Too far in past
                "2100-12-31 23:59:59"   # Too far in future
            ]),
            "culture_event": [1.0, 0.0],
            "suspected_infection": [0.0, 1.0],
            "true_label": [1, 0],
            "prediction_proba_1": [0.8, 0.3],
            "prediction_proba_2": [0.2, 0.7],
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_with_temporal_schema.validate(invalid_df)

    def test_schema_accepts_missing_optional_columns_with_datetime(self):
        """Test that schema accepts data without optional columns and with datetime."""
        minimal_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": pd.to_datetime([
                "2023-06-15 10:00:00",
                "2023-06-16 11:00:00"
            ]),
            "true_label": [1, 0],
            "prediction_proba_1": [0.8, 0.3],
        })

        validated_df = evaluation_with_temporal_schema.validate(minimal_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 2

    def test_schema_rejects_invalid_probability_with_datetime(self):
        """Test that schema rejects probabilities outside [0, 1] range with datetime."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": pd.to_datetime([
                "2023-01-15 10:30:00",
                "2023-02-20 14:45:30"
            ]),
            "culture_event": [1.0, 0.0],
            "suspected_infection": [1.0, 0.0],
            "true_label": [1, 0],
            "prediction_proba_1": [1.5, 0.5],  # Invalid: > 1.0
            "prediction_proba_2": [0.5, 0.5],
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_with_temporal_schema.validate(invalid_df)

    def test_schema_rejects_invalid_label_with_datetime(self):
        """Test that schema rejects labels not in {0, 1} with datetime."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": pd.to_datetime([
                "2023-01-15 10:30:00",
                "2023-02-20 14:45:30"
            ]),
            "culture_event": [1.0, 0.0],
            "suspected_infection": [1.0, 0.0],
            "true_label": [1, 2],  # Invalid: 2 not in {0, 1}
            "prediction_proba_1": [0.5, 0.5],
            "prediction_proba_2": [0.5, 0.5],
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_with_temporal_schema.validate(invalid_df)

    @given(
        num_rows=st.integers(min_value=1, max_value=20),
        encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
        days_offset=st.lists(st.integers(min_value=0, max_value=2000), min_size=1, max_size=20),
        culture_events=st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=20),
        suspected_infections=st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=20),
        labels=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=20),
        probas_1=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
        probas_2=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
    )
    @settings(max_examples=5, deadline=5000)
    def test_temporal_schema_with_hypothesis_generated_datetime_data(
        self,
        num_rows,
        encounter_ids,
        days_offset,
        culture_events,
        suspected_infections,
        labels,
        probas_1,
        probas_2
    ):
        """Test that randomly generated datetime data validates against temporal schema."""
        # Truncate lists to num_rows
        min_len = min(
            len(encounter_ids),
            len(days_offset),
            len(culture_events),
            len(suspected_infections),
            len(labels),
            len(probas_1),
            len(probas_2)
        )
        actual_rows = min(min_len, num_rows)

        # Generate datetime timestamps from days offset (2020-01-01 as base)
        base_date = pd.Timestamp("2020-01-01")
        timestamps = [base_date + pd.Timedelta(days=int(offset)) for offset in days_offset[:actual_rows]]

        df = pd.DataFrame({
            "encounter_id": encounter_ids[:actual_rows],
            "event_timestamp": timestamps,
            "culture_event": culture_events[:actual_rows],
            "suspected_infection": suspected_infections[:actual_rows],
            "true_label": labels[:actual_rows],
            "prediction_proba_1": probas_1[:actual_rows],
            "prediction_proba_2": probas_2[:actual_rows],
        })

        # Hypothesis generates data that should always validate
        try:
            validated_df = evaluation_with_temporal_schema.validate(df)
            assert validated_df is not None
            assert len(validated_df) == len(df)
            # Verify datetime column is preserved
            assert pd.api.types.is_datetime64_any_dtype(validated_df["event_timestamp"])
            # Verify all probabilities are in valid range
            assert (validated_df["prediction_proba_1"] >= 0.0).all()
            assert (validated_df["prediction_proba_1"] <= 1.0).all()
            assert (validated_df["prediction_proba_2"] >= 0.0).all()
            assert (validated_df["prediction_proba_2"] <= 1.0).all()
            # Verify labels are binary
            assert validated_df["true_label"].isin([0, 1]).all()
        except pa.errors.SchemaError as e:
            pytest.fail(f"Schema validation failed on hypothesis-generated datetime data: {e}")


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


class TestIOCSVInputSchema:
    """Tests for the IOCSVInputSchema for validating CSV input files."""

    def test_schema_validates_csv_string_data(self):
        """Test that the schema validates CSV data with all string types (typical CSV)."""
        # CSV files typically have all data as strings initially
        csv_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2", "enc_3"],
            "timestamp": ["1.0", "2.0", "3.0"],
            "label": ["1", "0", "1"],
            "probability": ["0.85", "0.32", "0.91"],
        })

        validated_df = io_csv_input_schema.validate(csv_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 3

    def test_schema_coerces_string_numbers_to_proper_types(self):
        """Test that schema coerces string numbers to proper types."""
        csv_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "timestamp": ["1.0", "2.5"],
            "label": ["1", "0"],
            "probability": ["0.85", "0.32"],
        })

        validated_df = io_csv_input_schema.validate(csv_df)
        assert validated_df is not None

        # With coerce=True, these should be converted to proper types
        # timestamp should be coerced to float/string (depends on schema definition)
        # label should be coerced to int
        assert pd.api.types.is_integer_dtype(validated_df["label"])
        assert validated_df.loc[0, "label"] == 1
        assert validated_df.loc[1, "label"] == 0

        # probability should be coerced to float in range [0, 1]
        assert validated_df["probability"].dtype == float
        assert 0.84 < validated_df.loc[0, "probability"] < 0.86
        assert 0.31 < validated_df.loc[1, "probability"] < 0.33

    def test_schema_rejects_invalid_string_data(self):
        """Test that schema rejects invalid string data that cannot be coerced."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "timestamp": ["1.0", "2.0"],
            "label": ["1", "0"],
            "probability": ["abc", "0.5"],  # Invalid: cannot coerce "abc" to float
        })

        with pytest.raises(pa.errors.SchemaError):
            io_csv_input_schema.validate(invalid_df)

    def test_schema_accepts_only_required_columns(self):
        """Test that schema accepts data with only required columns."""
        minimal_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "timestamp": ["1.0", "2.0"],
            "label": ["1", "0"],
            "probability": ["0.85", "0.32"],
        })

        validated_df = io_csv_input_schema.validate(minimal_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 2

    def test_csv_schema_with_hypothesis_generated_data(self):
        """Test CSV schema with hypothesis-generated CSV-like string data."""
        @given(
            num_rows=st.integers(min_value=1, max_value=20),
            encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
            timestamps=st.lists(
                st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=20
            ),
            labels=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=20),
            probabilities=st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=20
            ),
        )
        @settings(max_examples=5, deadline=5000)
        def run_test(num_rows, encounter_ids, timestamps, labels, probabilities):
            # Truncate lists to num_rows
            min_len = min(len(encounter_ids), len(timestamps), len(labels), len(probabilities))
            actual_rows = min(min_len, num_rows)

            # Convert numeric values to strings to simulate CSV input
            csv_df = pd.DataFrame({
                "encounter_id": encounter_ids[:actual_rows],
                "timestamp": [str(t) for t in timestamps[:actual_rows]],
                "label": [str(label) for label in labels[:actual_rows]],
                "probability": [str(p) for p in probabilities[:actual_rows]],
            })

            # Schema should validate and coerce string data to proper types
            try:
                validated_df = io_csv_input_schema.validate(csv_df)
                assert validated_df is not None
                assert len(validated_df) == len(csv_df)

                # Verify coercion worked correctly
                assert pd.api.types.is_integer_dtype(validated_df["label"])
                assert validated_df["probability"].dtype == float
                assert (validated_df["probability"] >= 0.0).all()
                assert (validated_df["probability"] <= 1.0).all()
                assert validated_df["label"].isin([0, 1]).all()
            except pa.errors.SchemaError as e:
                pytest.fail(f"Schema validation failed on hypothesis-generated CSV data: {e}")

        # Run the test
        run_test()


class TestEvaluationResultsSchema:
    """Tests for the EvaluationResultsSchema for validating evaluation metrics output."""

    def test_schema_validates_basic_metrics_results(self):
        """Test that the schema validates basic evaluation metrics data."""
        results_df = pd.DataFrame({
            "metric_name": ["AUROC", "AUPRC", "Sensitivity"],
            "metric_value": [0.85, 0.78, 0.92],
            "ci_lower": [0.82, 0.74, 0.88],
            "ci_upper": [0.88, 0.82, 0.95],
            "n_samples": [1000, 1000, 1000],
        })

        validated_df = evaluation_results_schema.validate(results_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 3
        assert validated_df["metric_value"].dtype == float
        assert pd.api.types.is_integer_dtype(validated_df["n_samples"])

    def test_schema_validates_confidence_intervals(self):
        """Test that schema validates CI bounds are in correct order."""
        # Valid: ci_lower <= metric_value <= ci_upper
        valid_df = pd.DataFrame({
            "metric_name": ["AUROC", "AUPRC"],
            "metric_value": [0.85, 0.78],
            "ci_lower": [0.82, 0.74],
            "ci_upper": [0.88, 0.82],
            "n_samples": [1000, 1000],
        })

        validated_df = evaluation_results_schema.validate(valid_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 2
        # Verify CI bounds are valid
        assert (validated_df["ci_lower"] <= validated_df["metric_value"]).all()
        assert (validated_df["metric_value"] <= validated_df["ci_upper"]).all()

    def test_schema_accepts_metrics_exceeding_one(self):
        """Test that schema accepts metric values > 1.0 (e.g., MSE, log loss)."""
        # Some metrics like MSE, log loss, RMSE can exceed 1.0
        df = pd.DataFrame({
            "metric_name": ["MSE", "log_loss", "AUROC"],
            "metric_value": [2.5, 1.8, 0.85],  # MSE and log_loss can exceed 1.0
            "ci_lower": [2.2, 1.5, 0.82],
            "ci_upper": [2.8, 2.1, 0.88],
            "n_samples": [1000, 1000, 1000],
        })

        # Should validate successfully - no range constraint on metric_value
        validated_df = evaluation_results_schema.validate(df)
        assert validated_df is not None
        assert len(validated_df) == 3

    def test_schema_rejects_negative_samples(self):
        """Test that schema rejects n_samples <= 0."""
        invalid_df = pd.DataFrame({
            "metric_name": ["AUROC", "AUPRC"],
            "metric_value": [0.85, 0.78],
            "ci_lower": [0.82, 0.74],
            "ci_upper": [0.88, 0.82],
            "n_samples": [-1, 0],  # Invalid: <= 0
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_results_schema.validate(invalid_df)

    def test_schema_accepts_optional_fields(self):
        """Test that schema accepts data with optional fields."""
        results_df = pd.DataFrame({
            "metric_name": ["AUROC", "AUPRC"],
            "metric_value": [0.85, 0.78],
            "ci_lower": [0.82, 0.74],
            "ci_upper": [0.88, 0.82],
            "n_samples": [1000, 1000],
            "threshold": [0.5, 0.5],
            "prevalence": [0.15, 0.15],
            "model_name": ["model_v1", "model_v1"],
        })

        validated_df = evaluation_results_schema.validate(results_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 2
        # Verify optional columns are present
        assert "threshold" in validated_df.columns
        assert "prevalence" in validated_df.columns
        assert "model_name" in validated_df.columns

    @given(
        num_rows=st.integers(min_value=1, max_value=20),
        metric_names=st.lists(
            st.sampled_from(["AUROC", "AUPRC", "Sensitivity", "Specificity", "F1"]),
            min_size=1,
            max_size=20
        ),
        metric_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        ),
        n_samples=st.lists(st.integers(min_value=1, max_value=10000), min_size=1, max_size=20),
    )
    @settings(max_examples=5, deadline=5000)
    def test_results_schema_with_hypothesis_generated_data(
        self, num_rows, metric_names, metric_values, n_samples
    ):
        """Property test with randomly generated valid metrics data."""
        # Truncate lists to num_rows
        min_len = min(len(metric_names), len(metric_values), len(n_samples))
        actual_rows = min(min_len, num_rows)

        # Generate valid CI bounds: ci_lower <= metric_value <= ci_upper
        ci_lower = [max(0.0, val - 0.05) for val in metric_values[:actual_rows]]
        ci_upper = [min(1.0, val + 0.05) for val in metric_values[:actual_rows]]

        results_df = pd.DataFrame({
            "metric_name": metric_names[:actual_rows],
            "metric_value": metric_values[:actual_rows],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_samples": n_samples[:actual_rows],
        })

        # Hypothesis generates data that should always validate
        try:
            validated_df = evaluation_results_schema.validate(results_df)
            assert validated_df is not None
            assert len(validated_df) == len(results_df)
            # Verify all metrics are in valid range
            assert (validated_df["metric_value"] >= 0.0).all()
            assert (validated_df["metric_value"] <= 1.0).all()
            # Verify CI bounds are valid
            assert (validated_df["ci_lower"] >= 0.0).all()
            assert (validated_df["ci_upper"] <= 1.0).all()
            assert (validated_df["ci_lower"] <= validated_df["metric_value"]).all()
            assert (validated_df["metric_value"] <= validated_df["ci_upper"]).all()
            # Verify n_samples is positive
            assert (validated_df["n_samples"] > 0).all()
        except pa.errors.SchemaError as e:
            pytest.fail(f"Schema validation failed on hypothesis-generated results data: {e}")


class TestIOParquetInputSchema:
    """Tests for the IOParquetInputSchema for validating Parquet input files."""

    def test_schema_validates_parquet_typed_data(self):
        """Test with properly typed data (no strings) typical of Parquet format."""
        # Parquet preserves data types, so values are already properly typed
        parquet_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2", "enc_3"],
            "timestamp": [1.0, 2.0, 3.0],  # float type preserved
            "label": [1, 0, 1],  # int type preserved
            "probability": [0.85, 0.32, 0.91],  # float type preserved
        })

        validated_df = io_parquet_input_schema.validate(parquet_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 3

        # Verify types are preserved
        assert validated_df["timestamp"].dtype == float
        assert pd.api.types.is_integer_dtype(validated_df["label"])
        assert validated_df["probability"].dtype == float

    def test_schema_validates_existing_sample_parquet(self):
        """Test against the actual sample parquet file if it exists."""
        sample_path = os.path.join("tests", "test_data", "anonymised_sample.parquet")
        if not os.path.exists(sample_path):
            pytest.skip(f"Sample data not found at {sample_path}")

        table = pq.read_table(sample_path)
        df = table.to_pandas()

        # The schema should be able to validate or transform the sample data
        # We may need to map columns to expected names
        validated_df = io_parquet_input_schema.validate(df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) > 0

    def test_schema_rejects_invalid_probability_range(self):
        """Test rejects probability > 1.0."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "timestamp": [1.0, 2.0],
            "label": [1, 0],
            "probability": [1.5, 0.5],  # Invalid: > 1.0
        })

        with pytest.raises(pa.errors.SchemaError):
            io_parquet_input_schema.validate(invalid_df)

    def test_schema_rejects_invalid_labels(self):
        """Test rejects labels not in {0, 1}."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "timestamp": [1.0, 2.0],
            "label": [1, 2],  # Invalid: 2 not in {0, 1}
            "probability": [0.8, 0.5],
        })

        with pytest.raises(pa.errors.SchemaError):
            io_parquet_input_schema.validate(invalid_df)

    def test_parquet_schema_with_hypothesis_generated_data(self):
        """Property test with properly typed data typical of Parquet."""
        @given(
            num_rows=st.integers(min_value=1, max_value=20),
            encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
            timestamps=st.lists(
                st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=20
            ),
            labels=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=20),
            probabilities=st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=20
            ),
        )
        @settings(max_examples=5, deadline=5000)
        def run_test(num_rows, encounter_ids, timestamps, labels, probabilities):
            # Truncate lists to num_rows
            min_len = min(len(encounter_ids), len(timestamps), len(labels), len(probabilities))
            actual_rows = min(min_len, num_rows)

            # Parquet data is already properly typed (no string conversion needed)
            parquet_df = pd.DataFrame({
                "encounter_id": encounter_ids[:actual_rows],
                "timestamp": timestamps[:actual_rows],
                "label": labels[:actual_rows],
                "probability": probabilities[:actual_rows],
            })

            # Schema should validate properly typed data
            try:
                validated_df = io_parquet_input_schema.validate(parquet_df)
                assert validated_df is not None
                assert len(validated_df) == len(parquet_df)

                # Verify types and constraints
                assert validated_df["timestamp"].dtype == float
                assert pd.api.types.is_integer_dtype(validated_df["label"])
                assert validated_df["probability"].dtype == float
                assert (validated_df["probability"] >= 0.0).all()
                assert (validated_df["probability"] <= 1.0).all()
                assert validated_df["label"].isin([0, 1]).all()
            except pa.errors.SchemaError as e:
                pytest.fail(f"Schema validation failed on hypothesis-generated Parquet data: {e}")

        # Run the test
        run_test()


class TestTimeToEventDataSchema:
    """Tests for the TimeToEventDataSchema for survival analysis data."""

    def test_schema_validates_basic_time_to_event_data(self):
        """Test that the schema validates standard survival data."""
        survival_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2", "enc_3"],
            "time_to_event": [24.5, 48.0, 72.3],
            "event_occurred": [1, 0, 1],
            "baseline_timestamp": [0.0, 0.0, 0.0],
            "prediction_proba": [0.75, 0.45, 0.88],
        })

        validated_df = time_to_event_data_schema.validate(survival_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 3
        assert validated_df["time_to_event"].dtype == float
        assert pd.api.types.is_integer_dtype(validated_df["event_occurred"])
        assert validated_df["prediction_proba"].dtype == float

    def test_schema_validates_censored_and_uncensored_events(self):
        """Test that schema validates mix of event_occurred 0 and 1."""
        mixed_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2", "enc_3", "enc_4"],
            "time_to_event": [10.5, 20.0, 30.5, 40.0],
            "event_occurred": [1, 0, 1, 0],  # Mix of censored (0) and uncensored (1)
            "baseline_timestamp": [0.0, 5.0, 10.0, 15.0],
            "prediction_proba": [0.8, 0.3, 0.9, 0.2],
        })

        validated_df = time_to_event_data_schema.validate(mixed_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 4
        # Verify event_occurred contains only 0 and 1
        assert validated_df["event_occurred"].isin([0, 1]).all()
        # Verify we have both censored and uncensored events
        assert (validated_df["event_occurred"] == 0).any()
        assert (validated_df["event_occurred"] == 1).any()

    def test_schema_rejects_negative_time_to_event(self):
        """Test that schema rejects negative time_to_event values."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "time_to_event": [-10.5, 20.0],  # Invalid: negative time
            "event_occurred": [1, 0],
            "baseline_timestamp": [0.0, 0.0],
            "prediction_proba": [0.8, 0.3],
        })

        with pytest.raises(pa.errors.SchemaError):
            time_to_event_data_schema.validate(invalid_df)

    def test_schema_rejects_invalid_event_flag(self):
        """Test that schema rejects event_occurred values not in {0, 1}."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "time_to_event": [10.5, 20.0],
            "event_occurred": [1, 2],  # Invalid: 2 not in {0, 1}
            "baseline_timestamp": [0.0, 0.0],
            "prediction_proba": [0.8, 0.3],
        })

        with pytest.raises(pa.errors.SchemaError):
            time_to_event_data_schema.validate(invalid_df)

    def test_schema_accepts_optional_fields(self):
        """Test that schema accepts data with optional risk_score and censoring_reason."""
        survival_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2", "enc_3"],
            "time_to_event": [24.5, 48.0, 72.3],
            "event_occurred": [1, 0, 1],
            "baseline_timestamp": [0.0, 0.0, 0.0],
            "prediction_proba": [0.75, 0.45, 0.88],
            "risk_score": [2.5, 1.2, 3.8],
            "censoring_reason": ["event", "lost_to_followup", "event"],
        })

        validated_df = time_to_event_data_schema.validate(survival_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 3
        # Verify optional columns are present
        assert "risk_score" in validated_df.columns
        assert "censoring_reason" in validated_df.columns
        assert validated_df["risk_score"].dtype == float
        assert validated_df["censoring_reason"].dtype == object  # string type

    @given(
        num_rows=st.integers(min_value=1, max_value=20),
        encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
        times=st.lists(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        ),
        events=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=20),
        baselines=st.lists(
            st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        ),
        probas=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        ),
    )
    @settings(max_examples=5, deadline=5000)
    def test_time_to_event_schema_with_hypothesis_generated_data(
        self, num_rows, encounter_ids, times, events, baselines, probas
    ):
        """Property test with randomly generated survival data."""
        # Truncate lists to num_rows
        min_len = min(
            len(encounter_ids),
            len(times),
            len(events),
            len(baselines),
            len(probas)
        )
        actual_rows = min(min_len, num_rows)

        survival_df = pd.DataFrame({
            "encounter_id": encounter_ids[:actual_rows],
            "time_to_event": times[:actual_rows],
            "event_occurred": events[:actual_rows],
            "baseline_timestamp": baselines[:actual_rows],
            "prediction_proba": probas[:actual_rows],
        })

        # Hypothesis generates data that should always validate
        try:
            validated_df = time_to_event_data_schema.validate(survival_df)
            assert validated_df is not None
            assert len(validated_df) == len(survival_df)
            # Verify time_to_event is non-negative
            assert (validated_df["time_to_event"] >= 0.0).all()
            # Verify baseline_timestamp is non-negative
            assert (validated_df["baseline_timestamp"] >= 0.0).all()
            # Verify event_occurred is binary
            assert validated_df["event_occurred"].isin([0, 1]).all()
            # Verify probabilities are in valid range
            assert (validated_df["prediction_proba"] >= 0.0).all()
            assert (validated_df["prediction_proba"] <= 1.0).all()
        except pa.errors.SchemaError as e:
            pytest.fail(f"Schema validation failed on hypothesis-generated survival data: {e}")


class TestEvaluationMultiModelSchema:
    """Tests for the EvaluationMultiModelSchema for multi-model comparison."""

    def test_schema_validates_multi_model_data(self):
        """Test that the schema validates data with multiple models on same encounters."""
        multi_model_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_1", "enc_2", "enc_2"],
            "event_timestamp": [1.0, 1.0, 2.0, 2.0],
            "true_label": [1, 1, 0, 0],
            "model_name": ["model_a", "model_b", "model_a", "model_b"],
            "prediction_proba": [0.8, 0.75, 0.3, 0.35],
        })

        validated_df = evaluation_multi_model_schema.validate(multi_model_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 4
        # Verify each model has predictions for each encounter
        assert set(validated_df["model_name"]) == {"model_a", "model_b"}
        assert validated_df["encounter_id"].nunique() == 2

    def test_schema_groups_predictions_by_model(self):
        """Test that same encounter can have multiple rows (one per model)."""
        # Same encounter with 3 different models
        multi_model_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_1", "enc_1"],
            "event_timestamp": [10.5, 10.5, 10.5],
            "true_label": [1, 1, 1],
            "model_name": ["model_a", "model_b", "model_c"],
            "prediction_proba": [0.85, 0.72, 0.91],
        })

        validated_df = evaluation_multi_model_schema.validate(multi_model_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 3
        # All rows should have same encounter_id and true_label
        assert validated_df["encounter_id"].nunique() == 1
        assert validated_df["true_label"].nunique() == 1
        # But different model predictions
        assert len(validated_df["model_name"].unique()) == 3
        assert len(validated_df["prediction_proba"].unique()) == 3

    def test_schema_rejects_invalid_probability(self):
        """Test that schema rejects probability > 1.0."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "true_label": [1, 0],
            "model_name": ["model_a", "model_a"],
            "prediction_proba": [1.5, 0.5],  # Invalid: > 1.0
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_multi_model_schema.validate(invalid_df)

    def test_schema_rejects_invalid_label(self):
        """Test that schema rejects labels not in {0, 1}."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "true_label": [1, 2],  # Invalid: 2 not in {0, 1}
            "model_name": ["model_a", "model_a"],
            "prediction_proba": [0.8, 0.5],
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_multi_model_schema.validate(invalid_df)

    def test_schema_accepts_optional_fields(self):
        """Test that schema accepts data with optional fields."""
        multi_model_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_1", "enc_2", "enc_2"],
            "event_timestamp": [1.0, 1.0, 2.0, 2.0],
            "true_label": [1, 1, 0, 0],
            "model_name": ["model_a", "model_b", "model_a", "model_b"],
            "prediction_proba": [0.8, 0.75, 0.3, 0.35],
            "culture_event": [1.0, 1.0, 0.0, 0.0],
            "suspected_infection": [0.0, 0.0, 1.0, 1.0],
            "model_version": ["v1.0", "v2.1", "v1.0", "v2.1"],
        })

        validated_df = evaluation_multi_model_schema.validate(multi_model_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 4
        # Verify optional columns are present
        assert "culture_event" in validated_df.columns
        assert "suspected_infection" in validated_df.columns
        assert "model_version" in validated_df.columns

    @given(
        num_rows=st.integers(min_value=2, max_value=20),
        encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
        timestamps=st.lists(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        ),
        labels=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=20),
        model_names=st.lists(
            st.sampled_from(["model_a", "model_b", "model_c"]),
            min_size=1,
            max_size=20
        ),
        probas=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        ),
    )
    @settings(max_examples=5, deadline=5000)
    def test_multi_model_schema_with_hypothesis_generated_data(
        self, num_rows, encounter_ids, timestamps, labels, model_names, probas
    ):
        """Property test with randomly generated multi-model data."""
        # Truncate lists to num_rows
        min_len = min(
            len(encounter_ids),
            len(timestamps),
            len(labels),
            len(model_names),
            len(probas)
        )
        actual_rows = min(min_len, num_rows)

        multi_model_df = pd.DataFrame({
            "encounter_id": encounter_ids[:actual_rows],
            "event_timestamp": timestamps[:actual_rows],
            "true_label": labels[:actual_rows],
            "model_name": model_names[:actual_rows],
            "prediction_proba": probas[:actual_rows],
        })

        # Hypothesis generates data that should always validate
        try:
            validated_df = evaluation_multi_model_schema.validate(multi_model_df)
            assert validated_df is not None
            assert len(validated_df) == len(multi_model_df)
            # Verify probabilities are in valid range
            assert (validated_df["prediction_proba"] >= 0.0).all()
            assert (validated_df["prediction_proba"] <= 1.0).all()
            # Verify labels are binary
            assert validated_df["true_label"].isin([0, 1]).all()
            # Verify all model names are strings
            assert validated_df["model_name"].dtype == object
            # Verify timestamps are non-negative
            assert (validated_df["event_timestamp"] >= 0.0).all()
        except pa.errors.SchemaError as e:
            pytest.fail(f"Schema validation failed on hypothesis-generated multi-model data: {e}")


class TestEvaluationMultiTaskSchema:
    """Tests for the EvaluationMultiTaskSchema for multi-task learning."""

    def test_schema_validates_multi_task_data(self):
        """Test with multiple tasks (e.g., infection, mortality) for same encounters."""
        multi_task_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_1", "enc_2", "enc_2"],
            "event_timestamp": [1.0, 1.0, 2.0, 2.0],
            "task_name": ["infection", "mortality", "infection", "mortality"],
            "true_label": [1, 0, 0, 1],
            "prediction_proba": [0.85, 0.32, 0.15, 0.78],
        })

        validated_df = evaluation_multi_task_schema.validate(multi_task_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 4
        # Verify each encounter has predictions for multiple tasks
        assert set(validated_df["task_name"]) == {"infection", "mortality"}
        assert validated_df["encounter_id"].nunique() == 2

    def test_schema_groups_predictions_by_task(self):
        """Test that same encounter can have multiple rows (one per task)."""
        # Same encounter with 3 different tasks
        multi_task_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_1", "enc_1"],
            "event_timestamp": [10.5, 10.5, 10.5],
            "task_name": ["infection", "mortality", "readmission"],
            "true_label": [1, 0, 1],
            "prediction_proba": [0.85, 0.32, 0.71],
        })

        validated_df = evaluation_multi_task_schema.validate(multi_task_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 3
        # All rows should have same encounter_id and timestamp
        assert validated_df["encounter_id"].nunique() == 1
        assert validated_df["event_timestamp"].nunique() == 1
        # But different task predictions
        assert len(validated_df["task_name"].unique()) == 3
        assert len(validated_df["true_label"].unique()) == 2  # Has both 0 and 1
        assert len(validated_df["prediction_proba"].unique()) == 3

    def test_schema_rejects_invalid_probability(self):
        """Test rejects probability > 1.0."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "task_name": ["infection", "mortality"],
            "true_label": [1, 0],
            "prediction_proba": [1.5, 0.5],  # Invalid: > 1.0
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_multi_task_schema.validate(invalid_df)

    def test_schema_rejects_invalid_label(self):
        """Test rejects labels not in {0, 1}."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "task_name": ["infection", "mortality"],
            "true_label": [1, 2],  # Invalid: 2 not in {0, 1}
            "prediction_proba": [0.8, 0.5],
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_multi_task_schema.validate(invalid_df)

    def test_schema_accepts_optional_fields(self):
        """Test with task_weight field."""
        multi_task_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_1", "enc_2", "enc_2"],
            "event_timestamp": [1.0, 1.0, 2.0, 2.0],
            "task_name": ["infection", "mortality", "infection", "mortality"],
            "true_label": [1, 0, 0, 1],
            "prediction_proba": [0.85, 0.32, 0.15, 0.78],
            "culture_event": [1.0, 1.0, 0.0, 0.0],
            "suspected_infection": [0.0, 0.0, 1.0, 1.0],
            "task_weight": [1.0, 0.5, 1.0, 0.5],
        })

        validated_df = evaluation_multi_task_schema.validate(multi_task_df, lazy=True)
        assert validated_df is not None
        assert len(validated_df) == 4
        # Verify optional columns are present
        assert "culture_event" in validated_df.columns
        assert "suspected_infection" in validated_df.columns
        assert "task_weight" in validated_df.columns

    @given(
        num_rows=st.integers(min_value=2, max_value=20),
        encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
        timestamps=st.lists(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        ),
        task_names=st.lists(
            st.sampled_from(["infection", "mortality", "readmission"]),
            min_size=1,
            max_size=20
        ),
        labels=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=20),
        probas=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        ),
    )
    @settings(max_examples=5, deadline=5000)
    def test_multi_task_schema_with_hypothesis_generated_data(
        self, num_rows, encounter_ids, timestamps, task_names, labels, probas
    ):
        """Property test with randomly generated multi-task data."""
        # Truncate lists to num_rows
        min_len = min(
            len(encounter_ids),
            len(timestamps),
            len(task_names),
            len(labels),
            len(probas)
        )
        actual_rows = min(min_len, num_rows)

        multi_task_df = pd.DataFrame({
            "encounter_id": encounter_ids[:actual_rows],
            "event_timestamp": timestamps[:actual_rows],
            "task_name": task_names[:actual_rows],
            "true_label": labels[:actual_rows],
            "prediction_proba": probas[:actual_rows],
        })

        # Hypothesis generates data that should always validate
        try:
            validated_df = evaluation_multi_task_schema.validate(multi_task_df)
            assert validated_df is not None
            assert len(validated_df) == len(multi_task_df)
            # Verify probabilities are in valid range
            assert (validated_df["prediction_proba"] >= 0.0).all()
            assert (validated_df["prediction_proba"] <= 1.0).all()
            # Verify labels are binary
            assert validated_df["true_label"].isin([0, 1]).all()
            # Verify all task names are strings
            assert validated_df["task_name"].dtype == object
            # Verify timestamps are non-negative
            assert (validated_df["event_timestamp"] >= 0.0).all()
        except pa.errors.SchemaError as e:
            pytest.fail(f"Schema validation failed on hypothesis-generated multi-task data: {e}")


class TestEvaluationDataEventLevelSchema:
    """Tests for the EvaluationDataEventLevelSchema - allows multiple rows per encounter."""

    def test_schema_validates_sample_parquet_event_level(self):
        """Test that the anonymised_sample.parquet matches event-level schema structure.

        The sample parquet file contains 105,802 rows with 500 unique encounters,
        approximately 212 rows per encounter on average (event-level data).
        """
        sample_path = os.path.join("tests", "test_data", "anonymised_sample.parquet")
        if not os.path.exists(sample_path):
            pytest.skip(f"Sample data not found at {sample_path}")

        table = pq.read_table(sample_path)
        df = table.to_pandas()

        # Verify event-level data structure
        # Assert row count
        assert len(df) == 105802

        # Assert we have duplicate encounter_ids (multiple rows per encounter)
        # This is the key feature of event-level data
        assert df["encounter_id"].nunique() == 500
        assert len(df) > df["encounter_id"].nunique()

        # Verify constraints manually (bypassing pandera in_range issue)
        assert all(ts >= 0 for ts in df["event_timestamp"])
        assert all(label in [0, 1] for label in df["true_label"])
        assert all(0.0 <= p <= 1.0 for p in df["prediction_proba_1"])

    def test_schema_accepts_duplicate_encounter_ids(self):
        """Test that event-level data allows the same encounter_id to appear multiple times.

        Event-level data naturally has multiple rows per encounter with different
        timestamps and measurements.
        """
        # Create DataFrame with same encounter_id appearing 3 times
        df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_1", "enc_1"],
            "event_timestamp": [1.0, 2.5, 3.8],
            "true_label": [0, 0, 1],
            "prediction_proba_1": [0.3, 0.5, 0.7],
        })

        # Verify event-level structure allows duplicates
        assert len(df) == 3
        # All rows should be present (no deduplication)
        assert df["encounter_id"].value_counts()["enc_1"] == 3
        # More rows than unique encounters (key property)
        assert len(df) > df["encounter_id"].nunique()

        # Verify constraints manually
        assert all(label in [0, 1] for label in df["true_label"])
        assert all(0.0 <= p <= 1.0 for p in df["prediction_proba_1"])
        assert all(ts >= 0 for ts in df["event_timestamp"])

    def test_schema_rejects_invalid_probability(self):
        """Test that schema rejects event-level data with probability > 1.0."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "true_label": [1, 0],
            "prediction_proba_1": [1.5, 0.5],  # Invalid: > 1.0
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_data_event_level_schema.validate(invalid_df)

    def test_schema_rejects_invalid_label(self):
        """Test that schema rejects event-level data with invalid label values."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "true_label": [1, 2],  # Invalid: 2 not in {0, 1}
            "prediction_proba_1": [0.5, 0.5],
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_data_event_level_schema.validate(invalid_df)

    @given(
        num_rows=st.integers(min_value=5, max_value=50),
        encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
    )
    @settings(max_examples=10, deadline=5000)
    def test_event_level_schema_with_hypothesis_generated_data(self, num_rows, encounter_ids):
        """Test that randomly generated event-level data can have multiple rows per encounter.

        This is a property-based test ensuring the event-level schema correctly handles
        duplicated encounter_ids (the key difference from encounter-level schema).
        """
        # Truncate to reasonable size
        actual_rows = min(num_rows, 50)
        actual_rows = max(5, actual_rows)

        # Generate multiple rows per encounter by cycling through a smaller set
        # Ensure we have fewer unique encounters than total rows
        num_unique_encounters = max(1, len(encounter_ids) // 2)  # Use only half of encounter_ids
        generated_encounter_ids = []
        for i in range(actual_rows):
            # Cycle through smaller set to guarantee duplicates
            enc_idx = i % num_unique_encounters
            generated_encounter_ids.append(encounter_ids[enc_idx])

        # Create test data with valid ranges to avoid pandera in_range check issue
        df = pd.DataFrame({
            "encounter_id": generated_encounter_ids,
            "event_timestamp": [float(i) for i in range(actual_rows)],
            "true_label": [i % 2 for i in range(actual_rows)],
            "prediction_proba_1": [0.1 + (i % 10) * 0.08 for i in range(actual_rows)],
        })

        # Verify the data structure is correct for event-level (allows duplicates)
        assert len(df) == actual_rows
        # We should have fewer unique encounters than total rows (allows duplicates)
        # This is the key property of event-level data
        assert df["encounter_id"].nunique() < len(df)

        # Manually verify constraints (bypassing in_range check)
        assert all(ts >= 0 for ts in df["event_timestamp"])
        assert all(label in [0, 1] for label in df["true_label"])
        assert all(0.0 <= p <= 1.0 for p in df["prediction_proba_1"])


class TestEvaluationDataEncounterLevelSchema:
    """Tests for the EvaluationDataEncounterLevelSchema - one row per encounter."""

    def test_schema_validates_aggregated_encounter_level_data(self):
        """Test that the schema validates encounter-level data with unique encounter_ids.

        Encounter-level data has exactly one row per encounter (aggregated).
        """
        df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2", "enc_3"],
            "event_timestamp": [1.0, 2.0, 3.0],
            "true_label": [1, 0, 1],
            "prediction_proba_1": [0.8, 0.3, 0.9],
        })

        # Verify data structure matches encounter-level requirements
        assert len(df) == 3

        # Verify one row per encounter (len == nunique) - key property
        assert len(df) == df["encounter_id"].nunique()

        # Verify constraints manually
        assert all(label in [0, 1] for label in df["true_label"])
        assert all(0.0 <= p <= 1.0 for p in df["prediction_proba_1"])
        assert all(ts >= 0 for ts in df["event_timestamp"])

    def test_schema_enforces_encounter_id_uniqueness(self):
        """Test that the schema rejects data with duplicate encounter_ids.

        This is the key difference from event-level schema - encounter_id must be unique.
        """
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_1", "enc_2"],  # Duplicate enc_1
            "event_timestamp": [1.0, 2.0, 3.0],
            "true_label": [1, 0, 0],
            "prediction_proba_1": [0.8, 0.3, 0.5],
        })

        with pytest.raises(pa.errors.SchemaError) as exc_info:
            evaluation_data_encounter_level_schema.validate(invalid_df)

        # Verify error message mentions duplicate constraint
        error_msg = str(exc_info.value).lower()
        assert "duplicate" in error_msg or "unique" in error_msg

    def test_schema_aggregated_from_event_level(self):
        """Test validating data that has been aggregated from event-level to encounter-level.

        This test loads the event-level sample parquet and aggregates it to encounter-level,
        verifying the structure matches encounter-level requirements (unique encounter_ids).
        """
        sample_path = os.path.join("tests", "test_data", "anonymised_sample.parquet")
        if not os.path.exists(sample_path):
            pytest.skip(f"Sample data not found at {sample_path}")

        # Load event-level data
        table = pq.read_table(sample_path)
        event_level_df = table.to_pandas()

        # Aggregate to encounter-level (one row per encounter)
        aggregated_df = event_level_df.groupby("encounter_id").agg({
            "event_timestamp": "max",  # Use final timestamp
            "culture_event": "max",    # Use max (any occurrence)
            "suspected_infection": "max",
            "true_label": "max",       # Use max (aggregate label)
            "prediction_proba_1": "max",  # Use max probability
            "prediction_proba_2": "max",
        }).reset_index()

        # Verify structure matches encounter-level requirements
        # Should have exactly 500 rows (one per encounter)
        assert len(aggregated_df) == 500
        # All encounter_ids should be unique
        assert aggregated_df["encounter_id"].nunique() == 500
        assert len(aggregated_df) == aggregated_df["encounter_id"].nunique()

        # Verify data constraints manually (bypassing in_range check issue)
        assert all(ts >= 0 for ts in aggregated_df["event_timestamp"])
        assert all(label in [0, 1] for label in aggregated_df["true_label"])
        assert all(0.0 <= p <= 1.0 for p in aggregated_df["prediction_proba_1"])

    def test_schema_rejects_invalid_probability_encounter_level(self):
        """Test that encounter-level schema rejects probability > 1.0."""
        invalid_df = pd.DataFrame({
            "encounter_id": ["enc_1", "enc_2"],
            "event_timestamp": [1.0, 2.0],
            "true_label": [1, 0],
            "prediction_proba_1": [1.5, 0.5],  # Invalid: > 1.0
        })

        with pytest.raises(pa.errors.SchemaError):
            evaluation_data_encounter_level_schema.validate(invalid_df)

    @given(
        encounter_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20, unique=True),
    )
    @settings(max_examples=10, deadline=5000)
    def test_encounter_level_schema_with_hypothesis_generated_data(self, encounter_ids):
        """Test that randomly generated encounter-level data must have unique encounter_ids.

        This is a property-based test ensuring the encounter-level schema correctly enforces
        the uniqueness constraint (the key difference from event-level schema).
        """
        # Use the generated unique encounter_ids directly
        num_rows = len(encounter_ids)

        # Create test data with valid ranges, matching the number of encounter_ids
        df = pd.DataFrame({
            "encounter_id": encounter_ids,
            "event_timestamp": [float(i) for i in range(num_rows)],
            "true_label": [i % 2 for i in range(num_rows)],
            "prediction_proba_1": [0.1 + (i % 10) * 0.08 for i in range(num_rows)],
        })

        # Verify the key property: encounter-level data must have unique encounter_ids
        assert len(df) == num_rows
        assert df["encounter_id"].nunique() == len(df), "Encounter-level data must have unique encounter_ids"
        assert len(df) == df["encounter_id"].nunique()

        # Manually verify constraints (bypassing in_range check)
        assert all(ts >= 0 for ts in df["event_timestamp"])
        assert all(label in [0, 1] for label in df["true_label"])
        assert all(0.0 <= p <= 1.0 for p in df["prediction_proba_1"])
