"""
Tests for time-to-event functionality in pysalient.evaluation module.

These tests focus on the new time-to-event metrics that calculate
median/mean time from alert to clinical events for true positives only.
"""

import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from pysalient.evaluation import evaluation, META_KEY_Y_PROBA, META_KEY_Y_LABEL, META_KEY_AGGREGATION_COLS

# Constants for test data
PROBA_COL = "y_prob"
LABEL_COL = "sepsis3_flag"
ENCOUNTER_COL = "ENCNTR_ID"
TIMESERIES_COL = "EVENT_DT_TM"
CULTURE_EVENT_COL = "culture_event"
ANTIBIOTICS_EVENT_COL = "antibiotics_event"


@pytest.fixture
def base_metadata():
    """Base metadata required for evaluation."""
    from pysalient.evaluation import META_KEY_TIMESERIES_COL
    return {
        META_KEY_Y_PROBA.encode("utf-8"): PROBA_COL.encode("utf-8"),
        META_KEY_Y_LABEL.encode("utf-8"): LABEL_COL.encode("utf-8"),
        META_KEY_AGGREGATION_COLS.encode("utf-8"): json.dumps([ENCOUNTER_COL]).encode("utf-8"),
        META_KEY_TIMESERIES_COL.encode("utf-8"): TIMESERIES_COL.encode("utf-8"),
    }


@pytest.fixture
def time_to_event_data():
    """Create synthetic time-to-event test data."""
    # Create data with 3 encounters, multiple events per encounter
    data = {
        ENCOUNTER_COL: ["ENC1", "ENC1", "ENC1", "ENC2", "ENC2", "ENC3", "ENC3", "ENC3"],
        PROBA_COL: [0.1, 0.6, 0.8, 0.3, 0.7, 0.2, 0.5, 0.9],
        LABEL_COL: [0, 1, 1, 0, 1, 0, 1, 1],
        TIMESERIES_COL: pd.to_datetime([
            "2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00",  # ENC1
            "2023-01-02 09:00:00", "2023-01-02 10:00:00",  # ENC2
            "2023-01-03 08:00:00", "2023-01-03 09:00:00", "2023-01-03 10:00:00"   # ENC3
        ]),
        CULTURE_EVENT_COL: pd.to_datetime([
            "2023-01-01 13:00:00", "2023-01-01 13:00:00", "2023-01-01 13:00:00",  # ENC1: 2 hrs after first alert
            "2023-01-02 08:00:00", "2023-01-02 08:00:00",  # ENC2: 1 hr before first alert
            "2023-01-03 11:00:00", "2023-01-03 11:00:00", "2023-01-03 11:00:00"   # ENC3: 2 hrs after first alert
        ]),
        ANTIBIOTICS_EVENT_COL: pd.to_datetime([
            "2023-01-01 14:00:00", "2023-01-01 14:00:00", "2023-01-01 14:00:00",  # ENC1: 3 hrs after first alert
            "2023-01-02 11:00:00", "2023-01-02 11:00:00",  # ENC2: 1 hr after first alert  
            "2023-01-03 07:00:00", "2023-01-03 07:00:00", "2023-01-03 07:00:00"   # ENC3: 1 hr before first alert
        ])
    }
    
    df = pd.DataFrame(data)
    return pa.table(df)


@pytest.fixture
def time_to_event_table(time_to_event_data, base_metadata):
    """Time-to-event data with proper metadata."""
    return time_to_event_data.replace_schema_metadata(base_metadata)


class TestTimeToEventParameterValidation:
    """Test parameter validation for time-to-event functionality."""
    
    def test_time_to_event_cols_validation_invalid_type(self, time_to_event_table):
        """Test that invalid time_to_event_cols type raises TypeError."""
        with pytest.raises(TypeError, match="must be a dictionary or None"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                time_to_event_cols="invalid"
            )
    
    def test_time_to_event_cols_validation_empty_dict(self, time_to_event_table):
        """Test that empty time_to_event_cols dict raises ValueError."""
        with pytest.raises(ValueError, match="cannot be an empty dictionary"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                time_to_event_cols={}
            )
    
    def test_time_to_event_cols_validation_invalid_keys_values(self, time_to_event_table):
        """Test that non-string keys/values in time_to_event_cols raise TypeError."""
        with pytest.raises(TypeError, match="must be strings"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                time_to_event_cols={123: "valid"}
            )
        
        with pytest.raises(TypeError, match="must be strings"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                time_to_event_cols={"valid": 123}
            )
    
    def test_time_to_event_cols_validation_empty_strings(self, time_to_event_table):
        """Test that empty string keys/values raise ValueError."""
        with pytest.raises(ValueError, match="must be non-empty strings"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                time_to_event_cols={"": "valid"}
            )
        
        with pytest.raises(ValueError, match="must be non-empty strings"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                time_to_event_cols={"valid": ""}
            )
    
    def test_aggregation_func_validation_invalid_type(self, time_to_event_table):
        """Test that invalid aggregation_func type raises TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                aggregation_func=123
            )
    
    def test_aggregation_func_validation_empty_string(self, time_to_event_table):
        """Test that empty aggregation_func raises ValueError."""
        with pytest.raises(ValueError, match="cannot be an empty string"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                aggregation_func=""
            )
    
    def test_aggregation_func_validation_invalid_function(self, time_to_event_table):
        """Test that invalid NumPy function name raises ValueError."""
        with pytest.raises(ValueError, match="not a valid NumPy aggregation function"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                aggregation_func="invalid_function"
            )
    
    def test_missing_time_to_event_columns(self, time_to_event_table):
        """Test that missing time-to-event columns raise ValueError."""
        with pytest.raises(ValueError, match="not found in table"):
            evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                time_to_event_cols={"bc": "non_existent_column"}
            )


class TestTimeToEventBackwardCompatibility:
    """Test that existing functionality is not broken."""
    
    def test_evaluation_without_time_to_event_cols(self, time_to_event_table):
        """Test that evaluation works normally when time_to_event_cols=None."""
        result = evaluation(
            time_to_event_table,
            "model", "filter", [0.0, 0.5, 1.0]
        )
        
        # Should return normal schema without time-to-event columns
        assert isinstance(result, pa.Table)
        assert result.num_rows == 3
        
        # Check that no time-to-event columns are present
        column_names = result.column_names
        for col in column_names:
            assert "_from_first_alert_to_" not in col
            assert "count_first_alerts_before_" not in col
            assert "count_first_alerts_after_or_at_" not in col
        
        # Should not have time-to-first-alert columns when timeseries_col is None
    
    def test_evaluation_with_time_to_event_but_no_aggregation_metadata(self, time_to_event_data):
        """Test warning when time_to_event_cols provided but no aggregation metadata."""
        # Create table without aggregation metadata
        metadata = {
            META_KEY_Y_PROBA.encode("utf-8"): PROBA_COL.encode("utf-8"),
            META_KEY_Y_LABEL.encode("utf-8"): LABEL_COL.encode("utf-8"),
        }
        table = time_to_event_data.replace_schema_metadata(metadata)
        
        with pytest.warns(UserWarning, match="aggregation metadata not found"):
            result = evaluation(
                table,
                "model", "filter", [0.5],
                time_to_event_cols={"bc": CULTURE_EVENT_COL}
            )
        
        # Should not have time-to-event columns
        column_names = result.column_names
        for col in column_names:
            assert "_from_first_alert_to_" not in col
        
        # Should not have time-to-first-alert columns when timeseries_col is None


class TestTimeToEventBasicFunctionality:
    """Test basic time-to-event functionality."""
    
    def test_single_event_median_aggregation(self, time_to_event_table):
        """Test time-to-event calculation with single event and median aggregation."""
        result = evaluation(
            time_to_event_table,
            "model", "filter", [0.5],
            time_to_event_cols={"bc": CULTURE_EVENT_COL},
            aggregation_func="median"
        )
        
        # Check schema includes new columns
        expected_columns = [
            "median_hours_from_first_alert_to_bc",
            "count_first_alerts_before_bc", 
            "count_first_alerts_after_or_at_bc"
        ]
        
        for col in expected_columns:
            assert col in result.column_names
        
        # Convert to dict for easier checking
        result_dict = result.to_pydict()
        
        # At threshold 0.5, we expect:
        # ENC1: alert at 11:00, culture at 13:00 -> 2 hrs (positive)
        # ENC2: alert at 10:00, culture at 08:00 -> -2 hrs (negative) 
        # ENC3: alert at 09:00, culture at 11:00 -> 2 hrs (positive)
        # After groupby max per encounter: [2, -2, 2]
        # Median: 2 hrs
        
        assert result_dict["median_hours_from_first_alert_to_bc"][0] == pytest.approx(2.0)
        assert result_dict["count_first_alerts_before_bc"][0] == 2  # 2 encounters with positive time
        assert result_dict["count_first_alerts_after_or_at_bc"][0] == 1  # 1 encounter with negative/zero time
    
    def test_multiple_events(self, time_to_event_table):
        """Test time-to-event calculation with multiple clinical events."""
        result = evaluation(
            time_to_event_table,
            "model", "filter", [0.5],
            
            time_to_event_cols={
                "bc": CULTURE_EVENT_COL,
                "ab": ANTIBIOTICS_EVENT_COL
            },
            aggregation_func="median"
        )
        
        # Check schema includes columns for both events
        expected_columns = [
            "median_hours_from_first_alert_to_bc",
            "count_first_alerts_before_bc",
            "count_first_alerts_after_or_at_bc",
            "median_hours_from_first_alert_to_ab", 
            "count_first_alerts_before_ab",
            "count_first_alerts_after_or_at_ab",
        ]
        
        for col in expected_columns:
            assert col in result.column_names
        
        result_dict = result.to_pydict()
        
        # Verify both events have metrics calculated
        assert result_dict["median_hours_from_first_alert_to_bc"][0] is not None
        assert result_dict["median_hours_from_first_alert_to_ab"][0] is not None
    
    def test_different_aggregation_functions(self, time_to_event_table):
        """Test different aggregation functions generate correct column names."""
        for agg_func in ["mean", "min", "max", "std"]:
            result = evaluation(
                time_to_event_table,
                "model", "filter", [0.5],
                
                time_to_event_cols={"bc": CULTURE_EVENT_COL},
                aggregation_func=agg_func
            )
            
            expected_col = f"{agg_func}_hours_from_first_alert_to_bc"
            assert expected_col in result.column_names
            
            # Count columns should remain the same regardless of aggregation function
            assert "count_first_alerts_before_bc" in result.column_names
            assert "count_first_alerts_after_or_at_bc" in result.column_names


class TestTimeToEventEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_no_true_positives_at_threshold(self, base_metadata):
        """Test behavior when no true positives exist at a threshold."""
        # Create data where we have predictions but no true positives at high threshold
        data = {
            ENCOUNTER_COL: ["ENC1", "ENC2"],
            PROBA_COL: [0.3, 0.4],  # All predictions < 0.5
            LABEL_COL: [1, 1],      # All positive labels
            TIMESERIES_COL: pd.to_datetime(["2023-01-01 10:00:00", "2023-01-02 10:00:00"]),
            CULTURE_EVENT_COL: pd.to_datetime(["2023-01-01 12:00:00", "2023-01-02 13:00:00"])
        }
        
        table = pa.table(data).replace_schema_metadata(base_metadata)
        
        result = evaluation(
            table,
            "model", "filter", [0.8],  # High threshold, no TPs since all probas < 0.5
            
            time_to_event_cols={"bc": CULTURE_EVENT_COL},
            force_threshold_zero=False  # Don't include 0.0 threshold automatically
        )
        
        result_dict = result.to_pydict()
        
        # Should have NaN for time metric and 0 for counts
        assert np.isnan(result_dict["median_hours_from_first_alert_to_bc"][0])
        assert result_dict["count_first_alerts_before_bc"][0] == 0
        assert result_dict["count_first_alerts_after_or_at_bc"][0] == 0
    
    def test_all_alerts_before_events(self, base_metadata):
        """Test when all alerts occur before clinical events."""
        # Create data where all alerts are before events
        data = {
            ENCOUNTER_COL: ["ENC1", "ENC2"],
            PROBA_COL: [0.8, 0.9],
            LABEL_COL: [1, 1],
            TIMESERIES_COL: pd.to_datetime(["2023-01-01 10:00:00", "2023-01-02 10:00:00"]),
            CULTURE_EVENT_COL: pd.to_datetime(["2023-01-01 12:00:00", "2023-01-02 13:00:00"])  # 2 and 3 hrs later
        }
        
        table = pa.table(data).replace_schema_metadata(base_metadata)
        
        result = evaluation(
            table,
            "model", "filter", [0.5],
            
            time_to_event_cols={"bc": CULTURE_EVENT_COL}
        )
        
        result_dict = result.to_pydict()
        
        # All encounters should have positive time differences
        assert result_dict["count_first_alerts_before_bc"][0] == 2
        assert result_dict["count_first_alerts_after_or_at_bc"][0] == 0
    
    def test_all_alerts_after_events(self, base_metadata):
        """Test when all alerts occur after clinical events."""
        # Create data where all alerts are after events
        data = {
            ENCOUNTER_COL: ["ENC1", "ENC2"],
            PROBA_COL: [0.8, 0.9],
            LABEL_COL: [1, 1],
            TIMESERIES_COL: pd.to_datetime(["2023-01-01 12:00:00", "2023-01-02 13:00:00"]),
            CULTURE_EVENT_COL: pd.to_datetime(["2023-01-01 10:00:00", "2023-01-02 10:00:00"])  # 2 and 3 hrs earlier
        }
        
        table = pa.table(data).replace_schema_metadata(base_metadata)
        
        result = evaluation(
            table,
            "model", "filter", [0.5],
            
            time_to_event_cols={"bc": CULTURE_EVENT_COL}
        )
        
        result_dict = result.to_pydict()
        
        # All encounters should have negative time differences
        assert result_dict["count_first_alerts_before_bc"][0] == 0
        assert result_dict["count_first_alerts_after_or_at_bc"][0] == 2
    
    def test_missing_aggregation_column(self, time_to_event_data, base_metadata):
        """Test error when aggregation column doesn't exist in data."""
        # Create metadata pointing to non-existent aggregation column
        metadata = dict(base_metadata)
        metadata[META_KEY_AGGREGATION_COLS.encode("utf-8")] = json.dumps(["non_existent_column"]).encode("utf-8")
        
        table = time_to_event_data.replace_schema_metadata(metadata)
        
        with pytest.raises(ValueError, match="not found in table"):
            evaluation(
                table,
                "model", "filter", [0.5],
                time_to_event_cols={"bc": CULTURE_EVENT_COL}
            )


class TestTimeToEventIntegration:
    """Integration tests for complete time-to-event pipeline."""
    
    def test_full_pipeline_with_rounding(self, time_to_event_table):
        """Test complete pipeline with decimal rounding."""
        result = evaluation(
            time_to_event_table,
            "model", "filter", [0.5],
            
            time_to_event_cols={"bc": CULTURE_EVENT_COL},
            decimal_places=2
        )
        
        result_dict = result.to_pydict()
        
        # Time metric should be rounded, count metrics should not be
        time_value = result_dict["median_hours_from_first_alert_to_bc"][0]
        assert isinstance(time_value, float)
        # Check that it's rounded to 2 decimal places
        assert time_value == round(time_value, 2)
    
    def test_multiple_thresholds(self, time_to_event_table):
        """Test time-to-event calculation across multiple thresholds."""
        thresholds = [0.0, 0.5, 0.8]
        
        result = evaluation(
            time_to_event_table,
            "model", "filter", thresholds,
            
            time_to_event_cols={"bc": CULTURE_EVENT_COL}
        )
        
        assert result.num_rows == len(thresholds)
        
        result_dict = result.to_pydict()
        
        # Each threshold should have time-to-event metrics
        for i in range(len(thresholds)):
            assert "median_hours_from_first_alert_to_bc" in result_dict
            # Values may be different for each threshold based on true positives
    
    def test_confidence_intervals_with_time_to_event(self, time_to_event_table):
        """Test that CI calculation works alongside time-to-event metrics."""
        result = evaluation(
            time_to_event_table,
            "model", "filter", [0.5],
            
            time_to_event_cols={"bc": CULTURE_EVENT_COL},
            calculate_threshold_ci=True,
            bootstrap_rounds=100  # Small number for fast test
        )
        
        # Should have both CI columns and time-to-event columns
        column_names = result.column_names
        
        # Check CI columns exist
        assert "PPV_Lower_CI" in column_names
        assert "PPV_Upper_CI" in column_names
        
        # Check time-to-event columns exist
        assert "median_hours_from_first_alert_to_bc" in column_names
        assert "count_first_alerts_before_bc" in column_names