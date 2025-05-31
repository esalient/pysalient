"""Tests for pysalient.evaluation._evaluation_process internal functions."""

import warnings
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pytest

from pysalient.evaluation._evaluation_process import (
    _process_single_evaluation,
)


# Test fixtures
@pytest.fixture
def basic_test_data():
    """Basic test data with known outcomes."""
    # probas: [0.1, 0.3, 0.6, 0.8]
    # labels: [0, 0, 1, 1]
    # At threshold 0.5: TP=2, TN=2, FP=0, FN=0
    probas = np.array([0.1, 0.3, 0.6, 0.8])
    labels = np.array([0, 0, 1, 1])
    return probas, labels


@pytest.fixture
def larger_test_data():
    """Larger dataset for CI testing."""
    np.random.seed(42)
    probas = np.random.rand(100)
    labels = (probas > 0.5).astype(int)  # Create correlation
    return probas, labels


@pytest.fixture
def timeseries_data():
    """Test data with timeseries information."""
    probas = np.array([0.2, 0.4, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])
    # Integer timeseries
    timeseries_int = np.array([1, 2, 3, 4])
    # Temporal timeseries
    timeseries_temporal = np.array(['2023-01-01T00:00:00', '2023-01-01T01:00:00',
                                   '2023-01-01T02:00:00', '2023-01-01T03:00:00'],
                                  dtype='datetime64[ns]')
    return probas, labels, timeseries_int, timeseries_temporal


class TestProcessSingleEvaluationBasic:
    """Test basic functionality of _process_single_evaluation."""

    def test_basic_evaluation_single_threshold(self, basic_test_data):
        """Test basic evaluation with single threshold."""
        probas, labels = basic_test_data
        results = _process_single_evaluation(
            probas=probas,
            labels=labels,
            modelid="test_model",
            filter_desc="test_filter",
            threshold_list=[0.5],
            timeseries_array=None,
            timeseries_pa_type=None,
            time_unit=None
        )

        assert len(results) == 1
        result = results[0]

        # Check metadata
        assert result["modelid"] == "test_model"
        assert result["filter_desc"] == "test_filter"
        assert result["threshold"] == 0.5

        # Check confusion matrix
        assert result["TP"] == 2
        assert result["TN"] == 2
        assert result["FP"] == 0
        assert result["FN"] == 0

        # Check metrics
        assert result["PPV"] == 1.0
        assert result["Sensitivity"] == 1.0
        assert result["Specificity"] == 1.0
        assert result["NPV"] == 1.0
        assert result["Accuracy"] == 1.0
        assert result["F1_Score"] == 1.0

        # Check overall metrics
        assert result["Sample_Size"] == 4
        assert result["Label_Count"] == 2
        assert result["Prevalence"] == 0.5

    def test_multiple_thresholds(self, basic_test_data):
        """Test evaluation with multiple thresholds."""
        probas, labels = basic_test_data
        results = _process_single_evaluation(
            probas=probas,
            labels=labels,
            modelid="test_model",
            filter_desc="test_filter",
            threshold_list=[0.0, 0.5, 1.0],
            timeseries_array=None,
            timeseries_pa_type=None,
            time_unit=None
        )

        assert len(results) == 3

        # Threshold 0.0: all predicted positive
        result_0 = results[0]
        assert result_0["threshold"] == 0.0
        assert result_0["TP"] == 2
        assert result_0["TN"] == 0
        assert result_0["FP"] == 2
        assert result_0["FN"] == 0

        # Threshold 1.0: all predicted negative
        result_1 = results[2]
        assert result_1["threshold"] == 1.0
        assert result_1["TP"] == 0
        assert result_1["TN"] == 2
        assert result_1["FP"] == 0
        assert result_1["FN"] == 2

    def test_empty_data_handling(self):
        """Test handling of empty data arrays."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _process_single_evaluation(
                probas=np.array([]),
                labels=np.array([]),
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=None,
                timeseries_pa_type=None,
                time_unit=None
            )

        assert len(results) == 0

    def test_single_class_all_positive(self):
        """Test handling of dataset with only positive labels."""
        probas = np.array([0.2, 0.5, 0.8])
        labels = np.array([1, 1, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _process_single_evaluation(
                probas=probas,
                labels=labels,
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=None,
                timeseries_pa_type=None,
                time_unit=None
            )

        assert len(results) == 1
        result = results[0]
        assert np.isnan(result["AUROC"])
        assert np.isnan(result["AUPRC"])

    def test_single_class_all_negative(self):
        """Test handling of dataset with only negative labels."""
        probas = np.array([0.2, 0.5, 0.8])
        labels = np.array([0, 0, 0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _process_single_evaluation(
                probas=probas,
                labels=labels,
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=None,
                timeseries_pa_type=None,
                time_unit=None
            )

        assert len(results) == 1
        result = results[0]
        assert np.isnan(result["AUROC"])
        assert np.isnan(result["AUPRC"])


class TestProcessSingleEvaluationTimeToFirstAlert:
    """Test time-to-first-alert functionality."""

    def test_time_to_first_alert_integer_timeseries(self, timeseries_data):
        """Test time-to-first-alert with integer timeseries."""
        probas, labels, timeseries_int, _ = timeseries_data

        results = _process_single_evaluation(
            probas=probas,
            labels=labels,
            modelid="test_model",
            filter_desc="test_filter",
            threshold_list=[0.5],
            timeseries_array=timeseries_int,
            timeseries_pa_type=pa.int64(),
            time_unit="minutes"
        )

        assert len(results) == 1
        result = results[0]

        # First alert should be at index 2 (threshold 0.5, probas=[0.2, 0.4, 0.7, 0.9])
        assert result["time_to_first_alert_value"] == 3.0  # timeseries_int[2]
        assert result["time_to_first_alert_unit"] == "minutes"

    def test_time_to_first_alert_temporal_timeseries(self, timeseries_data):
        """Test time-to-first-alert with temporal timeseries."""
        probas, labels, _, timeseries_temporal = timeseries_data

        results = _process_single_evaluation(
            probas=probas,
            labels=labels,
            modelid="test_model",
            filter_desc="test_filter",
            threshold_list=[0.5],
            timeseries_array=timeseries_temporal,
            timeseries_pa_type=pa.timestamp('ns'),
            time_unit=None  # Not required for temporal
        )

        assert len(results) == 1
        result = results[0]

        # First alert at index 2, time difference should be 2 hours = 7200 seconds
        assert result["time_to_first_alert_value"] == 7200.0
        assert result["time_to_first_alert_unit"] == "seconds"

    def test_time_to_first_alert_no_alerts(self):
        """Test time-to-first-alert when no alerts are triggered."""
        probas = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 1, 1])
        timeseries_int = np.array([1, 2, 3, 4])

        results = _process_single_evaluation(
            probas=probas,
            labels=labels,
            modelid="test_model",
            filter_desc="test_filter",
            threshold_list=[0.5],  # No probas >= 0.5
            timeseries_array=timeseries_int,
            timeseries_pa_type=pa.int64(),
            time_unit="minutes"
        )

        assert len(results) == 1
        result = results[0]

        assert result["time_to_first_alert_value"] is None
        assert result["time_to_first_alert_unit"] is None

    def test_time_to_first_alert_missing_time_unit(self):
        """Test error when time_unit is missing for integer/float timeseries."""
        probas = np.array([0.2, 0.7])
        labels = np.array([0, 1])
        timeseries_int = np.array([1, 2])

        with pytest.raises(ValueError, match="time_unit is required"):
            _process_single_evaluation(
                probas=probas,
                labels=labels,
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=timeseries_int,
                timeseries_pa_type=pa.int64(),
                time_unit=None  # Missing required time_unit
            )

    def test_time_to_first_alert_unsupported_type(self):
        """Test error with unsupported timeseries data type."""
        probas = np.array([0.2, 0.7])
        labels = np.array([0, 1])
        timeseries_str = np.array(["a", "b"])

        with pytest.raises(TypeError, match="Unsupported timeseries data type"):
            _process_single_evaluation(
                probas=probas,
                labels=labels,
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=timeseries_str,
                timeseries_pa_type=pa.string(),
                time_unit=None
            )


class TestProcessSingleEvaluationConfidenceIntervals:
    """Test confidence interval calculations."""

    @pytest.mark.parametrize("calculate_au_ci", [True, False])
    def test_au_ci_calculation(self, larger_test_data, calculate_au_ci):
        """Test overall AU CI calculation."""
        probas, labels = larger_test_data

        with patch('pysalient.evaluation._evaluation_process.calculate_bootstrap_ci') as mock_bootstrap:
            if calculate_au_ci:
                mock_bootstrap.return_value = (0.7, 0.9)  # Mock CI values

            results = _process_single_evaluation(
                probas=probas,
                labels=labels,
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=None,
                timeseries_pa_type=None,
                time_unit=None,
                calculate_au_ci=calculate_au_ci,
                bootstrap_rounds=100,
                bootstrap_seed=42
            )

            assert len(results) == 1
            result = results[0]

            if calculate_au_ci:
                assert result["AUROC_Lower_CI"] == 0.7
                assert result["AUROC_Upper_CI"] == 0.9
                assert result["AUPRC_Lower_CI"] == 0.7
                assert result["AUPRC_Upper_CI"] == 0.9
                assert mock_bootstrap.call_count == 2  # Called for AUROC and AUPRC
            else:
                assert result["AUROC_Lower_CI"] is None
                assert result["AUROC_Upper_CI"] is None
                assert result["AUPRC_Lower_CI"] is None
                assert result["AUPRC_Upper_CI"] is None
                assert not mock_bootstrap.called

    @pytest.mark.parametrize("calculate_threshold_ci", [True, False])
    def test_threshold_ci_bootstrap(self, larger_test_data, calculate_threshold_ci):
        """Test threshold CI calculation with bootstrap method."""
        probas, labels = larger_test_data

        with patch('pysalient.evaluation._evaluation_process.calculate_bootstrap_ci') as mock_bootstrap:
            if calculate_threshold_ci:
                mock_bootstrap.return_value = (0.6, 0.8)  # Mock CI values

            results = _process_single_evaluation(
                probas=probas,
                labels=labels,
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=None,
                timeseries_pa_type=None,
                time_unit=None,
                calculate_threshold_ci=calculate_threshold_ci,
                threshold_ci_method="bootstrap",
                bootstrap_rounds=100,
                bootstrap_seed=42
            )

            assert len(results) == 1
            result = results[0]

            if calculate_threshold_ci:
                # Should have CIs for all threshold metrics
                assert result["PPV_Lower_CI"] == 0.6
                assert result["PPV_Upper_CI"] == 0.8
                assert result["Sensitivity_Lower_CI"] == 0.6
                assert result["Sensitivity_Upper_CI"] == 0.8
                assert mock_bootstrap.call_count == 6  # 6 threshold metrics
            else:
                assert result["PPV_Lower_CI"] is None
                assert result["PPV_Upper_CI"] is None
                assert not mock_bootstrap.called

    @pytest.mark.parametrize("analytical_method", ["normal", "wilson", "agresti-coull"])
    def test_threshold_ci_analytical(self, larger_test_data, analytical_method):
        """Test threshold CI calculation with analytical methods."""
        probas, labels = larger_test_data

        with patch('pysalient.evaluation._evaluation_process.anaci') as mock_anaci:
            mock_ci_func = Mock(return_value=(0.5, 0.9))
            if analytical_method == "normal":
                mock_anaci.calculate_normal_approx_ci = mock_ci_func
            elif analytical_method == "wilson":
                mock_anaci.calculate_wilson_score_ci = mock_ci_func
            elif analytical_method == "agresti-coull":
                mock_anaci.calculate_agresti_coull_ci = mock_ci_func

            results = _process_single_evaluation(
                probas=probas,
                labels=labels,
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=None,
                timeseries_pa_type=None,
                time_unit=None,
                calculate_threshold_ci=True,
                threshold_ci_method=analytical_method,
                ci_alpha=0.05
            )

            assert len(results) == 1
            result = results[0]

            # Should have CIs for supported metrics (not F1)
            assert result["PPV_Lower_CI"] == 0.5
            assert result["PPV_Upper_CI"] == 0.9
            assert result["Sensitivity_Lower_CI"] == 0.5
            assert result["Sensitivity_Upper_CI"] == 0.9

            # F1 should be NaN for analytical methods
            assert np.isnan(result["F1_Score_Lower_CI"])
            assert np.isnan(result["F1_Score_Upper_CI"])


class TestProcessSingleEvaluationRounding:
    """Test decimal places rounding functionality."""

    def test_rounding_applied_to_metrics(self, basic_test_data):
        """Test that rounding is applied to all metrics."""
        probas, labels = basic_test_data

        results = _process_single_evaluation(
            probas=probas,
            labels=labels,
            modelid="test_model",
            filter_desc="test_filter",
            threshold_list=[0.333],  # Threshold that creates non-round metrics
            timeseries_array=None,
            timeseries_pa_type=None,
            time_unit=None,
            decimal_places=2
        )

        assert len(results) == 1
        result = results[0]

        # Check that values are rounded to 2 decimal places
        for key, value in result.items():
            if isinstance(value, float) and not np.isnan(value):
                # Check that the value has at most 2 decimal places
                assert abs(value * 100 - round(value * 100)) < 1e-9

    def test_rounding_applied_to_cis(self, larger_test_data):
        """Test that rounding is applied to confidence intervals."""
        probas, labels = larger_test_data

        with patch('pysalient.evaluation._evaluation_process.calculate_bootstrap_ci') as mock_bootstrap:
            mock_bootstrap.return_value = (0.123456, 0.987654)  # Unrounded CI values

            results = _process_single_evaluation(
                probas=probas,
                labels=labels,
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=None,
                timeseries_pa_type=None,
                time_unit=None,
                calculate_au_ci=True,
                bootstrap_rounds=100,
                decimal_places=3
            )

            assert len(results) == 1
            result = results[0]

            # Check CI values are rounded to 3 decimal places
            assert result["AUROC_Lower_CI"] == 0.123
            assert result["AUROC_Upper_CI"] == 0.988  # Rounded up


class TestProcessSingleEvaluationErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_sklearn_warning(self, basic_test_data):
        """Test warning when sklearn is not available."""
        probas, labels = basic_test_data

        with patch('pysalient.evaluation._evaluation_process.SKLEARN_AVAILABLE', False):
            with patch('pysalient.evaluation._evaluation_process.roc_auc_score', None):
                with warnings.catch_warnings():
                    warnings.simplefilter("always")

                    results = _process_single_evaluation(
                        probas=probas,
                        labels=labels,
                        modelid="test_model",
                        filter_desc="test_filter",
                        threshold_list=[0.5],
                        timeseries_array=None,
                        timeseries_pa_type=None,
                        time_unit=None
                    )

                    # Should still return results with NaN metrics
                    assert len(results) == 1
                    result = results[0]
                    assert np.isnan(result["AUROC"])
                    assert np.isnan(result["AUPRC"])

    def test_bootstrap_ci_failure_warning(self, larger_test_data):
        """Test warning when bootstrap CI calculation fails."""
        probas, labels = larger_test_data

        with patch('pysalient.evaluation._evaluation_process.calculate_bootstrap_ci') as mock_bootstrap:
            mock_bootstrap.side_effect = RuntimeError("Bootstrap failed")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                results = _process_single_evaluation(
                    probas=probas,
                    labels=labels,
                    modelid="test_model",
                    filter_desc="test_filter",
                    threshold_list=[0.5],
                    timeseries_array=None,
                    timeseries_pa_type=None,
                    time_unit=None,
                    calculate_au_ci=True,
                    bootstrap_rounds=100
                )

                # Should return results with None CIs
                assert len(results) == 1
                result = results[0]
                assert result["AUROC_Lower_CI"] is None
                assert result["AUROC_Upper_CI"] is None

                # Should have issued warning
                assert any("confidence interval calculation failed" in str(warning.message)
                          for warning in w)

    def test_temporal_nat_handling(self):
        """Test handling of NaT values in temporal timeseries."""
        probas = np.array([0.2, 0.7])
        labels = np.array([0, 1])
        timeseries_temporal = np.array(['2023-01-01T00:00:00', 'NaT'], dtype='datetime64[ns]')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = _process_single_evaluation(
                probas=probas,
                labels=labels,
                modelid="test_model",
                filter_desc="test_filter",
                threshold_list=[0.5],
                timeseries_array=timeseries_temporal,
                timeseries_pa_type=pa.timestamp('ns'),
                time_unit=None
            )

            assert len(results) == 1
            result = results[0]

            # Should handle NaT gracefully
            assert result["time_to_first_alert_value"] is None
            assert result["time_to_first_alert_unit"] is None

            # Should have issued warning
            assert any("NaT (Not a Time)" in str(warning.message) for warning in w)
