"""
Tests for the pysalient.evaluation.model_comparison module.
"""

import warnings

import numpy as np
import pyarrow as pa
import pytest

from pysalient.evaluation.model_comparison import compare_models


class TestCompareModels:
    """Test suite for compare_models function."""

    @pytest.fixture
    def sample_evaluation_result_1(self) -> pa.Table:
        """Create a sample evaluation result table for model 1."""
        data = {
            "modelid": ["model1"] * 3,
            "filter_desc": ["test"] * 3,
            "threshold": [0.1, 0.5, 0.9],
            "time_to_first_alert_value": [10.0, 15.0, 20.0],
            "time_to_first_alert_unit": ["hours"] * 3,
            "AUROC": [0.75, 0.75, 0.75],  # Threshold-independent
            "AUROC_Lower_CI": [0.70, 0.70, 0.70],
            "AUROC_Upper_CI": [0.80, 0.80, 0.80],
            "AUPRC": [0.65, 0.65, 0.65],  # Threshold-independent
            "AUPRC_Lower_CI": [0.60, 0.60, 0.60],
            "AUPRC_Upper_CI": [0.70, 0.70, 0.70],
            "Prevalence": [0.2, 0.2, 0.2],
            "Sample_Size": [1000, 1000, 1000],
            "Label_Count": [200, 200, 200],
            "TP": [50, 120, 180],
            "TN": [600, 400, 200],
            "FP": [200, 400, 600],
            "FN": [150, 80, 20],
            "PPV": [0.2, 0.23, 0.23],
            "PPV_Lower_CI": [0.15, 0.20, 0.20],
            "PPV_Upper_CI": [0.25, 0.26, 0.26],
            "Sensitivity": [0.25, 0.60, 0.90],
            "Sensitivity_Lower_CI": [0.20, 0.55, 0.85],
            "Sensitivity_Upper_CI": [0.30, 0.65, 0.95],
            "Specificity": [0.75, 0.50, 0.25],
            "Specificity_Lower_CI": [0.70, 0.45, 0.20],
            "Specificity_Upper_CI": [0.80, 0.55, 0.30],
            "NPV": [0.80, 0.83, 0.91],
            "NPV_Lower_CI": [0.75, 0.80, 0.88],
            "NPV_Upper_CI": [0.85, 0.86, 0.94],
            "Accuracy": [0.65, 0.52, 0.38],
            "Accuracy_Lower_CI": [0.60, 0.47, 0.33],
            "Accuracy_Upper_CI": [0.70, 0.57, 0.43],
            "F1_Score": [0.22, 0.32, 0.36],
            "F1_Score_Lower_CI": [0.18, 0.28, 0.32],
            "F1_Score_Upper_CI": [0.26, 0.36, 0.40],
        }
        return pa.Table.from_pydict(data)

    @pytest.fixture
    def sample_evaluation_result_2(self) -> pa.Table:
        """Create a sample evaluation result table for model 2."""
        data = {
            "modelid": ["model2"] * 3,
            "filter_desc": ["test"] * 3,
            "threshold": [0.1, 0.5, 0.9],
            "time_to_first_alert_value": [12.0, 18.0, 25.0],
            "time_to_first_alert_unit": ["hours"] * 3,
            "AUROC": [0.82, 0.82, 0.82],  # Better than model 1
            "AUROC_Lower_CI": [0.77, 0.77, 0.77],
            "AUROC_Upper_CI": [0.87, 0.87, 0.87],
            "AUPRC": [0.73, 0.73, 0.73],  # Better than model 1
            "AUPRC_Lower_CI": [0.68, 0.68, 0.68],
            "AUPRC_Upper_CI": [0.78, 0.78, 0.78],
            "Prevalence": [0.2, 0.2, 0.2],
            "Sample_Size": [1000, 1000, 1000],
            "Label_Count": [200, 200, 200],
            "TP": [60, 140, 190],
            "TN": [650, 450, 250],
            "FP": [150, 350, 550],
            "FN": [140, 60, 10],
            "PPV": [0.29, 0.29, 0.26],
            "PPV_Lower_CI": [0.24, 0.26, 0.23],
            "PPV_Upper_CI": [0.34, 0.32, 0.29],
            "Sensitivity": [0.30, 0.70, 0.95],
            "Sensitivity_Lower_CI": [0.25, 0.65, 0.90],
            "Sensitivity_Upper_CI": [0.35, 0.75, 1.00],
            "Specificity": [0.81, 0.56, 0.31],
            "Specificity_Lower_CI": [0.76, 0.51, 0.26],
            "Specificity_Upper_CI": [0.86, 0.61, 0.36],
            "NPV": [0.82, 0.88, 0.96],
            "NPV_Lower_CI": [0.77, 0.85, 0.93],
            "NPV_Upper_CI": [0.87, 0.91, 0.99],
            "Accuracy": [0.71, 0.59, 0.44],
            "Accuracy_Lower_CI": [0.66, 0.54, 0.39],
            "Accuracy_Upper_CI": [0.76, 0.64, 0.49],
            "F1_Score": [0.29, 0.42, 0.40],
            "F1_Score_Lower_CI": [0.25, 0.38, 0.36],
            "F1_Score_Upper_CI": [0.33, 0.46, 0.44],
        }
        return pa.Table.from_pydict(data)

    def test_basic_comparison(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test basic model comparison functionality."""
        results = compare_models(
            [sample_evaluation_result_1, sample_evaluation_result_2],
            model_labels=["LogRegressor", "LightGBM"],
        )

        # Check output structure
        expected_columns = {
            "threshold",
            "model",
            "metric",
            "value",
            "lower_ci",
            "upper_ci",
            "p_value",
        }
        assert set(results.column_names) == expected_columns

        # Check we have data for both models
        models = set(results["model"].to_pylist())
        assert models == {"LogRegressor", "LightGBM"}

        # Check we have multiple metrics
        metrics = set(results["metric"].to_pylist())
        assert "AUROC" in metrics
        assert "AUPRC" in metrics
        assert "F1_Score" in metrics

        # Check we have multiple thresholds
        thresholds = set(results["threshold"].to_pylist())
        assert thresholds == {0.1, 0.5, 0.9}

    def test_auto_generated_model_labels(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test that model labels are auto-generated when not provided."""
        results = compare_models(
            [sample_evaluation_result_1, sample_evaluation_result_2]
        )

        models = set(results["model"].to_pylist())
        assert models == {"Model_1", "Model_2"}

    def test_include_metrics_filter(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test filtering to include only specific metrics."""
        results = compare_models(
            [sample_evaluation_result_1, sample_evaluation_result_2],
            include_metrics=["AUROC", "F1_Score"],
        )

        metrics = set(results["metric"].to_pylist())
        assert metrics == {"AUROC", "F1_Score"}

    def test_decimal_places_rounding(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test that decimal_places parameter works correctly."""
        results = compare_models(
            [sample_evaluation_result_1, sample_evaluation_result_2],
            include_metrics=["AUROC"],
            decimal_places=2,
        )

        # Check that values are properly rounded
        values = results["value"].to_pylist()
        for value in values:
            if value is not None and not np.isnan(value):
                # Check that we have at most 2 decimal places
                rounded_value = round(value, 2)
                assert abs(value - rounded_value) < 1e-10

    def test_confidence_intervals_preserved(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test that confidence intervals are properly preserved."""
        results = compare_models(
            [sample_evaluation_result_1, sample_evaluation_result_2],
            include_metrics=["AUROC"],
        )

        # Filter for one model and threshold
        model1_auroc = results.filter(
            pa.compute.and_(
                pa.compute.equal(results["model"], "Model_1"),
                pa.compute.and_(
                    pa.compute.equal(results["metric"], "AUROC"),
                    pa.compute.equal(results["threshold"], 0.1),
                ),
            )
        )

        assert model1_auroc.num_rows == 1
        row = model1_auroc.to_pydict()

        # Check that CI values match the input data
        assert row["value"][0] == 0.75
        assert row["lower_ci"][0] == 0.70
        assert row["upper_ci"][0] == 0.80

    def test_input_validation_empty_list(self):
        """Test validation for empty evaluation results list."""
        with pytest.raises(ValueError, match="non-empty list"):
            compare_models([])

    def test_input_validation_single_model(self, sample_evaluation_result_1):
        """Test validation for single model (need at least 2)."""
        with pytest.raises(ValueError, match="At least 2 evaluation results"):
            compare_models([sample_evaluation_result_1])

    def test_input_validation_wrong_type(self):
        """Test validation for wrong input types."""
        with pytest.raises(TypeError, match="PyArrow Table"):
            compare_models([{"not": "a_table"}, {"also_not": "a_table"}])

    def test_input_validation_model_labels_length(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test validation for model_labels length mismatch."""
        with pytest.raises(ValueError, match="model_labels length"):
            compare_models(
                [sample_evaluation_result_1, sample_evaluation_result_2],
                model_labels=["OnlyOneLabel"],
            )

    def test_input_validation_missing_columns(self, sample_evaluation_result_1):
        """Test validation for missing required columns."""
        # Create table missing required column
        incomplete_data = {
            "modelid": ["model2"],
            "filter_desc": ["test"],
            # Missing 'threshold' column
            "AUROC": [0.8],
        }
        incomplete_table = pa.Table.from_pydict(incomplete_data)

        with pytest.raises(ValueError, match="missing required columns"):
            compare_models([sample_evaluation_result_1, incomplete_table])

    def test_input_validation_negative_decimal_places(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test validation for negative decimal_places."""
        with pytest.raises(ValueError, match="non-negative integer"):
            compare_models(
                [sample_evaluation_result_1, sample_evaluation_result_2],
                decimal_places=-1,
            )

    def test_threshold_mismatch_warning(self, sample_evaluation_result_1):
        """Test warning when thresholds don't match between models."""
        # Create model 2 with different thresholds
        mismatched_data = {
            "modelid": ["model2"] * 2,
            "filter_desc": ["test"] * 2,
            "threshold": [0.2, 0.6],  # Different thresholds
            "AUROC": [0.82, 0.82],
            "F1_Score": [0.3, 0.4],
        }
        mismatched_table = pa.Table.from_pydict(mismatched_data)

        with pytest.warns(UserWarning, match="Threshold mismatch"):
            compare_models([sample_evaluation_result_1, mismatched_table])

    def test_missing_metric_warning(self, sample_evaluation_result_1):
        """Test warning when a requested metric is missing from one model."""
        # Create model 2 missing F1_Score
        incomplete_data = {
            "modelid": ["model2"] * 3,
            "filter_desc": ["test"] * 3,
            "threshold": [0.1, 0.5, 0.9],
            "AUROC": [0.82, 0.82, 0.82],
            # Missing F1_Score
        }
        incomplete_table = pa.Table.from_pydict(incomplete_data)

        with pytest.warns(UserWarning, match="not found"):
            results = compare_models([sample_evaluation_result_1, incomplete_table])

            # Should still have AUROC for both models
            models_with_auroc = set(
                results.filter(pa.compute.equal(results["metric"], "AUROC"))[
                    "model"
                ].to_pylist()
            )
            assert models_with_auroc == {"Model_1", "Model_2"}

    def test_verbosity_suppresses_warnings(self, sample_evaluation_result_1):
        """Test that verbosity > 0 suppresses warnings."""
        # Create model with different thresholds
        mismatched_data = {
            "modelid": ["model2"] * 2,
            "filter_desc": ["test"] * 2,
            "threshold": [0.2, 0.6],
            "AUROC": [0.82, 0.82],
        }
        mismatched_table = pa.Table.from_pydict(mismatched_data)

        # Should not raise warning with verbosity=1
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            compare_models([sample_evaluation_result_1, mismatched_table], verbosity=1)

        # Filter out any unrelated warnings
        threshold_warnings = [
            w for w in warning_list if "Threshold mismatch" in str(w.message)
        ]
        assert len(threshold_warnings) == 0

    def test_real_world_scenario(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test a realistic usage scenario."""
        # Compare models focusing on key metrics
        results = compare_models(
            [sample_evaluation_result_1, sample_evaluation_result_2],
            model_labels=["LogRegressor", "LightGBM"],
            include_metrics=[
                "AUROC",
                "AUPRC",
                "F1_Score",
                "Sensitivity",
                "Specificity",
            ],
            decimal_places=3,
        )

        # Verify we can easily filter for specific comparisons
        auroc_comparison = results.filter(pa.compute.equal(results["metric"], "AUROC"))
        assert auroc_comparison.num_rows == 6  # 2 models × 3 thresholds

        # Check that LightGBM has better AUROC than LogRegressor
        lgb_auroc = auroc_comparison.filter(
            pa.compute.equal(auroc_comparison["model"], "LightGBM")
        )["value"].to_pylist()[0]

        lr_auroc = auroc_comparison.filter(
            pa.compute.equal(auroc_comparison["model"], "LogRegressor")
        )["value"].to_pylist()[0]

        assert lgb_auroc > lr_auroc  # 0.82 > 0.75

    def test_statistical_significance_basic(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test statistical significance functionality with mock bootstrap samples."""
        # Create mock bootstrap samples
        bootstrap_samples = [
            {
                "AUROC": np.array([0.74, 0.75, 0.76, 0.74, 0.75] * 200),  # Model 1: ~0.75
                "AUPRC": np.array([0.64, 0.65, 0.66, 0.64, 0.65] * 200),  # Model 1: ~0.65
            },
            {
                "AUROC": np.array([0.81, 0.82, 0.83, 0.81, 0.82] * 200),  # Model 2: ~0.82
                "AUPRC": np.array([0.72, 0.73, 0.74, 0.72, 0.73] * 200),  # Model 2: ~0.73
            },
        ]

        results = compare_models(
            [sample_evaluation_result_1, sample_evaluation_result_2],
            model_labels=["LogRegressor", "LightGBM"],
            include_metrics=["AUROC", "AUPRC"],
            calculate_statistical_significance=True,
            bootstrap_samples=bootstrap_samples,
            n_permutations=1000,
            permutation_seed=42,
        )

        # Check that p_value column has values
        p_values = results["p_value"].to_pylist()
        non_null_p_values = [p for p in p_values if p is not None and not np.isnan(p)]
        assert len(non_null_p_values) > 0, "Should have some non-null p-values"

        # Check that all rows for the same metric have the same p-value
        auroc_rows = results.filter(pa.compute.equal(results["metric"], "AUROC"))
        auroc_p_values = auroc_rows["p_value"].to_pylist()
        unique_auroc_p_values = set(auroc_p_values)
        assert len(unique_auroc_p_values) == 1, "All AUROC rows should have the same p-value"

        # P-values should be between 0 and 1
        for p_val in non_null_p_values:
            assert 0 <= p_val <= 1, f"P-value {p_val} should be between 0 and 1"

    def test_statistical_significance_validation(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test validation of statistical significance parameters."""
        # Test missing bootstrap_samples
        with pytest.raises(ValueError, match="bootstrap_samples is required"):
            compare_models(
                [sample_evaluation_result_1, sample_evaluation_result_2],
                calculate_statistical_significance=True,
                bootstrap_samples=None,
            )

        # Test mismatched lengths
        bootstrap_samples = [{"AUROC": np.array([0.75])}]  # Only one model
        with pytest.raises(ValueError, match="bootstrap_samples length"):
            compare_models(
                [sample_evaluation_result_1, sample_evaluation_result_2],
                calculate_statistical_significance=True,
                bootstrap_samples=bootstrap_samples,
            )

        # Test invalid significance_alpha
        bootstrap_samples = [
            {"AUROC": np.array([0.75])},
            {"AUROC": np.array([0.82])},
        ]
        with pytest.raises(ValueError, match="significance_alpha must be a float"):
            compare_models(
                [sample_evaluation_result_1, sample_evaluation_result_2],
                calculate_statistical_significance=True,
                bootstrap_samples=bootstrap_samples,
                significance_alpha=1.5,  # Invalid
            )

    def test_statistical_significance_disabled_by_default(
        self, sample_evaluation_result_1, sample_evaluation_result_2
    ):
        """Test that statistical significance is disabled by default."""
        results = compare_models(
            [sample_evaluation_result_1, sample_evaluation_result_2],
            model_labels=["LogRegressor", "LightGBM"],
        )

        # All p_values should be None/null when not calculated
        p_values = results["p_value"].to_pylist()
        non_null_p_values = [p for p in p_values if p is not None and not np.isnan(p)]
        assert len(non_null_p_values) == 0, "Should have no p-values when not calculated"
