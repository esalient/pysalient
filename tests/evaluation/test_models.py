"""
Tests for Pydantic evaluation config models.
"""

import pyarrow as pa
import pytest
from pydantic import ValidationError

from pysalient.evaluation._models import (
    ConfidenceIntervalConfig,
    EvaluationConfig,
    ThresholdCIMethod,
    ThresholdConfig,
    TimeToEventConfig,
    TimeUnit,
)


@pytest.fixture
def sample_table() -> pa.Table:
    """Create a small, valid pyarrow table for config tests."""
    return pa.table(
        {
            "y_true": [0, 1, 1, 0],
            "y_score": [0.1, 0.8, 0.9, 0.2],
        }
    )


def test_time_unit_accepts_existing_supported_aliases():
    """TimeUnit should accept aliases currently supported by evaluation()."""
    assert TimeUnit("hour") == TimeUnit.HOUR
    assert TimeUnit("h") == TimeUnit.H
    assert TimeUnit("minutes") == TimeUnit.MINUTES
    assert TimeUnit("w") == TimeUnit.W


def test_threshold_ci_method_enum_values():
    """Threshold CI method enum should expose currently supported methods."""
    assert ThresholdCIMethod("bootstrap") == ThresholdCIMethod.BOOTSTRAP
    assert ThresholdCIMethod("normal") == ThresholdCIMethod.NORMAL
    assert ThresholdCIMethod("wilson") == ThresholdCIMethod.WILSON
    assert ThresholdCIMethod("agresti-coull") == ThresholdCIMethod.AGRESTI_COULL


def test_threshold_config_rejects_empty_thresholds():
    """ThresholdConfig requires at least one threshold value."""
    with pytest.raises(ValidationError):
        ThresholdConfig(values=[])


def test_time_to_event_config_rejects_empty_event_columns():
    """TimeToEventConfig requires non-empty event column mapping."""
    with pytest.raises(ValidationError):
        TimeToEventConfig(event_columns={})


def test_confidence_interval_config_alpha_bounds():
    """ConfidenceIntervalConfig enforces 0 < alpha < 1 when CI is enabled."""
    with pytest.raises(ValidationError):
        ConfidenceIntervalConfig(calculate_au_ci=True, alpha=0.0)
    with pytest.raises(ValidationError):
        ConfidenceIntervalConfig(calculate_au_ci=True, alpha=1.0)


def test_confidence_interval_config_allows_any_alpha_when_ci_disabled():
    """alpha should only be constrained when any CI calculation is enabled."""
    config = ConfidenceIntervalConfig(
        calculate_au_ci=False,
        calculate_threshold_ci=False,
        alpha=2.0,
    )
    assert config.alpha == 2.0


def test_confidence_interval_config_allows_low_rounds_when_bootstrap_not_used():
    """bootstrap_rounds should not be constrained when bootstrap CI is not requested."""
    config = ConfidenceIntervalConfig(
        calculate_au_ci=False,
        calculate_threshold_ci=True,
        threshold_ci_method=ThresholdCIMethod.NORMAL,
        bootstrap_rounds=1,
    )
    assert config.bootstrap_rounds == 1


def test_confidence_interval_config_allows_zero_rounds_when_ci_disabled():
    """bootstrap_rounds should be ignored when all CI calculations are disabled."""
    config = ConfidenceIntervalConfig(
        calculate_au_ci=False,
        calculate_threshold_ci=False,
        bootstrap_rounds=0,
    )
    assert config.bootstrap_rounds == 0


def test_confidence_interval_config_allows_non_int_seed_when_ci_disabled():
    """bootstrap_seed should be ignored when CI is disabled."""
    config = ConfidenceIntervalConfig(
        calculate_au_ci=False,
        calculate_threshold_ci=False,
        bootstrap_seed="abc",
    )
    assert config.bootstrap_seed == "abc"


def test_confidence_interval_config_rejects_non_int_seed_for_au_ci():
    """bootstrap_seed should be validated when AU bootstrap CI is enabled."""
    with pytest.raises(ValidationError):
        ConfidenceIntervalConfig(
            calculate_au_ci=True,
            bootstrap_seed="abc",
        )


def test_confidence_interval_config_rejects_low_rounds_when_bootstrap_used():
    """bootstrap_rounds < 100 should be rejected when bootstrap CI is requested."""
    with pytest.raises(ValidationError):
        ConfidenceIntervalConfig(
            calculate_au_ci=True,
            bootstrap_rounds=99,
        )


def test_evaluation_config_accepts_minimal_valid_payload(sample_table):
    """EvaluationConfig should validate a minimal legacy-equivalent payload."""
    config = EvaluationConfig(
        data=sample_table,
        modelid="model-v1",
        filter_desc="all",
        thresholds=ThresholdConfig(values=[0.5]),
    )

    assert config.modelid == "model-v1"
    assert config.filter_desc == "all"
    assert config.thresholds.values == [0.5]
    assert config.confidence_intervals.threshold_ci_method == ThresholdCIMethod.BOOTSTRAP


def test_evaluation_config_rejects_negative_decimal_places(sample_table):
    """EvaluationConfig should enforce non-negative decimal_places."""
    with pytest.raises(ValidationError):
        EvaluationConfig(
            data=sample_table,
            modelid="model-v1",
            filter_desc="all",
            thresholds=ThresholdConfig(values=[0.5]),
            decimal_places=-1,
        )
