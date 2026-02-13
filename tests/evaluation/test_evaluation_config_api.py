"""
Tests for config-based evaluation API compatibility.
"""

import numpy as np
import pyarrow as pa
import pytest

from pysalient.evaluation import evaluation
from pysalient.evaluation._models import (
    ConfidenceIntervalConfig,
    EvaluationConfig,
    ThresholdCIMethod,
    ThresholdConfig,
)

META_KEY_Y_PROBA = "pysalient.io.y_proba_col"
META_KEY_Y_LABEL = "pysalient.io.y_label_col"
SYNTH_PROBA_COL = "synth_probas"
SYNTH_LABEL_COL = "synth_labels"

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:bootstrap_rounds < 500 may lead to less stable confidence intervals"
    ),
    pytest.mark.filterwarnings(
        "ignore:bootstrap_rounds is set to .* less reliable confidence interval estimates"
    ),
    pytest.mark.filterwarnings("ignore:Only one class is present in y_true"),
    pytest.mark.filterwarnings("ignore:No positive class found in y_true"),
    pytest.mark.filterwarnings("ignore:Bootstrap CI calculation failed.*seed must be an integer or None"),
]


def _build_table_with_metadata() -> pa.Table:
    data = {
        SYNTH_PROBA_COL: np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]),
        SYNTH_LABEL_COL: np.array([0, 0, 0, 1, 0, 1, 1, 1]),
    }
    metadata = {
        META_KEY_Y_PROBA.encode("utf-8"): SYNTH_PROBA_COL.encode("utf-8"),
        META_KEY_Y_LABEL.encode("utf-8"): SYNTH_LABEL_COL.encode("utf-8"),
    }
    return pa.table(data).replace_schema_metadata(metadata)


def test_evaluation_accepts_config_object_and_matches_legacy():
    """Config API should produce same deterministic result as legacy arguments."""
    table = _build_table_with_metadata()
    legacy = evaluation(table, "model-v1", "all", [0.5], force_threshold_zero=True)

    config = EvaluationConfig(
        data=table,
        modelid="model-v1",
        filter_desc="all",
        thresholds=ThresholdConfig(values=[0.5], force_threshold_zero=True),
    )
    with_config = evaluation(config)

    assert with_config.to_pydict() == legacy.to_pydict()


def test_evaluation_config_ci_fields_are_applied():
    """Config CI values should drive calculation and output schema."""
    table = _build_table_with_metadata()
    config = EvaluationConfig(
        data=table,
        modelid="model-v1",
        filter_desc="all",
        thresholds=ThresholdConfig(values=[0.5]),
        confidence_intervals=ConfidenceIntervalConfig(
            calculate_au_ci=True,
            alpha=0.05,
            bootstrap_rounds=100,
            bootstrap_seed=123,
        ),
    )
    result = evaluation(config)

    assert result.column("AUROC_Lower_CI")[0].as_py() is not None
    assert result.column("AUROC_Upper_CI")[0].as_py() is not None


def test_evaluation_config_allows_low_bootstrap_rounds_for_analytical_threshold_ci():
    """Config API should not enforce bootstrap round minimum for analytical threshold CI."""
    table = _build_table_with_metadata()
    config = EvaluationConfig(
        data=table,
        modelid="model-v1",
        filter_desc="all",
        thresholds=ThresholdConfig(values=[0.5]),
        confidence_intervals=ConfidenceIntervalConfig(
            calculate_threshold_ci=True,
            threshold_ci_method=ThresholdCIMethod.NORMAL,
            bootstrap_rounds=1,
            alpha=0.05,
        ),
    )

    result = evaluation(config)
    assert result.num_rows >= 1


def test_evaluation_config_ignores_bootstrap_rounds_when_ci_disabled():
    """Config API should accept ignored bootstrap settings when no CI is requested."""
    table = _build_table_with_metadata()
    config = EvaluationConfig(
        data=table,
        modelid="model-v1",
        filter_desc="all",
        thresholds=ThresholdConfig(values=[0.5]),
        confidence_intervals=ConfidenceIntervalConfig(
            calculate_au_ci=False,
            calculate_threshold_ci=False,
            bootstrap_rounds=0,
            alpha=2.0,
        ),
    )

    result = evaluation(config)
    assert result.num_rows >= 1


def test_evaluation_config_ignores_non_int_seed_for_analytical_threshold_ci():
    """Non-int bootstrap seed should be ignored for analytical threshold CI mode."""
    table = _build_table_with_metadata()
    config = EvaluationConfig(
        data=table,
        modelid="model-v1",
        filter_desc="all",
        thresholds=ThresholdConfig(values=[0.5]),
        confidence_intervals=ConfidenceIntervalConfig(
            calculate_threshold_ci=True,
            threshold_ci_method=ThresholdCIMethod.NORMAL,
            bootstrap_seed="abc",
            alpha=0.05,
        ),
    )

    result = evaluation(config)
    assert result.num_rows >= 1


def test_evaluation_config_non_int_seed_for_threshold_bootstrap_does_not_crash():
    """Non-int bootstrap seed should not crash threshold bootstrap evaluation path."""
    table = _build_table_with_metadata()
    config = EvaluationConfig(
        data=table,
        modelid="model-v1",
        filter_desc="all",
        thresholds=ThresholdConfig(values=[0.5]),
        confidence_intervals=ConfidenceIntervalConfig(
            calculate_threshold_ci=True,
            threshold_ci_method=ThresholdCIMethod.BOOTSTRAP,
            bootstrap_rounds=100,
            bootstrap_seed="abc",
            alpha=0.05,
        ),
    )

    result = evaluation(config)
    assert result.num_rows >= 1
