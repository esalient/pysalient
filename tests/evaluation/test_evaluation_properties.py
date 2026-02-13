"""Property-based tests for evaluation invariants."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

try:
    import hypothesis.strategies as st
    from hypothesis import HealthCheck, assume, given, settings
except ImportError:  # pragma: no cover - optional dependency in default env
    pytest.skip(
        "hypothesis is required for property-based tests", allow_module_level=True
    )

from pysalient.evaluation import META_KEY_Y_LABEL, META_KEY_Y_PROBA, evaluation
from tests.strategies import evaluation_data_strategy

PROPERTY_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

CI_PROPERTY_SETTINGS = settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _build_table(df) -> pa.Table:
    metadata = {
        META_KEY_Y_PROBA.encode("utf-8"): b"y_proba",
        META_KEY_Y_LABEL.encode("utf-8"): b"y_label",
    }
    return pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(
        metadata
    )


def _run_evaluation(df, thresholds: list[float], **kwargs):
    result_table = evaluation(
        _build_table(df),
        "property-model",
        "property-filter",
        thresholds,
        **kwargs,
    )
    return result_table.to_pandas()


def _row_for_threshold(results_df, threshold: float):
    mask = np.isclose(results_df["threshold"].to_numpy(dtype=float), threshold)
    assert mask.any(), f"Threshold {threshold} was not present in results."
    return results_df.loc[mask].iloc[0]


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_confusion_matrix_sums_to_n(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    assert row["TP"] + row["TN"] + row["FP"] + row["FN"] == len(data)


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_tp_plus_fn_equals_total_positives(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    assert row["TP"] + row["FN"] == int(data["y_label"].sum())


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_fp_plus_tn_equals_total_negatives(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    assert row["FP"] + row["TN"] == len(data) - int(data["y_label"].sum())


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_tp_plus_fp_equals_total_predicted_positives(data):
    threshold = 0.5
    results = _run_evaluation(data, thresholds=[0.0, threshold])
    row = _row_for_threshold(results, threshold)
    expected_predicted_positive = int((data["y_proba"].to_numpy() >= threshold).sum())
    assert row["TP"] + row["FP"] == expected_predicted_positive


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_threshold_zero_all_predicted_positive(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.0)
    assert row["FN"] == 0


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_threshold_one_above_max_means_no_predicted_positive(data):
    data = data.copy()
    data["y_proba"] = np.minimum(data["y_proba"], 0.999999)
    results = _run_evaluation(data, thresholds=[0.0, 1.0])
    row = _row_for_threshold(results, 1.0)
    assert row["TP"] == 0
    assert row["FP"] == 0


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_sensitivity_equals_tp_over_tp_plus_fn(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    tp = int(row["TP"])
    fn = int(row["FN"])
    denominator = tp + fn
    expected = tp / denominator if denominator > 0 else 0.0
    assert row["Sensitivity"] == pytest.approx(expected, abs=1e-12)


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_ppv_equals_tp_over_tp_plus_fp(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    tp = int(row["TP"])
    fp = int(row["FP"])
    denominator = tp + fp
    expected = tp / denominator if denominator > 0 else 0.0
    assert row["PPV"] == pytest.approx(expected, abs=1e-12)


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_specificity_equals_tn_over_tn_plus_fp(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    tn = int(row["TN"])
    fp = int(row["FP"])
    denominator = tn + fp
    expected = tn / denominator if denominator > 0 else 0.0
    assert row["Specificity"] == pytest.approx(expected, abs=1e-12)


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_npv_equals_tn_over_tn_plus_fn(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    tn = int(row["TN"])
    fn = int(row["FN"])
    denominator = tn + fn
    expected = tn / denominator if denominator > 0 else 0.0
    assert row["NPV"] == pytest.approx(expected, abs=1e-12)


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_accuracy_equals_tp_plus_tn_over_n(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    expected = (int(row["TP"]) + int(row["TN"])) / len(data)
    assert row["Accuracy"] == pytest.approx(expected, abs=1e-12)


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_f1_equals_harmonic_mean_of_ppv_and_sensitivity(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    ppv = float(row["PPV"])
    sensitivity = float(row["Sensitivity"])
    denominator = ppv + sensitivity
    expected = (2.0 * ppv * sensitivity / denominator) if denominator > 0 else 0.0
    assert row["F1_Score"] == pytest.approx(expected, abs=1e-12)


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_prevalence_equals_total_positives_over_n(data):
    results = _run_evaluation(data, thresholds=[0.0, 0.5, 0.9])
    expected_prevalence = int(data["y_label"].sum()) / len(data)
    for _, row in results.iterrows():
        assert row["Prevalence"] == pytest.approx(expected_prevalence, abs=1e-12)


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_auroc_bounded_zero_one_when_two_classes_present(data):
    assume(data["y_label"].nunique() == 2)
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    assert -1e-12 <= float(row["AUROC"]) <= 1.0 + 1e-12


@PROPERTY_SETTINGS
@given(data=evaluation_data_strategy())
def test_auprc_bounded_zero_one_when_two_classes_present(data):
    assume(data["y_label"].nunique() == 2)
    results = _run_evaluation(data, thresholds=[0.0, 0.5])
    row = _row_for_threshold(results, 0.5)
    assert -1e-12 <= float(row["AUPRC"]) <= 1.0 + 1e-12


@CI_PROPERTY_SETTINGS
@given(
    data=evaluation_data_strategy(
        n_rows=st.integers(min_value=40, max_value=120),
        n_encounters=st.integers(min_value=2, max_value=40),
    )
)
def test_ci_lower_le_point_le_upper_when_overall_ci_enabled(data):
    assume(data["y_label"].nunique() == 2)
    results = _run_evaluation(
        data,
        thresholds=[0.0, 0.5],
        calculate_au_ci=True,
        bootstrap_rounds=100,
    )
    row = _row_for_threshold(results, 0.5)
    if not np.isnan(row["AUROC_Lower_CI"]) and not np.isnan(row["AUROC_Upper_CI"]):
        assert row["AUROC_Lower_CI"] <= row["AUROC"] <= row["AUROC_Upper_CI"]
    if not np.isnan(row["AUPRC_Lower_CI"]) and not np.isnan(row["AUPRC_Upper_CI"]):
        assert row["AUPRC_Lower_CI"] <= row["AUPRC"] <= row["AUPRC_Upper_CI"]


@PROPERTY_SETTINGS
@given(
    data=evaluation_data_strategy(),
    thresholds=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    ),
)
def test_output_row_count_equals_unique_threshold_count_with_zero_included(
    data, thresholds
):
    threshold_spec = [0.0, *thresholds]
    results = _run_evaluation(data, thresholds=threshold_spec)
    assert len(results) == len(set(threshold_spec))
