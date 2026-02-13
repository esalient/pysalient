"""Hypothesis data strategies shared across the test suite."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import hypothesis.strategies as st
import pandas as pd


def _to_int_strategy(
    value: int | st.SearchStrategy[int],
) -> st.SearchStrategy[int]:
    if isinstance(value, int):
        return st.just(value)
    return value


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def evaluation_data_strategy(
    n_rows: int | st.SearchStrategy[int] = st.integers(min_value=2, max_value=200),
    n_encounters: int | st.SearchStrategy[int] = st.integers(min_value=1, max_value=50),
) -> st.SearchStrategy[pd.DataFrame]:
    """Generate a valid evaluation input table with correlated labels/probabilities."""

    @st.composite
    def _build(draw: st.DrawFn) -> pd.DataFrame:
        row_count = draw(_to_int_strategy(n_rows))
        encounter_count = min(row_count, draw(_to_int_strategy(n_encounters)))

        encounter_ids = draw(
            st.lists(
                st.text(
                    alphabet="ABCDEFGHJKLMNPQRSTUVWXYZ0123456789",
                    min_size=1,
                    max_size=8,
                ),
                min_size=encounter_count,
                max_size=encounter_count,
                unique=True,
            )
        )
        sampled_encounter_ids = draw(
            st.lists(
                st.sampled_from(encounter_ids),
                min_size=row_count,
                max_size=row_count,
            )
        )
        raw_scores = draw(
            st.lists(
                st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
                min_size=row_count,
                max_size=row_count,
            )
        )
        noise = draw(
            st.lists(
                st.floats(
                    min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False
                ),
                min_size=row_count,
                max_size=row_count,
            )
        )
        probabilities = [round(_clamp_unit_interval(score), 6) for score in raw_scores]
        labels = [
            1 if score + eps >= 0.5 else 0
            for score, eps in zip(raw_scores, noise, strict=True)
        ]
        event_timestamps = draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=1_000_000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=row_count,
                max_size=row_count,
            )
        )

        return pd.DataFrame(
            {
                "encounter_id": sampled_encounter_ids,
                "event_timestamp": event_timestamps,
                "y_proba": probabilities,
                "y_label": labels,
            }
        )

    return _build()


def temporal_data_strategy(
    n_rows: int | st.SearchStrategy[int] = st.integers(min_value=3, max_value=200),
    n_encounters: int | st.SearchStrategy[int] = st.integers(min_value=1, max_value=50),
) -> st.SearchStrategy[pd.DataFrame]:
    """Generate evaluation-like data with datetime event timestamps."""

    @st.composite
    def _build(draw: st.DrawFn) -> pd.DataFrame:
        base_df = draw(
            evaluation_data_strategy(
                n_rows=_to_int_strategy(n_rows),
                n_encounters=_to_int_strategy(n_encounters),
            )
        )
        datetimes = draw(
            st.lists(
                st.datetimes(
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime(2030, 12, 31),
                    timezones=st.none(),
                ),
                min_size=len(base_df),
                max_size=len(base_df),
            )
        )
        return base_df.assign(event_timestamp=pd.to_datetime(datetimes))

    return _build()


def encounter_grouped_strategy(
    n_encounters: int | st.SearchStrategy[int] = st.integers(min_value=1, max_value=20),
    events_per_encounter: int | st.SearchStrategy[int] = st.integers(
        min_value=2, max_value=10
    ),
) -> st.SearchStrategy[pd.DataFrame]:
    """Generate grouped encounter data with multiple events per encounter."""

    @st.composite
    def _build(draw: st.DrawFn) -> pd.DataFrame:
        encounter_count = draw(_to_int_strategy(n_encounters))
        events_per_group = draw(_to_int_strategy(events_per_encounter))
        total_rows = encounter_count * events_per_group

        encounter_ids = [f"enc_{idx:04d}" for idx in range(encounter_count)]
        repeated_encounter_ids = [
            encounter_id
            for encounter_id in encounter_ids
            for _ in range(events_per_group)
        ]
        base_time = draw(
            st.datetimes(
                min_value=datetime(2021, 1, 1),
                max_value=datetime(2030, 12, 31),
                timezones=st.none(),
            )
        )
        event_timestamps = [
            pd.Timestamp(base_time + timedelta(hours=offset))
            for offset in range(total_rows)
        ]

        probabilities = draw(
            st.lists(
                st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
                min_size=total_rows,
                max_size=total_rows,
            )
        )
        labels = [1 if probability >= 0.5 else 0 for probability in probabilities]

        return pd.DataFrame(
            {
                "encounter_id": repeated_encounter_ids,
                "event_timestamp": event_timestamps,
                "y_proba": probabilities,
                "y_label": labels,
            }
        )

    return _build()


def io_sample_strategy(
    n_rows: int | st.SearchStrategy[int] = st.integers(min_value=2, max_value=300),
) -> st.SearchStrategy[pd.DataFrame]:
    """Generate realistic I/O sample data compatible with load_evaluation_data tests."""

    @st.composite
    def _build(draw: st.DrawFn) -> pd.DataFrame:
        row_count = draw(_to_int_strategy(n_rows))
        encounter_count = max(1, min(row_count, row_count // 2))
        encounter_ids = [f"enc_{idx:03d}" for idx in range(encounter_count)]

        sampled_encounter_ids = draw(
            st.lists(
                st.sampled_from(encounter_ids),
                min_size=row_count,
                max_size=row_count,
            )
        )
        probabilities = draw(
            st.lists(
                st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
                min_size=row_count,
                max_size=row_count,
            )
        )
        labels = [1 if probability >= 0.5 else 0 for probability in probabilities]
        timestamps = draw(
            st.lists(
                st.datetimes(
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime(2030, 12, 31),
                    timezones=st.none(),
                ),
                min_size=row_count,
                max_size=row_count,
            )
        )

        return pd.DataFrame(
            {
                "encounter_id": sampled_encounter_ids,
                "event_timestamp": pd.to_datetime(timestamps),
                "prediction_probability": probabilities,
                "true_label": labels,
                "model_identifier": ["model-a"] * row_count,
                "task_identifier": ["task-a"] * row_count,
            }
        )

    return _build()


def evaluation_results_strategy(
    n_thresholds: int | st.SearchStrategy[int] = st.integers(min_value=1, max_value=15),
) -> st.SearchStrategy[pd.DataFrame]:
    """Generate evaluation result rows aligned with the output schema."""

    @st.composite
    def _build(draw: st.DrawFn) -> pd.DataFrame:
        threshold_count = draw(_to_int_strategy(n_thresholds))
        thresholds = draw(
            st.lists(
                st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
                min_size=threshold_count,
                max_size=threshold_count,
                unique=True,
            )
        )
        thresholds = sorted(thresholds)

        rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            sample_size = draw(st.integers(min_value=10, max_value=500))
            label_count = draw(st.integers(min_value=0, max_value=sample_size))
            tp = draw(st.integers(min_value=0, max_value=label_count))
            fn = label_count - tp
            negatives = sample_size - label_count
            fp = draw(st.integers(min_value=0, max_value=negatives))
            tn = negatives - fp

            ppv = _ratio(tp, tp + fp)
            sensitivity = _ratio(tp, tp + fn)
            specificity = _ratio(tn, tn + fp)
            npv = _ratio(tn, tn + fn)
            accuracy = _ratio(tp + tn, sample_size)
            f1_score = None
            if ppv is not None and sensitivity is not None and (ppv + sensitivity) > 0:
                f1_score = 2.0 * ppv * sensitivity / (ppv + sensitivity)

            auroc = draw(
                st.one_of(
                    st.none(),
                    st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                )
            )
            auprc = draw(
                st.one_of(
                    st.none(),
                    st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                )
            )

            def build_ci(value: float | None) -> tuple[float | None, float | None]:
                if value is None:
                    return None, None
                width = draw(
                    st.floats(
                        min_value=0.0,
                        max_value=0.2,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
                return _clamp_unit_interval(value - width), _clamp_unit_interval(
                    value + width
                )

            auroc_l, auroc_u = build_ci(auroc)
            auprc_l, auprc_u = build_ci(auprc)
            ppv_l, ppv_u = build_ci(ppv)
            sensitivity_l, sensitivity_u = build_ci(sensitivity)
            specificity_l, specificity_u = build_ci(specificity)
            npv_l, npv_u = build_ci(npv)
            accuracy_l, accuracy_u = build_ci(accuracy)
            f1_l, f1_u = build_ci(f1_score)

            rows.append(
                {
                    "modelid": "model-a",
                    "filter_desc": "all",
                    "threshold": threshold,
                    "AUROC": auroc,
                    "AUROC_Lower_CI": auroc_l,
                    "AUROC_Upper_CI": auroc_u,
                    "AUPRC": auprc,
                    "AUPRC_Lower_CI": auprc_l,
                    "AUPRC_Upper_CI": auprc_u,
                    "Prevalence": _ratio(label_count, sample_size),
                    "Sample_Size": sample_size,
                    "Label_Count": label_count,
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                    "PPV": ppv,
                    "PPV_Lower_CI": ppv_l,
                    "PPV_Upper_CI": ppv_u,
                    "Sensitivity": sensitivity,
                    "Sensitivity_Lower_CI": sensitivity_l,
                    "Sensitivity_Upper_CI": sensitivity_u,
                    "Specificity": specificity,
                    "Specificity_Lower_CI": specificity_l,
                    "Specificity_Upper_CI": specificity_u,
                    "NPV": npv,
                    "NPV_Lower_CI": npv_l,
                    "NPV_Upper_CI": npv_u,
                    "Accuracy": accuracy,
                    "Accuracy_Lower_CI": accuracy_l,
                    "Accuracy_Upper_CI": accuracy_u,
                    "F1_Score": f1_score,
                    "F1_Score_Lower_CI": f1_l,
                    "F1_Score_Upper_CI": f1_u,
                }
            )
        return pd.DataFrame(rows)

    return _build()
