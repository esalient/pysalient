"""Pandera schemas shared across the test suite."""

from __future__ import annotations

import pandera as pa

_UNIT_INTERVAL = pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)
_NON_NEGATIVE = pa.Check.greater_than_or_equal_to(0)


def _probability_column(*, nullable: bool = False) -> pa.Column:
    return pa.Column(float, checks=_UNIT_INTERVAL, nullable=nullable, coerce=True)


def _binary_int_column(*, nullable: bool = False) -> pa.Column:
    return pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=nullable, coerce=True)


evaluation_input_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False, coerce=True),
        "event_timestamp": pa.Column(
            float,
            checks=pa.Check.greater_than_or_equal_to(0.0),
            nullable=False,
            coerce=True,
        ),
        "y_proba": _probability_column(nullable=False),
        "y_label": _binary_int_column(nullable=False),
    },
    strict=False,
    coerce=True,
)


evaluation_results_schema = pa.DataFrameSchema(
    {
        "modelid": pa.Column(str, nullable=True, coerce=True),
        "filter_desc": pa.Column(str, nullable=True, coerce=True),
        "threshold": _probability_column(nullable=False),
        "AUROC": _probability_column(nullable=True),
        "AUROC_Lower_CI": _probability_column(nullable=True),
        "AUROC_Upper_CI": _probability_column(nullable=True),
        "AUPRC": _probability_column(nullable=True),
        "AUPRC_Lower_CI": _probability_column(nullable=True),
        "AUPRC_Upper_CI": _probability_column(nullable=True),
        "Prevalence": _probability_column(nullable=True),
        "Sample_Size": pa.Column(
            int, checks=_NON_NEGATIVE, nullable=False, coerce=True
        ),
        "Label_Count": pa.Column(
            int, checks=_NON_NEGATIVE, nullable=False, coerce=True
        ),
        "TP": pa.Column(int, checks=_NON_NEGATIVE, nullable=False, coerce=True),
        "TN": pa.Column(int, checks=_NON_NEGATIVE, nullable=False, coerce=True),
        "FP": pa.Column(int, checks=_NON_NEGATIVE, nullable=False, coerce=True),
        "FN": pa.Column(int, checks=_NON_NEGATIVE, nullable=False, coerce=True),
        "PPV": _probability_column(nullable=True),
        "PPV_Lower_CI": _probability_column(nullable=True),
        "PPV_Upper_CI": _probability_column(nullable=True),
        "Sensitivity": _probability_column(nullable=True),
        "Sensitivity_Lower_CI": _probability_column(nullable=True),
        "Sensitivity_Upper_CI": _probability_column(nullable=True),
        "Specificity": _probability_column(nullable=True),
        "Specificity_Lower_CI": _probability_column(nullable=True),
        "Specificity_Upper_CI": _probability_column(nullable=True),
        "NPV": _probability_column(nullable=True),
        "NPV_Lower_CI": _probability_column(nullable=True),
        "NPV_Upper_CI": _probability_column(nullable=True),
        "Accuracy": _probability_column(nullable=True),
        "Accuracy_Lower_CI": _probability_column(nullable=True),
        "Accuracy_Upper_CI": _probability_column(nullable=True),
        "F1_Score": _probability_column(nullable=True),
        "F1_Score_Lower_CI": _probability_column(nullable=True),
        "F1_Score_Upper_CI": _probability_column(nullable=True),
    },
    strict=False,
    coerce=True,
)


io_csv_input_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False, coerce=True),
        "prediction_probability": _probability_column(nullable=False),
        "true_label": _binary_int_column(nullable=False),
    },
    strict=False,
    coerce=True,
)


io_parquet_input_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False, coerce=True),
        "event_timestamp": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "prediction_probability": _probability_column(nullable=False),
        "true_label": _binary_int_column(nullable=False),
    },
    strict=False,
    coerce=True,
)


time_to_event_input_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False, coerce=True),
        "event_timestamp": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "y_proba": _probability_column(nullable=False),
        "y_label": _binary_int_column(nullable=False),
        "culture_event": _binary_int_column(nullable=False),
        "suspected_infection": _binary_int_column(nullable=False),
    },
    strict=False,
    coerce=True,
)


visualisation_input_schema = pa.DataFrameSchema(
    {
        "threshold": _probability_column(nullable=False),
        "AUROC": _probability_column(
            nullable=True,
        ),
        "AUPRC": _probability_column(nullable=True),
        "Prevalence": _probability_column(nullable=True),
        "PPV": _probability_column(nullable=True),
        "Sensitivity": _probability_column(nullable=True),
        "Specificity": _probability_column(nullable=True),
        "NPV": _probability_column(nullable=True),
        "Accuracy": _probability_column(nullable=True),
        "F1_Score": _probability_column(nullable=True),
        "Sample Size": pa.Column(
            int,
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=False,
            coerce=True,
        ),
    },
    strict=False,
    coerce=True,
)
