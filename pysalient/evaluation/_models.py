"""
Pydantic models for evaluation configuration.
"""

import warnings
from enum import Enum

import numpy as np
import pyarrow as pa
from pydantic import Field, field_validator, model_validator

from pysalient._shared_models import BaseConfig


class TimeUnit(str, Enum):
    SECOND = "second"
    SECONDS = "seconds"
    SEC = "sec"
    SECS = "secs"
    S = "s"
    MINUTE = "minute"
    MINUTES = "minutes"
    MIN = "min"
    MINS = "mins"
    M = "m"
    HOUR = "hour"
    HOURS = "hours"
    HR = "hr"
    HRS = "hrs"
    H = "h"
    DAY = "day"
    DAYS = "days"
    D = "d"
    WEEK = "week"
    WEEKS = "weeks"
    W = "w"


class ThresholdCIMethod(str, Enum):
    BOOTSTRAP = "bootstrap"
    NORMAL = "normal"
    WILSON = "wilson"
    AGRESTI_COULL = "agresti-coull"


class ThresholdConfig(BaseConfig):
    values: list[float] | tuple[float, ...] | tuple[float, float, float]
    force_threshold_zero: bool = True

    @field_validator("values")
    @classmethod
    def validate_thresholds_not_empty(
        cls, values: list[float] | tuple[float, ...] | tuple[float, float, float]
    ) -> list[float] | tuple[float, ...] | tuple[float, float, float]:
        if len(values) == 0:
            raise ValueError("thresholds must contain at least one value.")
        return values


class TimeToEventConfig(BaseConfig):
    event_columns: dict[str, str]
    aggregation_func: str = "median"
    time_unit: TimeUnit = TimeUnit.HOUR
    fillna: float | None = None

    @field_validator("event_columns")
    @classmethod
    def validate_event_columns(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("event_columns cannot be empty.")
        for key, column in value.items():
            if not key or not column:
                raise ValueError("event_columns keys and values must be non-empty.")
        return value

    @field_validator("aggregation_func")
    @classmethod
    def validate_aggregation_func(cls, value: str) -> str:
        if not hasattr(np, value) or not callable(getattr(np, value)):
            raise ValueError(
                f"aggregation_func '{value}' is not a valid NumPy aggregation function."
            )
        return value


class ConfidenceIntervalConfig(BaseConfig):
    calculate_au_ci: bool = False
    calculate_threshold_ci: bool = False
    threshold_ci_method: ThresholdCIMethod = ThresholdCIMethod.BOOTSTRAP
    alpha: float = 0.05
    bootstrap_rounds: int = 1000
    bootstrap_seed: object | None = None

    @model_validator(mode="after")
    def validate_bootstrap_rounds_when_bootstrap_used(self) -> "ConfidenceIntervalConfig":
        supported_threshold_ci_methods = {
            ThresholdCIMethod.BOOTSTRAP,
            ThresholdCIMethod.NORMAL,
            ThresholdCIMethod.WILSON,
            ThresholdCIMethod.AGRESTI_COULL,
        }
        if self.threshold_ci_method not in supported_threshold_ci_methods:
            raise ValueError(
                "threshold_ci_method must be one of: bootstrap, normal, wilson, agresti-coull."
            )

        ci_enabled = self.calculate_au_ci or self.calculate_threshold_ci
        if ci_enabled and not (0 < self.alpha < 1):
            raise ValueError("alpha must be between 0 and 1 when CI is enabled.")

        if self.calculate_au_ci and self.bootstrap_seed is not None and not isinstance(
            self.bootstrap_seed, int
        ):
            raise ValueError(
                "bootstrap_seed must be an integer or None when AU bootstrap CI is enabled."
            )

        bootstrap_required = self.calculate_au_ci or (
            self.calculate_threshold_ci
            and self.threshold_ci_method == ThresholdCIMethod.BOOTSTRAP
        )
        if bootstrap_required and self.bootstrap_rounds < 100:
            raise ValueError(
                "bootstrap_rounds must be >= 100 when bootstrap CI is enabled."
            )
        if bootstrap_required and self.bootstrap_rounds < 500:
            warnings.warn(
                "bootstrap_rounds < 500 may lead to less stable confidence intervals; "
                "consider >= 1000 for production runs.",
                UserWarning,
            )
        return self


class EvaluationConfig(BaseConfig):
    data: pa.Table
    modelid: str = Field(min_length=1)
    filter_desc: str = Field(min_length=1)
    thresholds: ThresholdConfig
    time_to_event: TimeToEventConfig | None = None
    confidence_intervals: ConfidenceIntervalConfig = Field(
        default_factory=ConfidenceIntervalConfig
    )
    decimal_places: int | None = Field(default=None, ge=0)
    verbosity: int = 0
    force_eval: bool = False
