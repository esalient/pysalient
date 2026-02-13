"""
Pydantic models for IO configuration.
"""

from typing import Any

import pandas as pd
from pydantic import Field

from pysalient._shared_models import BaseConfig


class ColumnConfig(BaseConfig):
    y_proba_col: str = Field(min_length=1)
    y_label_col: str = Field(min_length=1)
    aggregation_cols: str | list[str]
    timeseries_col: str = Field(min_length=1)
    model_col: str | None = None
    task_col: str | None = None


class LoadConfig(BaseConfig):
    source: str | pd.DataFrame
    columns: ColumnConfig
    assign_task_name: str | None = None
    assign_model_name: str | None = None
    source_type: str | None = None
    read_options: dict[str, Any] = Field(default_factory=dict)
    perform_aggregation: bool = False
    proba_agg_func: str = "mean"
    label_agg_func: str = "max"
