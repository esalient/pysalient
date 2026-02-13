"""
Pytest configuration for pysalient tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add the project root to Python path so tests can import pysalient without installation
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from hypothesis import HealthCheck, settings

    try:
        settings.register_profile(
            "dev",
            max_examples=50,
            deadline=500,
            suppress_health_check=[HealthCheck.too_slow],
        )
    except ValueError:
        pass

    try:
        settings.register_profile(
            "ci",
            max_examples=200,
            deadline=2000,
            suppress_health_check=[HealthCheck.too_slow],
        )
    except ValueError:
        pass

    settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
except ImportError:
    settings = None


@pytest.fixture
def valid_evaluation_df():
    """Return one schema-valid evaluation DataFrame from shared strategies."""
    pytest.importorskip("hypothesis")

    from tests.schemas import evaluation_input_schema
    from tests.strategies import evaluation_data_strategy

    return evaluation_input_schema.validate(evaluation_data_strategy().example())
