"""
Shared Pydantic model base classes and enums.
"""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base class for Pydantic config models used in pySALIENT."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
