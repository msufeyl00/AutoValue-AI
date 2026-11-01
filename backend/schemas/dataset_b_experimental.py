"""Request and response schemas for the isolated Dataset B experiment."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

DATASET_B_YEAR_MIN = 1980
DATASET_B_YEAR_MAX = 2025


class DatasetBPredictionRequest(BaseModel):
    """Fields available in the approved Dataset B source."""

    model_config = ConfigDict(extra="forbid")

    brand: Annotated[str, Field(min_length=1, max_length=100)]
    model_family: Annotated[str, Field(min_length=1, max_length=100)]
    manufacturing_year: Annotated[int, Field(ge=DATASET_B_YEAR_MIN, le=DATASET_B_YEAR_MAX)]
    kms_driven: Annotated[float, Field(ge=0, le=2_000_000)]
    fuel_type: Annotated[str, Field(min_length=1, max_length=100)]
    city: Annotated[str, Field(min_length=1, max_length=100)]

    @field_validator("brand", "model_family", "fuel_type", "city")
    @classmethod
    def strip_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("categorical values must be non-empty")
        return normalized


class DatasetBPredictionResponse(BaseModel):
    predicted_price_lakhs: float
    lower_bound_lakhs: float
    upper_bound_lakhs: float
    currency: str
    model_identifier: str
    experimental: bool
    warning: str | None = None
    input: dict[str, Any]
