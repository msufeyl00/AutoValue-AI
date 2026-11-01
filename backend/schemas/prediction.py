"""Validated API request and response models."""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import SUPPORTED_YEAR_MAX, SUPPORTED_YEAR_MIN

OptionalText = Annotated[str | None, Field(max_length=200)]
YEAR_ERROR = "Year must be between the supported dataset years and 2023."


def _parse_registration_year(value: str | int | float) -> int:
    text = str(value).strip()
    match = re.fullmatch(r"[A-Za-z]{3}-(\d{2})", text)
    if match:
        year = int(match.group(1))
        return 2000 + year if year <= 30 else 1900 + year
    try:
        year = int(float(text))
    except ValueError as exc:
        raise ValueError("registration_year must be an integer year or a value such as Jul-17") from exc
    return year + 2000 if 0 <= year <= 30 else year


class CarFeatures(BaseModel):
    """The exact eleven features used by the saved XGBoost preprocessor."""

    model_config = ConfigDict(extra="forbid")

    car_name: OptionalText = None
    registration_year: str | int | float
    manufacturing_year: Annotated[int, Field(ge=SUPPORTED_YEAR_MIN, le=SUPPORTED_YEAR_MAX)]
    fuel_type: OptionalText = None
    seats: Annotated[int, Field(ge=1, le=20)]
    kms_driven: Annotated[float, Field(ge=0, le=2_000_000)]
    ownership: OptionalText = None
    transmission: OptionalText = None
    insurance_validity: OptionalText = None
    mileage: Annotated[float | None, Field(ge=0, le=200)] = None
    engine: Annotated[float | None, Field(ge=0, le=20_000)] = None

    @field_validator(
        "car_name",
        "fuel_type",
        "ownership",
        "transmission",
        "insurance_validity",
        mode="before",
    )
    @classmethod
    def strip_optional_text(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string or null")
        return value.strip()

    @field_validator("manufacturing_year", mode="before")
    @classmethod
    def validate_manufacturing_year(cls, value: Any) -> Any:
        try:
            numeric_year = float(value)
            year = int(numeric_year)
        except (TypeError, ValueError) as exc:
            raise ValueError(YEAR_ERROR) from exc
        if numeric_year != year or not SUPPORTED_YEAR_MIN <= year <= SUPPORTED_YEAR_MAX:
            raise ValueError(YEAR_ERROR)
        return value

    @model_validator(mode="after")
    def validate_supported_years(self) -> "CarFeatures":
        registration_year = _parse_registration_year(self.registration_year)
        if not SUPPORTED_YEAR_MIN <= registration_year <= SUPPORTED_YEAR_MAX:
            raise ValueError(YEAR_ERROR)
        if registration_year < self.manufacturing_year:
            raise ValueError("registration_year cannot be earlier than manufacturing_year")
        return self


class PredictionResponse(BaseModel):
    predicted_price_lakhs: float
    predicted_price_inr: int
    formatted_inr: str
    vehicle: dict[str, Any]
    model_name: str
    prediction_range_lakhs: dict[str, float] | None = None
    range_note: str | None = None
    calibration_method: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    algorithm: str
    target_transformation: str
    features: list[str]
    metrics: dict[str, Any]
