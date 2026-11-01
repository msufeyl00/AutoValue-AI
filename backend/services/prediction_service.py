"""XGBoost inference and input normalization, independent of FastAPI."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..config import FEATURE_COLUMNS
from ..config import SUPPORTED_YEAR_MAX, SUPPORTED_YEAR_MIN
from ..schemas.prediction import CarFeatures
from .model_loader import LoadedModel


def parse_registration_year(value: str | int | float) -> int:
    """Parse numeric years and registration values such as ``Jul-17``."""
    text = str(value).strip()
    match = re.fullmatch(r"[A-Za-z]{3}-(\d{2})", text)
    if match:
        year = int(match.group(1))
        return 2000 + year if year <= 30 else 1900 + year
    try:
        year = int(float(text))
    except ValueError as exc:
        raise ValueError(
            "registration_year must be an integer year or a value such as Jul-17"
        ) from exc
    return year + 2000 if 0 <= year <= 30 else year


def _format_inr(value: int) -> str:
    return f"₹{value:,}"


class PredictionService:
    """Applies the persisted preprocessing pipeline and predicts in lakhs."""

    model_name = "xgboost_car_price_model"
    algorithm = "XGBRegressor"

    def __init__(self, loaded: LoadedModel):
        self.loaded = loaded
        feature_names = list(loaded.preprocessor.feature_names_in_)
        if feature_names != FEATURE_COLUMNS:
            raise RuntimeError("Saved preprocessor feature order does not match the API schema.")
        self.target_mode = str(loaded.metrics.get("target_mode", "raw"))
        self.calibration = loaded.calibration

    def _frame(self, features: CarFeatures) -> tuple[pd.DataFrame, dict[str, Any]]:
        values = features.model_dump()
        registration_year = parse_registration_year(values["registration_year"])
        if not SUPPORTED_YEAR_MIN <= registration_year <= SUPPORTED_YEAR_MAX:
            raise ValueError("Year must be between the supported dataset years and 2023.")
        if registration_year < values["manufacturing_year"]:
            raise ValueError("registration_year cannot precede manufacturing_year")
        values["registration_year"] = registration_year
        return pd.DataFrame([values], columns=FEATURE_COLUMNS), values

    def predict(self, features: CarFeatures) -> dict[str, Any]:
        frame, normalized = self._frame(features)
        transformed = self.loaded.preprocessor.transform(frame[FEATURE_COLUMNS])
        raw_prediction = float(self.loaded.model.predict(transformed)[0])
        prediction = float(np.expm1(raw_prediction) if self.target_mode == "log1p"
                           else raw_prediction)
        prediction = max(0.0, prediction)
        price_inr = int(round(prediction * 100_000))
        prediction_range = None
        calibration_method = None
        range_note = (
            "A calibrated prediction interval is not available. "
            "Actual resale prices may vary depending on vehicle condition and market demand."
        )
        if self.calibration is not None:
            quantile = float(self.calibration["calibration_quantile_lakhs"])
            prediction_range = {
                "lower": max(0.0, prediction - quantile),
                "upper": prediction + quantile,
            }
            calibration_method = str(self.calibration["calibration_method"])
            range_note = (
                "Prediction intervals are based on held-out calibration data. "
                "Conformal coverage is marginal and depends on calibration and "
                "deployment data being sufficiently representative. Actual resale "
                "prices may vary depending on vehicle condition and market demand."
            )
        return {
            "predicted_price_lakhs": prediction,
            "predicted_price_inr": price_inr,
            "formatted_inr": _format_inr(price_inr),
            "vehicle": normalized,
            "model_name": self.model_name,
            "prediction_range_lakhs": prediction_range,
            "calibration_method": calibration_method,
            "range_note": range_note,
        }
