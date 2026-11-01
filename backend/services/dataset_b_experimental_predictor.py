"""Inference service for the isolated Dataset B experimental model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from ..schemas.dataset_b_experimental import DatasetBPredictionRequest


class DatasetBExperimentalPredictor:
    """Loads and serves only the selected Dataset B interaction model."""

    model_identifier = "dataset_b_interaction_features_xgboost_experimental"
    expected_features = [
        "manufacturing_year",
        "kms_driven",
        "brand",
        "model_family",
        "fuel_type",
        "city",
        "brand_model_family",
    ]

    def __init__(
        self,
        model_path: Path,
        preprocessor_path: Path,
        config_path: Path,
    ) -> None:
        for path in (model_path, preprocessor_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"Dataset B experimental artifact is unavailable: {path.name}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.model.set_params(n_jobs=1)
        self.preprocessor = joblib.load(preprocessor_path)
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        actual_features = list(self.preprocessor.feature_names_in_)
        if actual_features != self.expected_features:
            raise RuntimeError(
                "Dataset B experimental feature order does not match the saved preprocessor."
            )
        self.calibration = self.config["calibration"]
        self.known_model_families = set(
            str(value)
            for value in self.preprocessor.named_transformers_["categorical"]
            .named_steps["one_hot"]
            .categories_[1]
        )

    def _frame(self, request: DatasetBPredictionRequest) -> pd.DataFrame:
        values = request.model_dump()
        values["brand_model_family"] = (
            f"{values['brand'].lower()}__{values['model_family'].lower()}"
        )
        return pd.DataFrame([values], columns=self.expected_features)

    def predict(self, request: DatasetBPredictionRequest) -> dict[str, Any]:
        frame = self._frame(request)
        transformed = self.preprocessor.transform(frame[self.expected_features])
        raw_prediction = float(self.model.predict(transformed)[0])
        prediction = float(np.expm1(raw_prediction))
        if not np.isfinite(prediction) or prediction <= 0:
            raise RuntimeError("Dataset B experimental model returned an invalid price.")
        quantile = float(self.calibration["quantile_lakhs"])
        lower = max(0.0, prediction - quantile)
        upper = prediction + quantile
        warning = None
        if request.model_family.lower() not in self.known_model_families:
            warning = (
                "This model family was not observed by the experimental preprocessor; "
                "prediction uncertainty may be higher."
            )
        return {
            "predicted_price_lakhs": prediction,
            "lower_bound_lakhs": lower,
            "upper_bound_lakhs": upper,
            "currency": "INR lakhs",
            "model_identifier": self.model_identifier,
            "experimental": True,
            "warning": warning,
            "input": request.model_dump(),
        }
