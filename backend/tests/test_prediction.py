"""Prediction behavior and offline consistency tests."""

from pathlib import Path

import joblib
import numpy as np
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from backend.config import FEATURE_COLUMNS
from backend.main import app
from backend.schemas.prediction import CarFeatures
from backend.services.prediction_service import PredictionService
from backend.services.model_loader import load_model_artifacts

ROOT = Path(__file__).resolve().parents[2]
LEGACY_ARTIFACTS_AVAILABLE = all(
    path.is_file()
    for path in (
        ROOT / "artifacts" / "xgboost_car_price_model.json",
        ROOT / "artifacts" / "xgboost_preprocessor.joblib",
        ROOT / "artifacts" / "xgboost_metrics.json",
    )
)
pytestmark = pytest.mark.skipif(
    not LEGACY_ARTIFACTS_AVAILABLE,
    reason="Legacy Dataset A artifacts are optional and excluded from clean clones.",
)
PAYLOAD = {
    "car_name": "2018 BMW X1 sDrive 20d xLine",
    "registration_year": "Sep-18",
    "manufacturing_year": 2018,
    "fuel_type": "Petrol",
    "seats": 5,
    "kms_driven": 24000,
    "ownership": "First Owner",
    "transmission": "Automatic",
    "mileage": 20.68,
    "engine": 1995,
    "insurance_validity": "Comprehensive",
}


def test_valid_unknown_and_multiple_predictions() -> None:
    with TestClient(app) as client:
        first = client.post("/api/predict", json=PAYLOAD)
        unknown = client.post(
            "/api/predict",
            json={**PAYLOAD, "fuel_type": "Hydrogen", "car_name": "Unknown model"},
        )
        multiple = [client.post("/api/predict", json=PAYLOAD) for _ in range(2)]
    assert first.status_code == 200
    assert first.json()["predicted_price_lakhs"] >= 0
    interval = first.json()["prediction_range_lakhs"]
    assert interval is not None
    assert interval["lower"] >= 0
    assert interval["upper"] >= first.json()["predicted_price_lakhs"]
    assert first.json()["calibration_method"] == "split_conformal_absolute_residual"
    assert unknown.status_code == 200
    assert all(response.status_code == 200 for response in multiple)


def test_missing_optional_values_and_invalid_input() -> None:
    with TestClient(app) as client:
        missing = client.post(
            "/api/predict",
            json={key: value for key, value in PAYLOAD.items()
                  if key not in {"mileage", "engine", "ownership"}},
        )
        invalid = client.post("/api/predict", json={**PAYLOAD, "seats": 0})
    assert missing.status_code == 200
    assert invalid.status_code == 422


def test_supported_year_boundaries_and_order_validation() -> None:
    with TestClient(app) as client:
        manufacturing_2023 = client.post(
            "/api/predict",
            json={**PAYLOAD, "manufacturing_year": 2023, "registration_year": 2023},
        )
        registration_2023 = client.post(
            "/api/predict",
            json={**PAYLOAD, "registration_year": 2023},
        )
        manufacturing_2024 = client.post(
            "/api/predict",
            json={**PAYLOAD, "manufacturing_year": 2024, "registration_year": 2024},
        )
        registration_2026 = client.post(
            "/api/predict",
            json={**PAYLOAD, "registration_year": 2026},
        )
        registration_before_manufacturing = client.post(
            "/api/predict",
            json={**PAYLOAD, "manufacturing_year": 2020, "registration_year": 2019},
        )

    assert manufacturing_2023.status_code == 200
    assert registration_2023.status_code == 200
    assert manufacturing_2024.status_code == 422
    assert registration_2026.status_code == 422
    assert registration_before_manufacturing.status_code == 422


def test_api_matches_direct_saved_model_prediction() -> None:
    loaded = load_model_artifacts(
        ROOT / "artifacts" / "xgboost_car_price_model.json",
        ROOT / "artifacts" / "xgboost_preprocessor.joblib",
        ROOT / "artifacts" / "xgboost_metrics.json",
    )
    service = PredictionService(loaded)
    api_features = CarFeatures(**PAYLOAD)
    api_result = service.predict(api_features)
    frame, _ = service._frame(api_features)
    direct_raw = loaded.model.predict(
        loaded.preprocessor.transform(frame[FEATURE_COLUMNS])
    )[0]
    direct_price = float(np.expm1(direct_raw))
    assert np.isclose(api_result["predicted_price_lakhs"], direct_price, rtol=1e-6)


def test_original_model_without_calibration_returns_no_interval() -> None:
    loaded = load_model_artifacts(
        ROOT / "artifacts" / "xgboost_car_price_model.json",
        ROOT / "artifacts" / "xgboost_preprocessor.joblib",
        ROOT / "artifacts" / "xgboost_metrics.json",
    )
    result = PredictionService(loaded).predict(CarFeatures(**PAYLOAD))
    assert result["prediction_range_lakhs"] is None
    assert result["calibration_method"] is None


def test_calibration_artifact_is_loaded() -> None:
    with TestClient(app) as client:
        response = client.post("/api/predict", json=PAYLOAD)
    body = response.json()
    assert response.status_code == 200
    assert body["prediction_range_lakhs"]["lower"] >= 0
    assert body["prediction_range_lakhs"]["upper"] > body["prediction_range_lakhs"]["lower"]
    assert body["calibration_method"] == "split_conformal_absolute_residual"
