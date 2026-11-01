"""Tests for the isolated Dataset B experimental inference bundle."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.dataset_b_experimental import DatasetBPredictionRequest
from backend.services.dataset_b_experimental_predictor import (
    DatasetBExperimentalPredictor,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL = ROOT / "artifacts" / "dataset_b_improvements"
PAYLOAD = {
    "brand": "Toyota",
    "model_family": "Innova",
    "manufacturing_year": 2018,
    "kms_driven": 45000,
    "fuel_type": "Diesel",
    "city": "Delhi",
}


def _predictor() -> DatasetBExperimentalPredictor:
    return DatasetBExperimentalPredictor(
        EXPERIMENTAL / "selected_dataset_b_xgboost_model.json",
        EXPERIMENTAL / "selected_dataset_b_preprocessor.joblib",
        EXPERIMENTAL / "dataset_b_experimental_config.json",
    )


def test_experimental_bundle_loads_and_predicts() -> None:
    result = _predictor().predict(DatasetBPredictionRequest(**PAYLOAD))
    assert np.isfinite(result["predicted_price_lakhs"])
    assert result["predicted_price_lakhs"] > 0
    assert 0 <= result["lower_bound_lakhs"] <= result["predicted_price_lakhs"]
    assert result["upper_bound_lakhs"] >= result["predicted_price_lakhs"]
    assert result["experimental"] is True


def test_unknown_categories_are_safe_and_warn_for_unknown_family() -> None:
    result = _predictor().predict(
        DatasetBPredictionRequest(
            **{**PAYLOAD, "brand": "Unknown Brand", "model_family": "Unknown Family"}
        )
    )
    assert result["predicted_price_lakhs"] > 0
    assert result["warning"] is not None


def test_invalid_dataset_b_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        DatasetBPredictionRequest(**{**PAYLOAD, "manufacturing_year": 1979})
    with pytest.raises(ValueError):
        DatasetBPredictionRequest(**{**PAYLOAD, "kms_driven": -1})
    with pytest.raises(ValueError):
        DatasetBPredictionRequest(**{**PAYLOAD, "fuel_type": ""})
    with pytest.raises(ValueError):
        DatasetBPredictionRequest(**{key: value for key, value in PAYLOAD.items()
                                    if key != "city"})


def test_missing_experimental_artifact_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DatasetBExperimentalPredictor(
            tmp_path / "model.json",
            tmp_path / "preprocessor.joblib",
            tmp_path / "config.json",
        )


def test_experimental_endpoint_is_disabled_by_default() -> None:
    with TestClient(app) as client:
        response = client.post("/api/experimental/dataset-b/predict", json=PAYLOAD)
    assert response.status_code == 503
    assert "disabled" in response.json()["detail"].lower()


def test_experimental_endpoint_predicts_when_state_is_enabled() -> None:
    with TestClient(app) as client:
        app.state.dataset_b_experimental_predictor = _predictor()
        response = client.post("/api/experimental/dataset-b/predict", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["experimental"] is True
    assert body["currency"] == "INR lakhs"
    assert body["upper_bound_lakhs"] >= body["predicted_price_lakhs"]
