"""Health and metadata endpoint tests."""

from fastapi.testclient import TestClient

from backend.config import (
    CALIBRATED_MODEL_PATH,
    CALIBRATED_PREPROCESSOR_PATH,
    METRICS_PATH,
    PREPROCESSOR_PATH,
    XGBOOST_MODEL_PATH,
)
from backend.main import app


def test_health_and_model_info() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        info = client.get("/api/model-info")
    assert health.status_code == 200
    expected_loaded = (
        METRICS_PATH.is_file()
        and (
            (
                CALIBRATED_MODEL_PATH.is_file()
                and CALIBRATED_PREPROCESSOR_PATH.is_file()
            )
            or (XGBOOST_MODEL_PATH.is_file() and PREPROCESSOR_PATH.is_file())
        )
    )
    assert health.json() == {"status": "ok", "model_loaded": expected_loaded}
    if expected_loaded:
        assert info.status_code == 200
        assert info.json()["algorithm"] == "XGBRegressor"
        assert info.json()["target_transformation"] == "log1p"
    else:
        assert info.status_code == 503
