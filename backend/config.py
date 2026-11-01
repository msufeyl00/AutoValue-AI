"""Backend configuration and artifact locations."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
XGBOOST_MODEL_PATH = ARTIFACTS_DIR / "xgboost_car_price_model.json"
CALIBRATED_MODEL_PATH = ARTIFACTS_DIR / "xgboost_calibrated_car_price_model.json"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "xgboost_preprocessor.joblib"
CALIBRATED_PREPROCESSOR_PATH = ARTIFACTS_DIR / "xgboost_calibrated_preprocessor.joblib"
METRICS_PATH = ARTIFACTS_DIR / "xgboost_metrics.json"
CALIBRATION_PATH = ARTIFACTS_DIR / "xgboost_calibration.json"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"
DATASET_B_EXPERIMENTAL_DIR = ARTIFACTS_DIR / "dataset_b_improvements"
DATASET_B_EXPERIMENTAL_MODEL_PATH = (
    DATASET_B_EXPERIMENTAL_DIR / "selected_dataset_b_xgboost_model.json"
)
DATASET_B_EXPERIMENTAL_PREPROCESSOR_PATH = (
    DATASET_B_EXPERIMENTAL_DIR / "selected_dataset_b_preprocessor.joblib"
)
DATASET_B_EXPERIMENTAL_CONFIG_PATH = (
    DATASET_B_EXPERIMENTAL_DIR / "dataset_b_experimental_config.json"
)
ENABLE_DATASET_B_EXPERIMENTAL = (
    os.getenv("ENABLE_DATASET_B_EXPERIMENTAL", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
SUPPORTED_YEAR_MIN = 2007
SUPPORTED_YEAR_MAX = 2023

FEATURE_COLUMNS = [
    "car_name",
    "registration_year",
    "manufacturing_year",
    "fuel_type",
    "seats",
    "kms_driven",
    "ownership",
    "transmission",
    "mileage",
    "engine",
    "insurance_validity",
]

NUMERIC_COLUMNS = [
    "registration_year",
    "manufacturing_year",
    "seats",
    "kms_driven",
    "mileage",
    "engine",
]

CATEGORICAL_COLUMNS = [
    "fuel_type",
    "ownership",
    "transmission",
    "insurance_validity",
    "car_name",
]
