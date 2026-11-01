"""Startup-only loading of persisted XGBoost artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import xgboost as xgb


@dataclass(frozen=True)
class LoadedModel:
    """All persisted objects required for inference."""

    model: xgb.XGBRegressor
    preprocessor: Any
    metrics: dict[str, Any]
    calibration: dict[str, Any] | None = None


def load_model_artifacts(
    model_path: Path,
    preprocessor_path: Path,
    metrics_path: Path,
    calibration_path: Path | None = None,
) -> LoadedModel:
    """Load model, preprocessor, and metadata once during application startup."""
    for path in (model_path, preprocessor_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError("Required model artifact is unavailable.")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    model.set_params(n_jobs=1)
    preprocessor = joblib.load(preprocessor_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    calibration = None
    if calibration_path is not None and calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    return LoadedModel(
        model=model,
        preprocessor=preprocessor,
        metrics=metrics,
        calibration=calibration,
    )
