"""Production-style FastAPI application for XGBoost inference."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    CALIBRATED_MODEL_PATH,
    CALIBRATED_PREPROCESSOR_PATH,
    CALIBRATION_PATH,
    DATASET_B_EXPERIMENTAL_CONFIG_PATH,
    DATASET_B_EXPERIMENTAL_MODEL_PATH,
    DATASET_B_EXPERIMENTAL_PREPROCESSOR_PATH,
    ENABLE_DATASET_B_EXPERIMENTAL,
    METRICS_PATH,
    PREPROCESSOR_PATH,
    XGBOOST_MODEL_PATH,
)
from .routes import health, model_info, prediction
from .routes import dataset_b_experimental
from .services.model_loader import load_model_artifacts
from .services.prediction_service import PredictionService
from .services.dataset_b_experimental_predictor import DatasetBExperimentalPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = CALIBRATED_MODEL_PATH if CALIBRATED_MODEL_PATH.is_file() else XGBOOST_MODEL_PATH
    preprocessor_path = (
        CALIBRATED_PREPROCESSOR_PATH
        if CALIBRATED_PREPROCESSOR_PATH.is_file()
        else PREPROCESSOR_PATH
    )
    app.state.prediction_service = None
    if model_path.is_file() and preprocessor_path.is_file() and METRICS_PATH.is_file():
        loaded = load_model_artifacts(
            model_path,
            preprocessor_path,
            METRICS_PATH,
            CALIBRATION_PATH,
        )
        app.state.prediction_service = PredictionService(loaded)
    app.state.dataset_b_experimental_predictor = None
    if ENABLE_DATASET_B_EXPERIMENTAL:
        app.state.dataset_b_experimental_predictor = DatasetBExperimentalPredictor(
            DATASET_B_EXPERIMENTAL_MODEL_PATH,
            DATASET_B_EXPERIMENTAL_PREPROCESSOR_PATH,
            DATASET_B_EXPERIMENTAL_CONFIG_PATH,
        )
    yield
    app.state.prediction_service = None
    app.state.dataset_b_experimental_predictor = None


app = FastAPI(
    title="Used Car Price Prediction API",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(prediction.router)
app.include_router(dataset_b_experimental.router)
