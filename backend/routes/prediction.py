"""Prediction endpoint."""

from fastapi import APIRouter, HTTPException, Request

from ..schemas.prediction import CarFeatures, PredictionResponse

router = APIRouter()


@router.post("/api/predict", response_model=PredictionResponse)
def predict(features: CarFeatures, request: Request) -> PredictionResponse:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        return PredictionResponse(**service.predict(features))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

