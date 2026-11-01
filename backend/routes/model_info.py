"""Model metadata and evaluation endpoints."""

from fastapi import APIRouter, HTTPException, Request

from ..config import FEATURE_COLUMNS
from ..schemas.prediction import ModelInfoResponse

router = APIRouter()


def _service(request: Request):
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return service


@router.get("/api/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    service = _service(request)
    return ModelInfoResponse(
        model_name=service.model_name,
        algorithm=service.algorithm,
        target_transformation=service.target_mode,
        features=FEATURE_COLUMNS,
        metrics=service.loaded.metrics,
    )


@router.get("/api/metrics")
def metrics(request: Request) -> dict[str, object]:
    return _service(request).loaded.metrics

