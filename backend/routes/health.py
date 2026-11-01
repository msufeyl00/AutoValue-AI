"""Health endpoint."""

from fastapi import APIRouter, Request

from ..schemas.prediction import HealthResponse

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=getattr(request.app.state, "prediction_service", None) is not None,
    )

