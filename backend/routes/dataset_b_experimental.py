"""Default-disabled Dataset B experimental prediction endpoint."""

from fastapi import APIRouter, HTTPException, Request

from ..schemas.dataset_b_experimental import (
    DatasetBPredictionRequest,
    DatasetBPredictionResponse,
)

router = APIRouter()


@router.post(
    "/api/experimental/dataset-b/predict",
    response_model=DatasetBPredictionResponse,
)
def predict_dataset_b(
    request_data: DatasetBPredictionRequest,
    request: Request,
) -> DatasetBPredictionResponse:
    predictor = getattr(request.app.state, "dataset_b_experimental_predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset B experimental inference is disabled.",
        )
    try:
        return DatasetBPredictionResponse(**predictor.predict(request_data))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Experimental prediction failed") from exc
