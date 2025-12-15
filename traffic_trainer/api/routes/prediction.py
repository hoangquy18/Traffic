"""Prediction endpoints."""

from fastapi import APIRouter, HTTPException

from traffic_trainer.api.models import (
    PredictionRequest,
    PredictionResponse,
    SegmentPrediction,
    HorizonPrediction,
)
from traffic_trainer.api.services import model_service

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict traffic LOS for specified segments.

    - segment_ids: List of segment IDs to predict
    - target_time: Target datetime (default: now)
    - return_probabilities: Include full probability distribution
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")

    invalid_ids = [
        sid for sid in request.segment_ids if sid not in model_service.segment_to_idx
    ]
    if invalid_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Segments not found: {invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}",
        )

    try:
        results = model_service.predict(
            segment_ids=request.segment_ids,
            target_time=request.target_time,
            return_probabilities=request.return_probabilities,
        )

        predictions = []
        for seg_id, data in results.items():
            horizon_preds = [
                HorizonPrediction(
                    horizon=p["horizon"],
                    los_class=p["los_class"],
                    los_index=p["los_index"],
                    confidence=p["confidence"],
                    probabilities=p.get("probabilities"),
                )
                for p in data["predictions"]
            ]
            predictions.append(
                SegmentPrediction(
                    segment_id=seg_id,
                    street_name=data.get("street_name"),
                    street_type=data.get("street_type"),
                    predictions=horizon_preds,
                )
            )

        return PredictionResponse(
            request_time=request.target_time,
            predictions=predictions,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{segment_id}")
async def predict_single(segment_id: int):
    """Quick prediction for a single segment."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")

    if segment_id not in model_service.segment_to_idx:
        raise HTTPException(status_code=404, detail="Segment not found")

    results = model_service.predict(
        segment_ids=[segment_id], return_probabilities=False
    )

    if segment_id not in results:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return results[segment_id]
