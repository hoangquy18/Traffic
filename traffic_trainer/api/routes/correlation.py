"""Correlation endpoints."""

from fastapi import APIRouter, HTTPException, Query

from traffic_trainer.api.models import (
    CorrelationRequest,
    CorrelationResponse,
    CorrelatedStreet,
)
from traffic_trainer.api.services import model_service

router = APIRouter(prefix="/correlations", tags=["Correlation"])


@router.post("", response_model=CorrelationResponse)
async def get_correlations(request: CorrelationRequest):
    """
    Get correlated streets for a segment.

    correlation_type:
    - auto: Auto-select based on model type
    - graph: Based on adjacency matrix
    - spatial_attention: Based on attention weights
    - combined: Graph + attention (for GMAN)
    - learned_graph: Learned adjacency (for GWNET, MTGNN)
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")

    if request.segment_id not in model_service.segment_to_idx:
        raise HTTPException(status_code=404, detail="Segment not found")

    correlations = model_service.get_street_correlations(
        segment_id=request.segment_id,
        top_k=request.top_k,
        include_self=request.include_self,
        correlation_type=request.correlation_type,
    )

    return CorrelationResponse(
        segment_id=request.segment_id,
        street_name=model_service.get_street_name(request.segment_id),
        correlation_type=request.correlation_type,
        correlations=[CorrelatedStreet(**c) for c in correlations],
    )


@router.get("/{segment_id}")
async def get_correlations_quick(
    segment_id: int,
    top_k: int = Query(10, ge=1, le=100),
    correlation_type: str = Query("auto"),
):
    """Quick correlation lookup for a segment."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")

    if segment_id not in model_service.segment_to_idx:
        raise HTTPException(status_code=404, detail="Segment not found")

    correlations = model_service.get_street_correlations(
        segment_id=segment_id,
        top_k=top_k,
        correlation_type=correlation_type,
    )

    return {
        "segment_id": segment_id,
        "street_name": model_service.get_street_name(segment_id),
        "correlation_type": correlation_type,
        "correlations": correlations,
    }
