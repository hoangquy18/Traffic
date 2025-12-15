"""Health check endpoints."""

from fastapi import APIRouter

from traffic_trainer.api.models import HealthResponse, ModelInfo
from traffic_trainer.api.services import model_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded,
        model_type=model_service.model_type,
        num_segments=model_service.num_nodes if model_service.is_loaded else None,
    )


@router.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded model."""
    return ModelInfo(**model_service.get_model_info())
