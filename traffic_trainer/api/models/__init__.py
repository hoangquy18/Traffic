"""Pydantic models for API request/response."""

from traffic_trainer.api.models.requests import (
    PredictionRequest,
    CorrelationRequest,
    LoadModelRequest,
)
from traffic_trainer.api.models.responses import (
    PredictionResponse,
    SegmentPrediction,
    HorizonPrediction,
    CorrelationResponse,
    CorrelatedStreet,
    StreetInfo,
    StreetListResponse,
    ModelInfo,
    HealthResponse,
)

__all__ = [
    # Requests
    "PredictionRequest",
    "CorrelationRequest",
    "LoadModelRequest",
    # Responses
    "PredictionResponse",
    "SegmentPrediction",
    "HorizonPrediction",
    "CorrelationResponse",
    "CorrelatedStreet",
    "StreetInfo",
    "StreetListResponse",
    "ModelInfo",
    "HealthResponse",
]
