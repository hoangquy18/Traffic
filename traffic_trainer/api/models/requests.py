"""Request models for API endpoints."""

from datetime import datetime as dt
from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request for traffic prediction."""

    segment_ids: List[int] = Field(..., description="List of segment IDs to predict")
    target_time: dt = Field(
        default_factory=dt.now,
        description="Target datetime for prediction (default: now)",
    )
    return_probabilities: bool = Field(
        default=False, description="Return full probability distribution"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "segment_ids": [206641347733512, 72693918077428],
                "target_time": "2025-01-15T08:30:00",
                "return_probabilities": False,
            }
        }


class CorrelationRequest(BaseModel):
    """Request for street correlation."""

    segment_id: int = Field(..., description="Source segment ID")
    top_k: int = Field(
        default=10, ge=1, le=100, description="Number of top correlations"
    )
    correlation_type: str = Field(
        default="auto",
        description="Correlation type: auto, graph, spatial_attention, combined, learned_graph",
    )
    include_self: bool = Field(default=False, description="Include self-correlation")

    class Config:
        json_schema_extra = {
            "example": {
                "segment_id": 206641347733512,
                "top_k": 10,
                "correlation_type": "auto",
            }
        }


class LoadModelRequest(BaseModel):
    """Request to load a model."""

    model: str = Field(
        ...,
        description="Model name (folder name in weights/)",
        json_schema_extra={"example": "gman"},
    )
    device: str = Field(default="cpu", description="Device: cpu or cuda")
