"""Response models for API endpoints."""

from datetime import datetime as dt
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HorizonPrediction(BaseModel):
    """Prediction for a single horizon."""

    horizon: int = Field(..., description="Prediction horizon (timesteps ahead)")
    los_class: str = Field(..., description="Predicted LOS class (A-F)")
    los_index: int = Field(..., description="LOS class index (0-5)")
    confidence: float = Field(..., description="Prediction confidence")
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Full probability distribution"
    )


class SegmentPrediction(BaseModel):
    """Prediction result for a segment."""

    segment_id: int
    street_name: Optional[str] = None
    street_type: Optional[str] = None
    predictions: List[HorizonPrediction]


class PredictionResponse(BaseModel):
    """Response for prediction request."""

    request_time: dt
    predictions: List[SegmentPrediction]


class CorrelatedStreet(BaseModel):
    """A correlated street."""

    segment_id: int
    street_name: Optional[str] = None
    street_type: Optional[str] = None
    correlation_score: float
    relationship_type: str
    model_type: str


class CorrelationResponse(BaseModel):
    """Response for correlation request."""

    segment_id: int
    street_name: Optional[str] = None
    correlation_type: str
    correlations: List[CorrelatedStreet]


class StreetInfo(BaseModel):
    """Street information."""

    segment_id: int
    street_id: int
    street_name: str
    street_type: str
    street_level: Optional[int] = None


class StreetListResponse(BaseModel):
    """Response for street list."""

    total: int
    streets: List[StreetInfo]


class ModelInfo(BaseModel):
    """Model information."""

    loaded: bool
    model_type: Optional[str] = None
    num_nodes: Optional[int] = None
    num_horizons: Optional[int] = None
    sequence_length: Optional[int] = None
    prediction_horizons: Optional[List[int]] = None
    input_dim: Optional[int] = None
    device: Optional[str] = None
    has_adjacency: Optional[bool] = None
    has_edge_index: Optional[bool] = None
    num_streets_loaded: Optional[int] = None
    supported_correlation_types: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    num_segments: Optional[int] = None
