"""Traffic Prediction API module."""

from traffic_trainer.api.main import app
from traffic_trainer.api.services import ModelService, model_service

__all__ = [
    "app",
    "ModelService",
    "model_service",
]
