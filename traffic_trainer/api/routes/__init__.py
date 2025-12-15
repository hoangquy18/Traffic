"""API routes."""

from traffic_trainer.api.routes.health import router as health_router
from traffic_trainer.api.routes.admin import router as admin_router
from traffic_trainer.api.routes.streets import router as streets_router
from traffic_trainer.api.routes.prediction import router as prediction_router
from traffic_trainer.api.routes.correlation import router as correlation_router

__all__ = [
    "health_router",
    "admin_router",
    "streets_router",
    "prediction_router",
    "correlation_router",
]
