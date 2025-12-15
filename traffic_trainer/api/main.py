"""FastAPI application for Traffic Prediction API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from traffic_trainer.api.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    WEIGHTS_DIR,
)
from traffic_trainer.api.routes import (
    health_router,
    admin_router,
    streets_router,
    prediction_router,
    correlation_router,
)
from traffic_trainer.api.routes.admin import list_models


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(admin_router)
app.include_router(streets_router)
app.include_router(prediction_router)
app.include_router(correlation_router)


@app.on_event("startup")
async def startup_event():
    """Show available models on startup."""
    models = list_models()
    if models:
        print(f"✓ Found {len(models)} models in weights/:")
        for m in models:
            print(f"  - {m['name']}")
        print(f"\nCall POST /load-model to load a model")
    else:
        print(f"⚠ No models found in {WEIGHTS_DIR}/")
