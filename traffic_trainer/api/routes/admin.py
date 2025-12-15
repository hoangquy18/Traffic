"""Admin endpoints for model management."""

import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException

from traffic_trainer.api.config import WEIGHTS_DIR, DATA_FILE
from traffic_trainer.api.models import LoadModelRequest
from traffic_trainer.api.services import model_service

router = APIRouter(tags=["Admin"])


def find_model_files(model_name: str) -> dict:
    """Find all required files in a model folder."""
    model_dir = WEIGHTS_DIR / model_name

    if not model_dir.exists():
        available = [m["name"] for m in list_models()]
        raise FileNotFoundError(
            f"Model not found: {model_name}. Available: {available}"
        )

    # Find checkpoint
    checkpoint_path = None
    for ckpt_name in ["best_model.ckpt", "best_model.pt", "model.ckpt", "model.pt"]:
        ckpt_path = model_dir / ckpt_name
        if ckpt_path.exists():
            checkpoint_path = ckpt_path
            break

    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in weights/{model_name}/")

    # Find config
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        config_path = None

    # Find metadata
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        metadata_path = None

    # Find adjacency matrix
    adjacency_path = None
    for adj_name in ["adjacency.npy", "adjacency.npz", "adj_matrix.npy"]:
        adj_path = model_dir / adj_name
        if adj_path.exists():
            adjacency_path = adj_path
            break

    return {
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "metadata_path": metadata_path,
        "adjacency_path": adjacency_path,
        "data_path": DATA_FILE if DATA_FILE.exists() else None,
        "street_info_path": DATA_FILE if DATA_FILE.exists() else None,
    }


def list_models() -> list:
    """List all available models."""
    if not WEIGHTS_DIR.exists():
        return []

    models = []
    for model_dir in sorted(WEIGHTS_DIR.iterdir()):
        if model_dir.is_dir():
            has_checkpoint = any(
                (model_dir / name).exists()
                for name in ["best_model.ckpt", "best_model.pt", "model.ckpt"]
            )
            if has_checkpoint:
                models.append(
                    {
                        "name": model_dir.name,
                        "has_config": (model_dir / "config.yaml").exists(),
                        "has_metadata": (model_dir / "metadata.json").exists(),
                    }
                )
    return models


@router.get("/models")
async def get_models():
    """List all available models in weights/ folder."""
    models = list_models()
    return {"total": len(models), "models": models}


@router.post("/load-model")
async def load_model(request: LoadModelRequest):
    """Load a trained model."""
    try:
        files = find_model_files(request.model)

        model_service.load_model(
            checkpoint_path=files["checkpoint_path"],
            config_path=files["config_path"],
            metadata_path=files["metadata_path"],
            street_info_path=files["street_info_path"],
            adjacency_path=files["adjacency_path"],
            data_path=files["data_path"],
            device=request.device,
        )

        return {
            "status": "success",
            "message": f"Model '{request.model}' loaded successfully",
            "model_type": model_service.model_type,
            "num_nodes": model_service.num_nodes,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
