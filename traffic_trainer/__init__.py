"""
Traffic Trainer Package - Deep Learning for Traffic Congestion Prediction.

This package provides models and training utilities for traffic LOS (Level of Service)
prediction using various neural network architectures.

Modules:
- data: Data loading and preprocessing (sequential and graph-based)
- models: Model architectures (RNN, GNN, Transformer, GMAN)
- trainers: Training scripts for each model type
- configs: Configuration files
- utils: Utility functions
- api: REST API for model inference (FastAPI)

Example usage:
    from traffic_trainer.data import load_dataset, LOS_LEVELS
    from traffic_trainer.models import create_model
    from traffic_trainer.trainers import RNNTrainer, TrainingConfig

    # For API:
    # python -m traffic_trainer.api.run_server
"""

# Re-export commonly used classes and functions
from traffic_trainer.data import (
    LOS_LEVELS,
    # Sequential
    SequenceSplit,
    TrafficWeatherDataset,
    load_dataset,
    # Graph
    RoadGraph,
    GraphTrafficDataset,
    build_road_graph,
    load_graph_dataset,
    collate_graph_batch,
)

from traffic_trainer.models import (
    # RNN
    SequenceClassifier,
    create_model,
    # GNN
    SpatioTemporalGNN,
    create_graph_model,
    # Transformer
    SpatioTemporalTransformer,
    create_transformer_model,
    # GMAN
    GMAN,
    create_sota_model,
)

__version__ = "0.2.0"
__all__ = [
    # Constants
    "LOS_LEVELS",
    # Data
    "SequenceSplit",
    "TrafficWeatherDataset",
    "load_dataset",
    "RoadGraph",
    "GraphTrafficDataset",
    "build_road_graph",
    "load_graph_dataset",
    "collate_graph_batch",
    # Models
    "SequenceClassifier",
    "create_model",
    "SpatioTemporalGNN",
    "create_graph_model",
    "SpatioTemporalTransformer",
    "create_transformer_model",
    "GMAN",
    "create_sota_model",
]
