"""Training modules for traffic prediction models."""

from traffic_trainer.trainers.base import BaseTrainer, BaseConfig, load_yaml_config
from traffic_trainer.trainers.rnn_trainer import Trainer as RNNTrainer, TrainingConfig
from traffic_trainer.trainers.gnn_trainer import GraphTrainer, GraphTrainingConfig
from traffic_trainer.trainers.transformer_trainer import TransformerTrainer, TransformerTrainingConfig
from traffic_trainer.trainers.gman_trainer import (
    SOTATrainer,
    SOTATrainingConfig,
    LabelSmoothingCrossEntropy,
    FocalLoss,
    OrdinalFocalLoss,
)
from traffic_trainer.trainers.tcn_trainer import TCNTrainer, TCNTrainingConfig
from traffic_trainer.trainers.informer_trainer import InformerTrainer, InformerTrainingConfig
from traffic_trainer.trainers.autoformer_trainer import AutoformerTrainer, AutoformerTrainingConfig
from traffic_trainer.trainers.dlinear_trainer import DLinearTrainer, DLinearTrainingConfig
from traffic_trainer.trainers.patchtst_trainer import PatchTSTTrainer, PatchTSTTrainingConfig

__all__ = [
    # Base
    "BaseTrainer",
    "BaseConfig",
    "load_yaml_config",
    # RNN
    "RNNTrainer",
    "TrainingConfig",
    # GNN
    "GraphTrainer",
    "GraphTrainingConfig",
    # Transformer
    "TransformerTrainer",
    "TransformerTrainingConfig",
    # GMAN
    "SOTATrainer",
    "SOTATrainingConfig",
    "LabelSmoothingCrossEntropy",
    "FocalLoss",
    "OrdinalFocalLoss",
    # TCN
    "TCNTrainer",
    "TCNTrainingConfig",
    # Informer
    "InformerTrainer",
    "InformerTrainingConfig",
    # Autoformer
    "AutoformerTrainer",
    "AutoformerTrainingConfig",
    # DLinear
    "DLinearTrainer",
    "DLinearTrainingConfig",
    # PatchTST
    "PatchTSTTrainer",
    "PatchTSTTrainingConfig",
]
