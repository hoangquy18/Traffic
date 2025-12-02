"""RNN-based trainer for traffic prediction."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from traffic_trainer.data import LOS_LEVELS, load_dataset
from traffic_trainer.models import create_model
from traffic_trainer.trainers.base import BaseConfig, BaseTrainer, load_yaml_config


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for RNN training."""
    
    # Model
    rnn_type: str = "lstm"
    num_layers: int = 2
    bidirectional: bool = True
    time_embedding_dim: Optional[int] = None
    segment_embedding_dim: Optional[int] = None


def load_config(config_path: Path) -> TrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    optim = cfg.get("optimization", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    early_stop = cfg.get("early_stopping", {})
    checkpoint = cfg.get("checkpoint", {})
    
    return TrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=data.get("sequence_length", 8),
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        hidden_dim=model.get("hidden_dim", 128),
        dropout=model.get("dropout", 0.3),
        rnn_type=model.get("rnn_type", "lstm"),
        num_layers=model.get("num_layers", 2),
        bidirectional=model.get("bidirectional", True),
        time_embedding_dim=model.get("time_embedding_dim"),
        segment_embedding_dim=model.get("segment_embedding_dim"),
        batch_size=optim.get("batch_size", 128),
        learning_rate=optim.get("learning_rate", 0.001),
        weight_decay=optim.get("weight_decay", 0.0001),
        gradient_clip_norm=optim.get("gradient_clip_norm", 1.0),
        epochs=training.get("epochs", 50),
        num_workers=training.get("num_workers", 4),
        device=training.get("device", "cuda"),
        wandb_project=logging_cfg.get("wandb_project"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        wandb_run_name=logging_cfg.get("wandb_run_name", "rnn-run"),
        wandb_mode=logging_cfg.get("wandb_mode", "disabled"),
        early_stopping_patience=early_stop.get("early_stopping_patience", 5),
        early_stopping_delta=early_stop.get("early_stopping_delta", 0.0),
        checkpoint_every=checkpoint.get("checkpoint_every", 1),
        save_optimizer_state=checkpoint.get("save_optimizer_state", True),
    )


class Trainer(BaseTrainer):
    """Trainer for RNN-based sequence classifier."""
    
    config: TrainingConfig
    
    def _load_data(self) -> None:
        """Load sequential dataset."""
        print(f"Loading dataset from {self.config.csv_path}...")
        
        use_time_embedding = (
            self.config.time_embedding_dim is not None 
            and self.config.time_embedding_dim > 0
        )
        use_segment_embedding = (
            self.config.segment_embedding_dim is not None 
            and self.config.segment_embedding_dim > 0
        )
        
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.feature_names,
            self.scaler,
            metadata,
        ) = load_dataset(
            csv_path=self.config.csv_path,
            sequence_length=self.config.sequence_length,
            feature_columns={
                "numerical": self.config.numerical_features,
                "categorical": self.config.categorical_features,
            },
            prediction_horizons=tuple(self.config.prediction_horizons),
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            resample_rule=self.config.resample_rule,
            use_time_embedding=use_time_embedding,
            use_segment_embedding=use_segment_embedding,
        )
        
        self.segment_encoder = metadata.get("segment_encoder")
        self.segment_vocab_size = metadata.get("segment_vocab_size")
        self.metadata = metadata
    
    def _create_model(self) -> nn.Module:
        """Create RNN model."""
        return create_model(
            input_dim=len(self.feature_names),
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_classes=len(LOS_LEVELS),
            rnn_type=self.config.rnn_type,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
            num_horizons=self.num_horizons,
            time_embedding_dim=self.config.time_embedding_dim,
            segment_embedding_dim=self.config.segment_embedding_dim,
            segment_vocab_size=self.segment_vocab_size,
        )
    
    def _forward_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass on a batch."""
        batch_inputs, targets = batch
        
        sequences = batch_inputs["features"].to(self.device)
        time_ids = batch_inputs.get("time_ids")
        segment_ids = batch_inputs.get("segment_ids")
        
        if time_ids is not None:
            time_ids = time_ids.to(self.device)
        if segment_ids is not None:
            segment_ids = segment_ids.to(self.device)
        
        targets = targets.to(self.device)
        
        logits = self.model(sequences, time_ids=time_ids, segment_ids=segment_ids)
        
        return logits, targets, None  # No mask for sequential data
    
    def _get_model_type(self) -> str:
        return f"RNN ({self.config.rnn_type.upper()})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNN for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
