"""TimesNet++ trainer for traffic prediction."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from traffic_trainer.data import LOS_LEVELS, collate_graph_batch, load_graph_dataset
from traffic_trainer.models.timesnet_plus_plus import create_timesnet_plus_plus_model
from traffic_trainer.trainers.base import BaseConfig, BaseTrainer, load_yaml_config


@dataclass
class TimesNetPlusPlusTrainingConfig(BaseConfig):
    """Configuration for TimesNet++ training."""
    
    # Model - TimesNet++ specific
    seq_len: int = 96
    pred_len: int = 24
    e_layers: int = 2
    top_k: int = 5
    d_ff: int = 2048
    num_kernels: int = 6
    num_periods: int = 3


def load_config(config_path: Path) -> TimesNetPlusPlusTrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    timesnet_cfg = cfg.get("timesnet_plus_plus", {})
    optim = cfg.get("optimization", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    early_stop = cfg.get("early_stopping", {})
    checkpoint = cfg.get("checkpoint", {})
    
    return TimesNetPlusPlusTrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/timesnet_plus_plus_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=data.get("sequence_length", 8),
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        hidden_dim=model.get("hidden_dim", 128),
        dropout=model.get("dropout", 0.3),
        seq_len=timesnet_cfg.get("seq_len", 96),
        pred_len=timesnet_cfg.get("pred_len", 24),
        e_layers=timesnet_cfg.get("e_layers", 2),
        top_k=timesnet_cfg.get("top_k", 5),
        d_ff=timesnet_cfg.get("d_ff", 2048),
        num_kernels=timesnet_cfg.get("num_kernels", 6),
        num_periods=timesnet_cfg.get("num_periods", 3),
        batch_size=optim.get("batch_size", 16),
        learning_rate=optim.get("learning_rate", 0.001),
        weight_decay=optim.get("weight_decay", 0.0001),
        gradient_clip_norm=optim.get("gradient_clip_norm", 1.0),
        epochs=training.get("epochs", 50),
        num_workers=training.get("num_workers", 0),
        device=training.get("device", "cuda"),
        wandb_project=logging_cfg.get("wandb_project"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        wandb_run_name=logging_cfg.get("wandb_run_name", "timesnet-plus-plus-run"),
        wandb_mode=logging_cfg.get("wandb_mode", "disabled"),
        early_stopping_patience=early_stop.get("early_stopping_patience", 10),
        early_stopping_delta=early_stop.get("early_stopping_delta", 0.0),
        checkpoint_every=checkpoint.get("checkpoint_every", 5),
        save_optimizer_state=checkpoint.get("save_optimizer_state", True),
    )


class TimesNetPlusPlusTrainer(BaseTrainer):
    """Trainer for TimesNet++ model."""
    
    config: TimesNetPlusPlusTrainingConfig
    
    def _load_data(self) -> None:
        """Load dataset for TimesNet++."""
        print(f"Loading dataset for TimesNet++ from {self.config.csv_path}...")
        
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.road_graph,
            self.feature_names,
            self.scaler,
            self.metadata,
        ) = load_graph_dataset(
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
            graph_mode="none",  # TimesNet++ doesn't need graph topology
            use_time_embedding=False,
            use_segment_embedding=False,
        )
        
        self.num_segments = self.road_graph.num_nodes
    
    def _get_collate_fn(self):
        """Return graph collate function."""
        return collate_graph_batch
    
    def _create_model(self) -> nn.Module:
        """Create TimesNet++ model."""
        return create_timesnet_plus_plus_model(
            input_dim=len(self.feature_names),
            hidden_dim=self.config.hidden_dim,
            num_classes=len(LOS_LEVELS),
            num_horizons=self.num_horizons,
            seq_len=self.config.sequence_length,
            pred_len=max(self.config.prediction_horizons),
            e_layers=self.config.e_layers,
            top_k=self.config.top_k,
            d_ff=self.config.d_ff,
            num_kernels=self.config.num_kernels,
            num_periods=self.config.num_periods,
            dropout=self.config.dropout,
        )
    
    def _forward_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass on a batch."""
        node_features = batch["node_features"].to(self.device)
        targets = batch["targets"].to(self.device)
        mask = batch["mask"].to(self.device)
        
        logits = self.model(
            node_features,
            mask=mask,
        )
        
        return logits, targets, mask
    
    def _get_model_type(self) -> str:
        return "TimesNet++"
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Add TimesNet++-specific info to W&B config."""
        config = super()._get_wandb_config()
        config.update({
            "num_segments": self.num_segments,
            "e_layers": self.config.e_layers,
            "top_k": self.config.top_k,
            "num_kernels": self.config.num_kernels,
            "num_periods": self.config.num_periods,
        })
        return config
    
    def _print_model_info(self) -> None:
        """Print model info including segment count."""
        super()._print_model_info()
        print(f"Segments: {self.num_segments}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TimesNet++ for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "timesnet_plus_plus_config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = TimesNetPlusPlusTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()

