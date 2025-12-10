"""TCN-based trainer for traffic prediction."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from traffic_trainer.data import LOS_LEVELS, collate_graph_batch, load_graph_dataset
from traffic_trainer.models import create_tcn_model
from traffic_trainer.trainers.base import BaseConfig, BaseTrainer, load_yaml_config


@dataclass
class TCNTrainingConfig(BaseConfig):
    """Configuration for TCN training."""
    
    # Model - TCN
    model_type: str = "standard"  # "standard" or "multiscale"
    num_channels: Optional[List[int]] = None  # Channel progression for standard TCN
    kernel_size: int = 3
    scales: Optional[List[int]] = None  # Kernel sizes for multiscale TCN
    use_temporal_attention: bool = True
    use_spatial_attention: bool = False
    num_attention_heads: int = 4
    time_embedding_dim: Optional[int] = None
    segment_embedding_dim: Optional[int] = None


def load_config(config_path: Path) -> TCNTrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    tcn_cfg = cfg.get("tcn", {})
    optim = cfg.get("optimization", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    early_stop = cfg.get("early_stopping", {})
    checkpoint = cfg.get("checkpoint", {})
    
    return TCNTrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/tcn_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=data.get("sequence_length", 8),
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        hidden_dim=model.get("hidden_dim", 128),
        dropout=model.get("dropout", 0.2),
        model_type=tcn_cfg.get("model_type", "standard"),
        num_channels=tcn_cfg.get("num_channels"),
        kernel_size=tcn_cfg.get("kernel_size", 3),
        scales=tcn_cfg.get("scales"),
        use_temporal_attention=tcn_cfg.get("use_temporal_attention", True),
        use_spatial_attention=tcn_cfg.get("use_spatial_attention", False),
        num_attention_heads=tcn_cfg.get("num_attention_heads", 4),
        time_embedding_dim=model.get("time_embedding_dim"),
        segment_embedding_dim=model.get("segment_embedding_dim"),
        batch_size=optim.get("batch_size", 32),
        learning_rate=optim.get("learning_rate", 0.001),
        weight_decay=optim.get("weight_decay", 0.0001),
        gradient_clip_norm=optim.get("gradient_clip_norm", 1.0),
        epochs=training.get("epochs", 50),
        num_workers=training.get("num_workers", 4),
        device=training.get("device", "cuda"),
        scheduler_type=training.get("scheduler_type", "plateau"),
        wandb_project=logging_cfg.get("wandb_project"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        wandb_run_name=logging_cfg.get("wandb_run_name", "tcn-run"),
        wandb_mode=logging_cfg.get("wandb_mode", "disabled"),
        early_stopping_patience=early_stop.get("early_stopping_patience", 10),
        early_stopping_delta=early_stop.get("early_stopping_delta", 0.0),
        checkpoint_every=checkpoint.get("checkpoint_every", 5),
        save_optimizer_state=checkpoint.get("save_optimizer_state", True),
    )


class TCNTrainer(BaseTrainer):
    """Trainer for Temporal Convolutional Network."""
    
    config: TCNTrainingConfig
    
    def _load_data(self) -> None:
        """Load dataset (graph format for consistency with other models)."""
        print(f"Loading dataset for TCN from {self.config.csv_path}...")
        
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
            graph_mode="none",  # TCN doesn't need graph edges
            use_time_embedding=use_time_embedding,
            use_segment_embedding=use_segment_embedding,
        )
        
        self.num_segments = self.road_graph.num_nodes
        self.segment_vocab_size = self.metadata.get("segment_vocab_size")
    
    def _get_collate_fn(self):
        """Return graph collate function."""
        return collate_graph_batch
    
    def _create_model(self) -> nn.Module:
        """Create TCN model."""
        # Default channel progression if not specified
        num_channels = self.config.num_channels
        if num_channels is None:
            num_channels = [
                self.config.hidden_dim,
                self.config.hidden_dim,
                self.config.hidden_dim * 2,
                self.config.hidden_dim * 2,
            ]
        
        return create_tcn_model(
            model_type=self.config.model_type,
            input_dim=len(self.feature_names),
            hidden_dim=self.config.hidden_dim,
            num_classes=len(LOS_LEVELS),
            num_horizons=self.num_horizons,
            num_channels=num_channels if self.config.model_type == "standard" else None,
            kernel_size=self.config.kernel_size,
            scales=self.config.scales,
            dropout=self.config.dropout,
            use_temporal_attention=self.config.use_temporal_attention,
            use_spatial_attention=self.config.use_spatial_attention,
            num_attention_heads=self.config.num_attention_heads,
            time_embedding_dim=self.config.time_embedding_dim,
            segment_embedding_dim=self.config.segment_embedding_dim,
            segment_vocab_size=self.segment_vocab_size,
        )
    
    def _forward_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass on a batch."""
        node_features = batch["node_features"].to(self.device)
        targets = batch["targets"].to(self.device)
        mask = batch["mask"].to(self.device)
        segment_ids = batch["segment_ids"].to(self.device)
        
        time_ids = batch.get("time_ids")
        if time_ids is not None:
            time_ids = time_ids.to(self.device)
        
        # TCN doesn't use edge_index but we pass it for API consistency
        logits = self.model(
            node_features,
            mask=mask,
            time_ids=time_ids,
            segment_ids=segment_ids,
        )
        
        return logits, targets, mask
    
    def _get_model_type(self) -> str:
        model_type_str = "Multi-Scale TCN" if self.config.model_type == "multiscale" else "TCN"
        attention_str = ""
        if self.config.use_temporal_attention and self.config.use_spatial_attention:
            attention_str = " (Temporal + Spatial Attention)"
        elif self.config.use_temporal_attention:
            attention_str = " (Temporal Attention)"
        elif self.config.use_spatial_attention:
            attention_str = " (Spatial Attention)"
        return f"{model_type_str}{attention_str}"
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Add TCN-specific info to W&B config."""
        config = super()._get_wandb_config()
        config.update({
            "num_segments": self.num_segments,
            "tcn_model_type": self.config.model_type,
            "num_channels": self.config.num_channels,
            "kernel_size": self.config.kernel_size,
            "use_temporal_attention": self.config.use_temporal_attention,
            "use_spatial_attention": self.config.use_spatial_attention,
        })
        return config
    
    def _print_model_info(self) -> None:
        """Print model info including TCN-specific details."""
        super()._print_model_info()
        print(f"Segments: {self.num_segments}")
        print(f"TCN Type: {self.config.model_type}")
        if self.config.model_type == "standard":
            print(f"Channels: {self.config.num_channels or 'default'}")
            print(f"Kernel Size: {self.config.kernel_size}")
        else:
            print(f"Scales: {self.config.scales or [3, 5, 7]}")
        print(f"Temporal Attention: {self.config.use_temporal_attention}")
        print(f"Spatial Attention: {self.config.use_spatial_attention}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCN for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "tcn_config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = TCNTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()

