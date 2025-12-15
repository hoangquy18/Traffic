"""DLinear-based trainer for traffic prediction."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from traffic_trainer.data import LOS_LEVELS, collate_graph_batch, load_graph_dataset
from traffic_trainer.models.dlinear import create_dlinear_model
from traffic_trainer.trainers.base import BaseConfig, BaseTrainer, load_yaml_config


@dataclass
class DLinearTrainingConfig(BaseConfig):
    """Configuration for DLinear training."""
    
    # Model - DLinear
    seq_len: int = 96  # Input sequence length
    pred_len: int = 24  # Prediction length (not used for classification)
    kernel_size: int = 25  # Moving average window for decomposition
    individual: bool = False  # Use individual linear layers per feature
    model_type: str = "dlinear"  # "dlinear" or "nlinear"
    time_embedding_dim: Optional[int] = None
    segment_embedding_dim: Optional[int] = None


def load_config(config_path: Path) -> DLinearTrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    dlinear_cfg = cfg.get("dlinear", {})
    optim = cfg.get("optimization", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    early_stop = cfg.get("early_stopping", {})
    checkpoint = cfg.get("checkpoint", {})
    
    # Get sequence length from data config
    seq_len = data.get("sequence_length", 96)
    
    return DLinearTrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/dlinear_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=seq_len,
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        hidden_dim=model.get("hidden_dim", 128),
        dropout=model.get("dropout", 0.1),
        seq_len=dlinear_cfg.get("seq_len", seq_len),
        pred_len=dlinear_cfg.get("pred_len", seq_len // 4),
        kernel_size=dlinear_cfg.get("kernel_size", 25),
        individual=dlinear_cfg.get("individual", False),
        model_type=dlinear_cfg.get("model_type", "dlinear"),
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
        wandb_run_name=logging_cfg.get("wandb_run_name", "dlinear-run"),
        wandb_mode=logging_cfg.get("wandb_mode", "disabled"),
        early_stopping_patience=early_stop.get("early_stopping_patience", 10),
        early_stopping_delta=early_stop.get("early_stopping_delta", 0.0),
        checkpoint_every=checkpoint.get("checkpoint_every", 5),
        save_optimizer_state=checkpoint.get("save_optimizer_state", True),
    )


class DLinearTrainer(BaseTrainer):
    """Trainer for DLinear model."""
    
    config: DLinearTrainingConfig
    
    def _load_data(self) -> None:
        """Load dataset for DLinear."""
        print(f"Loading dataset for DLinear from {self.config.csv_path}...")
        
        use_time_embedding = (
            self.config.time_embedding_dim is not None 
            and self.config.time_embedding_dim > 0
        )
        use_segment_embedding = (
            self.config.segment_embedding_dim is not None 
            and self.config.segment_embedding_dim > 0
        )
        
        # Use seq_len for sequence_length
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
            sequence_length=self.config.seq_len,
            feature_columns={
                "numerical": self.config.numerical_features,
                "categorical": self.config.categorical_features,
            },
            prediction_horizons=tuple(self.config.prediction_horizons),
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            resample_rule=self.config.resample_rule,
            graph_mode="none",  # DLinear doesn't need graph edges
            use_time_embedding=use_time_embedding,
            use_segment_embedding=use_segment_embedding,
        )
        
        self.num_segments = self.road_graph.num_nodes
        self.segment_vocab_size = self.metadata.get("segment_vocab_size")
    
    def _get_collate_fn(self):
        """Return graph collate function."""
        return collate_graph_batch
    
    def _create_model(self) -> nn.Module:
        """Create DLinear model."""
        return create_dlinear_model(
            input_dim=len(self.feature_names),
            hidden_dim=self.config.hidden_dim,
            num_classes=len(LOS_LEVELS),
            num_horizons=self.num_horizons,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            kernel_size=self.config.kernel_size,
            individual=self.config.individual,
            model_type=self.config.model_type,
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
        
        # DLinear forward pass
        logits = self.model(
            node_features,
            time_ids=time_ids,
            segment_ids=segment_ids,
            mask=mask,
        )
        
        return logits, targets, mask
    
    def _get_model_type(self) -> str:
        model_name = "DLinear" if self.config.model_type == "dlinear" else "NLinear"
        individual_str = " (Individual)" if self.config.individual else ""
        return f"{model_name}{individual_str}"
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Add DLinear-specific info to W&B config."""
        config = super()._get_wandb_config()
        config.update({
            "num_segments": self.num_segments,
            "seq_len": self.config.seq_len,
            "pred_len": self.config.pred_len,
            "kernel_size": self.config.kernel_size,
            "individual": self.config.individual,
            "model_type": self.config.model_type,
        })
        return config
    
    def _print_model_info(self) -> None:
        """Print model info including DLinear-specific details."""
        super()._print_model_info()
        print(f"Segments: {self.num_segments}")
        print(f"Sequence Length: {self.config.seq_len}")
        print(f"Prediction Length: {self.config.pred_len}")
        print(f"Decomposition Kernel: {self.config.kernel_size}")
        print(f"Individual Layers: {self.config.individual}")
        print(f"Model Type: {self.config.model_type.upper()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DLinear for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "dlinear_config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = DLinearTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()

