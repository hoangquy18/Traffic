"""PatchTST-based trainer for traffic prediction."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from traffic_trainer.data import LOS_LEVELS, collate_graph_batch, load_graph_dataset
from traffic_trainer.models.patchtst import create_patchtst_model
from traffic_trainer.trainers.base import BaseConfig, BaseTrainer, load_yaml_config


@dataclass
class PatchTSTTrainingConfig(BaseConfig):
    """Configuration for PatchTST training."""
    
    # Model - PatchTST
    seq_len: int = 96  # Input sequence length
    patch_len: int = 16  # Length of each patch
    stride: int = 8  # Stride for patching
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 3  # Number of Transformer layers
    d_ff: int = 512  # Feed-forward dimension
    activation: str = "gelu"  # Activation function
    channel_independence: bool = True  # Process each channel separately
    time_embedding_dim: Optional[int] = None
    segment_embedding_dim: Optional[int] = None


def load_config(config_path: Path) -> PatchTSTTrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    patchtst_cfg = cfg.get("patchtst", {})
    optim = cfg.get("optimization", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    early_stop = cfg.get("early_stopping", {})
    checkpoint = cfg.get("checkpoint", {})
    
    # Get sequence length from data config
    seq_len = data.get("sequence_length", 96)
    
    return PatchTSTTrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/patchtst_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=seq_len,
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        hidden_dim=model.get("hidden_dim", 128),
        dropout=model.get("dropout", 0.1),
        seq_len=patchtst_cfg.get("seq_len", seq_len),
        patch_len=patchtst_cfg.get("patch_len", 16),
        stride=patchtst_cfg.get("stride", 8),
        n_heads=patchtst_cfg.get("n_heads", 8),
        n_layers=patchtst_cfg.get("n_layers", 3),
        d_ff=patchtst_cfg.get("d_ff", 512),
        activation=patchtst_cfg.get("activation", "gelu"),
        channel_independence=patchtst_cfg.get("channel_independence", True),
        time_embedding_dim=model.get("time_embedding_dim"),
        segment_embedding_dim=model.get("segment_embedding_dim"),
        batch_size=optim.get("batch_size", 16),
        learning_rate=optim.get("learning_rate", 0.0001),
        weight_decay=optim.get("weight_decay", 0.0001),
        gradient_clip_norm=optim.get("gradient_clip_norm", 1.0),
        epochs=training.get("epochs", 50),
        num_workers=training.get("num_workers", 4),
        device=training.get("device", "cuda"),
        scheduler_type=training.get("scheduler_type", "plateau"),
        wandb_project=logging_cfg.get("wandb_project"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        wandb_run_name=logging_cfg.get("wandb_run_name", "patchtst-run"),
        wandb_mode=logging_cfg.get("wandb_mode", "disabled"),
        early_stopping_patience=early_stop.get("early_stopping_patience", 10),
        early_stopping_delta=early_stop.get("early_stopping_delta", 0.0),
        checkpoint_every=checkpoint.get("checkpoint_every", 5),
        save_optimizer_state=checkpoint.get("save_optimizer_state", True),
    )


class PatchTSTTrainer(BaseTrainer):
    """Trainer for PatchTST model."""
    
    config: PatchTSTTrainingConfig
    
    def _load_data(self) -> None:
        """Load dataset for PatchTST."""
        print(f"Loading dataset for PatchTST from {self.config.csv_path}...")
        
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
            graph_mode="none",  # PatchTST doesn't need graph edges
            use_time_embedding=use_time_embedding,
            use_segment_embedding=use_segment_embedding,
        )
        
        self.num_segments = self.road_graph.num_nodes
        self.segment_vocab_size = self.metadata.get("segment_vocab_size")
    
    def _get_collate_fn(self):
        """Return graph collate function."""
        return collate_graph_batch
    
    def _create_model(self) -> nn.Module:
        """Create PatchTST model."""
        return create_patchtst_model(
            input_dim=len(self.feature_names),
            hidden_dim=self.config.hidden_dim,
            num_classes=len(LOS_LEVELS),
            num_horizons=self.num_horizons,
            seq_len=self.config.seq_len,
            patch_len=self.config.patch_len,
            stride=self.config.stride,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
            activation=self.config.activation,
            channel_independence=self.config.channel_independence,
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
        
        # PatchTST forward pass
        logits = self.model(
            node_features,
            time_ids=time_ids,
            segment_ids=segment_ids,
            mask=mask,
        )
        
        return logits, targets, mask
    
    def _get_model_type(self) -> str:
        ci_str = "Channel-Independent" if self.config.channel_independence else "Channel-Dependent"
        return f"PatchTST ({ci_str})"
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Add PatchTST-specific info to W&B config."""
        config = super()._get_wandb_config()
        
        # Calculate num_patches
        num_patches = (self.config.seq_len - self.config.patch_len) // self.config.stride + 1
        
        config.update({
            "num_segments": self.num_segments,
            "seq_len": self.config.seq_len,
            "patch_len": self.config.patch_len,
            "stride": self.config.stride,
            "num_patches": num_patches,
            "n_heads": self.config.n_heads,
            "n_layers": self.config.n_layers,
            "d_ff": self.config.d_ff,
            "activation": self.config.activation,
            "channel_independence": self.config.channel_independence,
        })
        return config
    
    def _print_model_info(self) -> None:
        """Print model info including PatchTST-specific details."""
        super()._print_model_info()
        
        num_patches = (self.config.seq_len - self.config.patch_len) // self.config.stride + 1
        
        print(f"Segments: {self.num_segments}")
        print(f"Sequence Length: {self.config.seq_len}")
        print(f"Patch Length: {self.config.patch_len}")
        print(f"Stride: {self.config.stride}")
        print(f"Number of Patches: {num_patches} (reduced from {self.config.seq_len}!)")
        print(f"Transformer Layers: {self.config.n_layers}")
        print(f"Attention Heads: {self.config.n_heads}")
        print(f"Feed-Forward Dim: {self.config.d_ff}")
        print(f"Channel Independence: {self.config.channel_independence}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchTST for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "patchtst_config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = PatchTSTTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()


