"""GMAN (Graph Multi-Attention Network) trainer for traffic prediction."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from traffic_trainer.data import LOS_LEVELS, collate_graph_batch, load_graph_dataset
from traffic_trainer.models import create_sota_model
from traffic_trainer.trainers.base import BaseConfig, BaseTrainer, load_yaml_config


# ============================================================================
# Custom Loss Functions
# ============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None, 
                 ignore_index: int = -1) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)

        pred = pred[valid_mask]
        target = target[valid_mask]

        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = torch.log_softmax(pred, dim=-1)

        if self.weight is not None:
            weight = self.weight[target]
            loss = -(one_hot * log_prob * weight.unsqueeze(-1)).sum(dim=-1)
        else:
            loss = -(one_hot * log_prob).sum(dim=-1)

        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, 
                 ignore_index: int = -1) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)

        pred = pred[valid_mask]
        target = target[valid_mask]

        ce_loss = F.cross_entropy(pred, target, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class OrdinalFocalLoss(nn.Module):
    """
    Combined Focal Loss + Ordinal penalty for LOS prediction.
    LOS levels are ordinal: A(0) < B(1) < C(2) < D(3) < E(4) < F(5)
    """
    
    def __init__(self, num_classes: int = 6, gamma: float = 2.0, ordinal_weight: float = 0.3,
                 alpha: Optional[torch.Tensor] = None, ignore_index: int = -1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ordinal_weight = ordinal_weight
        self.alpha = alpha
        self.ignore_index = ignore_index
        
        # Pre-compute ordinal distance matrix
        distances = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                distances[i, j] = abs(i - j) / (num_classes - 1)
        self.register_buffer("distances", distances)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)

        pred = pred[valid_mask]
        target = target[valid_mask]

        # Focal loss component
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss

        # Ordinal loss component
        probs = torch.softmax(pred, dim=-1)
        distances = self.distances.to(pred.device)
        ordinal_distances = distances[target]
        ordinal_loss = (probs * ordinal_distances).sum(dim=-1)

        total_loss = focal_loss + self.ordinal_weight * ordinal_loss
        return total_loss.mean()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SOTATrainingConfig(BaseConfig):
    """Configuration for GMAN training."""
    
    # Graph
    graph_mode: str = "none"  # "topology", "fully_connected", "none"
    add_reverse_edges: bool = True
    
    # Model
    model_type: str = "gman"
    num_layers: int = 3
    num_heads: int = 4
    use_spatial_embedding: bool = True
    use_temporal_conv: bool = True
    time_embedding_dim: Optional[int] = None
    segment_embedding_dim: Optional[int] = None
    
    # Loss
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    use_focal_loss: bool = False
    use_ordinal_loss: bool = True
    focal_gamma: float = 2.0
    ordinal_weight: float = 0.3


def load_config(config_path: Path) -> SOTATrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    graph_cfg = cfg.get("graph", {})
    gman_cfg = cfg.get("gman", {})
    optim = cfg.get("optimization", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    early_stop = cfg.get("early_stopping", {})
    checkpoint = cfg.get("checkpoint", {})
    
    return SOTATrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/sota_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=data.get("sequence_length", 12),
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1h"),
        graph_mode=graph_cfg.get("graph_mode", "none"),
        add_reverse_edges=graph_cfg.get("add_reverse_edges", True),
        hidden_dim=model.get("hidden_dim", 128),
        dropout=model.get("dropout", 0.2),
        model_type=model.get("type", "gman"),
        num_layers=model.get("num_layers", 3),
        num_heads=gman_cfg.get("num_heads", 4),
        use_spatial_embedding=gman_cfg.get("use_spatial_embedding", True),
        use_temporal_conv=gman_cfg.get("use_temporal_conv", True),
        time_embedding_dim=model.get("time_embedding_dim"),
        segment_embedding_dim=model.get("segment_embedding_dim"),
        label_smoothing=cfg.get("label_smoothing", 0.1),
        use_class_weights=cfg.get("use_class_weights", True),
        use_focal_loss=cfg.get("use_focal_loss", False),
        use_ordinal_loss=cfg.get("use_ordinal_loss", True),
        focal_gamma=cfg.get("focal_gamma", 2.0),
        ordinal_weight=cfg.get("ordinal_weight", 0.3),
        batch_size=optim.get("batch_size", 8),
        learning_rate=optim.get("learning_rate", 0.0003),
        weight_decay=optim.get("weight_decay", 0.0001),
        gradient_clip_norm=optim.get("gradient_clip_norm", 1.0),
        scheduler_type=cfg.get("scheduler_type", "plateau"),
        epochs=training.get("epochs", 100),
        num_workers=training.get("num_workers", 0),
        device=training.get("device", "cuda"),
        wandb_project=logging_cfg.get("wandb_project"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        wandb_run_name=logging_cfg.get("wandb_run_name", "gman-run"),
        wandb_mode=logging_cfg.get("wandb_mode", "disabled"),
        early_stopping_patience=early_stop.get("early_stopping_patience", 15),
        early_stopping_delta=early_stop.get("early_stopping_delta", 0.001),
        checkpoint_every=checkpoint.get("checkpoint_every", 5),
        save_optimizer_state=checkpoint.get("save_optimizer_state", True),
    )


# ============================================================================
# Trainer
# ============================================================================

class SOTATrainer(BaseTrainer):
    """Trainer for GMAN++ model."""
    
    config: SOTATrainingConfig
    
    def _load_data(self) -> None:
        """Load graph dataset."""
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
            add_reverse_edges=self.config.add_reverse_edges,
            graph_mode=self.config.graph_mode,
            use_time_embedding=use_time_embedding,
            use_segment_embedding=use_segment_embedding,
        )
        
        self.edge_index = self.road_graph.edge_index.to(self.device)
        self.num_nodes = self.road_graph.num_nodes
        self.segment_vocab_size = self.metadata.get("segment_vocab_size")
    
    def _get_collate_fn(self):
        """Return graph collate function."""
        return collate_graph_batch
    
    def _compute_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        all_targets = []
        all_masks = []

        for batch in self.train_loader:
            targets = batch["targets"].numpy()
            masks = batch["mask"].numpy()
            all_targets.append(targets)
            all_masks.append(masks)

        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # Expand mask
        all_masks_expanded = np.expand_dims(all_masks, axis=-1)
        all_masks_expanded = np.broadcast_to(all_masks_expanded, all_targets.shape)

        valid_mask = all_masks_expanded & (all_targets >= 0)
        valid_targets = all_targets[valid_mask].flatten()

        class_counts = np.bincount(valid_targets.astype(int), minlength=self.num_classes)
        class_counts = np.maximum(class_counts, 1)

        weights = 1.0 / class_counts
        weights = weights / weights.sum() * self.num_classes

        print(f"Class distribution: {dict(zip(LOS_LEVELS.keys(), class_counts))}")
        print(f"Class weights: {dict(zip(LOS_LEVELS.keys(), weights.round(3)))}")

        return torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def _create_criterion(self) -> nn.Module:
        """Create custom loss function."""
        class_weights = self._compute_class_weights() if self.config.use_class_weights else None
        
        if self.config.use_ordinal_loss:
            print(f"Using Ordinal Focal Loss (gamma={self.config.focal_gamma}, ordinal_weight={self.config.ordinal_weight})")
            return OrdinalFocalLoss(
                num_classes=self.num_classes,
                gamma=self.config.focal_gamma,
                ordinal_weight=self.config.ordinal_weight,
                alpha=class_weights,
                ignore_index=-1,
            )
        elif self.config.use_focal_loss:
            print(f"Using Focal Loss (gamma={self.config.focal_gamma})")
            return FocalLoss(gamma=self.config.focal_gamma, alpha=class_weights, ignore_index=-1)
        elif self.config.label_smoothing > 0:
            print(f"Using Label Smoothing (smoothing={self.config.label_smoothing})")
            return LabelSmoothingCrossEntropy(
                smoothing=self.config.label_smoothing, weight=class_weights, ignore_index=-1
            )
        else:
            return nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    def _create_model(self) -> nn.Module:
        """Create GMAN model."""
        return create_sota_model(
            model_type="gman",
            input_dim=len(self.feature_names),
            hidden_dim=self.config.hidden_dim,
            num_classes=self.num_classes,
            num_nodes=self.num_nodes,
            num_layers=self.config.num_layers,
            num_horizons=self.num_horizons,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            use_spatial_embedding=self.config.use_spatial_embedding,
            use_temporal_conv=self.config.use_temporal_conv,
            time_embedding_dim=self.config.time_embedding_dim,
            segment_embedding_dim=self.config.segment_embedding_dim,
            segment_vocab_size=self.segment_vocab_size,
        )
    
    def _forward_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass on a batch."""
        node_features = batch["node_features"].to(self.device)
        targets = batch["targets"].to(self.device)
        mask = batch["mask"].to(self.device)
        
        time_ids = batch.get("time_ids")
        segment_ids = batch.get("segment_ids")
        
        if time_ids is not None:
            time_ids = time_ids.to(self.device)
        if segment_ids is not None:
            segment_ids = segment_ids.to(self.device)
        
        logits = self.model(
            node_features, 
            self.edge_index, 
            mask, 
            time_ids=time_ids, 
            segment_ids=segment_ids,
        )
        
        return logits, targets, mask
    
    def _get_model_type(self) -> str:
        return "GMAN++"
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Add GMAN-specific info to W&B config."""
        config = super()._get_wandb_config()
        config.update({
            "num_nodes": self.num_nodes,
            "num_edges": self.road_graph.edge_index.shape[1],
            "num_heads": self.config.num_heads,
            "use_ordinal_loss": self.config.use_ordinal_loss,
        })
        return config
    
    def _print_model_info(self) -> None:
        """Print model info including graph stats."""
        super()._print_model_info()
        print(f"Graph: {self.num_nodes} nodes, {self.road_graph.edge_index.shape[1]} edges")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GMAN for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "sota_config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = SOTATrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
