"""GNN-based trainer for traffic prediction."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from traffic_trainer.data import LOS_LEVELS, collate_graph_batch, load_graph_dataset
from traffic_trainer.models import create_graph_model
from traffic_trainer.trainers.base import BaseConfig, BaseTrainer, load_yaml_config


@dataclass
class GraphTrainingConfig(BaseConfig):
    """Configuration for GNN training."""
    
    # Graph
    add_reverse_edges: bool = True
    graph_mode: str = "topology"  # "topology", "fully_connected", "none"
    
    # Model
    rnn_type: str = "gru"
    gnn_type: str = "gcn"  # "gcn" or "gat"
    num_rnn_layers: int = 2
    num_gnn_layers: int = 2
    gat_heads: int = 4
    bidirectional: bool = True
    time_embedding_dim: Optional[int] = None
    segment_embedding_dim: Optional[int] = None


def load_config(config_path: Path) -> GraphTrainingConfig:
    """Load configuration from YAML file."""
    cfg = load_yaml_config(config_path)
    
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    graph_cfg = cfg.get("graph", {})
    optim = cfg.get("optimization", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    early_stop = cfg.get("early_stopping", {})
    checkpoint = cfg.get("checkpoint", {})
    
    return GraphTrainingConfig(
        csv_path=Path(paths.get("csv_path", "data.csv")),
        output_dir=Path(paths.get("output_dir", "experiments/graph_run01")),
        numerical_features=data.get("numerical_features", []),
        categorical_features=data.get("categorical_features", []),
        prediction_horizons=data.get("prediction_horizons", [1]),
        sequence_length=data.get("sequence_length", 8),
        train_ratio=data.get("train_ratio", 0.7),
        val_ratio=data.get("val_ratio", 0.15),
        resample_rule=data.get("resample_rule", "1H"),
        add_reverse_edges=graph_cfg.get("add_reverse_edges", True),
        graph_mode=graph_cfg.get("graph_mode", "topology"),
        hidden_dim=model.get("hidden_dim", 128),
        dropout=model.get("dropout", 0.3),
        rnn_type=model.get("rnn_type", "gru"),
        gnn_type=graph_cfg.get("gnn_type", "gcn"),
        num_rnn_layers=model.get("num_layers", 2),
        num_gnn_layers=graph_cfg.get("num_gnn_layers", 2),
        gat_heads=graph_cfg.get("gat_heads", 4),
        bidirectional=model.get("bidirectional", True),
        time_embedding_dim=model.get("time_embedding_dim"),
        segment_embedding_dim=model.get("segment_embedding_dim"),
        batch_size=optim.get("batch_size", 16),
        learning_rate=optim.get("learning_rate", 0.001),
        weight_decay=optim.get("weight_decay", 0.0001),
        gradient_clip_norm=optim.get("gradient_clip_norm", 1.0),
        epochs=training.get("epochs", 50),
        num_workers=training.get("num_workers", 0),
        device=training.get("device", "cuda"),
        wandb_project=logging_cfg.get("wandb_project"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        wandb_run_name=logging_cfg.get("wandb_run_name", "gnn-run"),
        wandb_mode=logging_cfg.get("wandb_mode", "disabled"),
        early_stopping_patience=early_stop.get("early_stopping_patience", 10),
        early_stopping_delta=early_stop.get("early_stopping_delta", 0.0),
        checkpoint_every=checkpoint.get("checkpoint_every", 5),
        save_optimizer_state=checkpoint.get("save_optimizer_state", True),
    )


class GraphTrainer(BaseTrainer):
    """Trainer for Spatio-Temporal GNN."""
    
    config: GraphTrainingConfig
    
    def _load_data(self) -> None:
        """Load graph dataset."""
        print(f"Loading graph dataset from {self.config.csv_path}...")
        
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
    
    def _create_model(self) -> nn.Module:
        """Create GNN model."""
        return create_graph_model(
            input_dim=len(self.feature_names),
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_rnn_layers,
            num_classes=len(LOS_LEVELS),
            num_horizons=self.num_horizons,
            rnn_type=self.config.rnn_type,
            gnn_type=self.config.gnn_type,
            num_gnn_layers=self.config.num_gnn_layers,
            gat_heads=self.config.gat_heads,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
            time_embedding_dim=self.config.time_embedding_dim,
            segment_embedding_dim=self.config.segment_embedding_dim,
            segment_vocab_size=self.segment_vocab_size,
        )
    
    def _forward_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass on a graph batch."""
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
        return f"GNN ({self.config.gnn_type.upper()})"
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Add graph-specific info to W&B config."""
        config = super()._get_wandb_config()
        config.update({
            "num_nodes": self.num_nodes,
            "num_edges": self.road_graph.edge_index.shape[1],
            "gnn_type": self.config.gnn_type,
        })
        return config
    
    def _print_model_info(self) -> None:
        """Print model info including graph stats."""
        super()._print_model_info()
        print(f"Graph: {self.num_nodes} nodes, {self.road_graph.edge_index.shape[1]} edges")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN for traffic prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "graph_config.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = GraphTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
