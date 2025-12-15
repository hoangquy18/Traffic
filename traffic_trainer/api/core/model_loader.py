"""Model loading utilities for different model types."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from traffic_trainer.models import GMAN, SpatioTemporalTransformer, create_sota_model


SUPPORTED_MODELS = [
    "gman",
    "transformer",
    "spatio_temporal_transformer",
    "stgcn",
    "astgcn",
    "gwnet",
    "mtgnn",
    "dcrnn",
]


class ModelLoader:
    """Handles loading models and related data."""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}
        self.model_type: str = "gman"
        self.num_nodes: int = 0
        self.num_horizons: int = 1
        self.sequence_length: int = 8
        self.prediction_horizons: list = [1, 2, 3]
        self.feature_names: list = []
        self.numerical_features: list = []
        self.categorical_features: list = []
        self.segment_to_idx: Dict[int, int] = {}
        self.idx_to_segment: Dict[int, int] = {}
        self.street_info: Dict[int, Dict[str, Any]] = {}
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None
        self.scaler = None
        self._input_dim: Optional[int] = None  # Store from checkpoint

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        config_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        adjacency_path: Optional[Path] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load model from checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Load config
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f) or {}
        elif "config" in checkpoint:
            self.config = checkpoint.get("config") or {}

        self._parse_config()
        self._load_metadata(metadata_path, checkpoint)
        self._load_adjacency(adjacency_path, checkpoint)

        if "scaler" in checkpoint:
            self.scaler = checkpoint["scaler"]
        if "feature_names" in checkpoint:
            self.feature_names = checkpoint["feature_names"]

        # Extract input_dim from state_dict BEFORE creating model
        self._extract_input_dim(checkpoint)

        model = self._create_model()
        self._load_weights(model, checkpoint)
        model.to(device)
        model.eval()

        return model, checkpoint

    def _extract_input_dim(self, checkpoint: Dict) -> None:
        """Extract input_dim from checkpoint state_dict."""
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )

        if not isinstance(state_dict, dict):
            return

        # Common keys for input projection layer
        for key in [
            "input_proj.0.weight",
            "input_projection.weight",
            "fc_in.weight",
            "encoder.0.weight",
        ]:
            if key in state_dict:
                self._input_dim = state_dict[key].shape[1]
                print(f"Input dim from checkpoint: {self._input_dim}")
                return

    def _load_weights(self, model: nn.Module, checkpoint: Dict) -> None:
        """Load model weights from checkpoint."""
        state_dict = None
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if state_dict:
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

    def _parse_config(self) -> None:
        """Parse config to extract model parameters."""
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        self.numerical_features = data_cfg.get("numerical_features", [])
        self.categorical_features = data_cfg.get("categorical_features", [])
        self.sequence_length = data_cfg.get("sequence_length", 8)
        self.prediction_horizons = data_cfg.get("prediction_horizons", [1, 2, 3])
        self.num_horizons = len(self.prediction_horizons)
        self.model_type = model_cfg.get("type", "gman").lower()

    def _load_metadata(self, path: Optional[Path], checkpoint: Dict) -> None:
        """Load segment mappings."""
        metadata = None
        if path and path.exists():
            with open(path) as f:
                metadata = json.load(f)
        elif "metadata" in checkpoint:
            metadata = checkpoint["metadata"]

        if metadata:
            self.segment_to_idx = {
                int(k): v for k, v in metadata.get("segment_to_idx", {}).items()
            }
            self.idx_to_segment = {v: int(k) for k, v in self.segment_to_idx.items()}
            self.num_nodes = metadata.get("num_nodes", len(self.segment_to_idx))

    def _load_adjacency(self, path: Optional[Path], checkpoint: Dict) -> None:
        """Load adjacency matrix."""
        if path and path.exists():
            if path.suffix == ".npz":
                data = np.load(path)
                self.adjacency_matrix = data.get("adjacency", data[data.files[0]])
            elif path.suffix == ".npy":
                self.adjacency_matrix = np.load(path)
            if self.adjacency_matrix is not None:
                self._build_edge_index()
        elif "adjacency_matrix" in checkpoint:
            self.adjacency_matrix = checkpoint["adjacency_matrix"]
            self._build_edge_index()
        elif "edge_index" in checkpoint:
            self.edge_index = checkpoint["edge_index"]
            self.edge_weight = checkpoint.get("edge_weight")

    def _build_edge_index(self) -> None:
        """Build edge_index from adjacency matrix."""
        if self.adjacency_matrix is None:
            return
        rows, cols = np.where(self.adjacency_matrix > 0)
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long)
        self.edge_weight = torch.tensor(
            self.adjacency_matrix[rows, cols], dtype=torch.float32
        )

    def get_input_dim(self) -> int:
        """Get input dimension (from checkpoint or config)."""
        # Priority: stored from checkpoint > config > feature_names > calculate
        if self._input_dim is not None:
            return self._input_dim

        # Try from config
        model_cfg = self.config.get("model", {})
        if "input_dim" in model_cfg:
            return model_cfg["input_dim"]

        data_cfg = self.config.get("data", {})
        if "input_dim" in data_cfg:
            return data_cfg["input_dim"]

        # From feature names
        if self.feature_names:
            return len(self.feature_names)

        # Calculate from features
        dim = len(self.numerical_features)
        for cat in self.categorical_features:
            dim += {"weekday": 7, "period": 4}.get(cat, 5)
        return dim if dim > 0 else 50

    def _create_model(self) -> nn.Module:
        """Create model instance."""
        cfg = self.config.get("model", {})
        input_dim = self.get_input_dim()
        hidden_dim = cfg.get("hidden_dim", 256)
        num_layers = cfg.get("num_layers", 3)
        dropout = cfg.get("dropout", 0.1)
        num_heads = cfg.get("num_heads", 4)

        if self.model_type == "gman":
            return GMAN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=6,
                num_nodes=self.num_nodes,
                num_layers=num_layers,
                num_heads=num_heads,
                num_horizons=self.num_horizons,
                dropout=dropout,
                use_spatial_embedding=True,
                use_temporal_conv=True,
                segment_vocab_size=self.num_nodes,
            )
        elif self.model_type in ["transformer", "spatio_temporal_transformer"]:
            return SpatioTemporalTransformer(
                num_segments=self.num_nodes,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=6,
                num_layers=num_layers,
                num_heads=num_heads,
                num_horizons=self.num_horizons,
                dropout=dropout,
                sequence_length=self.sequence_length,
            )
        elif self.model_type in ["stgcn", "astgcn", "gwnet", "mtgnn", "dcrnn"]:
            if self.edge_index is None:
                self._create_fully_connected_graph()
            return create_sota_model(
                model_type=self.model_type,
                num_nodes=self.num_nodes,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=6,
                num_layers=num_layers,
                num_horizons=self.num_horizons,
                dropout=dropout,
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
                sequence_length=self.sequence_length,
            )
        raise ValueError(f"Unsupported model: {self.model_type}")

    def _create_fully_connected_graph(self) -> None:
        """Create fully connected graph."""
        n = self.num_nodes
        rows = [i for i in range(n) for j in range(n) if i != j]
        cols = [j for i in range(n) for j in range(n) if i != j]
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long)
        self.edge_weight = torch.ones(len(rows))

    def load_street_info(self, csv_path: Path) -> None:
        """Load street information from CSV."""
        df = pd.read_csv(csv_path)
        for _, row in df.drop_duplicates(subset=["segment_id"]).iterrows():
            seg_id = int(row["segment_id"])
            self.street_info[seg_id] = {
                "street_id": int(row.get("street_id", seg_id)),
                "street_name": row.get("street_name", "Unknown"),
                "street_type": row.get("street_type", "Unknown"),
                "street_level": (
                    int(row["street_level"])
                    if pd.notna(row.get("street_level"))
                    else None
                ),
                "length": float(row["length"]) if pd.notna(row.get("length")) else None,
                "max_velocity": (
                    float(row["max_velocity"])
                    if pd.notna(row.get("max_velocity"))
                    else None
                ),
            }
            if seg_id not in self.segment_to_idx:
                idx = len(self.segment_to_idx)
                self.segment_to_idx[seg_id] = idx
                self.idx_to_segment[idx] = seg_id
        if not self.num_nodes:
            self.num_nodes = len(self.segment_to_idx)
