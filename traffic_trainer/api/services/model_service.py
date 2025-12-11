"""Model service for loading and running inference on trained models."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from traffic_trainer.api.core import (
    ModelLoader,
    CorrelationExtractor,
    InferenceEngine,
    FeaturePreprocessor,
    data_service,
)


class ModelService:
    """Service for loading trained models and running inference."""

    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.device: torch.device = torch.device("cpu")

        # Sub-components
        self.loader: Optional[ModelLoader] = None
        self.correlator: Optional[CorrelationExtractor] = None
        self.inference: Optional[InferenceEngine] = None
        self.preprocessor: Optional[FeaturePreprocessor] = None

    # ==================== Properties ====================

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_type(self) -> Optional[str]:
        return self.loader.model_type if self.loader else None

    @property
    def num_nodes(self) -> int:
        return self.loader.num_nodes if self.loader else 0

    @property
    def segment_to_idx(self) -> Dict[int, int]:
        return self.loader.segment_to_idx if self.loader else {}

    @property
    def idx_to_segment(self) -> Dict[int, int]:
        return self.loader.idx_to_segment if self.loader else {}

    @property
    def street_info(self) -> Dict[int, Dict[str, Any]]:
        return self.loader.street_info if self.loader else {}

    # ==================== Load Model ====================

    def load_model(
        self,
        checkpoint_path: Path,
        config_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        street_info_path: Optional[Path] = None,
        adjacency_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        """Load a trained model from checkpoint."""
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )

        self.loader = ModelLoader()
        self.model, _ = self.loader.load_checkpoint(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            metadata_path=metadata_path,
            adjacency_path=adjacency_path,
            device=self.device,
        )

        if street_info_path and street_info_path.exists():
            self.loader.load_street_info(street_info_path)

        # Load data for real features
        if data_path and data_path.exists():
            data_service.load_data(data_path)
            # Set expected input dim for padding
            data_service.expected_input_dim = self.loader.get_input_dim()
        elif street_info_path and street_info_path.exists():
            # Try using street_info_path as data source
            data_service.load_data(street_info_path)
            data_service.expected_input_dim = self.loader.get_input_dim()

        self._init_components()

        print(
            f"Model loaded: {self.model_type}, "
            f"{self.num_nodes} segments, device={self.device}"
        )

    def _init_components(self) -> None:
        """Initialize sub-components."""
        input_dim = self.loader.get_input_dim()

        self.correlator = CorrelationExtractor(
            model=self.model,
            model_type=self.loader.model_type,
            num_nodes=self.loader.num_nodes,
            sequence_length=self.loader.sequence_length,
            input_dim=input_dim,
            device=self.device,
            adjacency_matrix=self.loader.adjacency_matrix,
            edge_index=self.loader.edge_index,
            edge_weight=self.loader.edge_weight,
        )

        self.inference = InferenceEngine(
            model=self.model,
            model_type=self.loader.model_type,
            num_nodes=self.loader.num_nodes,
            sequence_length=self.loader.sequence_length,
            input_dim=input_dim,
            prediction_horizons=self.loader.prediction_horizons,
            device=self.device,
            edge_index=self.loader.edge_index,
            edge_weight=self.loader.edge_weight,
        )

        self.preprocessor = FeaturePreprocessor(
            numerical_features=self.loader.numerical_features,
            categorical_features=self.loader.categorical_features,
            scaler=self.loader.scaler,
        )

    # ==================== Street Info ====================

    def get_street_name(self, segment_id: int) -> Optional[str]:
        """Get street name for a segment ID."""
        info = self.street_info.get(segment_id)
        return info.get("street_name") if info else None

    def get_segment_id_by_name(self, street_name: str) -> List[int]:
        """Get segment IDs for a street name."""
        return [
            seg_id
            for seg_id, info in self.street_info.items()
            if info.get("street_name", "").lower() == street_name.lower()
        ]

    def get_all_streets(self) -> List[Dict[str, Any]]:
        """Get information about all available streets."""
        return [
            {
                "segment_id": seg_id,
                "street_id": info.get("street_id", seg_id),
                "street_name": info.get("street_name", "Unknown"),
                "street_type": info.get("street_type", "Unknown"),
                "street_level": info.get("street_level"),
            }
            for seg_id, info in self.street_info.items()
        ]

    # ==================== Prediction ====================

    @torch.no_grad()
    def predict(
        self,
        segment_ids: Optional[List[int]] = None,
        target_time: Optional[datetime] = None,
        features: Optional[torch.Tensor] = None,
        return_probabilities: bool = True,
    ) -> Dict[str, Any]:
        """
        Run prediction for specified segments.

        Args:
            segment_ids: List of segment IDs to predict
            target_time: Target datetime for prediction
            features: Pre-computed features (optional, will load from data if None)
            return_probabilities: Include probability distribution
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if target_time is None:
            target_time = datetime.now()

        # Get features from data service if not provided
        if features is None and data_service.df is not None:
            features_np = data_service.get_all_segments_sequence(
                target_time=target_time,
                sequence_length=self.loader.sequence_length,
                segment_to_idx=self.segment_to_idx,
            )
            features = torch.from_numpy(features_np).unsqueeze(
                0
            )  # [1, num_nodes, seq_len, features]

        raw_results = self.inference.predict(
            features=features,
            segment_ids=segment_ids,
            segment_to_idx=self.segment_to_idx,
            idx_to_segment=self.idx_to_segment,
            return_probabilities=return_probabilities,
        )

        # Enrich with street info
        return {
            seg_id: {
                "segment_id": seg_id,
                "street_name": self.get_street_name(seg_id),
                "street_type": self.street_info.get(seg_id, {}).get("street_type"),
                "predictions": data["predictions"],
            }
            for seg_id, data in raw_results.items()
        }

    # ==================== Correlation ====================

    @torch.no_grad()
    def get_spatial_attention(self) -> torch.Tensor:
        """Get spatial attention/correlation matrix."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.correlator.get_correlation_matrix("spatial_attention")

    def get_street_correlations(
        self,
        segment_id: int,
        top_k: int = 10,
        include_self: bool = False,
        correlation_type: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Get top correlated streets for a segment."""
        if segment_id not in self.segment_to_idx:
            return []

        node_idx = self.segment_to_idx[segment_id]

        if correlation_type == "auto":
            correlation_type = self.correlator.get_auto_type()

        corr_matrix = self.correlator.get_correlation_matrix(correlation_type)
        correlations = corr_matrix[node_idx].numpy()

        if not include_self:
            correlations[node_idx] = -np.inf

        top_indices = np.argsort(correlations)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if correlations[idx] == -np.inf:
                continue

            target_seg_id = self.idx_to_segment.get(idx, -1)
            if target_seg_id == -1:
                continue

            results.append(
                {
                    "segment_id": target_seg_id,
                    "street_name": self.get_street_name(target_seg_id),
                    "street_type": self.street_info.get(target_seg_id, {}).get(
                        "street_type"
                    ),
                    "correlation_score": float(correlations[idx]),
                    "relationship_type": correlation_type,
                    "model_type": self.model_type,
                }
            )

        return results

    def get_full_correlation_matrix(
        self, correlation_type: str = "auto"
    ) -> Dict[str, Any]:
        """Get full correlation matrix with metadata."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if correlation_type == "auto":
            correlation_type = self.correlator.get_auto_type()

        matrix = self.correlator.get_correlation_matrix(correlation_type)

        return {
            "matrix": matrix.numpy().tolist(),
            "shape": list(matrix.shape),
            "correlation_type": correlation_type,
            "model_type": self.model_type,
            "segment_ids": [
                self.idx_to_segment.get(i, i) for i in range(self.num_nodes)
            ],
        }

    # ==================== Info ====================

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_type": self.model_type,
            "num_nodes": self.num_nodes,
            "num_horizons": self.loader.num_horizons,
            "sequence_length": self.loader.sequence_length,
            "prediction_horizons": self.loader.prediction_horizons,
            "input_dim": self.loader.get_input_dim(),
            "device": str(self.device),
            "has_adjacency": self.loader.adjacency_matrix is not None,
            "has_edge_index": self.loader.edge_index is not None,
            "num_streets_loaded": len(self.street_info),
            "supported_correlation_types": self.correlator.get_supported_types(),
        }

    def preprocess_features(
        self,
        weather_data: Dict[str, float],
        segment_id: int,
        weekday: int,
        period: str,
    ) -> np.ndarray:
        """Preprocess features for prediction."""
        seg_info = self.street_info.get(segment_id, {})
        return self.preprocessor.preprocess(weather_data, seg_info, weekday, period)


# Global model service instance
model_service = ModelService()
