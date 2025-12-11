"""Inference utilities for traffic prediction."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from traffic_trainer.data.constants import LOS_LEVELS


# Reverse mapping: index -> LOS class
IDX_TO_LOS = {v: k for k, v in LOS_LEVELS.items()}


class InferenceEngine:
    """Handles model inference for predictions."""

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        num_nodes: int,
        sequence_length: int,
        input_dim: int,
        prediction_horizons: List[int],
        device: torch.device,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.prediction_horizons = prediction_horizons
        self.device = device
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    @torch.no_grad()
    def predict(
        self,
        features: Optional[torch.Tensor] = None,
        segment_ids: Optional[List[int]] = None,
        segment_to_idx: Optional[Dict[int, int]] = None,
        idx_to_segment: Optional[Dict[int, int]] = None,
        return_probabilities: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """Run prediction for specified segments."""
        if features is None:
            features = torch.randn(
                1, self.num_nodes, self.sequence_length, self.input_dim
            )

        features = features.to(self.device)
        logits = self._run_model(features)

        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        results = {}

        if segment_to_idx is None:
            segment_to_idx = {i: i for i in range(self.num_nodes)}
        if idx_to_segment is None:
            idx_to_segment = {i: i for i in range(self.num_nodes)}

        target_segments = segment_ids if segment_ids else list(idx_to_segment.values())

        for seg_id in target_segments:
            if seg_id not in segment_to_idx:
                continue

            node_idx = segment_to_idx[seg_id]
            seg_results = []

            for h_idx, horizon in enumerate(self.prediction_horizons):
                pred_idx = predictions[0, node_idx, h_idx].item()
                pred_probs = probs[0, node_idx, h_idx].cpu().numpy()

                horizon_result = {
                    "horizon": horizon,
                    "los_class": IDX_TO_LOS[pred_idx],
                    "los_index": pred_idx,
                    "confidence": float(pred_probs[pred_idx]),
                }

                if return_probabilities:
                    horizon_result["probabilities"] = {
                        IDX_TO_LOS[i]: float(p) for i, p in enumerate(pred_probs)
                    }

                seg_results.append(horizon_result)

            results[seg_id] = {"predictions": seg_results}

        return results

    def _run_model(self, features: torch.Tensor) -> torch.Tensor:
        """Run model inference based on model type."""
        if self.model_type in ["transformer", "spatio_temporal_transformer", "gman"]:
            segment_indices = torch.arange(self.num_nodes).unsqueeze(0).to(self.device)
            time_ids = torch.zeros(
                1, self.num_nodes, self.sequence_length, dtype=torch.long
            ).to(self.device)

            return self.model(
                node_features=features,
                segment_ids=segment_indices,
                time_ids=time_ids,
            )

        elif self.model_type in ["stgcn", "astgcn", "gwnet", "mtgnn", "dcrnn"]:
            edge_index = (
                self.edge_index.to(self.device) if self.edge_index is not None else None
            )
            edge_weight = (
                self.edge_weight.to(self.device)
                if self.edge_weight is not None
                else None
            )

            return self.model(
                x=features,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

        raise ValueError(f"Unsupported model type: {self.model_type}")


class FeaturePreprocessor:
    """Preprocess features for model input."""

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        scaler=None,
    ) -> None:
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scaler = scaler

    def preprocess(
        self,
        weather_data: Dict[str, float],
        segment_info: Dict[str, Any],
        weekday: int,
        period: str,
    ) -> np.ndarray:
        """Preprocess features for a single segment/timestamp."""
        features = {
            "length": segment_info.get("length", 1000.0),
            "max_velocity": segment_info.get("max_velocity", 30.0),
        }

        for feat in self.numerical_features:
            if feat in weather_data:
                features[feat] = weather_data[feat]
            elif feat not in features:
                features[feat] = 0.0

        numerical = np.array([features.get(f, 0.0) for f in self.numerical_features])

        categorical_arrays = []

        for cat_feat in self.categorical_features:
            if cat_feat == "weekday":
                one_hot = np.zeros(7)
                one_hot[weekday] = 1.0
                categorical_arrays.append(one_hot)

            elif cat_feat == "period":
                periods = [
                    "period_00_06",
                    "period_06_12",
                    "period_12_18",
                    "period_18_00",
                ]
                one_hot = np.zeros(len(periods))
                for i, p in enumerate(periods):
                    if p in period or period in p:
                        one_hot[i] = 1.0
                        break
                categorical_arrays.append(one_hot)

            elif cat_feat == "street_type":
                types = [
                    "Đường chính",
                    "Đường phụ",
                    "Đường quan trọng",
                    "Đường nhánh",
                    "Other",
                ]
                one_hot = np.zeros(len(types))
                st_type = segment_info.get("street_type", "Other")
                for i, t in enumerate(types):
                    if t == st_type:
                        one_hot[i] = 1.0
                        break
                categorical_arrays.append(one_hot)

            elif cat_feat == "street_level":
                one_hot = np.zeros(5)
                level = segment_info.get("street_level", 0)
                if level and 1 <= level <= 5:
                    one_hot[level - 1] = 1.0
                categorical_arrays.append(one_hot)

        if categorical_arrays:
            categorical = np.concatenate(categorical_arrays)
            feature_vector = np.concatenate([numerical, categorical])
        else:
            feature_vector = numerical

        if self.scaler is not None:
            feature_vector = (feature_vector - self.scaler.mean_) / self.scaler.scale_

        return feature_vector.astype(np.float32)
