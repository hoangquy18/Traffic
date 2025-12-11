"""Correlation and attention extraction utilities."""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrelationExtractor:
    """Extract correlation/attention weights from different model types."""

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        num_nodes: int,
        sequence_length: int,
        input_dim: int,
        device: torch.device,
        adjacency_matrix: Optional[np.ndarray] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.device = device
        self.adjacency_matrix = adjacency_matrix
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        # Cache correlation matrices for consistency
        self._correlation_cache: Dict[str, torch.Tensor] = {}

    def get_supported_types(self) -> List[str]:
        """Get supported correlation types for current model."""
        types = ["auto"]

        if self.model_type in ["transformer", "spatio_temporal_transformer"]:
            types.append("spatial_attention")

        if self.model_type == "gman":
            types.extend(["spatial_attention", "graph", "combined"])

        if self.model_type in ["stgcn", "astgcn", "gwnet", "mtgnn", "dcrnn"]:
            types.append("graph")

        if self.model_type in ["gwnet", "mtgnn"]:
            types.append("learned_graph")

        return types

    def get_auto_type(self) -> str:
        """Get default correlation type for model."""
        # Always prefer graph-based correlation when adjacency is available
        # because attention-based requires real features (not random)
        has_graph = self.adjacency_matrix is not None or self.edge_index is not None

        if has_graph:
            # Use graph for all models when available
            if self.model_type in ["gwnet", "mtgnn"]:
                return "learned_graph"  # These models learn their own graph
            return "graph"

        # Fallback to attention only if no graph available
        # Note: attention with random features gives inconsistent results
        return "spatial_attention"

    @torch.no_grad()
    def get_correlation_matrix(self, correlation_type: str = "auto") -> torch.Tensor:
        """Get full correlation matrix (cached for consistency)."""
        if correlation_type == "auto":
            correlation_type = self.get_auto_type()

        # Return cached result if available
        if correlation_type in self._correlation_cache:
            return self._correlation_cache[correlation_type]

        if correlation_type == "graph":
            result = self._get_graph_correlation()
        elif correlation_type == "combined":
            result = self._get_combined_correlation()
        elif correlation_type == "learned_graph":
            result = self._get_learned_graph_matrix()
        else:  # spatial_attention
            result = self._get_attention_correlation()

        # Cache the result
        self._correlation_cache[correlation_type] = result
        return result

    def _get_graph_correlation(self) -> torch.Tensor:
        """Get correlation from graph adjacency matrix."""
        if self.adjacency_matrix is not None:
            adj = self.adjacency_matrix.copy()
            np.fill_diagonal(adj, 1.0)
            row_sum = adj.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            return torch.tensor(adj / row_sum, dtype=torch.float32)

        if self.edge_index is not None:
            adj = np.zeros((self.num_nodes, self.num_nodes))
            edge_np = self.edge_index.cpu().numpy()

            if self.edge_weight is not None:
                weights = self.edge_weight.cpu().numpy()
                for i, (s, d) in enumerate(zip(edge_np[0], edge_np[1])):
                    adj[s, d] = weights[i]
            else:
                for s, d in zip(edge_np[0], edge_np[1]):
                    adj[s, d] = 1.0

            np.fill_diagonal(adj, 1.0)
            row_sum = adj.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            return torch.tensor(adj / row_sum, dtype=torch.float32)

        return torch.ones(self.num_nodes, self.num_nodes) / self.num_nodes

    def _get_attention_correlation(self) -> torch.Tensor:
        """Get correlation from attention weights using deterministic features."""
        # Create deterministic features based on node indices
        # This ensures consistent results across calls
        features = torch.zeros(
            1,
            self.num_nodes,
            self.sequence_length,
            self.input_dim,
            dtype=torch.float32,
            device=self.device,
        )
        # Add node-specific values for differentiation
        for i in range(self.num_nodes):
            features[0, i, :, :] = (i + 1) / (self.num_nodes + 1)

        if self.model_type in ["transformer", "spatio_temporal_transformer"]:
            attn = self._extract_transformer_attention(features)
        elif self.model_type == "gman":
            attn = self._extract_gman_attention(features)
        else:
            attn = None

        if attn is not None:
            return attn.cpu()

        # Fallback: return identity-like matrix (self-correlation highest)
        identity = torch.eye(self.num_nodes, dtype=torch.float32)
        uniform = torch.ones(self.num_nodes, self.num_nodes) / self.num_nodes
        return 0.5 * identity + 0.5 * uniform

    def _get_combined_correlation(self) -> torch.Tensor:
        """Get combined correlation (graph + attention) for GMAN."""
        graph_corr = self._get_graph_correlation()

        try:
            # Use deterministic features (same as _get_attention_correlation)
            features = torch.zeros(
                1,
                self.num_nodes,
                self.sequence_length,
                self.input_dim,
                dtype=torch.float32,
                device=self.device,
            )
            for i in range(self.num_nodes):
                features[0, i, :, :] = (i + 1) / (self.num_nodes + 1)

            attn_corr = self._extract_gman_attention(features)

            if attn_corr is not None:
                combined = 0.7 * graph_corr + 0.3 * attn_corr.cpu()
                combined = combined / combined.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                return combined
        except Exception as e:
            print(f"Could not get attention, using graph only: {e}")

        return graph_corr

    def _get_learned_graph_matrix(self) -> torch.Tensor:
        """Get learned graph from GWNET/MTGNN."""
        try:
            if hasattr(self.model, "gc") and hasattr(self.model.gc, "adp"):
                adp = self.model.gc.adp
                if adp is not None:
                    return adp.detach().cpu()

            if hasattr(self.model, "adp"):
                adp = self.model.adp
                if adp is not None:
                    return adp.detach().cpu()
        except Exception as e:
            print(f"Error getting learned graph: {e}")

        return self._get_graph_correlation()

    def _extract_transformer_attention(
        self, features: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention from SpatioTemporalTransformer."""
        try:
            if not hasattr(self.model, "encoder_blocks"):
                return None
            if len(self.model.encoder_blocks) == 0:
                return None

            block = self.model.encoder_blocks[0]
            if not hasattr(block, "spatial_attn"):
                return None

            x = features

            if (
                hasattr(self.model, "segment_id_embedding")
                and self.model.segment_id_embedding is not None
            ):
                seg_ids = torch.arange(self.num_nodes).unsqueeze(0).to(self.device)
                seg_embed = self.model.segment_id_embedding(seg_ids)
                seg_embed = seg_embed.unsqueeze(2).expand(
                    -1, -1, self.sequence_length, -1
                )
                x = torch.cat([x, seg_embed], dim=-1)

            x = self.model.input_proj(x)

            if self.model.spatial_embed is not None:
                x = x + self.model.spatial_embed[:, : self.num_nodes, :, :]

            x = x + self.model.temporal_embed[:, :, : self.sequence_length, :]

            if self.model.use_temporal_conv:
                x = x + self.model.temporal_conv(x)

            x_t = block.norm2(x)[:, :, -1, :]

            spatial_attn = block.spatial_attn
            batch_size, num_nodes, hidden = x_t.shape

            qkv = spatial_attn.qkv(x_t).reshape(
                batch_size, num_nodes, 3, spatial_attn.num_heads, spatial_attn.head_dim
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * spatial_attn.scale
            attn = F.softmax(attn, dim=-1)

            return attn.mean(dim=(0, 1))

        except Exception as e:
            print(f"Error extracting transformer attention: {e}")
            return None

    def _extract_gman_attention(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract attention from GMAN."""
        try:
            if not hasattr(self.model, "encoder_blocks"):
                return None
            if len(self.model.encoder_blocks) == 0:
                return None

            block = self.model.encoder_blocks[0]
            if not hasattr(block, "spatial_attn"):
                return None

            x = features

            if (
                hasattr(self.model, "segment_id_embedding")
                and self.model.segment_id_embedding is not None
            ):
                seg_ids = torch.arange(self.num_nodes).unsqueeze(0).to(self.device)
                seg_embed = self.model.segment_id_embedding(seg_ids)
                seg_embed = seg_embed.unsqueeze(2).expand(
                    -1, -1, self.sequence_length, -1
                )
                x = torch.cat([x, seg_embed], dim=-1)

            x = self.model.input_proj(x)

            if (
                hasattr(self.model, "spatial_embed")
                and self.model.spatial_embed is not None
            ):
                x = x + self.model.spatial_embed[:, : self.num_nodes, :, :]

            if hasattr(self.model, "temporal_embed"):
                x = x + self.model.temporal_embed[:, :, : self.sequence_length, :]

            x_t = block.norm2(x)[:, :, -1, :]

            spatial_attn = block.spatial_attn
            batch_size, num_nodes, hidden = x_t.shape

            qkv = spatial_attn.qkv(x_t).reshape(
                batch_size, num_nodes, 3, spatial_attn.num_heads, spatial_attn.head_dim
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * spatial_attn.scale
            attn = F.softmax(attn, dim=-1)
            print("Extracted GMAN attention successfully")
            return attn.mean(dim=(0, 1))

        except Exception as e:
            print(f"Error extracting GMAN attention: {e}")
            return None
