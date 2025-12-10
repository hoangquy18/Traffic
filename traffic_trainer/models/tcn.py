"""
Temporal Convolutional Network (TCN) with Attention for Traffic Prediction.

TCN is a powerful architecture for sequence modeling that uses dilated causal convolutions
to capture long-range temporal dependencies efficiently. Unlike RNNs, TCN can be trained
in parallel and is more stable for long sequences.

Key Features:
1. Dilated Causal Convolutions - exponentially increasing receptive field
2. Residual Connections - stable training for deep networks
3. Temporal Attention - focus on important time steps
4. Multi-scale Feature Extraction - captures patterns at different time scales
5. Spatial Aggregation - optional cross-segment attention

Architecture:
1. Input projection with embeddings (time, segment)
2. Multi-scale TCN blocks with increasing dilation
3. Temporal attention pooling
4. Optional spatial attention across segments
5. Classification head for multi-horizon prediction
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that ensures the output at time t only depends on inputs <= t.
    This is essential for time series prediction to prevent information leakage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, seq_len]"""
        out = self.conv(x)
        # Remove future time steps to maintain causality
        if self.padding > 0:
            out = out[..., : -self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Residual block with two dilated causal convolutions.
    Uses weight normalization and dropout for regularization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # First causal conv
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal conv
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, seq_len]"""
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TemporalAttention(nn.Module):
    """
    Attention mechanism over time steps.
    Learns to focus on the most relevant time steps for prediction.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden_dim]
        Returns: [batch, seq_len, hidden_dim]
        """
        # Self-attention over time
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        return self.norm(x + attn_out)


class SpatialAttention(nn.Module):
    """
    Cross-attention between different segments/nodes.
    Allows the model to learn spatial dependencies without explicit graph structure.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [batch, num_nodes, hidden_dim]
        mask: [batch, num_nodes] - True for valid nodes
        Returns: [batch, num_nodes, hidden_dim]
        """
        # Create attention mask (True = ignore)
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None

        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        attn_out = self.dropout(attn_out)
        return self.norm(x + attn_out)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for traffic prediction.

    The network uses a stack of dilated causal convolutions to efficiently
    capture long-range temporal dependencies while maintaining causality.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_horizons: int = 1,
        num_channels: list[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_temporal_attention: bool = True,
        use_spatial_attention: bool = False,
        num_attention_heads: int = 4,
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_horizons = num_horizons
        self.use_temporal_attention = use_temporal_attention
        self.use_spatial_attention = use_spatial_attention
        self.time_embedding_dim = time_embedding_dim if time_embedding_dim else 0
        self.segment_embedding_dim = segment_embedding_dim if segment_embedding_dim else 0

        # Default channel progression: gradually increase capacity
        if num_channels is None:
            num_channels = [hidden_dim, hidden_dim, hidden_dim * 2, hidden_dim * 2]

        # Time embedding (hour of day)
        if time_embedding_dim and time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(time_vocab_size, time_embedding_dim)
        else:
            self.time_embedding = None

        # Segment embedding (road segment identity)
        if segment_embedding_dim and segment_embedding_dim > 0 and segment_vocab_size is not None:
            self.segment_embedding = nn.Embedding(segment_vocab_size, segment_embedding_dim)
        else:
            self.segment_embedding = None

        # Adjust input dim for embeddings
        feature_dim = input_dim
        if self.time_embedding is not None:
            feature_dim += time_embedding_dim
        if self.segment_embedding is not None:
            feature_dim += segment_embedding_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, num_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # TCN layers with exponentially increasing dilation
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_channels[i - 1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*layers)

        # Temporal attention over time steps
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                num_channels[-1], num_attention_heads, dropout
            )
        else:
            self.temporal_attention = None

        # Spatial attention across segments
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(
                num_channels[-1], num_attention_heads, dropout
            )
        else:
            self.spatial_attention = None

        # Global temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes * num_horizons),
        )

    def forward(
        self,
        node_features: torch.Tensor,  # [batch, num_nodes, seq_len, features]
        mask: Optional[torch.Tensor] = None,  # [batch, num_nodes]
        time_ids: Optional[torch.Tensor] = None,  # [batch, num_nodes, seq_len]
        segment_ids: Optional[torch.Tensor] = None,  # [batch, num_nodes]
        edge_index: Optional[torch.Tensor] = None,  # Not used, for API compatibility
    ) -> torch.Tensor:
        """
        Args:
            node_features: Time series for each node [batch, num_nodes, seq_len, features]
            mask: Valid node mask [batch, num_nodes]
            time_ids: Hour-of-day for each timestep [batch, num_nodes, seq_len]
            segment_ids: Segment index for each node [batch, num_nodes]
            edge_index: Not used (for API compatibility with GNN models)

        Returns:
            Logits [batch, num_nodes, num_horizons, num_classes]
        """
        batch_size, num_nodes, seq_len, num_features = node_features.shape

        # Start with base features
        features = node_features

        # Add time embeddings if available
        if self.time_embedding is not None and time_ids is not None:
            time_embed = self.time_embedding(time_ids)  # [batch, num_nodes, seq_len, time_dim]
            features = torch.cat([features, time_embed], dim=-1)

        # Add segment embeddings if available
        if self.segment_embedding is not None and segment_ids is not None:
            seg_embed = self.segment_embedding(segment_ids)  # [batch, num_nodes, seg_dim]
            seg_embed = seg_embed.unsqueeze(2).expand(-1, -1, seq_len, -1)
            features = torch.cat([features, seg_embed], dim=-1)

        # Input projection: [batch, num_nodes, seq_len, features] -> [batch, num_nodes, seq_len, channels]
        x = self.input_proj(features)

        # Reshape for TCN: [batch * num_nodes, channels, seq_len]
        x = x.view(batch_size * num_nodes, seq_len, -1).transpose(1, 2)

        # Apply TCN layers
        x = self.tcn(x)  # [batch * num_nodes, channels, seq_len]

        # Reshape back: [batch, num_nodes, channels, seq_len]
        x = x.view(batch_size, num_nodes, -1, seq_len)

        # Temporal attention (optional)
        if self.temporal_attention is not None:
            # Reshape for attention: [batch * num_nodes, seq_len, channels]
            x_attn = x.transpose(2, 3).contiguous().view(batch_size * num_nodes, seq_len, -1)
            x_attn = self.temporal_attention(x_attn)
            # Reshape back
            x = x_attn.view(batch_size, num_nodes, seq_len, -1).transpose(2, 3)

        # Temporal pooling: aggregate over time
        # [batch, num_nodes, channels, seq_len] -> [batch, num_nodes, channels, 1]
        x = x.view(batch_size * num_nodes, -1, seq_len)
        x = self.temporal_pool(x).squeeze(-1)
        x = x.view(batch_size, num_nodes, -1)  # [batch, num_nodes, channels]

        # Spatial attention (optional)
        if self.spatial_attention is not None:
            x = self.spatial_attention(x, mask)

        # Classification
        logits = self.classifier(x)  # [batch, num_nodes, num_classes * num_horizons]
        logits = logits.view(batch_size, num_nodes, self.num_horizons, self.num_classes)

        return logits


class MultiScaleTCN(nn.Module):
    """
    Multi-scale TCN that processes the input at different temporal resolutions
    and combines the results. This captures both fine-grained and coarse-grained patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_horizons: int = 1,
        scales: list[int] = None,  # Different kernel sizes for multi-scale
        dropout: float = 0.2,
        use_temporal_attention: bool = True,
        num_attention_heads: int = 4,
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_horizons = num_horizons
        self.time_embedding_dim = time_embedding_dim if time_embedding_dim else 0
        self.segment_embedding_dim = segment_embedding_dim if segment_embedding_dim else 0

        if scales is None:
            scales = [3, 5, 7]  # Different kernel sizes

        # Time embedding
        if time_embedding_dim and time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(time_vocab_size, time_embedding_dim)
        else:
            self.time_embedding = None

        # Segment embedding
        if segment_embedding_dim and segment_embedding_dim > 0 and segment_vocab_size is not None:
            self.segment_embedding = nn.Embedding(segment_vocab_size, segment_embedding_dim)
        else:
            self.segment_embedding = None

        # Adjust input dim
        feature_dim = input_dim
        if self.time_embedding is not None:
            feature_dim += time_embedding_dim
        if self.segment_embedding is not None:
            feature_dim += segment_embedding_dim

        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Multiple TCN branches with different scales
        self.tcn_branches = nn.ModuleList()
        for kernel_size in scales:
            branch = nn.Sequential(
                TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation=1, dropout=dropout),
                TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation=2, dropout=dropout),
                TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation=4, dropout=dropout),
            )
            self.tcn_branches.append(branch)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * len(scales), hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal attention
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                hidden_dim, num_attention_heads, dropout
            )
        else:
            self.temporal_attention = None

        # Pooling and classification
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes * num_horizons),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_nodes, seq_len, num_features = node_features.shape

        # Add embeddings
        features = node_features
        if self.time_embedding is not None and time_ids is not None:
            time_embed = self.time_embedding(time_ids)
            features = torch.cat([features, time_embed], dim=-1)
        if self.segment_embedding is not None and segment_ids is not None:
            seg_embed = self.segment_embedding(segment_ids)
            seg_embed = seg_embed.unsqueeze(2).expand(-1, -1, seq_len, -1)
            features = torch.cat([features, seg_embed], dim=-1)

        # Input projection
        x = self.input_proj(features)
        x = x.view(batch_size * num_nodes, seq_len, -1).transpose(1, 2)

        # Multi-scale processing
        branch_outputs = []
        for branch in self.tcn_branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)

        # Concatenate and fuse
        x = torch.cat(branch_outputs, dim=1)
        x = self.fusion(x)

        # Temporal attention
        if self.temporal_attention is not None:
            x = x.transpose(1, 2)  # [batch * nodes, seq_len, channels]
            x = self.temporal_attention(x)
            x = x.transpose(1, 2)  # [batch * nodes, channels, seq_len]

        # Temporal pooling
        x = self.temporal_pool(x).squeeze(-1)
        x = x.view(batch_size, num_nodes, -1)

        # Classification
        logits = self.classifier(x)
        logits = logits.view(batch_size, num_nodes, self.num_horizons, self.num_classes)

        return logits


def create_tcn_model(
    model_type: Literal["standard", "multiscale"] = "standard",
    input_dim: int = 1,
    hidden_dim: int = 64,
    num_classes: int = 4,
    num_horizons: int = 1,
    num_channels: Optional[list[int]] = None,
    kernel_size: int = 3,
    dropout: float = 0.2,
    use_temporal_attention: bool = True,
    use_spatial_attention: bool = False,
    num_attention_heads: int = 4,
    time_embedding_dim: Optional[int] = None,
    segment_embedding_dim: Optional[int] = None,
    segment_vocab_size: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a TCN model.

    Args:
        model_type: "standard" or "multiscale"
        input_dim: Number of input features per time step
        hidden_dim: Hidden dimension size
        num_classes: Number of output classes (e.g., LOS levels)
        num_horizons: Number of future time steps to predict
        num_channels: Channel sizes for each TCN layer (standard only)
        kernel_size: Convolution kernel size (standard only)
        dropout: Dropout rate
        use_temporal_attention: Whether to use temporal attention
        use_spatial_attention: Whether to use spatial attention (standard only)
        num_attention_heads: Number of attention heads
        time_embedding_dim: Dimension of time embedding (hour of day)
        segment_embedding_dim: Dimension of segment embedding
        segment_vocab_size: Number of unique segments

    Returns:
        TCN model instance
    """
    if model_type == "multiscale":
        return MultiScaleTCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_horizons=num_horizons,
            scales=kwargs.get("scales", [3, 5, 7]),
            dropout=dropout,
            use_temporal_attention=use_temporal_attention,
            num_attention_heads=num_attention_heads,
            time_embedding_dim=time_embedding_dim,
            segment_embedding_dim=segment_embedding_dim,
            segment_vocab_size=segment_vocab_size,
        )
    else:
        return TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_horizons=num_horizons,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_temporal_attention=use_temporal_attention,
            use_spatial_attention=use_spatial_attention,
            num_attention_heads=num_attention_heads,
            time_embedding_dim=time_embedding_dim,
            segment_embedding_dim=segment_embedding_dim,
            segment_vocab_size=segment_vocab_size,
        )



