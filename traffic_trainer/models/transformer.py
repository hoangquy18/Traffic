"""
Spatio-Temporal Transformer for Traffic Prediction.

This model learns relationships between road segments automatically via self-attention,
without requiring explicit graph topology (no need for s_node_id, e_node_id).

Architecture:
1. Temporal Encoder (per-segment): GRU/LSTM processes each segment's time series
2. Spatial Transformer: Self-attention across all segments to learn interactions
3. Classification Head: Predicts LOS for each segment

The key insight is that self-attention can learn which segments influence each other
based on their feature patterns, without explicit connectivity information.
"""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SpatialPositionalEncoding(nn.Module):
    """
    Optional spatial encoding based on coordinates (lat, lon).
    If coordinates are available, this adds location-aware embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.coord_proj = nn.Linear(2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [batch, num_segments, d_model]
        coords: [batch, num_segments, 2] (lat, lon) - optional
        """
        if coords is not None:
            spatial_embed = self.coord_proj(coords)
            x = x + spatial_embed
        return self.dropout(x)


class SegmentEmbedding(nn.Module):
    """Learnable embedding for each segment (like token embedding in NLP)."""

    def __init__(self, num_segments: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_segments, d_model)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        """segment_ids: [batch, num_segments] or [num_segments]"""
        return self.embedding(segment_ids)


class SpatioTemporalTransformer(nn.Module):
    """
    Transformer-based model for traffic prediction.

    Key features:
    - Self-attention learns which segments influence each other
    - No need for explicit graph structure
    - Can optionally use coordinates for spatial awareness
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_horizons: int = 1,
        # Temporal encoder
        rnn_type: Literal["lstm", "gru"] = "gru",
        num_rnn_layers: int = 2,
        bidirectional: bool = True,
        # Transformer
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        # Regularization
        dropout: float = 0.1,
        # Optional embeddings
        num_segments: Optional[int] = None,  # For learnable segment embeddings
        use_spatial_coords: bool = False,  # If coordinates are available
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_horizons = num_horizons
        self.use_spatial_coords = use_spatial_coords
        self.time_embedding_dim = time_embedding_dim if time_embedding_dim else 0
        self.segment_embedding_dim = segment_embedding_dim if segment_embedding_dim else 0

        # Time embedding (hour of day)
        if time_embedding_dim and time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(time_vocab_size, time_embedding_dim)
        else:
            self.time_embedding = None

        # Segment embedding (explicit, via config)
        if segment_embedding_dim and segment_embedding_dim > 0 and segment_vocab_size is not None:
            self.segment_id_embedding = nn.Embedding(segment_vocab_size, segment_embedding_dim)
        else:
            self.segment_id_embedding = None

        # Adjust input dim for embeddings
        feature_dim = input_dim
        if self.time_embedding is not None:
            feature_dim += time_embedding_dim
        if self.segment_id_embedding is not None:
            feature_dim += segment_embedding_dim

        # === Temporal Encoder (per-segment) ===
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.temporal_encoder = rnn_cls(
            feature_dim,
            hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional

        temporal_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Project to transformer dimension
        self.input_proj = nn.Linear(temporal_out_dim, hidden_dim)

        # === Optional Segment Embedding ===
        self.segment_embedding = (
            SegmentEmbedding(num_segments, hidden_dim)
            if num_segments is not None
            else None
        )

        # === Optional Spatial Encoding ===
        self.spatial_encoding = (
            SpatialPositionalEncoding(hidden_dim, dropout)
            if use_spatial_coords
            else None
        )

        # === Spatial Transformer (cross-segment attention) ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # === Classification Head ===
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes * num_horizons),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        node_features: torch.Tensor,  # [batch, num_segments, seq_len, features]
        segment_ids: Optional[torch.Tensor] = None,  # [batch, num_segments]
        coords: Optional[torch.Tensor] = None,  # [batch, num_segments, 2]
        mask: Optional[torch.Tensor] = None,  # [batch, num_segments] - valid segments
        time_ids: Optional[torch.Tensor] = None,  # [batch, num_segments, seq_len]
    ) -> torch.Tensor:
        """
        Args:
            node_features: Time series for each segment [batch, num_segments, seq_len, features]
            segment_ids: Optional segment indices for learnable embeddings
            coords: Optional (lat, lon) coordinates for spatial encoding
            mask: Boolean mask for valid segments
            time_ids: Hour-of-day for each timestep [batch, num_segments, seq_len]

        Returns:
            Logits [batch, num_segments, num_horizons, num_classes]
        """
        batch_size, num_segments, seq_len, num_features = node_features.shape

        # Start with base features
        features = node_features

        # Add time embeddings if available
        if self.time_embedding is not None and time_ids is not None:
            time_embed = self.time_embedding(time_ids)  # [batch, num_segments, seq_len, time_dim]
            features = torch.cat([features, time_embed], dim=-1)

        # Add segment embeddings if available (via config, not implicit)
        if self.segment_id_embedding is not None and segment_ids is not None:
            seg_embed = self.segment_id_embedding(segment_ids)  # [batch, num_segments, seg_dim]
            seg_embed = seg_embed.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, num_segments, seq_len, seg_dim]
            features = torch.cat([features, seg_embed], dim=-1)

        # 1. Temporal encoding: process each segment's time series
        # Reshape to [batch * num_segments, seq_len, features]
        x = features.view(batch_size * num_segments, seq_len, -1)

        # Run through RNN
        _, hidden = self.temporal_encoder(x)

        # Get final hidden state
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        if self.bidirectional:
            temporal_repr = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            temporal_repr = hidden[-1]

        # Reshape to [batch, num_segments, temporal_dim]
        temporal_repr = temporal_repr.view(batch_size, num_segments, -1)

        # 2. Project to transformer dimension
        x = self.input_proj(temporal_repr)  # [batch, num_segments, hidden_dim]

        # 3. Add segment embeddings (if available)
        if self.segment_embedding is not None and segment_ids is not None:
            seg_embed = self.segment_embedding(segment_ids)
            x = x + seg_embed

        # 4. Add spatial encoding (if coordinates available)
        if self.spatial_encoding is not None and coords is not None:
            x = self.spatial_encoding(x, coords)

        # 5. Spatial Transformer: self-attention across segments
        # Create attention mask for invalid segments
        if mask is not None:
            # Transformer expects: True = ignore, False = attend
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        x = self.spatial_transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 6. Classification
        logits = self.classifier(x)  # [batch, num_segments, num_classes * num_horizons]
        logits = logits.view(batch_size, num_segments, self.num_horizons, self.num_classes)

        return logits

    def get_attention_weights(
        self,
        node_features: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract attention weights to visualize which segments attend to which.
        Useful for interpretability.
        """
        batch_size, num_segments, seq_len, num_features = node_features.shape

        # Temporal encoding
        x = node_features.view(batch_size * num_segments, seq_len, num_features)
        _, hidden = self.temporal_encoder(x)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        if self.bidirectional:
            temporal_repr = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            temporal_repr = hidden[-1]
        temporal_repr = temporal_repr.view(batch_size, num_segments, -1)

        x = self.input_proj(temporal_repr)

        if self.segment_embedding is not None and segment_ids is not None:
            x = x + self.segment_embedding(segment_ids)

        if self.spatial_encoding is not None and coords is not None:
            x = self.spatial_encoding(x, coords)

        # Get attention from first layer (for visualization)
        # This is a simplified version - full attention extraction requires hooks
        attn_layer = self.spatial_transformer.layers[0].self_attn
        attn_output, attn_weights = attn_layer(x, x, x, need_weights=True)

        return attn_weights  # [batch, num_segments, num_segments]


def create_transformer_model(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_horizons: int = 1,
    rnn_type: Literal["lstm", "gru"] = "gru",
    num_rnn_layers: int = 2,
    bidirectional: bool = True,
    num_transformer_layers: int = 2,
    num_heads: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    num_segments: Optional[int] = None,
    use_spatial_coords: bool = False,
    time_embedding_dim: Optional[int] = None,
    segment_embedding_dim: Optional[int] = None,
    segment_vocab_size: Optional[int] = None,
) -> SpatioTemporalTransformer:
    """Factory function to create a SpatioTemporalTransformer model."""
    return SpatioTemporalTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_horizons=num_horizons,
        rnn_type=rnn_type,
        num_rnn_layers=num_rnn_layers,
        bidirectional=bidirectional,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_segments=num_segments,
        use_spatial_coords=use_spatial_coords,
        time_embedding_dim=time_embedding_dim,
        segment_embedding_dim=segment_embedding_dim,
        segment_vocab_size=segment_vocab_size,
    )


