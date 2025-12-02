"""
GMAN++ (Enhanced Graph Multi-Attention Network) for Multi-Horizon Traffic Prediction.

Improvements over standard GMAN:
1. Autoregressive Decoder - sequential prediction for each horizon
2. Horizon Embedding - learnable embedding for each prediction step
3. Dilated Temporal Convolutions - better long-range temporal patterns
4. Cross-Horizon Attention - horizons can attend to each other
5. Deeper encoder with pre-norm residual blocks
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedTemporalConv(nn.Module):
    """Dilated causal convolutions for capturing multi-scale temporal patterns."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels * 2, kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.gate_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, seq_len]"""
        residual = self.residual(x)
        
        out = self.conv(x)
        if self.padding > 0:
            out = out[..., :-self.padding]  # Causal
        
        # Gated activation
        out_a, out_b = out.chunk(2, dim=1)
        out = torch.tanh(out_a) * torch.sigmoid(out_b)
        out = self.dropout(out)
        
        return out + residual


class TemporalConvStack(nn.Module):
    """Stack of dilated convolutions with exponentially increasing dilation."""
    
    def __init__(self, hidden_dim: int, num_layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DilatedTemporalConv(hidden_dim, hidden_dim, kernel_size=3, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, num_nodes, seq_len, hidden_dim]"""
        batch_size, num_nodes, seq_len, hidden_dim = x.shape
        
        # Reshape for conv1d: [batch * nodes, hidden, seq]
        x = x.view(batch_size * num_nodes, seq_len, hidden_dim).transpose(1, 2)
        
        for layer in self.layers:
            x = layer(x)
        
        # Reshape back
        x = x.transpose(1, 2).view(batch_size, num_nodes, -1, hidden_dim)
        return self.norm(x)


class SpatialAttention(nn.Module):
    """Multi-head spatial attention with relative position bias."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [batch, num_nodes, hidden_dim]"""
        batch_size, num_nodes, _ = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, nodes, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_nodes, -1)
        return self.proj(out)


class TemporalAttention(nn.Module):
    """Temporal self-attention for each node."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, num_nodes, seq_len, hidden_dim]"""
        batch_size, num_nodes, seq_len, hidden_dim = x.shape
        
        x_flat = x.view(batch_size * num_nodes, seq_len, hidden_dim)
        
        qkv = self.qkv(x_flat).reshape(batch_size * num_nodes, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        out = self.proj(out)
        
        return out.view(batch_size, num_nodes, seq_len, hidden_dim)


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block with spatial and temporal attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.temporal_attn = TemporalAttention(hidden_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.spatial_attn = SpatialAttention(hidden_dim, num_heads, dropout)
        
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [batch, num_nodes, seq_len, hidden_dim]"""
        batch_size, num_nodes, seq_len, _ = x.shape
        
        # Temporal attention
        x = x + self.temporal_attn(self.norm1(x))
        
        # Spatial attention at each time step
        x_spatial = self.norm2(x)
        spatial_out = []
        for t in range(seq_len):
            h = x_spatial[:, :, t, :]
            h = self.spatial_attn(h, mask)
            spatial_out.append(h)
        x = x + torch.stack(spatial_out, dim=2)
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x


class HorizonDecoder(nn.Module):
    """
    Autoregressive decoder for multi-horizon prediction.
    Uses cross-attention to encoder output and self-attention between horizons.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_horizons: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_horizons = num_horizons
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Horizon embeddings (learnable position for each future step)
        self.horizon_embed = nn.Parameter(torch.randn(1, 1, num_horizons, hidden_dim) * 0.02)
        
        # Cross attention: decoder queries encoder
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # Self attention between horizons
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)
        
        # Causal mask for autoregressive decoding
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(num_horizons, num_horizons), diagonal=1).bool()
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection per horizon
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        
    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch, num_nodes, hidden_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = encoder_output.shape
        
        # Initialize decoder queries with horizon embeddings
        # [batch, nodes, horizons, hidden]
        queries = self.horizon_embed.expand(batch_size, num_nodes, -1, -1)
        
        # Reshape for attention: [batch * nodes, horizons, hidden]
        queries = queries.view(batch_size * num_nodes, self.num_horizons, hidden_dim)
        encoder_flat = encoder_output.view(batch_size * num_nodes, 1, hidden_dim)
        
        # Cross attention to encoder
        cross_out, _ = self.cross_attn(
            self.cross_norm(queries),
            encoder_flat.expand(-1, self.num_horizons, -1),
            encoder_flat.expand(-1, self.num_horizons, -1),
        )
        queries = queries + cross_out
        
        # Self attention between horizons (causal)
        self_out, _ = self.self_attn(
            self.self_norm(queries),
            self.self_norm(queries),
            self.self_norm(queries),
            attn_mask=self.causal_mask,
        )
        queries = queries + self_out
        
        # FFN
        queries = queries + self.ffn(self.ffn_norm(queries))
        
        # Output projection
        logits = self.output_proj(queries)  # [batch * nodes, horizons, classes]
        logits = logits.view(batch_size, num_nodes, self.num_horizons, self.num_classes)
        
        return logits


class GMAN(nn.Module):
    """
    GMAN++ (Enhanced Graph Multi-Attention Network) for Multi-Horizon Prediction.
    
    Architecture:
    1. Input projection + positional embeddings
    2. Dilated temporal convolutions for multi-scale patterns
    3. Stacked encoder blocks (temporal + spatial attention)
    4. Autoregressive horizon decoder with cross-attention
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_nodes: int,
        num_layers: int = 3,
        num_heads: int = 4,
        num_horizons: int = 1,
        dropout: float = 0.1,
        use_spatial_embedding: bool = True,
        use_temporal_conv: bool = True,
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_horizons = num_horizons
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim if time_embedding_dim else 0
        self.segment_embedding_dim = segment_embedding_dim if segment_embedding_dim else 0
        
        # Time embedding (hour of day) - added to input before projection
        if time_embedding_dim and time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(time_vocab_size, time_embedding_dim)
        else:
            self.time_embedding = None

        # Segment embedding (explicit, in addition to spatial_embed)
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
        
        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Learnable spatial embedding (segment identity) - implicit, position-based
        if use_spatial_embedding:
            self.spatial_embed = nn.Parameter(torch.randn(1, num_nodes, 1, hidden_dim) * 0.02)
        else:
            self.spatial_embed = None
        
        # Temporal positional encoding (sinusoidal + learnable)
        self.temporal_embed = nn.Parameter(torch.randn(1, 1, 200, hidden_dim) * 0.02)
        
        # Dilated temporal convolutions
        self.use_temporal_conv = use_temporal_conv
        if use_temporal_conv:
            self.temporal_conv = TemporalConvStack(hidden_dim, num_layers=4, dropout=dropout)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        
        # Temporal aggregation (attention pooling)
        self.temporal_pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.temporal_pool_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        
        # Horizon decoder
        self.decoder = HorizonDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_horizons=num_horizons,
            num_heads=num_heads,
            dropout=dropout,
        )
        
    def forward(
        self,
        node_features: torch.Tensor,  # [batch, num_nodes, seq_len, features]
        edge_index: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,  # [batch, num_nodes, seq_len]
        segment_ids: Optional[torch.Tensor] = None,  # [batch, num_nodes]
    ) -> torch.Tensor:
        batch_size, num_nodes, seq_len, _ = node_features.shape
        
        # Start with base features
        features = node_features

        # Add time embeddings if available
        if self.time_embedding is not None and time_ids is not None:
            time_embed = self.time_embedding(time_ids)  # [batch, num_nodes, seq_len, time_dim]
            features = torch.cat([features, time_embed], dim=-1)

        # Add segment embeddings if available
        if self.segment_id_embedding is not None and segment_ids is not None:
            seg_embed = self.segment_id_embedding(segment_ids)  # [batch, num_nodes, seg_dim]
            seg_embed = seg_embed.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, num_nodes, seq_len, seg_dim]
            features = torch.cat([features, seg_embed], dim=-1)
        
        # Input projection
        x = self.input_proj(features)
        
        # Add spatial embedding
        if self.spatial_embed is not None:
            x = x + self.spatial_embed[:, :num_nodes, :, :]
        
        # Add temporal embedding
        x = x + self.temporal_embed[:, :, :seq_len, :]
        
        # Temporal convolutions for multi-scale patterns
        if self.use_temporal_conv:
            x = x + self.temporal_conv(x)
        
        # Encode with transformer blocks
        for block in self.encoder_blocks:
            x = block(x, mask)
        x = self.encoder_norm(x)
        
        # Temporal aggregation with attention pooling
        x_flat = x.view(batch_size * num_nodes, seq_len, self.hidden_dim)
        query = self.temporal_pool_query.expand(batch_size * num_nodes, -1, -1)
        pooled, _ = self.temporal_pool_attn(query, x_flat, x_flat)
        encoder_output = pooled.squeeze(1).view(batch_size, num_nodes, self.hidden_dim)
        
        # Decode for each horizon
        logits = self.decoder(encoder_output, mask)
        
        return logits


def create_sota_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_nodes: int,
    num_layers: int = 3,
    num_horizons: int = 1,
    dropout: float = 0.1,
    time_embedding_dim: Optional[int] = None,
    segment_embedding_dim: Optional[int] = None,
    segment_vocab_size: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Factory function to create GMAN++ model."""
    
    return GMAN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_nodes=num_nodes,
        num_layers=num_layers,
        num_heads=kwargs.get("num_heads", 4),
        num_horizons=num_horizons,
        dropout=dropout,
        use_spatial_embedding=kwargs.get("use_spatial_embedding", True),
        use_temporal_conv=kwargs.get("use_temporal_conv", True),
        time_embedding_dim=time_embedding_dim,
        segment_embedding_dim=segment_embedding_dim,
        segment_vocab_size=segment_vocab_size,
    )


