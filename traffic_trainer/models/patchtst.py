"""
PatchTST: A Time Series is Worth 64 Words (ICLR 2023)

PatchTST is a state-of-the-art Transformer-based model that applies patching 
(similar to Vision Transformer) to time series forecasting. It achieves SOTA 
performance while being more efficient than traditional Transformers.

Key Innovations:
1. Patching - Divides time series into patches (like ViT for images)
2. Channel Independence - Processes each feature/channel separately
3. Positional Encoding - Learns position of patches
4. Efficient - Reduces sequence length dramatically (e.g., 96 → 16 patches)

Why PatchTST Works:
- Patching reduces sequence length → less computation
- Channel independence → simpler, more effective
- Local semantic info preserved in patches
- Better than point-wise Transformers

Performance:
- SOTA on many benchmarks (ETTh1, ETTh2, Weather, etc.)
- More efficient than Informer/Autoformer
- Better generalization

Reference:
Nie, Y., et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting 
with Transformers." ICLR 2023.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Patching(nn.Module):
    """
    Patching module that divides time series into non-overlapping patches.
    
    Example: seq_len=96, patch_len=16 → 6 patches
    """

    def __init__(self, patch_len: int, stride: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
            
        Returns:
            patches: [batch, num_patches, features, patch_len]
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Calculate number of patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        
        # Extract patches using unfold
        # [batch, features, seq_len] -> [batch, features, num_patches, patch_len]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        
        # [batch, features, num_patches, patch_len] -> [batch, num_patches, features, patch_len]
        patches = patches.transpose(1, 2)
        
        return patches


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patches."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            x + pos: [batch, seq_len, d_model]
        """
        return x + self.pos_embed[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Self-attention
        x2, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # Feed-forward
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        return x


class PatchTST(nn.Module):
    """
    PatchTST: Patching + Transformer for time series forecasting.
    
    Key features:
    1. Patching: Reduces sequence length
    2. Channel independence: Each feature processed separately
    3. Transformer: Captures dependencies between patches
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_horizons: int = 1,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        channel_independence: bool = True,
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension (d_model)
            num_classes: Number of output classes
            num_horizons: Number of prediction horizons
            seq_len: Input sequence length
            patch_len: Length of each patch
            stride: Stride for patching
            n_heads: Number of attention heads
            n_layers: Number of Transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
            channel_independence: Process each channel separately
            time_embedding_dim: Time embedding dimension
            segment_embedding_dim: Segment embedding dimension
            time_vocab_size: Time vocabulary size
            segment_vocab_size: Segment vocabulary size
        """
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.channel_independence = channel_independence

        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Time embedding
        self.time_embedding_dim = time_embedding_dim if time_embedding_dim else 0
        if time_embedding_dim and time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(time_vocab_size, time_embedding_dim)
        else:
            self.time_embedding = None

        # Segment embedding
        self.segment_embedding_dim = segment_embedding_dim if segment_embedding_dim else 0
        if segment_embedding_dim and segment_embedding_dim > 0 and segment_vocab_size is not None:
            self.segment_embedding = nn.Embedding(segment_vocab_size, segment_embedding_dim)
        else:
            self.segment_embedding = None

        # Calculate feature dimension
        feature_dim = input_dim
        if self.time_embedding is not None:
            feature_dim += time_embedding_dim
        if self.segment_embedding is not None:
            feature_dim += segment_embedding_dim

        # Patching
        self.patching = Patching(patch_len, stride)

        # Patch embedding: project patch to d_model
        # Each patch has shape [features, patch_len]
        if channel_independence:
            # Process each channel separately
            self.patch_embedding = nn.Linear(patch_len, hidden_dim)
            self.n_channels = feature_dim
        else:
            # Process all channels together
            self.patch_embedding = nn.Linear(patch_len * feature_dim, hidden_dim)
            self.n_channels = 1

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=self.num_patches)

        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dim,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            ) for _ in range(n_layers)
        ])

        # Classification head
        if channel_independence:
            # Aggregate across channels
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.num_patches * hidden_dim * self.n_channels, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes * num_horizons),
            )
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.num_patches * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes * num_horizons),
            )

    def forward(
        self,
        node_features: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch, num_nodes, seq_len, features]
            time_ids: [batch, num_nodes, seq_len]
            segment_ids: [batch, num_nodes]
            mask: Optional mask
            edge_index: Not used (for API compatibility)
            
        Returns:
            logits: [batch, num_nodes, num_horizons, num_classes]
        """
        batch_size, num_nodes, seq_len, num_features = node_features.shape

        # Process each node independently
        outputs = []
        for i in range(num_nodes):
            node_data = node_features[:, i, :, :]  # [batch, seq_len, features]
            node_time_ids = time_ids[:, i, :] if time_ids is not None else None
            node_segment_ids = segment_ids[:, i] if segment_ids is not None else None

            # Add embeddings
            features = node_data
            if self.time_embedding is not None and node_time_ids is not None:
                time_embed = self.time_embedding(node_time_ids)
                features = torch.cat([features, time_embed], dim=-1)
            if self.segment_embedding is not None and node_segment_ids is not None:
                seg_embed = self.segment_embedding(node_segment_ids)
                seg_embed = seg_embed.unsqueeze(1).expand(-1, seq_len, -1)
                features = torch.cat([features, seg_embed], dim=-1)

            # Patching: [batch, seq_len, features] -> [batch, num_patches, features, patch_len]
            patches = self.patching(features)
            batch, num_patches, n_vars, patch_len = patches.shape

            if self.channel_independence:
                # Process each channel separately
                # [batch, num_patches, n_vars, patch_len] -> [batch * n_vars, num_patches, patch_len]
                patches = patches.reshape(batch * n_vars, num_patches, patch_len)
                
                # Patch embedding
                x = self.patch_embedding(patches)  # [batch * n_vars, num_patches, d_model]
                
                # Positional encoding
                x = self.pos_encoding(x)
                
                # Transformer encoder
                for layer in self.encoder_layers:
                    x = layer(x)
                
                # [batch * n_vars, num_patches, d_model] -> [batch, n_vars, num_patches, d_model]
                x = x.reshape(batch, n_vars, num_patches, -1)
                
                # [batch, n_vars, num_patches, d_model] -> [batch, n_vars * num_patches * d_model]
                x = x.reshape(batch, -1)
            else:
                # Process all channels together
                # [batch, num_patches, n_vars, patch_len] -> [batch, num_patches, n_vars * patch_len]
                patches = patches.reshape(batch, num_patches, -1)
                
                # Patch embedding
                x = self.patch_embedding(patches)  # [batch, num_patches, d_model]
                
                # Positional encoding
                x = self.pos_encoding(x)
                
                # Transformer encoder
                for layer in self.encoder_layers:
                    x = layer(x)
                
                # [batch, num_patches, d_model] -> [batch, num_patches * d_model]
                x = x.reshape(batch, -1)

            outputs.append(x)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, num_nodes, num_patches * hidden_dim]

        # Classification - process each node separately
        batch, nodes, features = outputs.shape
        outputs_flat = outputs.reshape(batch * nodes, features)  # [batch * nodes, features]
        logits_flat = self.head(outputs_flat)  # [batch * nodes, num_classes * num_horizons]
        logits = logits_flat.reshape(batch, nodes, self.num_horizons, self.num_classes)

        return logits


def create_patchtst_model(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_horizons: int = 1,
    seq_len: int = 96,
    patch_len: int = 16,
    stride: int = 8,
    n_heads: int = 8,
    n_layers: int = 3,
    d_ff: int = 512,
    dropout: float = 0.1,
    activation: str = "gelu",
    channel_independence: bool = True,
    time_embedding_dim: Optional[int] = None,
    segment_embedding_dim: Optional[int] = None,
    segment_vocab_size: Optional[int] = None,
    **kwargs,
) -> PatchTST:
    """
    Factory function to create a PatchTST model.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension (d_model)
        num_classes: Number of output classes
        num_horizons: Number of prediction horizons
        seq_len: Input sequence length
        patch_len: Length of each patch
        stride: Stride for patching
        n_heads: Number of attention heads
        n_layers: Number of Transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        channel_independence: Process each channel separately
        time_embedding_dim: Time embedding dimension
        segment_embedding_dim: Segment embedding dimension
        segment_vocab_size: Number of unique segments
        
    Returns:
        PatchTST model instance
    """
    return PatchTST(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_horizons=num_horizons,
        seq_len=seq_len,
        patch_len=patch_len,
        stride=stride,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        channel_independence=channel_independence,
        time_embedding_dim=time_embedding_dim,
        segment_embedding_dim=segment_embedding_dim,
        segment_vocab_size=segment_vocab_size,
    )


