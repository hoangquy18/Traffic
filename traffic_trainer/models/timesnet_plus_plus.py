"""
TimesNet++: Enhanced Temporal 2D-Variation Modeling for General Time Series Analysis

TimesNet++ is an improved version of TimesNet with enhanced features:
1. Multi-scale 2D FFT - Multiple period detection
2. Enhanced Inception Blocks - Better feature extraction
3. Adaptive Period Selection - Dynamic period detection
4. Cross-scale Feature Fusion - Better feature integration

This implementation is adapted for traffic prediction with spatial-temporal features.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class AdaptiveFFT2D(nn.Module):
    """Adaptive 2D FFT with multiple period detection."""

    def __init__(self, top_k: int = 5, num_periods: int = 3) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_periods = num_periods

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Convert 1D time series to multiple 2D representations with different periods.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            List of 2D representations, each [batch, top_k, period, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # FFT to get frequency domain
        x_fft = torch.fft.rfft(x.transpose(1, 2), dim=-1)
        amplitude = torch.abs(x_fft)

        # Select top-k frequencies
        top_k_amplitude, top_k_indices = torch.topk(amplitude, self.top_k, dim=-1)

        # Generate multiple period-based 2D representations
        representations = []
        for period_idx in range(self.num_periods):
            # Different periods for multi-scale analysis
            period = max(seq_len // (self.top_k * (period_idx + 1)), 1)
            if period * self.top_k > seq_len:
                period = seq_len // self.top_k

            # Reshape to 2D
            x_2d = x[:, : period * self.top_k, :].view(
                batch_size, period, self.top_k, d_model
            )
            x_2d = x_2d.transpose(1, 2)  # [batch, top_k, period, d_model]
            representations.append(x_2d)

        return representations


class EnhancedInceptionBlock(nn.Module):
    """Enhanced Inception block with residual connections and attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.num_kernels = num_kernels
        self.use_attention = use_attention

        # Multi-scale convolutions
        kernels = []
        for i in range(num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    stride=1,
                    padding=i,
                )
            )
        self.kernels = nn.ModuleList(kernels)

        # Channel attention
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels // 4, out_channels, 1),
                nn.Sigmoid(),
            )

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]

        Returns:
            [batch, out_channels, height, width]
        """
        residual = self.residual(x)

        # Multi-scale convolutions
        res_list = []
        for kernel in self.kernels:
            res_list.append(kernel(x))
        x = torch.stack(res_list, dim=-1).mean(dim=-1)

        # Channel attention
        if self.use_attention:
            attn = self.channel_attention(x)
            x = x * attn

        # Residual connection
        x = x + residual

        return x


class CrossScaleFusion(nn.Module):
    """Fuse features from multiple scales."""

    def __init__(self, d_model: int, num_scales: int = 3) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.fusion = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, scale_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            scale_features: List of [batch, seq_len, d_model] tensors

        Returns:
            [batch, seq_len, d_model]
        """
        # Stack and fuse
        stacked = torch.cat(
            scale_features, dim=-1
        )  # [batch, seq_len, d_model * num_scales]
        fused = self.fusion(stacked)  # [batch, seq_len, d_model]
        return fused


class TimesBlockPlusPlus(nn.Module):
    """Enhanced TimesBlock with multi-scale processing and cross-scale fusion."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
        num_periods: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.d_model = d_model
        self.num_periods = num_periods

        # Adaptive 2D FFT
        self.adaptive_fft = AdaptiveFFT2D(top_k=top_k, num_periods=num_periods)

        # Enhanced Inception blocks for each scale
        self.conv_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        EnhancedInceptionBlock(
                            d_model, d_model, num_kernels=num_kernels
                        )
                        for _ in range(2)  # Two layers per scale
                    ]
                )
                for _ in range(num_periods)
            ]
        )

        # Cross-scale fusion
        self.cross_scale_fusion = CrossScaleFusion(d_model, num_scales=num_periods)

        # Feed-forward network
        self.conv3 = nn.Conv2d(d_model, d_ff, kernel_size=1)
        self.conv4 = nn.Conv2d(d_ff, d_model, kernel_size=1)

        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Multi-scale 2D FFT
        x_2d_list = self.adaptive_fft(x)

        # Process each scale
        scale_outputs = []
        for scale_idx, x_2d in enumerate(x_2d_list):
            # Reshape for 2D convolution: [batch, d_model, top_k, period]
            x_2d = x_2d.permute(0, 3, 1, 2)

            # Enhanced Inception blocks
            for conv_block in self.conv_blocks[scale_idx]:
                x_2d = conv_block(x_2d)
                x_2d = F.gelu(x_2d)

            # Reshape back: [batch, top_k, period, d_model]
            x_2d = x_2d.permute(0, 2, 3, 1)

            # Reshape to 1D: [batch, top_k * period, d_model]
            period = x_2d.shape[2]
            x_2d = x_2d.contiguous().view(batch_size, self.top_k * period, d_model)

            # Pad or truncate to original length
            if x_2d.shape[1] < seq_len:
                padding = torch.zeros(
                    batch_size,
                    seq_len - x_2d.shape[1],
                    d_model,
                    device=x.device,
                    dtype=x.dtype,
                )
                x_2d = torch.cat([x_2d, padding], dim=1)
            else:
                x_2d = x_2d[:, :seq_len, :]

            scale_outputs.append(x_2d)

        # Cross-scale fusion
        x_2d_fused = self.cross_scale_fusion(scale_outputs)

        # Residual connection and normalization
        x = x + self.dropout(x_2d_fused)
        x = self.norm1(x)

        # Feed-forward network
        x_2d_ff = x.view(batch_size, seq_len, d_model).transpose(1, 2).unsqueeze(-1)
        x_2d_ff = self.conv3(x_2d_ff)
        x_2d_ff = F.gelu(x_2d_ff)
        x_2d_ff = self.conv4(x_2d_ff)
        x_2d_ff = x_2d_ff.squeeze(-1).transpose(1, 2)

        # Residual connection and normalization
        x = x + self.dropout(x_2d_ff)
        x = self.norm2(x)

        return x


class TimesNetPlusPlus(nn.Module):
    """
    TimesNet++ model with enhanced features for time series analysis.

    Architecture:
    1. Input embedding
    2. Stack of TimesBlockPlusPlus
    3. Classification head
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_horizons: int = 1,
        seq_len: int = 96,
        pred_len: int = 24,
        e_layers: int = 2,
        top_k: int = 5,
        d_ff: int = 2048,
        num_kernels: int = 6,
        num_periods: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_horizons = num_horizons
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.e_layers = e_layers

        # Input projection
        self.enc_embedding = nn.Linear(input_dim, hidden_dim)

        # Stack of TimesBlockPlusPlus
        self.encoder = nn.ModuleList(
            [
                TimesBlockPlusPlus(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    top_k=top_k,
                    d_model=hidden_dim,
                    d_ff=d_ff,
                    num_kernels=num_kernels,
                    num_periods=num_periods,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )

        # Classification head
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
        mask: Optional[torch.Tensor] = None,  # [batch, num_segments]
    ) -> torch.Tensor:
        """
        Args:
            node_features: Time series for each segment [batch, num_segments, seq_len, features]
            mask: Boolean mask for valid segments

        Returns:
            Logits [batch, num_segments, num_horizons, num_classes]
        """
        batch_size, num_segments, seq_len, num_features = node_features.shape

        # Reshape to process all segments: [batch * num_segments, seq_len, features]
        x = node_features.view(batch_size * num_segments, seq_len, num_features)

        # Input embedding
        x = self.enc_embedding(x)  # [batch * num_segments, seq_len, hidden_dim]

        # Pass through encoder layers
        for layer in self.encoder:
            x = layer(x)

        # Use the last timestep for classification
        x = x[:, -1, :]  # [batch * num_segments, hidden_dim]

        # Classification
        logits = self.classifier(
            x
        )  # [batch * num_segments, num_classes * num_horizons]
        logits = logits.view(
            batch_size, num_segments, self.num_horizons, self.num_classes
        )

        return logits


def create_timesnet_plus_plus_model(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_horizons: int = 1,
    seq_len: int = 96,
    pred_len: int = 24,
    e_layers: int = 2,
    top_k: int = 5,
    d_ff: int = 2048,
    num_kernels: int = 6,
    num_periods: int = 3,
    dropout: float = 0.1,
) -> TimesNetPlusPlus:
    """Factory function to create a TimesNet++ model."""
    return TimesNetPlusPlus(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_horizons=num_horizons,
        seq_len=seq_len,
        pred_len=pred_len,
        e_layers=e_layers,
        top_k=top_k,
        d_ff=d_ff,
        num_kernels=num_kernels,
        num_periods=num_periods,
        dropout=dropout,
    )
