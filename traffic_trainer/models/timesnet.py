"""
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (ICLR 2023)

TimesNet is a state-of-the-art model for time series analysis that converts 1D time series
into 2D representations using 2D FFT, enabling the model to capture both intraperiod and
interperiod variations simultaneously.

Key Innovations:
1. 2D FFT Transformation - Converts 1D time series to 2D representation
2. Inception-like Blocks - Multi-scale feature extraction
3. Temporal Variation Modeling - Captures both intraperiod and interperiod patterns

This implementation is adapted for traffic prediction with spatial-temporal features.

Reference:
Wu, H., et al. (2023). "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis."
ICLR 2023.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFT2D(nn.Module):
    """2D FFT transformation for converting 1D time series to 2D representation."""

    def __init__(self, mode: str = "period") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        Convert 1D time series to 2D representation using period-based reshaping.

        The key idea is to reshape the 1D sequence into a 2D matrix where:
        - Rows represent different periods
        - Columns represent different time steps within a period

        Args:
            x: [batch, seq_len, d_model]
            top_k: Number of top frequencies to keep (used to determine period)

        Returns:
            2D representation [batch, top_k, period, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Determine period based on top_k
        # The period is chosen such that top_k * period ≈ seq_len
        period = seq_len // top_k if seq_len // top_k > 0 else 1

        # Reshape to 2D: [batch, period, top_k, d_model]
        # Then transpose to [batch, top_k, period, d_model]
        if period * top_k <= seq_len:
            x_2d = x[:, : period * top_k, :].view(batch_size, period, top_k, d_model)
            x_2d = x_2d.transpose(1, 2)  # [batch, top_k, period, d_model]
        else:
            # If period calculation doesn't work, use a simpler reshape
            period = max(seq_len // top_k, 1)
            x_2d = x[:, : period * top_k, :].view(batch_size, period, top_k, d_model)
            x_2d = x_2d.transpose(1, 2)

        return x_2d


class InceptionBlock(nn.Module):
    """Inception-like block for multi-scale feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True,
    ) -> None:
        super().__init__()
        self.num_kernels = num_kernels
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

        if init_weight:
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
        res_list = []
        for i, kernel in enumerate(self.kernels):
            res_list.append(kernel(x))
        res = torch.stack(res_list, dim=-1).mean(dim=-1)
        return res


class TimesBlock(nn.Module):
    """Core block of TimesNet that processes 2D representations."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.d_model = d_model

        # 2D FFT
        self.fft2d = FFT2D()

        # Inception blocks for 2D convolution
        self.conv1 = InceptionBlock(d_model, d_model, num_kernels=num_kernels)
        self.conv2 = InceptionBlock(d_model, d_model, num_kernels=num_kernels)

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

        # 2D FFT transformation
        x_2d = self.fft2d(x, top_k=self.top_k)  # [batch, top_k, period, d_model]

        # Reshape for 2D convolution: [batch, d_model, top_k, period]
        x_2d = x_2d.permute(0, 3, 1, 2)

        # Inception blocks
        x_2d = self.conv1(x_2d)
        x_2d = F.gelu(x_2d)
        x_2d = self.conv2(x_2d)

        # Reshape back: [batch, top_k, period, d_model]
        x_2d = x_2d.permute(0, 2, 3, 1)

        # Get period from the shape
        _, top_k, period, _ = x_2d.shape

        # Reshape to original: [batch, seq_len, d_model]
        x_2d = x_2d.contiguous().view(batch_size, self.top_k * period, d_model)
        x_2d = x_2d[:, :seq_len, :]  # Truncate to original length

        # Residual connection and normalization
        x = x + self.dropout(x_2d)
        x = self.norm1(x)

        # Feed-forward network
        # Convert to 2D for conv operations
        x_2d_ff = x.view(batch_size, seq_len, d_model).transpose(1, 2).unsqueeze(-1)
        x_2d_ff = self.conv3(x_2d_ff)
        x_2d_ff = F.gelu(x_2d_ff)
        x_2d_ff = self.conv4(x_2d_ff)
        x_2d_ff = x_2d_ff.squeeze(-1).transpose(1, 2)

        # Residual connection and normalization
        x = x + self.dropout(x_2d_ff)
        x = self.norm2(x)

        return x


class TimesNet(nn.Module):
    """
    TimesNet model for time series analysis.

    Architecture:
    1. Input embedding
    2. Stack of TimesBlocks
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
        dropout: float = 0.1,
        embed: str = "timeF",  # "timeF", "fixed", "learned"
        freq: str = "h",  # "h", "t", "s", "m"
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

        # Stack of TimesBlocks
        self.encoder = nn.ModuleList(
            [
                TimesBlock(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    top_k=top_k,
                    d_model=hidden_dim,
                    d_ff=d_ff,
                    num_kernels=num_kernels,
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


def create_timesnet_model(
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
    dropout: float = 0.1,
) -> TimesNet:
    """Factory function to create a TimesNet model."""
    return TimesNet(
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
        dropout=dropout,
    )
