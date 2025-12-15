"""
DLinear: Are Transformers Effective for Time Series Forecasting? (AAAI 2023)

DLinear is a surprisingly simple yet effective model that challenges the necessity of 
complex Transformer architectures for time series forecasting. It achieves competitive 
or better performance than Transformers with just linear layers!

Key Innovations:
1. Series Decomposition - Separates trend and seasonal components
2. Two Linear Layers - One for trend, one for seasonal
3. Extreme Simplicity - No attention, no convolutions, just linear projections
4. Channel Independence - Processes each feature separately

Why DLinear Works:
- Time series forecasting is fundamentally different from NLP/CV
- Linear mappings can capture temporal patterns effectively
- Decomposition helps model learn cleaner patterns
- Much faster and more memory efficient than Transformers

Performance:
- Often matches or beats Transformer models
- 10-100x faster training
- 10-50x less memory
- Much easier to tune

Reference:
Zeng, A., et al. (2023). "Are Transformers Effective for Time Series Forecasting?"
AAAI 2023.
"""

from typing import Optional

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Moving average block to extract trend from time series.
    Uses average pooling with padding to maintain sequence length.
    """

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
            
        Returns:
            trend: [batch, seq_len, features]
        """
        # Padding on both sides
        num_pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_pad, 1)
        end = x[:, -1:, :].repeat(1, num_pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # Apply moving average
        x_padded = x_padded.permute(0, 2, 1)  # [batch, features, seq_len]
        trend = self.avg(x_padded)
        trend = trend.permute(0, 2, 1)  # [batch, seq_len, features]
        
        return trend


class SeriesDecomp(nn.Module):
    """
    Series decomposition block that separates trend and seasonal components.
    """

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, features]
            
        Returns:
            seasonal: [batch, seq_len, features]
            trend: [batch, seq_len, features]
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear: Decomposition + Linear layers for time series forecasting.
    
    Extremely simple yet effective:
    1. Decompose input into trend and seasonal
    2. Apply separate linear layers to each component
    3. Combine predictions
    
    No attention, no convolutions, just linear projections!
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_horizons: int = 1,
        seq_len: int = 96,
        pred_len: int = 24,
        kernel_size: int = 25,
        individual: bool = False,
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Not used (for API compatibility)
            num_classes: Number of output classes
            num_horizons: Number of prediction horizons
            seq_len: Input sequence length
            pred_len: Prediction length (not used for classification)
            kernel_size: Moving average window size for decomposition
            individual: If True, use separate linear layers for each feature
            time_embedding_dim: Time embedding dimension
            segment_embedding_dim: Segment embedding dimension
            time_vocab_size: Time vocabulary size
            segment_vocab_size: Segment vocabulary size
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.individual = individual

        # Decomposition
        self.decomposition = SeriesDecomp(kernel_size)

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

        # Calculate total feature dimension
        feature_dim = input_dim
        if self.time_embedding is not None:
            feature_dim += time_embedding_dim
        if self.segment_embedding is not None:
            feature_dim += segment_embedding_dim

        if self.individual:
            # Individual: separate linear layer for each feature
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(self.seq_len, self.seq_len) for _ in range(feature_dim)
            ])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(self.seq_len, self.seq_len) for _ in range(feature_dim)
            ])
        else:
            # Shared: single linear layer for all features
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.seq_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.seq_len)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * self.seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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

            # Decomposition
            seasonal, trend = self.decomposition(features)

            # Apply linear layers
            if self.individual:
                # Individual: process each feature separately
                seasonal_output = torch.zeros_like(seasonal)
                trend_output = torch.zeros_like(trend)
                for idx in range(features.shape[-1]):
                    seasonal_output[:, :, idx] = self.Linear_Seasonal[idx](
                        seasonal[:, :, idx]
                    )
                    trend_output[:, :, idx] = self.Linear_Trend[idx](
                        trend[:, :, idx]
                    )
            else:
                # Shared: process all features together
                # [batch, seq_len, features] -> [batch, features, seq_len]
                seasonal = seasonal.permute(0, 2, 1)
                trend = trend.permute(0, 2, 1)
                
                seasonal_output = self.Linear_Seasonal(seasonal)
                trend_output = self.Linear_Trend(trend)
                
                # [batch, features, seq_len] -> [batch, seq_len, features]
                seasonal_output = seasonal_output.permute(0, 2, 1)
                trend_output = trend_output.permute(0, 2, 1)

            # Combine seasonal and trend
            x = seasonal_output + trend_output  # [batch, seq_len, features]

            # Flatten for classification
            x = x.reshape(batch_size, -1)  # [batch, seq_len * features]
            outputs.append(x)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, num_nodes, seq_len * features]

        # Classification
        logits = self.classifier(outputs)  # [batch, num_nodes, num_classes * num_horizons]
        logits = logits.view(batch_size, num_nodes, self.num_horizons, self.num_classes)

        return logits


class NLinear(nn.Module):
    """
    NLinear: Normalized Linear layer variant.
    
    Applies normalization before linear projection, which can help with
    non-stationary time series.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_horizons: int = 1,
        seq_len: int = 96,
        pred_len: int = 24,
        individual: bool = False,
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.individual = individual

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

        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.seq_len) for _ in range(feature_dim)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.seq_len)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * self.seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        batch_size, num_nodes, seq_len, num_features = node_features.shape

        outputs = []
        for i in range(num_nodes):
            node_data = node_features[:, i, :, :]
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

            # Normalization
            seq_last = features[:, -1:, :].detach()
            features = features - seq_last

            # Apply linear layers
            if self.individual:
                output = torch.zeros_like(features)
                for idx in range(features.shape[-1]):
                    output[:, :, idx] = self.Linear[idx](features[:, :, idx])
            else:
                features = features.permute(0, 2, 1)
                output = self.Linear(features)
                output = output.permute(0, 2, 1)

            # De-normalization
            output = output + seq_last

            # Flatten
            x = output.reshape(batch_size, -1)
            outputs.append(x)

        outputs = torch.stack(outputs, dim=1)
        logits = self.classifier(outputs)
        logits = logits.view(batch_size, num_nodes, self.num_horizons, self.num_classes)

        return logits


def create_dlinear_model(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_horizons: int = 1,
    seq_len: int = 96,
    pred_len: int = 24,
    kernel_size: int = 25,
    individual: bool = False,
    model_type: str = "dlinear",
    time_embedding_dim: Optional[int] = None,
    segment_embedding_dim: Optional[int] = None,
    segment_vocab_size: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create DLinear or NLinear model.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension for classifier
        num_classes: Number of output classes
        num_horizons: Number of prediction horizons
        seq_len: Input sequence length
        pred_len: Prediction length
        kernel_size: Moving average window for decomposition
        individual: Use individual linear layers per feature
        model_type: "dlinear" or "nlinear"
        time_embedding_dim: Time embedding dimension
        segment_embedding_dim: Segment embedding dimension
        segment_vocab_size: Number of unique segments
        
    Returns:
        DLinear or NLinear model instance
    """
    if model_type == "nlinear":
        return NLinear(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_horizons=num_horizons,
            seq_len=seq_len,
            pred_len=pred_len,
            individual=individual,
            time_embedding_dim=time_embedding_dim,
            segment_embedding_dim=segment_embedding_dim,
            segment_vocab_size=segment_vocab_size,
        )
    else:
        return DLinear(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_horizons=num_horizons,
            seq_len=seq_len,
            pred_len=pred_len,
            kernel_size=kernel_size,
            individual=individual,
            time_embedding_dim=time_embedding_dim,
            segment_embedding_dim=segment_embedding_dim,
            segment_vocab_size=segment_vocab_size,
        )

