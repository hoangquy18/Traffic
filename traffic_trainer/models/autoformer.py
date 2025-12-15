"""
Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (NeurIPS 2021)

Autoformer is a state-of-the-art model that addresses the limitations of vanilla Transformers
for time series forecasting through two key innovations:

Key Innovations:
1. Auto-Correlation Mechanism - replaces self-attention with series-wise connection based on
   autocorrelation, which is more suitable for time series data
2. Series Decomposition Architecture - progressively decomposes time series into trend and
   seasonal components at each layer, making the model more interpretable

Advantages over Informer:
- Better captures periodic patterns through auto-correlation
- More interpretable with explicit trend-seasonal decomposition
- Often achieves better accuracy on long-term forecasting tasks
- More stable training due to decomposition

Reference:
Wu, H., et al. (2021). "Autoformer: Decomposition Transformers with Auto-Correlation 
for Long-Term Series Forecasting." NeurIPS 2021.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block that separates trend and seasonal components.
    Uses moving average for trend extraction.
    """

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, features]
            
        Returns:
            seasonal: [batch, seq_len, features]
            trend: [batch, seq_len, features]
        """
        # Padding on both sides
        num_pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_pad, 1)
        end = x[:, -1:, :].repeat(1, num_pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # Moving average for trend
        x_padded = x_padded.transpose(1, 2)  # [batch, features, seq_len]
        trend = self.avg(x_padded)
        trend = trend.transpose(1, 2)  # [batch, seq_len, features]

        # Seasonal = original - trend
        seasonal = x - trend
        
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism that finds period-based dependencies.
    
    Instead of computing attention between all pairs, it:
    1. Computes autocorrelation in frequency domain (FFT)
    2. Selects top-k periods with highest autocorrelation
    3. Aggregates values based on these periods
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 1,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(
        self, values: torch.Tensor, corr: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate values based on autocorrelation delays.
        
        Args:
            values: [batch, head, seq_len, d_keys]
            corr: [batch, head, seq_len]
            
        Returns:
            aggregated: [batch, head, seq_len, d_keys]
        """
        head = values.shape[1]
        channel = values.shape[3]
        length = values.shape[2]

        # Find top k autocorrelation delays
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        
        # Normalize weights
        tmp_corr = torch.softmax(weights, dim=-1)

        # Aggregate values with time delays
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), dims=2)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
                1, head, length, channel
            )

        return delays_agg

    def time_delay_agg_inference(
        self, values: torch.Tensor, corr: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized aggregation for inference (batch processing).
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[3]
        length = values.shape[2]

        # Find top k delays
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, index = torch.topk(mean_value, top_k, dim=-1)
        
        # Normalize
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Aggregate
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = torch.roll(values, -int(index[0, i]), dims=2)
            delays_agg = delays_agg + tmp_delay * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
                1, head, length, channel
            )

        return delays_agg

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            queries: [batch, head, seq_len, d_keys]
            keys: [batch, head, seq_len, d_keys]
            values: [batch, head, seq_len, d_keys]
            
        Returns:
            output: [batch, head, seq_len, d_keys]
            attn: None (for API compatibility)
        """
        B, H, L, E = queries.shape
        _, _, S, D = values.shape
        
        if L > S:
            # Pad values to match query length
            zeros = torch.zeros_like(queries[:, :, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=2)
            keys = torch.cat([keys, zeros], dim=2)
        else:
            values = values[:, :, :L, :]
            keys = keys[:, :, :L, :]

        # Compute autocorrelation using FFT
        q_fft = torch.fft.rfft(queries.contiguous(), dim=2)
        k_fft = torch.fft.rfft(keys.contiguous(), dim=2)
        
        # Autocorrelation in frequency domain
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=2)

        # Time delay aggregation
        if self.training:
            V = self.time_delay_agg_training(values, corr)
        else:
            V = self.time_delay_agg_inference(values, corr)

        return V, None


class AutoCorrelationLayer(nn.Module):
    """Multi-head auto-correlation layer."""

    def __init__(
        self,
        correlation: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ) -> None:
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """Autoformer encoder layer with auto-correlation and series decomposition."""

    def __init__(
        self,
        attention: AutoCorrelationLayer,
        d_model: int,
        d_ff: int = 2048,
        moving_avg: int = 25,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seasonal: [batch, seq_len, d_model]
            trend: [batch, seq_len, d_model]
        """
        # Auto-correlation + decomposition
        new_x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, trend1 = self.decomp1(x)

        # Feed-forward + decomposition
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        seasonal, trend2 = self.decomp2(x + y)

        return seasonal, trend1 + trend2


class Encoder(nn.Module):
    """Autoformer encoder with progressive decomposition."""

    def __init__(
        self,
        attn_layers: list[EncoderLayer],
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seasonal: [batch, seq_len, d_model]
            trend: [batch, seq_len, d_model]
        """
        seasonal_list = []
        trend_list = []

        for attn_layer in self.attn_layers:
            seasonal, trend = attn_layer(x, attn_mask=attn_mask)
            seasonal_list.append(seasonal)
            trend_list.append(trend)
            x = seasonal

        if self.norm is not None:
            x = self.norm(x)

        return x, sum(trend_list)


class DecoderLayer(nn.Module):
    """Autoformer decoder layer."""

    def __init__(
        self,
        self_attention: AutoCorrelationLayer,
        cross_attention: AutoCorrelationLayer,
        d_model: int,
        d_ff: int = 2048,
        moving_avg: int = 25,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seasonal: [batch, seq_len, d_model]
            trend: [batch, seq_len, d_model]
        """
        # Self auto-correlation
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)

        # Cross auto-correlation
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)

        # Feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        seasonal, trend3 = self.decomp3(x + y)

        return seasonal, trend1 + trend2 + trend3


class Decoder(nn.Module):
    """Autoformer decoder."""

    def __init__(
        self,
        layers: list[DecoderLayer],
        norm_layer: Optional[nn.Module] = None,
        projection: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        trend: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            output: [batch, seq_len, d_model] or projected output
        """
        seasonal_list = []
        trend_list = []

        for layer in self.layers:
            seasonal, trend_part = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            seasonal_list.append(seasonal)
            trend_list.append(trend_part)
            x = seasonal

        if self.norm is not None:
            x = self.norm(x)

        # Combine seasonal and trend
        if trend is not None:
            x = x + trend + sum(trend_list)
        else:
            x = x + sum(trend_list)

        if self.projection is not None:
            x = self.projection(x)

        return x


class DataEmbedding(nn.Module):
    """Embedding layer with value, positional, and temporal encoding."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        dropout: float = 0.1,
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.time_embedding_dim = time_embedding_dim if time_embedding_dim else 0
        self.segment_embedding_dim = segment_embedding_dim if segment_embedding_dim else 0
        
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
        
        # Adjust input dimension
        feature_dim = c_in
        if self.time_embedding is not None:
            feature_dim += time_embedding_dim
        if self.segment_embedding is not None:
            feature_dim += segment_embedding_dim
            
        self.value_embedding = nn.Linear(feature_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
            time_ids: [batch, seq_len]
            segment_ids: [batch]
        """
        features = x
        
        # Add time embeddings
        if self.time_embedding is not None and time_ids is not None:
            time_embed = self.time_embedding(time_ids)
            features = torch.cat([features, time_embed], dim=-1)
        
        # Add segment embeddings
        if self.segment_embedding is not None and segment_ids is not None:
            seg_embed = self.segment_embedding(segment_ids)
            seg_embed = seg_embed.unsqueeze(1).expand(-1, x.size(1), -1)
            features = torch.cat([features, seg_embed], dim=-1)
        
        x = self.value_embedding(features) + self.position_embedding(x)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class Autoformer(nn.Module):
    """
    Autoformer model for long sequence time-series forecasting.
    
    Uses auto-correlation and series decomposition for better time series modeling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_horizons: int = 1,
        seq_len: int = 96,
        label_len: int = 48,
        out_len: int = 24,
        factor: int = 1,
        d_ff: int = 2048,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        moving_avg: int = 25,
        dropout: float = 0.1,
        activation: str = 'gelu',
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = out_len
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Decomposition for input
        self.decomp = SeriesDecomposition(moving_avg)

        # Encoding
        self.enc_embedding = DataEmbedding(
            input_dim, hidden_dim, dropout,
            time_embedding_dim, segment_embedding_dim,
            time_vocab_size, segment_vocab_size
        )
        self.dec_embedding = DataEmbedding(
            input_dim, hidden_dim, dropout,
            time_embedding_dim, segment_embedding_dim,
            time_vocab_size, segment_vocab_size
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout),
                        hidden_dim, n_heads
                    ),
                    hidden_dim,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_dim)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout),
                        hidden_dim, n_heads
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout),
                        hidden_dim, n_heads
                    ),
                    hidden_dim,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_dim),
            projection=None
        )

        # Output projection
        self.projection = nn.Linear(hidden_dim, num_classes * num_horizons)

    def forward(
        self,
        node_features: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        dec_inp: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch, num_nodes, seq_len, features]
            time_ids: [batch, num_nodes, seq_len]
            segment_ids: [batch, num_nodes]
            dec_inp: Optional decoder input
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
            
            # Decompose input
            seasonal_init, trend_init = self.decomp(node_data)
            
            # Encoder
            enc_out = self.enc_embedding(seasonal_init, node_time_ids, node_segment_ids)
            enc_seasonal, enc_trend = self.encoder(enc_out, attn_mask=None)
            
            # Decoder input (use last part of encoder input + zeros for prediction)
            if dec_inp is None:
                dec_inp_node = torch.zeros(
                    batch_size, self.label_len + self.pred_len, num_features,
                    device=node_data.device
                )
                dec_inp_node[:, :self.label_len, :] = seasonal_init[:, -self.label_len:, :]
            else:
                dec_inp_node = dec_inp[:, i, :, :]
            
            # Decoder time_ids
            if node_time_ids is not None:
                dec_time_ids = torch.zeros(
                    batch_size, self.label_len + self.pred_len,
                    dtype=torch.long, device=node_data.device
                )
                dec_time_ids[:, :self.label_len] = node_time_ids[:, -self.label_len:]
                dec_time_ids[:, self.label_len:] = node_time_ids[:, -1:].expand(-1, self.pred_len)
            else:
                dec_time_ids = None
            
            # Decoder trend
            trend_part = torch.zeros(
                batch_size, self.label_len + self.pred_len, self.hidden_dim,
                device=node_data.device
            )
            
            dec_out = self.dec_embedding(dec_inp_node, dec_time_ids, node_segment_ids)
            dec_out = self.decoder(dec_out, enc_seasonal, x_mask=None, cross_mask=None, trend=trend_part)
            
            # Take the last prediction
            dec_out = dec_out[:, -1, :]  # [batch, hidden_dim]
            outputs.append(dec_out)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, num_nodes, hidden_dim]
        
        # Project to output
        logits = self.projection(outputs)  # [batch, num_nodes, num_classes * num_horizons]
        logits = logits.view(batch_size, num_nodes, self.num_horizons, self.num_classes)
        
        return logits


def create_autoformer_model(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_horizons: int = 1,
    seq_len: int = 96,
    label_len: int = 48,
    out_len: int = 24,
    factor: int = 1,
    d_ff: int = 2048,
    n_heads: int = 8,
    e_layers: int = 2,
    d_layers: int = 1,
    moving_avg: int = 25,
    dropout: float = 0.1,
    activation: str = 'gelu',
    time_embedding_dim: Optional[int] = None,
    segment_embedding_dim: Optional[int] = None,
    segment_vocab_size: Optional[int] = None,
    **kwargs,
) -> Autoformer:
    """
    Factory function to create an Autoformer model.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension (d_model)
        num_classes: Number of output classes
        num_horizons: Number of prediction horizons
        seq_len: Input sequence length
        label_len: Start token length for decoder
        out_len: Prediction sequence length
        factor: Auto-correlation factor
        d_ff: Feed-forward dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        d_layers: Number of decoder layers
        moving_avg: Moving average window size for decomposition
        dropout: Dropout rate
        activation: Activation function ('relu' or 'gelu')
        time_embedding_dim: Time embedding dimension
        segment_embedding_dim: Segment embedding dimension
        segment_vocab_size: Number of unique segments
        
    Returns:
        Autoformer model instance
    """
    return Autoformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_horizons=num_horizons,
        seq_len=seq_len,
        label_len=label_len,
        out_len=out_len,
        factor=factor,
        d_ff=d_ff,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        moving_avg=moving_avg,
        dropout=dropout,
        activation=activation,
        time_embedding_dim=time_embedding_dim,
        segment_embedding_dim=segment_embedding_dim,
        segment_vocab_size=segment_vocab_size,
    )
