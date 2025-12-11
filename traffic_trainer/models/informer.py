"""
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI 2021)

Informer is a state-of-the-art model specifically designed for long sequence time series forecasting.
It addresses three main challenges of vanilla Transformers:
1. Quadratic time complexity and memory usage
2. Stack of decoders causes memory bottleneck
3. Slow inference speed

Key Innovations:
1. ProbSparse Self-Attention - O(L log L) complexity instead of O(L²)
2. Self-Attention Distilling - Reduces dimension and highlights dominant features
3. Generative Style Decoder - One forward step for long sequence prediction
4. Multi-head ProbSparse Attention - Efficient attention mechanism

This implementation is adapted for traffic prediction with spatial-temporal features.

Reference:
Zhou, H., et al. (2021). "Informer: Beyond Efficient Transformer for Long Sequence 
Time-Series Forecasting." AAAI 2021.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbAttention(nn.Module):
    """
    ProbSparse Self-Attention Mechanism.
    
    Instead of computing all attention weights, it samples the most "active" queries
    based on their sparsity measurement, reducing complexity from O(L²) to O(L log L).
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(
        self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate sparsity measurement M(q_i, K) and select top queries.
        
        M(q_i, K) = max(q_i * K^T / sqrt(d)) - mean(q_i * K^T / sqrt(d))
        
        Higher M means the query is more "focused" on specific keys.
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calculate sample indices for keys
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        
        # Calculate Q * K^T for sampled keys
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Calculate sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        
        # Select top queries with highest sparsity
        M_top = M.topk(n_top, sorted=False)[1]

        # Reduce Q to selected queries
        Q_reduce = Q[torch.arange(B)[:, None, None], 
                     torch.arange(H)[None, :, None], 
                     M_top, :]
        
        return Q_reduce, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        """Get initial context by mean pooling."""
        B, H, L_V, D = V.shape
        
        if not self.mask_flag:
            # Mean pooling for all values
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            # Cumulative mean for causal attention
            assert L_Q == L_V  # requires that L_Q == L_V for causal attention
            context = V.cumsum(dim=-2)
        
        return context

    def _update_context(
        self,
        context_in: torch.Tensor,
        V: torch.Tensor,
        scores: torch.Tensor,
        index: torch.Tensor,
        L_Q: int,
    ) -> torch.Tensor:
        """Update context with attention scores."""
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -torch.inf)

        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        
        return context_in

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            queries: [B, H, L_Q, D]
            keys: [B, H, L_K, D]
            values: [B, H, L_V, D]
            attn_mask: Optional attention mask
            
        Returns:
            context: [B, H, L_Q, D]
            attn: Optional attention weights (None for efficiency)
        """
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        # Sampling factor
        U_part = self.factor * torch.ceil(torch.log(torch.tensor(L_K, dtype=torch.float))).int().item()
        u = self.factor * torch.ceil(torch.log(torch.tensor(L_Q, dtype=torch.float))).int().item()

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        # Calculate scale
        scale = self.scale or 1.0 / math.sqrt(D)

        # ProbSparse attention
        if L_Q > u:
            queries_reduce, top_index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
            
            # Calculate attention for selected queries
            scores_top = torch.matmul(queries_reduce, keys.transpose(-2, -1)) * scale
            
            # Get initial context
            context = self._get_initial_context(values, L_Q)
            
            # Update context with top queries
            context = self._update_context(context, values, scores_top, top_index, L_Q)
        else:
            # Standard attention for short sequences
            scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
            
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L_Q, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -torch.inf)
            
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            context = torch.matmul(attn, values)

        return context, None


class AttentionLayer(nn.Module):
    """Multi-head attention layer wrapper."""

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ) -> None:
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
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

        out, attn = self.inner_attention(
            queries.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        return self.out_projection(out), attn


class ProbMask:
    """Mask for ProbSparse attention."""

    def __init__(
        self,
        B: int,
        H: int,
        L: int,
        index: torch.Tensor,
        scores: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self.mask = indicator.view(scores.shape)


class TriangularCausalMask:
    """Causal mask for decoder."""

    def __init__(self, B: int, L: int, device: str = "cpu") -> None:
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self.mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)


class ConvLayer(nn.Module):
    """Distilling operation for self-attention."""

    def __init__(self, c_in: int) -> None:
        super().__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downConv(x.transpose(1, 2))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """Informer encoder layer with ProbSparse attention and distilling."""

    def __init__(
        self,
        attention: AttentionLayer,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention
        new_x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)


class Encoder(nn.Module):
    """Informer encoder with distilling."""

    def __init__(self, attn_layers: list[EncoderLayer], conv_layers: Optional[list[ConvLayer]] = None, norm_layer: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encoder with distilling
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    """Informer decoder layer."""

    def __init__(
        self,
        self_attention: AttentionLayer,
        cross_attention: AttentionLayer,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        # Cross-attention
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x = self.norm2(x)

        # Feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """Informer decoder."""

    def __init__(self, layers: list[DecoderLayer], norm_layer: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

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


class Informer(nn.Module):
    """
    Informer model for long sequence time-series forecasting.
    
    Optimized for traffic prediction with spatial-temporal features.
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
        factor: int = 5,
        d_ff: int = 2048,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        dropout: float = 0.1,
        attn: str = 'prob',
        activation: str = 'gelu',
        distil: bool = True,
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

        # Attention
        Attn = ProbAttention if attn == 'prob' else nn.MultiheadAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout) if attn == 'prob' 
                        else Attn(hidden_dim, n_heads, dropout=dropout, batch_first=True),
                        hidden_dim, n_heads
                    ),
                    hidden_dim,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [ConvLayer(hidden_dim) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(hidden_dim)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout) if attn == 'prob'
                        else Attn(hidden_dim, n_heads, dropout=dropout, batch_first=True),
                        hidden_dim, n_heads
                    ),
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout) if attn == 'prob'
                        else Attn(hidden_dim, n_heads, dropout=dropout, batch_first=True),
                        hidden_dim, n_heads
                    ),
                    hidden_dim,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_dim)
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
            
            # Encoder
            enc_out = self.enc_embedding(node_data, node_time_ids, node_segment_ids)
            enc_out = self.encoder(enc_out, attn_mask=None)
            
            # Decoder input (use last part of encoder input + zeros for prediction)
            if dec_inp is None:
                dec_inp_node = torch.zeros(batch_size, self.label_len + self.pred_len, num_features, device=node_data.device)
                dec_inp_node[:, :self.label_len, :] = node_data[:, -self.label_len:, :]
            else:
                dec_inp_node = dec_inp[:, i, :, :]
            
            # Decoder time_ids (for label_len + pred_len)
            if node_time_ids is not None:
                # Take last label_len time_ids and extend with zeros for pred_len
                dec_time_ids = torch.zeros(batch_size, self.label_len + self.pred_len, dtype=torch.long, device=node_data.device)
                dec_time_ids[:, :self.label_len] = node_time_ids[:, -self.label_len:]
                # For pred_len, we could use future time_ids but for simplicity use last time_id
                dec_time_ids[:, self.label_len:] = node_time_ids[:, -1:].expand(-1, self.pred_len)
            else:
                dec_time_ids = None
            
            dec_out = self.dec_embedding(dec_inp_node, dec_time_ids, node_segment_ids)
            dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
            
            # Take the last prediction
            dec_out = dec_out[:, -1, :]  # [batch, hidden_dim]
            outputs.append(dec_out)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, num_nodes, hidden_dim]
        
        # Project to output
        logits = self.projection(outputs)  # [batch, num_nodes, num_classes * num_horizons]
        logits = logits.view(batch_size, num_nodes, self.num_horizons, self.num_classes)
        
        return logits


def create_informer_model(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_horizons: int = 1,
    seq_len: int = 96,
    label_len: int = 48,
    out_len: int = 24,
    factor: int = 5,
    d_ff: int = 2048,
    n_heads: int = 8,
    e_layers: int = 3,
    d_layers: int = 2,
    dropout: float = 0.1,
    attn: str = 'prob',
    activation: str = 'gelu',
    distil: bool = True,
    time_embedding_dim: Optional[int] = None,
    segment_embedding_dim: Optional[int] = None,
    segment_vocab_size: Optional[int] = None,
    **kwargs,
) -> Informer:
    """
    Factory function to create an Informer model.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension (d_model)
        num_classes: Number of output classes
        num_horizons: Number of prediction horizons
        seq_len: Input sequence length
        label_len: Start token length for decoder
        out_len: Prediction sequence length
        factor: ProbSparse attention factor
        d_ff: Feed-forward dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        d_layers: Number of decoder layers
        dropout: Dropout rate
        attn: Attention type ('prob' or 'full')
        activation: Activation function ('relu' or 'gelu')
        distil: Whether to use distilling in encoder
        time_embedding_dim: Time embedding dimension
        segment_embedding_dim: Segment embedding dimension
        segment_vocab_size: Number of unique segments
        
    Returns:
        Informer model instance
    """
    return Informer(
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
        dropout=dropout,
        attn=attn,
        activation=activation,
        distil=distil,
        time_embedding_dim=time_embedding_dim,
        segment_embedding_dim=segment_embedding_dim,
        segment_vocab_size=segment_vocab_size,
    )



