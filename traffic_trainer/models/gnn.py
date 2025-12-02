"""
Spatio-Temporal Graph Neural Network for Traffic Prediction.

Architecture:
1. Temporal Encoder (GRU/LSTM): Processes time series for each node independently
2. Spatial Encoder (GCN/GAT): Aggregates information from neighboring road segments
3. Prediction Head: Outputs LOS classification for each node

The model learns both:
- How traffic evolves over time on each road segment
- How traffic propagates between connected road segments
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolutional Layer.
    Implements: H' = σ(D^(-1/2) A D^(-1/2) H W)

    For simplicity, we use a normalized adjacency approach.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: torch.Tensor,  # [batch, num_nodes, features]
        edge_index: torch.Tensor,  # [2, num_edges]
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [batch, num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Updated node features [batch, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.shape
        src, dst = edge_index[0], edge_index[1]

        # Compute degree for normalization
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg = deg.clamp(min=1)  # Avoid division by zero
        deg_inv_sqrt = deg.pow(-0.5)

        # Normalize edge weights
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=x.device)
        norm = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]

        # Message passing: aggregate neighbor features
        # For each destination node, sum normalized features from source nodes
        out = torch.zeros_like(x)
        for b in range(batch_size):
            # Gather source features
            src_features = x[b, src]  # [num_edges, features]
            # Weight by normalization
            weighted_features = src_features * norm.unsqueeze(-1)
            # Scatter add to destination nodes
            out[b].scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_features), weighted_features)

        # Apply linear transformation
        return self.linear(out)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    Learns attention weights between connected nodes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(
        self,
        x: torch.Tensor,  # [batch, num_nodes, features]
        edge_index: torch.Tensor,  # [2, num_edges]
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        src, dst = edge_index[0], edge_index[1]

        # Linear transformation
        h = self.W(x)  # [batch, num_nodes, num_heads * out_features]
        h = h.view(batch_size, num_nodes, self.num_heads, self.out_features)

        # Compute attention scores
        h_src = h[:, src]  # [batch, num_edges, num_heads, out_features]
        h_dst = h[:, dst]  # [batch, num_edges, num_heads, out_features]

        # Concatenate and compute attention
        edge_features = torch.cat([h_src, h_dst], dim=-1)  # [batch, num_edges, heads, 2*out]
        attention = (edge_features * self.a).sum(dim=-1)  # [batch, num_edges, heads]
        attention = self.leaky_relu(attention)

        # Softmax over neighbors (per destination node)
        attention_exp = attention.exp()
        attention_sum = torch.zeros(batch_size, num_nodes, self.num_heads, device=x.device)
        for b in range(batch_size):
            attention_sum[b].scatter_add_(
                0,
                dst.unsqueeze(-1).expand(-1, self.num_heads),
                attention_exp[b],
            )
        attention_norm = attention_exp / (attention_sum[:, dst] + 1e-10)
        attention_norm = self.dropout(attention_norm)

        # Aggregate
        out = torch.zeros(batch_size, num_nodes, self.num_heads, self.out_features, device=x.device)
        weighted_h = h_src * attention_norm.unsqueeze(-1)
        for b in range(batch_size):
            out[b].scatter_add_(
                0,
                dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.out_features),
                weighted_h[b],
            )

        if self.concat:
            return out.view(batch_size, num_nodes, -1)  # [batch, nodes, heads*out]
        else:
            return out.mean(dim=2)  # [batch, nodes, out]


class SpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for traffic prediction.

    Architecture:
    1. Per-node temporal encoding (GRU/LSTM)
    2. Graph convolution for spatial aggregation
    3. Classification head for LOS prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        num_horizons: int = 1,
        rnn_type: Literal["lstm", "gru"] = "gru",
        gnn_type: Literal["gcn", "gat"] = "gcn",
        num_gnn_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False,
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_horizons = num_horizons
        self.gnn_type = gnn_type
        self.time_embedding_dim = time_embedding_dim if time_embedding_dim else 0
        self.segment_embedding_dim = segment_embedding_dim if segment_embedding_dim else 0

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

        # Temporal encoder (processes each node's time series)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.temporal_encoder = rnn_cls(
            feature_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional

        temporal_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Spatial encoder (graph convolution layers)
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        current_dim = temporal_out_dim
        for i in range(num_gnn_layers):
            if gnn_type == "gat":
                # GAT layer
                out_dim = hidden_dim if i < num_gnn_layers - 1 else hidden_dim
                self.gnn_layers.append(
                    GraphAttentionLayer(
                        current_dim,
                        out_dim // gat_heads if i < num_gnn_layers - 1 else out_dim,
                        num_heads=gat_heads if i < num_gnn_layers - 1 else 1,
                        dropout=dropout,
                        concat=(i < num_gnn_layers - 1),
                    )
                )
                current_dim = out_dim
            else:
                # GCN layer
                out_dim = hidden_dim
                self.gnn_layers.append(GraphConvLayer(current_dim, out_dim))
                current_dim = out_dim

            self.gnn_norms.append(nn.LayerNorm(current_dim))

        self.dropout = nn.Dropout(dropout)

        # Classification head (per node)
        self.classifier = nn.Sequential(
            nn.Linear(current_dim + temporal_out_dim, hidden_dim),  # Skip connection
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes * num_horizons),
        )

    def forward(
        self,
        node_features: torch.Tensor,  # [batch, num_nodes, seq_len, features]
        edge_index: torch.Tensor,  # [2, num_edges]
        mask: Optional[torch.Tensor] = None,  # [batch, num_nodes]
        time_ids: Optional[torch.Tensor] = None,  # [batch, num_nodes, seq_len]
        segment_ids: Optional[torch.Tensor] = None,  # [batch, num_nodes]
    ) -> torch.Tensor:
        """
        Args:
            node_features: Per-node time series [batch, num_nodes, seq_len, features]
            edge_index: Graph connectivity [2, num_edges]
            mask: Valid node mask [batch, num_nodes]
            time_ids: Hour-of-day for each timestep [batch, num_nodes, seq_len]
            segment_ids: Segment index for each node [batch, num_nodes]

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
            seg_embed = seg_embed.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, num_nodes, seq_len, seg_dim]
            features = torch.cat([features, seg_embed], dim=-1)

        # 1. Temporal encoding: process each node's time series
        # Reshape to [batch * num_nodes, seq_len, features]
        x = features.view(batch_size * num_nodes, seq_len, -1)

        # Run through RNN
        rnn_out, hidden = self.temporal_encoder(x)

        # Get final hidden state
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        if self.bidirectional:
            temporal_repr = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            temporal_repr = hidden[-1]

        # Reshape back to [batch, num_nodes, hidden_dim]
        temporal_repr = temporal_repr.view(batch_size, num_nodes, -1)

        # 2. Spatial encoding: graph convolution
        spatial_repr = temporal_repr
        for gnn_layer, norm in zip(self.gnn_layers, self.gnn_norms):
            if self.gnn_type == "gat":
                spatial_repr = gnn_layer(spatial_repr, edge_index)
            else:
                spatial_repr = gnn_layer(spatial_repr, edge_index)
            spatial_repr = norm(spatial_repr)
            spatial_repr = F.relu(spatial_repr)
            spatial_repr = self.dropout(spatial_repr)

        # 3. Combine temporal and spatial representations (skip connection)
        combined = torch.cat([temporal_repr, spatial_repr], dim=-1)

        # 4. Classification
        logits = self.classifier(combined)  # [batch, num_nodes, num_classes * num_horizons]
        logits = logits.view(batch_size, num_nodes, self.num_horizons, self.num_classes)

        return logits


def create_graph_model(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_classes: int,
    num_horizons: int = 1,
    rnn_type: Literal["lstm", "gru"] = "gru",
    gnn_type: Literal["gcn", "gat"] = "gcn",
    num_gnn_layers: int = 2,
    gat_heads: int = 4,
    dropout: float = 0.1,
    bidirectional: bool = False,
    time_embedding_dim: Optional[int] = None,
    segment_embedding_dim: Optional[int] = None,
    segment_vocab_size: Optional[int] = None,
) -> SpatioTemporalGNN:
    """Factory function to create a SpatioTemporalGNN model."""
    return SpatioTemporalGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        num_horizons=num_horizons,
        rnn_type=rnn_type,
        gnn_type=gnn_type,
        num_gnn_layers=num_gnn_layers,
        gat_heads=gat_heads,
        dropout=dropout,
        bidirectional=bidirectional,
        time_embedding_dim=time_embedding_dim,
        segment_embedding_dim=segment_embedding_dim,
        segment_vocab_size=segment_vocab_size,
    )


