"""Model architectures for traffic prediction."""

from traffic_trainer.models.rnn import SequenceClassifier, create_model
from traffic_trainer.models.gnn import (
    GraphConvLayer,
    GraphAttentionLayer,
    SpatioTemporalGNN,
    create_graph_model,
)
from traffic_trainer.models.transformer import (
    PositionalEncoding,
    SpatialPositionalEncoding,
    SegmentEmbedding,
    SpatioTemporalTransformer,
    create_transformer_model,
)
from traffic_trainer.models.gman import (
    DilatedTemporalConv,
    TemporalConvStack,
    SpatialAttention,
    TemporalAttention,
    EncoderBlock,
    HorizonDecoder,
    GMAN,
    create_sota_model,
)
from traffic_trainer.models.tcn import (
    CausalConv1d,
    TemporalBlock,
    TemporalConvNet,
    MultiScaleTCN,
    create_tcn_model,
)
from traffic_trainer.models.informer import (
    Informer,
    ProbAttention,
    create_informer_model,
)

__all__ = [
    # RNN
    "SequenceClassifier",
    "create_model",
    # GNN
    "GraphConvLayer",
    "GraphAttentionLayer",
    "SpatioTemporalGNN",
    "create_graph_model",
    # Transformer
    "PositionalEncoding",
    "SpatialPositionalEncoding",
    "SegmentEmbedding",
    "SpatioTemporalTransformer",
    "create_transformer_model",
    # GMAN
    "DilatedTemporalConv",
    "TemporalConvStack",
    "SpatialAttention",
    "TemporalAttention",
    "EncoderBlock",
    "HorizonDecoder",
    "GMAN",
    "create_sota_model",
    # TCN
    "CausalConv1d",
    "TemporalBlock",
    "TemporalConvNet",
    "MultiScaleTCN",
    "create_tcn_model",
    # Informer
    "Informer",
    "ProbAttention",
    "create_informer_model",
]


