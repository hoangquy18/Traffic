"""RNN-based sequence classifier for traffic prediction."""

from typing import Literal, Optional

import torch
from torch import nn


class SequenceClassifier(nn.Module):
    """Generic sequence classifier using an RNN backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        dropout: float = 0.1,
        bidirectional: bool = False,
        num_horizons: int = 1,
        time_embedding_dim: Optional[int] = None,
        segment_embedding_dim: Optional[int] = None,
        time_vocab_size: int = 24,
        segment_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.time_embedding_dim = time_embedding_dim if time_embedding_dim else 0
        self.segment_embedding_dim = (
            segment_embedding_dim if segment_embedding_dim else 0
        )

        feature_dim = input_dim

        if time_embedding_dim and time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(
                num_embeddings=time_vocab_size,
                embedding_dim=time_embedding_dim,
            )
            feature_dim += time_embedding_dim
        else:
            self.time_embedding = None

        if (
            segment_embedding_dim
            and segment_embedding_dim > 0
            and segment_vocab_size is not None
        ):
            self.segment_embedding = nn.Embedding(
                num_embeddings=segment_vocab_size,
                embedding_dim=segment_embedding_dim,
            )
            feature_dim += segment_embedding_dim
        else:
            self.segment_embedding = None

        self.rnn = rnn_cls(
            feature_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes * num_horizons),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = x
        if self.time_embedding is not None and time_ids is not None:
            time_embed = self.time_embedding(time_ids)
            features = torch.cat([features, time_embed], dim=-1)
        if self.segment_embedding is not None and segment_ids is not None:
            segment_embed = self.segment_embedding(segment_ids)
            features = torch.cat([features, segment_embed], dim=-1)

        outputs, hidden = self.rnn(features)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        if self.bidirectional:
            last_hidden = torch.cat(
                (hidden[-2], hidden[-1]),
                dim=-1,
            )
        else:
            last_hidden = hidden[-1]
        logits = self.head(last_hidden)
        return logits.view(-1, self.num_horizons, self.num_classes)


def create_model(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_classes: int,
    rnn_type: Literal["lstm", "gru"],
    dropout: float,
    bidirectional: bool,
    num_horizons: int = 1,
    time_embedding_dim: Optional[int] = None,
    segment_embedding_dim: Optional[int] = None,
    segment_vocab_size: Optional[int] = None,
) -> SequenceClassifier:
    return SequenceClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        rnn_type=rnn_type,
        dropout=dropout,
        bidirectional=bidirectional,
        num_horizons=num_horizons,
        time_embedding_dim=time_embedding_dim,
        segment_embedding_dim=segment_embedding_dim,
        segment_vocab_size=segment_vocab_size,
    )


