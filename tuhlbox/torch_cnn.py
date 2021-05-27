"""Basic CNN model."""
from typing import List, Tuple

import torch
import torch.nn as nn


def _generate_conv_layers(
    embedding_dim: int,
    conv_layer_configurations: List[Tuple[int, int, int, int, int, int]],
) -> List[nn.Module]:
    result: List[nn.Module] = []
    for i, layer in enumerate(conv_layer_configurations):
        input_size = layer[0] if i > 0 else embedding_dim
        result.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_size,
                    out_channels=layer[1],
                    kernel_size=(layer[2],),
                    stride=(layer[3],),
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=layer[4], stride=layer[5]),
            )
        )
    return result


def _generate_fc_layers(
    n_classes: int, dropout: float, fc_layer_configurations: List[int]
) -> List[nn.Module]:
    result: List[nn.Module] = []
    last_output_size = 0
    for i, layer in enumerate(fc_layer_configurations):
        input_size = layer
        if i < len(fc_layer_configurations) - 1:
            last_output_size = fc_layer_configurations[i + 1]
            print(f"adding fc layer with ({input_size}, {last_output_size})")
            result.append(
                nn.Sequential(
                    nn.Linear(input_size, last_output_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
    print(f"adding fc layer with ({last_output_size}, {n_classes})")
    result.append(nn.Linear(last_output_size, n_classes))
    result.append(nn.LogSoftmax(dim=-1))
    return result


class CharCNN(nn.Module):
    """Basic CNN model that can be built with variable amounts of layers etc."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_classes: int,
        max_seq_length: int,
        conv_layer_configurations: List[Tuple[int, int, int, int, int, int]],
        fc_layer_configurations: List[int],
        dropout: float = 0.0,
    ):
        """
        Create a new CNN model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Size of embedding vectors
            n_classes: Number of classes used
            max_seq_length: Max. sequence length for each token
            dropout: random dropout fraction
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.n_classes = n_classes
        self.conv_layer_configurations = conv_layer_configurations
        self.fc_layer_configurations = fc_layer_configurations

        self.emb = nn.Embedding(
            num_embeddings=self.vocab_size + 1,
            embedding_dim=self.embedding_dim,
        )

        self.conv_layers = _generate_conv_layers(
            self.embedding_dim, self.conv_layer_configurations
        )
        self.fc_layers = _generate_fc_layers(
            self.n_classes, self.dropout, self.fc_layer_configurations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x).permute(0, 2, 1)
        for conv in self.conv_layers:
            x = conv(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        for fc in self.fc_layers:
            x = fc(x)
        return x
