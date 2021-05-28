"""Basic CNN model."""
import math
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn

ConvLayerConfig = namedtuple(
    "ConvLayerConfig",
    [
        "in_channels",
        "out_channels",
        "conv_kernel_size",
        "conv_stride",
        "max_kernel_size",
        "max_stride",
    ],
)

FcLayerConfig = namedtuple("FcLayerConfig", ["in_features", "out_features"])


def create_conv_layers(configs: List[ConvLayerConfig]) -> List[nn.Sequential]:
    result = []
    for config in configs:
        result.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    kernel_size=(config.conv_kernel_size,),
                    stride=(config.conv_stride,),
                ),
                nn.ReLU(),
                nn.MaxPool1d(
                    kernel_size=(config.max_kernel_size,),
                    stride=(config.max_stride,),
                ),
            ),
        )
    return result


def create_fc_layers(
    fc_configs: List[FcLayerConfig], conv_configs: List[ConvLayerConfig], start_n: int
) -> List[nn.Sequential]:
    result = []
    start_n = compute_first_fc_layer_input_size(conv_configs, start_n)
    for i, config in enumerate(fc_configs):
        in_features = start_n if i == 0 else config.in_features
        result.append(
            nn.Sequential(
                nn.Linear(in_features=in_features, out_features=config.out_features)
            )
        )
    return result


def compute_first_fc_layer_input_size(
    conv_configs: List[ConvLayerConfig], n: int
) -> int:
    """
    Calculates the input dimension of the first fully connected layer.

    See https://datascience.stackexchange.com/a/40991/9281

    Args:
        conv_configs: Configurations of the convolution layers
        n: starting value (max sequence length)

    Returns:
        the dimension of the first fc layer
    """

    def get_output_dim(in_size: int, kernel: int, stride: int) -> int:
        return math.floor(((in_size - kernel) / stride) + 1)

    for config in conv_configs:
        n = get_output_dim(n, config.conv_kernel_size, config.conv_stride)
        n = get_output_dim(n, config.max_kernel_size, config.max_stride)

    last_conv_out_channels = conv_configs[-1].out_channels
    return n * last_conv_out_channels


class CharCNN(nn.Module):
    """Basic CNN model that can be built with variable amounts of layers etc."""

    def __init__(
        self,
        n_classes: int,
        max_seq_len: int,
        emb_layer: nn.Embedding,
        conv_layer_configs: List[ConvLayerConfig],
        fc_layer_configs: List[FcLayerConfig],
    ):
        super().__init__()
        self.emb_layer = emb_layer
        self.conv_layers = create_conv_layers(conv_layer_configs)
        self.fc_layers = create_fc_layers(
            fc_layer_configs, conv_layer_configs, max_seq_len
        )
        self.last_layer = nn.Sequential(
            nn.Linear(fc_layer_configs[-1].out_features, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the embedding layer returns the values in a different order than is required
        # by the convolution layers, so we have to swap them
        x = self.emb_layer(x).permute(0, 2, 1)
        for conv in self.conv_layers:
            x = conv(x)
        # flatten all values
        x = x.view(x.size(0), -1)
        for fc in self.fc_layers:
            x = fc(x)
        return self.last_layer(x)

    @staticmethod
    def conv_layer(
        in_channels: int,
        out_channels: int,
        conv_kernel: int,
        conv_stride: int = 1,
        max_kernel: int = 3,
        max_stride: int = 3,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(conv_kernel,),
                stride=(conv_stride,),
            ),
            nn.ReLU(),
            nn.MaxPool1d(max_kernel, max_stride),
        )

    @staticmethod
    def fc_layer(
        in_size: int,
        out_size: int,
        dropout: float = 0.0,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features=in_size, out_features=out_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
