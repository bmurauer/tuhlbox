"""Basic CNN model."""
import torch
import torch.nn as nn


class CharCNN(nn.Module):
    """Basic CNN model that can be built with variable amounts of layers etc."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        max_seq_length: int,
        dropout: float = 0.0,
    ):
        """
        Create a new CNN model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Size of embedding vectors
            num_classes: Number of classes used
            max_seq_length: Max. sequence length for each token
            dropout: random dropout fraction
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

        # Embedding Input dimensions (x, y):
        # x: batch size
        # y: max. seq. length

        self.emb = nn.Embedding(
            num_embeddings=self.vocab_size + 1,
            embedding_dim=self.embedding_dim,
        )

        # Embedding Output dimensions (x, y, z):
        # x: batch size
        # y: max. seq. length
        # z: embedding_dim

        # Conv1D dimensions (x, y, z):
        # x = batch size
        # y = number of channels
        # z = number of features

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=50,
                kernel_size=7,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=50,
                out_channels=50,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )

        self.fc1 = nn.Sequential(nn.Linear(450, 256), nn.ReLU(), nn.Dropout(p=dropout))

        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=dropout))

        self.fc3 = nn.Linear(128, self.num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)

        # Embedding Output dimensions (x, y, z):
        # x: batch size
        # y: max. seq. length
        # z: embedding_dim

        # Conv1D dimensions (x, y, z):
        # x = batch size
        # y = number of channels
        # z = number of features

        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.conv2(x)

        # collapse
        x = x.view(x.size(0), -1)

        # linear layer
        x = self.fc1(x)

        # linear layer
        x = self.fc2(x)

        # linear layer
        x = self.fc3(x)

        # output layer
        x = self.log_softmax(x)

        return x
