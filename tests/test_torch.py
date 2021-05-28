"""Test basic skorch models with CNN network."""

import torch
from dstoolbox.transformers import Padder2d, TextFeaturizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from torch import nn
from tuhlbox.torch_classifier import TorchClassifier
from tuhlbox.torch_cnn import CharCNN, ConvLayerConfig, FcLayerConfig
from tuhlbox.torch_lstm import RNNClassifier

x, y = fetch_20newsgroups(return_X_y=True)
VOCAB_SIZE = 1000
EMB_DIM = 300
MAX_SEQ_LEN = 100


def test_cnn() -> None:
    pipe = make_pipeline(
        TextFeaturizer(max_features=VOCAB_SIZE),
        Padder2d(pad_value=VOCAB_SIZE, max_len=MAX_SEQ_LEN, dtype=int),
        TorchClassifier(
            module=CharCNN,
            max_seq_len=MAX_SEQ_LEN,
            device="cpu",
            batch_size=54,
            max_epochs=5,
            learn_rate=0.01,
            optimizer=torch.optim.Adam,
            model_kwargs=dict(
                module__emb_layer=nn.Embedding(VOCAB_SIZE + 1, EMB_DIM),
                module__conv_layer_configs=[
                    ConvLayerConfig(EMB_DIM, 50, 7, 1, 3, 3),
                    ConvLayerConfig(50, 50, 5, 1, 3, 3),
                ],
                module__fc_layer_configs=[
                    FcLayerConfig(None, 256),  # will be calculated automagically
                    FcLayerConfig(256, 128),
                    FcLayerConfig(128, 64),
                ],
            ),
        ),
    )

    pipe.fit(x, y)


def test_lstm() -> None:
    pipe = make_pipeline(
        TextFeaturizer(max_features=VOCAB_SIZE),
        Padder2d(pad_value=VOCAB_SIZE, max_len=MAX_SEQ_LEN, dtype=int),
        TorchClassifier(
            module=RNNClassifier,
            device="cuda",
            batch_size=54,
            max_epochs=5,
            learn_rate=0.01,
            optimizer=torch.optim.Adam,
        ),
    )

    pipe.fit(x, y)
