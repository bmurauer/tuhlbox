"""Test basic skorch models with CNN network."""

import torch
from dstoolbox.transformers import Padder2d, TextFeaturizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tuhlbox.torch_classifier import TorchClassifier
from tuhlbox.torch_cnn import CharCNN
from tuhlbox.torch_lstm import RNNClassifier

x, y = fetch_20newsgroups(
    return_X_y=True, categories=["alt.atheism", "talk.religion.misc"]
)

x_train, x_test, y_train, y_test = train_test_split(x, y)

VOCAB_SIZE = 1000
EMB_DIM = 300
MAX_SEQ_LEN = 100


def test_cnn() -> None:
    pipe = make_pipeline(
        TextFeaturizer(max_features=VOCAB_SIZE),
        Padder2d(pad_value=VOCAB_SIZE, max_len=MAX_SEQ_LEN, dtype=int),
        TorchClassifier(
            module=CharCNN,
            device="cpu",  # the gitlab CI does not have cuda
            batch_size=54,
            max_epochs=5,
            learn_rate=0.01,
            optimizer=torch.optim.Adam,
            model_kwargs=dict(
                module__max_seq_len=MAX_SEQ_LEN,
                module__embedding_dim=EMB_DIM,
                module__num_features=VOCAB_SIZE,
            ),
        ),
    )

    pipe.fit(x_train, y_train)
    pipe.predict(x_test)


def test_lstm() -> None:
    pipe = make_pipeline(
        TextFeaturizer(max_features=VOCAB_SIZE),
        Padder2d(pad_value=VOCAB_SIZE, max_len=MAX_SEQ_LEN, dtype=int),
        TorchClassifier(
            module=RNNClassifier,
            device="cpu",  # the gitlab CI does not have cuda
            batch_size=4,
            max_epochs=5,
            learn_rate=0.01,
            optimizer=torch.optim.Adam,
        ),
    )

    pipe.fit(x_train, y_train)
    pipe.predict(x_test)
