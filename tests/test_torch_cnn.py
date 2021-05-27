"""Test basic skorch models with CNN network."""

import torch
from dstoolbox.transformers import TextFeaturizer, Padder2d
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from skorch import NeuralNetClassifier

from tuhlbox.torch_cnn import CharCNN

x, y = fetch_20newsgroups(return_X_y=True)

VOCAB_SIZE = 1000
EMB_DIM = 300
MAX_SEQ_LEN = 100

pipe = make_pipeline(
    TextFeaturizer(max_features=VOCAB_SIZE),
    Padder2d(pad_value=VOCAB_SIZE, max_len=MAX_SEQ_LEN, dtype=int),
    NeuralNetClassifier(
        module=CharCNN,
        device='cuda',
        batch_size=54,
        max_epochs=5,
        lr=0.01,
        optimizer=torch.optim.Adam,
        module__embedding_dim=EMB_DIM,
        module__vocab_size=VOCAB_SIZE,
        module__max_seq_length=MAX_SEQ_LEN,
        module__num_classes=len(set(y)),
    )
)

pipe.fit(x, y)
