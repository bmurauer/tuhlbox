"""Contains models that use keras for neural networks."""
import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Embedding
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

logger = logging.getLogger(__name__)


def count_num_words(x):
    return int(max([max(document) for document in x])) + 1


class KerasModel(BaseEstimator, ClassifierMixin):
    """A generic model that can use external model_fn functions."""

    def __init__(
        self,
        model_fn=None,
        embedding_dim=300,
        epochs=10,
        batch_size=2,
        validation_data=None,
        exclude_embedding=False,
    ):
        """
        Initialize the model.
        Args:
            model_fn: a keras model building function. See keras for details.
            embedding_dim: the dimension of the (first) embedding layer.
            epochs: epoch to train.
            batch_size: batch size of network training
            validation_data: optional data that will be passed to the model's
                fit function to get running validation scores during fitting.
            exclude_embedding: if set, does not use a first embedding layer.
        """
        # passed parameters
        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.model_fn = model_fn
        self.batch_size = batch_size  # for sklearn cloning
        self.model_kwargs = {}
        self.validation_data = validation_data
        self.exclude_embedding = exclude_embedding

    def fit(self, x, y, **fit_params):
        """Fit the model."""
        num_words = count_num_words(x)
        sequence_length = len(x[0])

        if not self.exclude_embedding:
            embedding_layer = Embedding(
                num_words,
                self.embedding_dim,
                input_length=sequence_length,
                trainable=True,
            )
            self.model_kwargs['embedding_layer'] = embedding_layer

        self.model_kwargs['num_classes'] = len(set(y))
        self.model_kwargs['epochs'] = self.epochs
        self.model_kwargs['batch_size'] = self.batch_size
        fit_params['validation_data'] = self.validation_data
        self.model_kwargs['callbacks'] = [
            TensorBoard(
                log_dir='tensorboard-logs',
            )
        ]
        self.model = KerasClassifier(self.model_fn, **self.model_kwargs)
        x = np.asarray(x).astype('float32')
        self.model.fit(x, y, **fit_params)

    def predict(self, x):
        """Predict the data."""
        return self.model.predict(x)

    @property
    def configuration(self):
        """Return a database representation for this model."""
        return self.get_params()
