from collections.abc import Iterable

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin

# gensim is really loud.
import logging

logging.getLogger('gensim').setLevel(logging.WARNING)
logging.getLogger('gensim').propagate = False


def _sanity_check(documents):
    if len(documents) < 1:
        raise ValueError('empty document set provided')

    if type(documents[0]) == str or not isinstance(documents[1], Iterable):
        raise TypeError('this transformer only works on pre-split data')


class Doc2VecTransformer(TransformerMixin, BaseEstimator):
    """

    input: a list of list of strings
    output: a list of list of numbers

    """

    def __init__(
        self,
        learning_rate=0.02,
        epochs=20,
        vector_size=100,
        alpha=0.025,
        min_alpha=0.00025,
        min_count=2,
        dm=1,
        workers=1,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vector_size = vector_size
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.dm = dm
        self.workers = workers

        self.model = Doc2Vec(
            vector_size=vector_size,
            alpha=alpha,
            min_alpha=min_alpha,
            min_count=min_count,
            dm=dm,
            workers=workers,
        )

    def fit(self, documents, labels=None, **fit_params):
        documents = [
            [str(x) for x in document]
            for document in documents
        ]

        _sanity_check(documents)
        tagged_x = [
            TaggedDocument(words=row, tags=[])
            for _, row in enumerate(documents)
        ]
        self.model.build_vocab(tagged_x)
        self.model.train(tagged_x, total_examples=self.model.corpus_count,
                         epochs=self.epochs)
        return self

    def transform(self, documents):
        documents = [
            [str(x) for x in document]
            for document in documents
        ]
        _sanity_check(documents)
        vectors = [self.model.infer_vector(doc) for doc in documents]
        return np.array(vectors)
