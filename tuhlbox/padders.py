from dstoolbox.transformers import Padder2d
from sklearn.base import TransformerMixin, BaseEstimator


class MaxLengthPadder(TransformerMixin, BaseEstimator):

    def __init__(self, pad_value, min_length=5):
        self.pad_value = pad_value
        self.min_length = min_length

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, documents):
        max_len = max([len(document) for document in documents]
                      + [self.min_length])
        print('max len is', max_len)
        padder = Padder2d(max_len=max_len, pad_value=self.pad_value)
        return padder.transform(documents)
