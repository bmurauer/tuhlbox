# This transformers are a re-implementation of the feature extraction method
# presented in Llorens(2016).

import random
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LifeVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, fragment_sizes=None, samples=200, sample_type="bow",
                 force=True):
        if fragment_sizes is None:
            fragment_sizes = [200, 500, 800, 1000, 1500, 2000, 3000, 4000]
        valid_sample_types = ["bow", "fragment", "both"]
        if sample_type not in valid_sample_types:
            raise ValueError(
                f"unknown sample type: {sample_type}. valid values: "
                f"{valid_sample_types}"
            )
        self.fragment_sizes = fragment_sizes
        self.samples = samples
        self.sample_type = sample_type
        self.force = force

    def fit(self, x, y=None):
        return self

    def sample(self, words, fragment_size, method):
        ret = []
        wordcount = len(words)
        if wordcount < fragment_size:
            if self.force:
                fragment_size = wordcount
            else:
                raise ValueError(
                    f"fragment size ({fragment_size}) is larger than document "
                    f"size ({wordcount}) for document starting with: \n\n"
                    f'{" ".join(words[:50])}\n\n'
                )
        for _ in range(self.samples):
            if method == "fragment":
                left = random.randint(0, wordcount - fragment_size)
                right = left + fragment_size
                ret.append(words[left:right])
            if method == "bow":
                ret.append(random.sample(words, fragment_size))
        return ret

    def get_features_for_sample(self, sample):
        counts = defaultdict(int)
        for word in sample:
            counts[word] += 1
        v0 = len(counts.keys())
        v1, v2, v3 = 0, 0, 0
        for occurrances in counts.values():
            if occurrances <= 1:
                v1 += 1
            elif occurrances <= 4:
                v2 += 1
            elif occurrances <= 10:
                v3 += 1

        return [v0, v1, v2, v3]

    def get_features(self, document, sample_size):
        if self.sample_type == "both":
            return np.concatenate(
                [
                    self._get_features(document, sample_size, "bow"),
                    self._get_features(document, sample_size, "fragment"),
                ]
            )
        else:
            return self._get_features(document, sample_size, self.sample_type)

    def _get_features(self, document, fragment_size, method):
        samples = self.sample(document, fragment_size, method)
        features = []
        for sample in samples:
            features.append(self.get_features_for_sample(sample))
        features = np.array(features)
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        return np.concatenate(
            [means,
             np.divide(means, stds, out=np.zeros_like(means), where=stds != 0)]
        )

    def transform(self, x, y=None):
        ret = []
        for document in x:
            doc = [self.get_features(document, size) for size in
                   self.fragment_sizes]
            ret.append(np.concatenate(doc))

        # some classifiers like XGBoost require a numpy array if nested
        return np.array(ret)
