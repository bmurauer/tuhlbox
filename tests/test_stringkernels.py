"""Test String-Kernel based models."""

import numpy as np
from dstoolbox.transformers import TextFeaturizer
from numpy.testing import assert_array_almost_equal, assert_array_equal
from tuhlbox.stringkernels import (intersection_kernel, pqgram_kernel,
                                   presence_kernel, spectrum_kernel)

docs = [
    "i like this old movie. the movie is very nice.",
    "in my opinion the book tells a very nice story. i really like it.",
    "i wonder if you could drink this juice. it tastes so bad. isn’t it bad?",
    "your dish is too spicy. you must be a such bad cook. "
    "don’t worry, i am as bad as you.",
]

ngram_min, ngram_max = 1, 4


def test_intersection_kernel() -> None:
    """Test intersection kernel by comparing with original code."""
    # obtained from:
    # java ComputeStringKernel intersection 1 4 sentences.txt <outfile>
    expected = np.array(
        [[178, 95, 66, 49], [95, 254, 72, 72], [66, 72, 278, 112], [49, 72, 112, 334]],
        dtype=int,
    )

    tf = TextFeaturizer(analyzer="char", ngram_range=(ngram_min, ngram_max))
    ngrams = tf.fit_transform(docs)
    assert_array_equal(expected, intersection_kernel(ngrams, ngrams))


def test_presence_kernel() -> None:
    """Test presence kernel by comparing with original code."""
    # obtained from:
    # java ComputeStringKernel presence 1 4 sentences.txt <outfile>
    expected = np.array(
        [[128, 67, 42, 29], [67, 197, 38, 42], [42, 38, 209, 64], [29, 42, 64, 235]],
        dtype=int,
    )
    tf = TextFeaturizer(analyzer="char", ngram_range=(ngram_min, ngram_max))
    ngrams = tf.fit_transform(docs)
    assert_array_equal(expected, presence_kernel(ngrams, ngrams))


def test_spectrum_kernel() -> None:
    """Test spectrum kernel by comparing with original code."""
    # obtained from:
    # java ComputeStringKernel spectrum 1 4 sentences.txt <outfile>
    expected = np.array(
        [
            [390, 335, 300, 313],
            [335, 598, 393, 458],
            [300, 393, 680, 585],
            [313, 458, 585, 1006],
        ],
        dtype=int,
    )
    tf = TextFeaturizer(analyzer="char", ngram_range=(ngram_min, ngram_max))
    ngrams = tf.fit_transform(docs)
    assert_array_equal(expected, spectrum_kernel(ngrams, ngrams))


def test_pqgram_kernel() -> None:

    # these example values are taken from the paper
    x = np.array(
        [
            [
                "*-a-*-*-a",
                "a-a-*-*-e",
                "a-e-*-*-*",
                "a-a-*-e-b",
                "a-b-*-*-*",
                "a-a-e-b-*",
                "a-a-b-*-*",
                "*-a-*-a-b",
                "a-b-*-*-*",
                "*-a-a-b-c",
                "a-c-*-*-*",
                "*-a-b-c-*",
                "*-a-c-*-*",
            ],
            [
                "*-a-*-*-a",
                "a-a-*-*-e",
                "a-e-*-*-*",
                "a-a-*-e-b",
                "a-b-*-*-*",
                "a-a-e-b-*",
                "a-a-b-*-*",
                "*-a-*-a-b",
                "a-b-*-*-*",
                "*-a-a-b-d",
                "a-d-*-*-*",
                "*-a-b-d-*",
                "*-a-d-*-*",
            ],
        ]
    )

    expected = np.array([[0, 0.470588], [0.470588, 0]])
    assert_array_almost_equal(expected, pqgram_kernel(x, x))


def test_kernels_with_non_strings() -> None:
    """Tests the presence kernel with documents not consisting of strings."""

    ngrams = np.array([[1, 2, 3, 3, 2], [2, 3, 3, 1, 4], [6, 7, 5, 1, 2]])

    expected = np.array(
        [
            [3, 3, 2],  # the first document has 4 elements common to itself,
            # 4 elements in common with the second document
            # and 2 documents in common with the last document.
            [3, 4, 2],
            [2, 2, 5],
        ]
    )
    actual = presence_kernel(ngrams, ngrams)
    assert_array_equal(expected, actual)
