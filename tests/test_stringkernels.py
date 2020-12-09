"""Test String-Kernel based models."""
import numpy as np
from numpy.testing import assert_array_equal

from tuhlbox.stringkernels import (
    intersection_kernel,
    presence_kernel,
    spectrum_kernel,
)

docs = [
    'I like this old movie. The movie is very nice.',
    'In my opinion the book tells a very nice story. I really like it.',
    'I wonder if you could drink this juice. It tastes so bad. Isn’t it bad?',
    'Your dish is too spicy. You must be a such bad cook. '
    'Don’t worry, I am as bad as you.',
]

ngram_min, ngram_max = 1, 4


def test_intersection_kernel():
    """Test intersection kernel by comparing with original code."""
    # obtained from:
    # java ComputeStringKernel intersection 1 4 sentences.txt <outfile>
    expected = np.array(
        [[178, 95, 66, 49], [95, 254, 72, 72], [66, 72, 278, 112],
         [49, 72, 112, 334]],
        dtype=int,
    )
    assert_array_equal(expected,
                       intersection_kernel(docs, docs, ngram_min, ngram_max))


def test_presence_kernel():
    """Test presence kernel by comparing with original code."""
    # obtained from:
    # java ComputeStringKernel presence 1 4 sentences.txt <outfile>

    expected = np.array(
        [[128, 67, 42, 29], [67, 197, 38, 42], [42, 38, 209, 64],
         [29, 42, 64, 235]],
        dtype=int,
    )
    assert_array_equal(expected,
                       presence_kernel(docs, docs, ngram_min, ngram_max))


def test_spectrum_kernel():
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
    assert_array_equal(expected,
                       spectrum_kernel(docs, docs, ngram_min, ngram_max))
