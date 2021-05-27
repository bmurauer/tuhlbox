"""Tests LIFE models."""
import itertools
import unittest

import numpy as np
from tuhlbox.life import LifeVectorizer


class TestLife(unittest.TestCase):
    """Tests LIFE models."""

    def test_short_fragment(self) -> None:
        """Calculate correct FRAGMENT kernel results."""
        fragment_sizes = [1000]  # larger than text
        text = [[str(x) for x in range(100)]]

        transformer = LifeVectorizer(fragment_sizes, 1, "fragment", force=True)
        actual = transformer.transform(text)[0].tolist()
        predicted = [100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(actual, predicted)

    def test_short_bow(self) -> None:
        """Calculate correct BOW kernel results."""
        fragment_sizes = [1000]  # larger than text
        text = [[str(x) for x in range(100)]]

        transformer = LifeVectorizer(fragment_sizes, 1, "bow", force=True)
        actual = transformer.transform(text)[0].tolist()
        predicted = [100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(actual, predicted)

    def test_short_both(self) -> None:
        """Calculate correct BOTH kernel results."""
        fragment_sizes = [1000]  # larger than text
        text = [str(x) for x in range(100)]

        transformer = LifeVectorizer(fragment_sizes, 1, "both", force=True)
        actual = transformer.transform([text])[0].tolist()
        predicted = [
            100.0,
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            100.0,
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.assertEqual(actual, predicted)

    def test_short_document_exceptions(self) -> None:
        """Use short documents."""
        text = [str(x) for x in range(100)]
        for m in ["fragment", "bow", "both"]:
            transformer = LifeVectorizer([1000], 1, m, force=False)
            self.assertRaises(ValueError, transformer.transform, text)

    def test_random_shape_sizes(self) -> None:
        """Use random fragment sizes."""
        n_sizes = np.random.randint(1, 10)
        fragment_sizes = list(np.random.randint(2, 50, size=n_sizes))
        text = [[str(x) for x in range(100)]]

        vec1 = LifeVectorizer(fragment_sizes, 1, "fragment")
        vec2 = LifeVectorizer(fragment_sizes, 1, "bow")
        vec3 = LifeVectorizer(fragment_sizes, 1, "both")

        self.assertEqual(vec1.transform(text)[0].shape, (n_sizes * 8,))
        self.assertEqual(vec2.transform(text)[0].shape, (n_sizes * 8,))
        self.assertEqual(vec3.transform(text)[0].shape, (n_sizes * 16,))

    def test_life_single_fragment(self) -> None:
        """Use a single window size with FRAGMENT."""
        transformer = LifeVectorizer([42], 50, "fragment")
        text = [str(x) for x in range(100)]
        actual = transformer.transform([text])[0].tolist()
        predicted = [42.0, 42.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(actual, predicted)

    def test_life_single_bow(self) -> None:
        """Use a single window size with BOW."""
        transformer = LifeVectorizer([42], 100, "bow")
        text = [str(x) for x in range(100)]
        actual = transformer.transform([text])[0].tolist()
        predicted = [42.0, 42.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(actual, predicted)

    def test_life_single_bfs(self) -> None:
        """Use a single window size with BOTH."""
        transformer = LifeVectorizer([42], 100, "both")
        text = [str(x) for x in range(100)]
        actual = transformer.transform([text])[0].tolist()
        predicted = [
            42.0,
            42.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            42.0,
            42.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.assertEqual(actual, predicted)

    def test_life_double_fragment(self) -> None:
        """Use a two window size with FRAGMENT."""
        transformer = LifeVectorizer([42, 41], 50, "fragment")
        text = [str(x) for x in range(100)]
        actual = transformer.transform([text])[0].tolist()
        predicted = [
            42.0,
            42.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            41.0,
            41.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.assertEqual(actual, predicted)

    def test_life_double_bow(self) -> None:
        """Use a two window size with BOW."""
        transformer = LifeVectorizer([42, 41], 100, "bow")
        text = [str(x) for x in range(100)]
        actual = transformer.transform([text])[0].tolist()
        predicted = [
            42.0,
            42.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            41.0,
            41.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.assertEqual(actual, predicted)

    def test_life_double_bfs(self) -> None:
        """Use a two window size with BOTH."""
        rand_1 = np.random.randint(2, 100)
        rand_2 = np.random.randint(2, 100)
        transformer = LifeVectorizer([rand_1, rand_2], 100, "both")
        text = [str(x) for x in range(100)]
        actual = transformer.transform([text])[0].tolist()
        predicted = [
            float(rand_1),
            float(rand_1),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(rand_1),
            float(rand_1),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(rand_2),
            float(rand_2),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(rand_2),
            float(rand_2),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.assertEqual(actual, predicted)

    def test_life_half_fragment(self) -> None:
        """Use more detailed expectations."""
        frag_size = np.random.randint(2, 100)
        transformer = LifeVectorizer([frag_size], 100, "fragment")
        # this text consists of pairs of two identical words
        pairs = [(str(y), str(y)) for y in range(100)]
        text = list(itertools.chain.from_iterable(pairs))
        actual = transformer.transform([text])[0].tolist()

        vocabulary_size = actual[0]
        freq_1_count = actual[1]
        freq_4_count = actual[2]
        freq_10_count = actual[3]

        self.assertTrue(vocabulary_size <= frag_size / 2 + 1)
        self.assertTrue(vocabulary_size >= frag_size / 2)
        #  in the most extreme case, no samples have a hapax legomena
        self.assertTrue(freq_1_count >= 0)
        #  in the most extreme case, all samples have 2 hapax legomena
        self.assertTrue(freq_1_count <= 2)
        #  in the most extreme case, all samples have 2 hapax legomena and
        #  (frag_size-2)/2 dis legomena
        self.assertTrue(freq_4_count >= frag_size / 2 - 1)
        #  in the most extreme case, no samples have hapax legomena and
        #  frag_size/2 dis legomena
        self.assertTrue(freq_4_count <= frag_size / 2)
        #  no sample should have word occurring more often than twice
        self.assertAlmostEqual(freq_10_count, 0.0)
