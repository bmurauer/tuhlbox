"""Tests tree-based models."""
import unittest

import nltk

from tuhlbox.tree import (
    StringToTreeTransformer,
    TreeChainTransformer,
    WordToPosTransformer,
)


class TestTrees(unittest.TestCase):
    """Tests tree-based models."""

    def test_tree_parsing(self):
        """Test if strings are parsed correctly."""
        transformer = StringToTreeTransformer()
        # one document could contain many sentences, and each sentence is a
        # tree.
        documents = [['(S (NP I) (VP (V saw) (NP him)))']]
        expected = [[nltk.Tree.fromstring(documents[0][0])]]
        actual = transformer.transform(documents)
        self.assertEqual(actual, expected)

    def test_pos_extraction(self):
        """Test if POS tags are extracted correctly."""
        transformer = WordToPosTransformer()
        documents = [
            [nltk.Tree.fromstring('(S (NP I) (VP (V saw) (NP him)))')]]
        expected = [['NP', 'V', 'NP']]
        actual = transformer.transform(documents)
        self.assertEqual(actual, expected)

    def test_pos_short_chains(self):
        """Test if short chains are extracted correctly with single joining."""
        transformer = TreeChainTransformer(max_length=2,
                                           combine_chain_elements=' ')
        documents = [
            # document 1
            [
                #  tree 1
                nltk.Tree.fromstring(
                    '(S (DP (D the) (NP dog)) (VP (V chased) (DP (D the) (NP '
                    'cat))))'
                )
            ],
            # document 2
            [
                # tree 2
                nltk.Tree.fromstring('(S (NP I) (VP (V saw) (NP him)))'),
                # tree 3
                nltk.Tree.fromstring('(S (NP I) (VP (V saw) (NP him)))'),
            ],
        ]
        expected = [
            # document 1
            [
                # tree 1
                ['S DP', 'DP D', 'DP NP', 'S VP', 'VP V', 'VP DP', 'DP D',
                 'DP NP']
            ],
            # document 2
            [
                # tree 2
                ['S NP', 'S VP', 'VP V', 'VP NP'],
                # tree 3
                ['S NP', 'S VP', 'VP V', 'VP NP'],
            ],
        ]
        actual = list(transformer.transform(documents))
        self.assertEqual(actual, expected)

    def test_pos_short_chains_nocombine(self):
        """Test if short chains are extracted correctly without joining."""
        transformer = TreeChainTransformer(max_length=2,
                                           combine_chain_elements=None)
        documents = [
            # document 1
            [
                #  tree 1
                nltk.Tree.fromstring(
                    '(S (DP (D the) (NP dog)) (VP (V chased) (DP (D the) (NP '
                    'cat))))'
                )
            ],
            # document 2
            [
                # tree 2
                nltk.Tree.fromstring('(S (NP I) (VP (V saw) (NP him)))'),
                # tree 3
                nltk.Tree.fromstring('(S (NP I) (VP (V saw) (NP him)))'),
            ],
        ]
        expected = [
            # document 1
            [
                # tree 1
                [
                    ['S', 'DP'],
                    ['DP', 'D'],
                    ['DP', 'NP'],
                    ['S', 'VP'],
                    ['VP', 'V'],
                    ['VP', 'DP'],
                    ['DP', 'D'],
                    ['DP', 'NP'],
                ]
            ],
            # document 2
            [
                # tree 2
                [['S', 'NP'], ['S', 'VP'], ['VP', 'V'], ['VP', 'NP']],
                # tree 3
                [['S', 'NP'], ['S', 'VP'], ['VP', 'V'], ['VP', 'NP']],
            ],
        ]
        actual = list(transformer.transform(documents))
        self.assertEqual(actual, expected)

    def test_pos_long_chains(self):
        """Test if long chains are extracted correctly."""
        transformer = TreeChainTransformer(max_length=None,
                                           combine_chain_elements=' ')
        documents = [
            # document 1
            [
                #  tree 1
                nltk.Tree.fromstring(
                    '(S (DP (D the) (NP dog)) (VP (V chased) (DP (D the) (NP '
                    'cat))))'
                )
            ],
            # document 2
            [
                # tree 2
                nltk.Tree.fromstring('(S (NP I) (VP (V saw) (NP him)))')
            ],
        ]
        expected = [
            # document 1
            [
                # tree 1
                ['S DP D', 'S DP NP', 'S VP V', 'S VP DP D', 'S VP DP NP']
            ],
            # document 2
            [
                # tree 2
                ['S NP', 'S VP V', 'S VP NP']
            ],
        ]
        actual = list(transformer.transform(documents))
        self.assertEqual(actual, expected)

    def test_pos_short_chains_combine_chains(self):
        """Test if short chains are extracted correctly with double joining."""
        transformer = TreeChainTransformer(
            max_length=None,
            combine_chain_elements=' ',
            combine_chains='@',
            combine_strings=None,
        )
        documents = [
            # document 1
            [
                #  tree 1
                nltk.Tree.fromstring(
                    '(S (DP (D the) (NP dog)) (VP (V chased) (DP (D the) (NP '
                    'cat))))'
                ),
                #  tree 2
                nltk.Tree.fromstring('(S (NP I) (VP (V saw) (NP him)))'),
            ]
        ]
        expected = [
            # document 1 - note that the two sentences are separated by #
            [
                # tree 1
                'S DP D@S DP NP@S VP V@S VP DP D@S VP DP NP',
                # tree 2
                'S NP@S VP V@S VP NP',
            ]
        ]
        actual = list(transformer.transform(documents))
        self.assertEqual(actual, expected)

    def test_pos_short_chains_combine_all(self):
        """Test if short chains are extracted correctly with triple joining."""
        transformer = TreeChainTransformer(
            max_length=None,
            combine_chain_elements=' ',
            combine_chains='@',
            combine_strings='#',
        )
        documents = [
            # document 1
            [
                #  tree 1
                nltk.Tree.fromstring(
                    '(S (DP (D the) (NP dog)) (VP (V chased) (DP (D the) (NP '
                    'cat))))'
                ),
                #  tree 2
                nltk.Tree.fromstring('(S (NP I) (VP (V saw) (NP him)))'),
            ]
        ]
        expected = [
            # document 1 - note that the two sentences are separated by #
            'S DP D@S DP NP@S VP V@S VP DP D@S VP DP NP#S NP@S VP V@S VP NP'
        ]
        actual = list(transformer.transform(documents))
        self.assertEqual(actual, expected)

    def test_combine_chains_error(self):
        """Test if illegal parameters cause Error with joining."""
        kwargs = {'combine_chain_elements': None, 'combine_chains': '#'}
        self.assertRaises(ValueError, TreeChainTransformer, **kwargs)

    def test_combine_strings_error_1(self):
        """Test if illegal parameters cause Error with joining."""
        kwargs = {
            'combine_chain_elements': ' ',
            'combine_chains': None,
            'combine_strings': ' ',
        }
        self.assertRaises(ValueError, TreeChainTransformer, **kwargs)

    def test_combine_strings_error_2(self):
        """Test if illegal parameters cause Error with joining."""
        kwargs = {
            'combine_chain_elements': None,
            'combine_chains': ' ',
            'combine_strings': ' ',
        }
        self.assertRaises(ValueError, TreeChainTransformer, **kwargs)
