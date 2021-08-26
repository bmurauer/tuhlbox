from typing import List
from tuhlbox.stanza import StanzaNlpToFieldTransformer, StanzaParserTransformer

sentences: List


def setup_function() -> None:
    global sentences
    documents = [
        "I have been trying to reach you. This is another sentence in this document.",
        "This is a second document, independent from the others.",
    ]
    parser = StanzaParserTransformer("en", silent=True)
    sentences = parser.transform(documents)


def test_stanza_nlp_to_field_transformer_xpos() -> None:
    transformer = StanzaNlpToFieldTransformer("xpos")
    expected = [
        [  # document 1
            ["PRP", "VBP", "VBN", "VBG", "TO", "VB", "PRP", "."],  # sentence 1
            ["DT", "VBZ", "DT", "NN", "IN", "DT", "NN", "."],  # sentence 2
        ],
        [  # document 2 (one sentence)
            ["DT", "VBZ", "DT", "JJ", "NN", ",", "JJ", "IN", "DT", "NNS", "."],
        ],
    ]
    actual = transformer.transform(sentences)
    assert expected == actual


def test_stanza_nlp_to_field_transformer_upos() -> None:
    transformer = StanzaNlpToFieldTransformer("upos")
    expected = [
        [  # document 1
            ["PRON", "AUX", "AUX", "VERB", "PART", "VERB", "PRON", "PUNCT"],
            ["PRON", "AUX", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
        ],
        [  # document 2
            [
                "PRON",
                "AUX",
                "DET",
                "ADJ",
                "NOUN",
                "PUNCT",
                "ADJ",
                "ADP",
                "DET",
                "NOUN",
                "PUNCT",
            ]
        ],
    ]
    actual = transformer.transform(sentences)
    assert expected == actual
