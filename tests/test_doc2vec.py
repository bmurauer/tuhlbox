"""Test Doc2Vec Models."""
from nltk import word_tokenize
from tuhlbox.doc2vec import Doc2VecTransformer


def test_transformation() -> None:
    """Expect a correct vector size."""
    documents = [
        word_tokenize("This is an example sentence."),
        word_tokenize("This is a second piece of text."),
    ]

    transformer = Doc2VecTransformer(vector_size=123)
    actual = transformer.fit_transform(documents)
    assert len(actual) == len(documents)
    assert len(actual[0]) == 123
    assert len(actual[1]) == 123
