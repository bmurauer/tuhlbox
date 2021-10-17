"""Tests Second-Order Models."""
from tuhlbox.subfreq import SubFrequencyVectorizer


def test_subset_vectorizer() -> None:
    """Test if the transformer produces correct output shapes."""
    transformer = SubFrequencyVectorizer()

    texts = [
        "male boy man girly ",
        "female girly woman boy boy woman",
        "super duper super man ",
        "man manly boy man",
    ]
    targets = [0, 1, 2]

    actual = transformer.fit_transform(texts, targets)
    assert actual.shape == (4, 3)  # 4 samples, 3 classes
