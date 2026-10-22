import pytest
from processor.cleaner import clean_text


def test_clean_text_normalizes_whitespace():
    text = "  Hello    world!  \n\n  Python   is great.  "
    result = clean_text(text)
    assert result == "Hello world!\nPython is great."


def test_clean_text_handles_empty_string():
    text = ""
    result = clean_text(text)
    assert result == ""


def test_clean_text_preserves_punctuation():
    text = "Don't remove punctuation!"
    result = clean_text(text)
    assert result == "Don't remove punctuation!"

def test_clean_text_rejects_non_string():
    with pytest.raises(TypeError):
        clean_text(None)

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello    Python", "Hello Python"),
        ("  Hello  ", "Hello"),
        ("Hello\n\nPython", "Hello\nPython"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_clean_text(text, expected):
    assert clean_text(text) == expected