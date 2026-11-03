import pytest
from processor.chunker import chunk_text, chunk_text_with_metadata

def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []

def test_short_text_returns_one_chunk():
    assert chunk_text("Hello DocuMind") == ["Hello DocuMind"]

def test_long_text_is_split_into_chunks():
    text = "abcdefghij"

    assert chunk_text(text, chunk_size=4, overlap=1) == [
        "abcd",
        "defg",
        "ghij",
    ]

def test_chunks_preserve_overlap():
    chunks = chunk_text("abcdefgh", chunk_size=5, overlap=2)

    assert chunks == ["abcde", "defgh"]

def test_zero_or_negative_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)

def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=5, overlap=5)

def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=5, overlap=-1)

def test_chunks_include_source_and_index():
    chunks = chunk_text_with_metadata(
        "abcdefgh",
        source="example.txt",
        chunk_size=4,
        overlap=2,
    )

    assert chunks == [
        {
            "text": "abcd",
            "metadata": {
                "source": "example.txt",
                "chunk_index": 0,
            },
        },
        {
            "text": "cdef",
            "metadata": {
                "source": "example.txt",
                "chunk_index": 1,
            },
        },
        {
            "text": "efgh",
            "metadata": {
                "source": "example.txt",
                "chunk_index": 2,
            },
        },
    ]


def test_empty_text_returns_no_metadata_chunks():
    assert chunk_text_with_metadata(
        "",
        source="example.txt",
    ) == []
