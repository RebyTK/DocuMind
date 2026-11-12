import pytest
import numpy as np

from processor.vector_store import create_index, search_index


@pytest.fixture
def embedded_chunks():
    return [
        {
            "text": "Python is a programming language.",
            "metadata": {
                "source": "python.pdf",
                "chunk_index": 0
            },
            "embedding": [1.0, 0.0, 0.0]
        },
        {
            "text": "FAISS performs vector similarity search.",
            "metadata": {
                "source": "faiss.pdf",
                "chunk_index": 0
            },
            "embedding": [0.0, 1.0, 0.0]
        },
        {
            "text": "Machine learning uses numerical data.",
            "metadata": {
                "source": "ml.pdf",
                "chunk_index": 0
            },
            "embedding": [0.0, 0.0, 1.0]
        }
    ]

def test_create_index_success(embedded_chunks):
    index = create_index(embedded_chunks)

    assert index is not None
    assert index.ntotal == 3
    assert index.d == 3


def test_create_index_empty_chunks():
    index = create_index([])

    assert index is None


def test_search_index_returns_relevant_chunks(embedded_chunks):
    index = create_index(embedded_chunks)

    query_embedding = [0.0, 0.9, 0.1]

    results = search_index(
        query_embedding=query_embedding,
        index=index,
        embedded_chunks=embedded_chunks,
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["text"] == "FAISS performs vector similarity search."
    assert results[0]["metadata"]["source"] == "faiss.pdf"
    assert results[0]["score"] > results[1]["score"]


def test_search_index_empty_chunks():
    results = search_index(
        query_embedding=[1.0, 0.0, 0.0],
        index=None,
        embedded_chunks=[],
        top_k=3
    )

    assert results == []


def test_search_index_handles_invalid_indices(embedded_chunks):
    index = create_index(embedded_chunks[:2])

    results = search_index(
        query_embedding=[1.0, 0.0, 0.0],
        index=index,
        embedded_chunks=embedded_chunks[:2],
        top_k=5
    )

    assert len(results) == 2
    assert all(result["text"] for result in results)


def test_search_index_invalid_top_k(embedded_chunks):
    index = create_index(embedded_chunks)

    with pytest.raises(ValueError):
        search_index(
            query_embedding=[1.0, 0.0, 0.0],
            index=index,
            embedded_chunks=embedded_chunks,
            top_k=0
        )


def test_search_index_dimension_mismatch(embedded_chunks):
    index = create_index(embedded_chunks)

    with pytest.raises(ValueError):
        search_index(
            query_embedding=[1.0, 0.0],
            index=index,
            embedded_chunks=embedded_chunks,
            top_k=2
        )


def test_search_index_limits_chunks_per_document():
    embedded_chunks = [
        {
            "text": "A - first chunk",
            "metadata": {
                "document_id": "doc_a",
                "source": "a.pdf",
            },
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "text": "A - second chunk",
            "metadata": {
                "document_id": "doc_a",
                "source": "a.pdf",
            },
            "embedding": [0.99, 0.01, 0.0],
        },
        {
            "text": "A - third chunk",
            "metadata": {
                "document_id": "doc_a",
                "source": "a.pdf",
            },
            "embedding": [0.98, 0.02, 0.0],
        },
        {
            "text": "B - first chunk",
            "metadata": {
                "document_id": "doc_b",
                "source": "b.pdf",
            },
            "embedding": [0.95, 0.05, 0.0],
        },
    ]

    index = create_index(embedded_chunks)

    results = search_index(
        query_embedding=[1.0, 0.0, 0.0],
        index=index,
        embedded_chunks=embedded_chunks,
        top_k=3,
        max_per_document=2,
    )

    document_ids = [
        result["metadata"]["document_id"]
        for result in results
    ]

    assert len(results) == 3
    assert document_ids.count("doc_a") == 2
    assert document_ids.count("doc_b") == 1


def test_search_index_fills_results_when_only_one_document():
    embedded_chunks = [
        {
            "text": f"Chunk {i}",
            "metadata": {
                "document_id": "doc_a",
                "source": "a.pdf",
            },
            "embedding": [1.0 - i * 0.01, i * 0.01, 0.0],
        }
        for i in range(4)
    ]

    index = create_index(embedded_chunks)

    results = search_index(
        query_embedding=[1.0, 0.0, 0.0],
        index=index,
        embedded_chunks=embedded_chunks,
        top_k=3,
        max_per_document=2,
    )

    assert len(results) == 3
    assert all(
        result["metadata"]["document_id"] == "doc_a"
        for result in results
    )


def test_search_index_invalid_document_limit(embedded_chunks):
    index = create_index(embedded_chunks)

    with pytest.raises(ValueError):
        search_index(
            query_embedding=[1.0, 0.0, 0.0],
            index=index,
            embedded_chunks=embedded_chunks,
            top_k=3,
            max_per_document=0,
        )
