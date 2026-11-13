import pytest

from processor.retrieval import (
    diversify_results,
    retrieve_chunks,
)


@pytest.fixture
def sample_results():
    return [
        {
            "chunk_id": "a1",
            "text": "Document A chunk one",
            "metadata": {"document_id": "doc_a"},
            "score": 0.95,
        },
        {
            "chunk_id": "a2",
            "text": "Document A chunk two",
            "metadata": {"document_id": "doc_a"},
            "score": 0.90,
        },
        {
            "chunk_id": "b1",
            "text": "Document B chunk one",
            "metadata": {"document_id": "doc_b"},
            "score": 0.85,
        },
        {
            "chunk_id": "c1",
            "text": "Document C chunk one",
            "metadata": {"document_id": "doc_c"},
            "score": 0.80,
        },
    ]


def test_diversify_results_limits_chunks_per_document(sample_results):
    results = diversify_results(
        candidates=sample_results,
        top_k=3,
        max_per_document=1,
    )

    assert [result["chunk_id"] for result in results] == [
        "a1",
        "b1",
        "c1",
    ]


def test_diversify_results_fills_remaining_slots(sample_results):
    results = diversify_results(
        candidates=sample_results[:2],
        top_k=3,
        max_per_document=1,
    )

    assert len(results) == 2
    assert [result["chunk_id"] for result in results] == [
        "a1",
        "a2",
    ]


def test_diversify_results_invalid_top_k(sample_results):
    with pytest.raises(ValueError):
        diversify_results(sample_results, top_k=0)


def test_diversify_results_invalid_document_limit(sample_results):
    with pytest.raises(ValueError):
        diversify_results(
            sample_results,
            top_k=3,
            max_per_document=0,
        )


def test_retrieve_chunks_uses_postgres(monkeypatch, sample_results):
    received = {}

    def fake_search(**kwargs):
        received.update(kwargs)
        return sample_results

    monkeypatch.setattr(
        "processor.retrieval.search_similar_chunks",
        fake_search,
    )

    results = retrieve_chunks(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=2,
        backend="postgres",
        max_per_document=1,
    )

    assert received["top_k"] == 20
    assert len(results) == 2
    assert results[0]["chunk_id"] == "a1"
    assert results[1]["chunk_id"] == "b1"


def test_retrieve_chunks_uses_faiss(monkeypatch, sample_results):
    received = {}

    def fake_search(**kwargs):
        received.update(kwargs)
        return sample_results

    monkeypatch.setattr(
        "processor.retrieval.search_index",
        fake_search,
    )

    results = retrieve_chunks(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=2,
        backend="faiss",
        index=object(),
        embedded_chunks=[{"embedding": [0.1]}],
    )

    assert received["top_k"] == 20
    assert received["max_per_document"] == 20
    assert len(results) == 2


def test_retrieve_chunks_rejects_unknown_backend():
    with pytest.raises(ValueError):
        retrieve_chunks(
            query_embedding=[0.1, 0.2],
            backend="unknown",
        )


def test_retrieve_chunks_returns_empty_for_missing_faiss_index():
    results = retrieve_chunks(
        query_embedding=[0.1, 0.2, 0.3],
        backend="faiss",
    )

    assert results == []


def test_retrieve_chunks_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        retrieve_chunks(
            query_embedding=[0.1],
            top_k=0,
            backend="postgres",
        )
