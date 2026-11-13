import os
from uuid import uuid4

import pytest

from repositories.document_repository import (
    insert_document,
    delete_document,
)
from repositories.vector_repository import (
    validate_embedding,
    insert_chunks,
    search_similar_chunks,
    count_chunks,
)


RUN_DB_TESTS = os.getenv("DOCUMIND_RUN_DB_TESTS") == "1"

EMBEDDING_A = [1.0] + [0.0] * 383
EMBEDDING_B = [0.0, 1.0] + [0.0] * 382


def test_validate_embedding_success():
    validate_embedding(EMBEDDING_A)


def test_validate_embedding_rejects_wrong_dimensions():
    with pytest.raises(ValueError):
        validate_embedding([1.0, 0.0])


def test_validate_embedding_rejects_non_numeric_values():
    embedding = [1.0] * 383 + ["invalid"]

    with pytest.raises(ValueError):
        validate_embedding(embedding)


def test_validate_embedding_rejects_non_finite_values():
    embedding = [1.0] * 383 + [float("nan")]

    with pytest.raises(ValueError):
        validate_embedding(embedding)


def test_insert_chunks_empty_input():
    assert insert_chunks([]) == 0


def test_search_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        search_similar_chunks(
            query_embedding=EMBEDDING_A,
            top_k=0,
        )


@pytest.fixture
def registered_test_document():
    """Create and clean up an isolated database test document."""

    document_id = f"doc_vector_test_{uuid4().hex}"

    document = {
        "document_id": document_id,
        "filename": "vector_test.txt",
        "file_type": "txt",
        "file_hash": uuid4().hex,
        "total_chunks": 2,
        "metadata": {"purpose": "automated vector testing"},
    }

    insert_document(document)

    yield document_id

    delete_document(document_id)


def make_chunk(document_id, index, embedding, content):
    """Create an embedded chunk for database testing."""

    return {
        "chunk_id": f"{document_id}_chunk_{index:04d}",
        "text": content,
        "embedding": embedding,
        "metadata": {
            "document_id": document_id,
            "source": "vector_test.txt",
            "chunk_index": index,
            "page_number": None,
        },
    }


@pytest.mark.skipif(
    not RUN_DB_TESTS,
    reason="Set DOCUMIND_RUN_DB_TESTS=1 to run database tests.",
)
def test_insert_chunks_and_count(registered_test_document):
    document_id = registered_test_document

    chunks = [
        make_chunk(
            document_id,
            0,
            EMBEDDING_A,
            "First test chunk.",
        ),
        make_chunk(
            document_id,
            1,
            EMBEDDING_B,
            "Second test chunk.",
        ),
    ]

    inserted = insert_chunks(chunks)

    assert inserted == 2
    assert count_chunks(document_id) == 2


@pytest.mark.skipif(
    not RUN_DB_TESTS,
    reason="Set DOCUMIND_RUN_DB_TESTS=1 to run database tests.",
)
def test_search_returns_nearest_chunks(registered_test_document):
    document_id = registered_test_document

    chunks = [
        make_chunk(
            document_id,
            0,
            EMBEDDING_A,
            "First test chunk.",
        ),
        make_chunk(
            document_id,
            1,
            EMBEDDING_B,
            "Second test chunk.",
        ),
    ]

    insert_chunks(chunks)

    results = search_similar_chunks(
        query_embedding=EMBEDDING_A,
        top_k=2,
        document_id=document_id,
    )

    assert len(results) == 2
    assert results[0]["metadata"]["chunk_index"] == 0
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert results[1]["score"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.skipif(
    not RUN_DB_TESTS,
    reason="Set DOCUMIND_RUN_DB_TESTS=1 to run database tests.",
)
def test_search_filters_by_document(registered_test_document):
    document_id = registered_test_document

    chunks = [
        make_chunk(
            document_id,
            0,
            EMBEDDING_A,
            "Document-specific chunk.",
        ),
    ]

    insert_chunks(chunks)

    results = search_similar_chunks(
        query_embedding=EMBEDDING_A,
        top_k=5,
        document_id=document_id,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["document_id"] == document_id
