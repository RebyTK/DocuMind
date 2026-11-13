import os
from uuid import uuid4

import pytest
from psycopg.errors import UniqueViolation

from repositories.document_repository import (
    get_document,
    delete_document,
)
from repositories.vector_repository import count_chunks
from repositories.ingestion_repository import (
    persist_document_with_chunks,
)


pytestmark = pytest.mark.skipif(
    os.getenv("DOCUMIND_RUN_DB_TESTS") != "1",
    reason="Set DOCUMIND_RUN_DB_TESTS=1 to run database tests.",
)


EMBEDDING_A = [1.0] + [0.0] * 383
EMBEDDING_B = [0.0, 1.0] + [0.0] * 382


def make_document(document_id=None, file_hash=None, total_chunks=1):
    """Create a unique test document."""

    return {
        "document_id": document_id or f"doc_test_{uuid4().hex}",
        "filename": "ingestion_test.pdf",
        "file_type": "pdf",
        "file_hash": file_hash or uuid4().hex,
        "total_chunks": total_chunks,
        "metadata": {"purpose": "ingestion testing"},
    }


def make_chunk(document_id, chunk_index, chunk_id=None):
    """Create a valid embedded test chunk."""

    embedding = EMBEDDING_A if chunk_index % 2 == 0 else EMBEDDING_B

    return {
        "chunk_id": chunk_id or f"{document_id}_chunk_{chunk_index:04d}",
        "text": f"Test content for chunk {chunk_index}.",
        "embedding": embedding.copy(),
        "metadata": {
            "document_id": document_id,
            "source": "ingestion_test.pdf",
            "file_type": "pdf",
            "chunk_index": chunk_index,
            "page_number": chunk_index + 1,
        },
    }


@pytest.fixture
def cleanup_documents():
    """Track test documents and remove them after each test."""

    document_ids = []

    def track(document_id):
        document_ids.append(document_id)

    yield track

    for document_id in document_ids:
        delete_document(document_id)


def test_persist_document_with_chunks(cleanup_documents):
    """Test successful atomic document and chunk insertion."""

    document = make_document(total_chunks=2)
    document_id = document["document_id"]

    cleanup_documents(document_id)

    chunks = [
        make_chunk(document_id, 0),
        make_chunk(document_id, 1),
    ]

    result = persist_document_with_chunks(document, chunks)

    assert result["inserted"] is True
    assert result["document_id"] == document_id
    assert result["total_chunks"] == 2

    stored_document = get_document(document_id)

    assert stored_document is not None
    assert stored_document["filename"] == document["filename"]
    assert count_chunks(document_id) == 2


def test_duplicate_file_hash_is_not_inserted_twice(
    cleanup_documents,
):
    """Test duplicate detection using the SHA-256 file hash."""

    file_hash = uuid4().hex

    original = make_document(file_hash=file_hash)
    duplicate = make_document(file_hash=file_hash)

    cleanup_documents(original["document_id"])

    original_result = persist_document_with_chunks(
        original,
        [make_chunk(original["document_id"], 0)],
    )

    duplicate_result = persist_document_with_chunks(
        duplicate,
        [make_chunk(duplicate["document_id"], 0)],
    )

    assert original_result["inserted"] is True
    assert duplicate_result["inserted"] is False

    assert duplicate_result["document_id"] == original["document_id"]

    assert get_document(duplicate["document_id"]) is None
    assert count_chunks(original["document_id"]) == 1


def test_invalid_embedding_does_not_insert_document(
    cleanup_documents,
):
    """Test validation occurs before any database insertion."""

    document = make_document(total_chunks=1)
    document_id = document["document_id"]

    cleanup_documents(document_id)

    chunk = make_chunk(document_id, 0)

    chunk["embedding"] = [0.1, 0.2, 0.3]

    with pytest.raises(ValueError):
        persist_document_with_chunks(document, [chunk])

    assert get_document(document_id) is None


def test_chunk_insertion_failure_rolls_back_document(
    cleanup_documents,
):
    """Test that a database error rolls back the document insert."""

    blocker = make_document(total_chunks=1)
    candidate = make_document(total_chunks=1)

    cleanup_documents(blocker["document_id"])
    cleanup_documents(candidate["document_id"])

    conflicting_chunk_id = "shared_test_chunk_" + uuid4().hex

    # First insert a chunk that already owns this chunk ID.
    blocker_chunk = make_chunk(
        blocker["document_id"],
        0,
        chunk_id=conflicting_chunk_id,
    )

    persist_document_with_chunks(
        blocker,
        [blocker_chunk],
    )

    # Attempt to insert a different document using the same chunk ID.
    candidate_chunk = make_chunk(
        candidate["document_id"],
        0,
        chunk_id=conflicting_chunk_id,
    )

    with pytest.raises(UniqueViolation):
        persist_document_with_chunks(
            candidate,
            [candidate_chunk],
        )

    # The candidate document must have been rolled back.
    assert get_document(candidate["document_id"]) is None

    # The original document and its chunk must remain intact.
    assert get_document(blocker["document_id"]) is not None
    assert count_chunks(blocker["document_id"]) == 1
