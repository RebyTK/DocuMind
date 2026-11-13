
import os
import sys
from uuid import uuid4

import pytest

import main as main_module

from repositories.document_repository import (
    insert_document,
    get_document,
    delete_document,
)

from repositories.vector_repository import (
    insert_chunks,
    count_chunks,
)


pytestmark = pytest.mark.skipif(
    os.getenv("DOCUMIND_RUN_DB_TESTS") != "1",
    reason="Set DOCUMIND_RUN_DB_TESTS=1 to run database tests.",
)


EMBEDDING_A = [1.0] + [0.0] * 383
EMBEDDING_B = [0.0, 1.0] + [0.0] * 382


@pytest.fixture
def stored_document_with_chunks():
    """Create a temporary document with two PostgreSQL chunks."""

    document_id = f"doc_lifecycle_test_{uuid4().hex}"

    document = {
        "document_id": document_id,
        "filename": "lifecycle_test.pdf",
        "file_type": "pdf",
        "file_hash": uuid4().hex,
        "total_chunks": 2,
        "metadata": {
            "purpose": "document lifecycle testing"
        },
    }

    insert_document(document)

    chunks = [
        {
            "chunk_id": f"{document_id}_chunk_0000",
            "text": "First lifecycle test chunk.",
            "embedding": EMBEDDING_A,
            "metadata": {
                "document_id": document_id,
                "source": "lifecycle_test.pdf",
                "chunk_index": 0,
                "page_number": 1,
            },
        },
        {
            "chunk_id": f"{document_id}_chunk_0001",
            "text": "Second lifecycle test chunk.",
            "embedding": EMBEDDING_B,
            "metadata": {
                "document_id": document_id,
                "source": "lifecycle_test.pdf",
                "chunk_index": 1,
                "page_number": 2,
            },
        },
    ]

    insert_chunks(chunks)

    yield document_id

    # Cleanup even when a test fails.
    delete_document(document_id)


def test_delete_document_cascades_to_chunks(
    stored_document_with_chunks,
):
    """Deleting a document should automatically delete its chunks."""

    document_id = stored_document_with_chunks

    assert get_document(document_id) is not None
    assert count_chunks(document_id) == 2

    deleted = delete_document(document_id)

    assert deleted is True
    assert get_document(document_id) is None
    assert count_chunks(document_id) == 0


def test_delete_document_returns_false_when_missing():
    """Deleting a nonexistent document should return False."""

    document_id = f"doc_missing_{uuid4().hex}"

    result = delete_document(document_id)

    assert result is False


def test_delete_document_twice_is_safe(
    stored_document_with_chunks,
):
    """A second deletion should not raise an exception."""

    document_id = stored_document_with_chunks

    assert delete_document(document_id) is True
    assert delete_document(document_id) is False

    assert count_chunks(document_id) == 0


def test_cli_delete_document(
    stored_document_with_chunks,
    monkeypatch,
    capsys,
):
    """Test deletion through the actual CLI command."""

    document_id = stored_document_with_chunks

    monkeypatch.setattr(
        main_module,
        "setup_logging",
        lambda output_dir: None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--delete",
            document_id,
        ],
    )

    main_module.main()

    output = capsys.readouterr().out

    assert "Document deleted successfully" in output
    assert document_id in output

    # Verify actual database state after CLI deletion.
    assert get_document(document_id) is None
    assert count_chunks(document_id) == 0
