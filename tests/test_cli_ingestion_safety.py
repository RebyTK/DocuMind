
import os
import sys
from uuid import uuid4

import pytest

import main as main_module

from processor.document_manager import calculate_file_hash
from repositories.document_repository import (
    get_document_by_hash,
    delete_document,
)
from repositories.ingestion_repository import (
    persist_document_with_chunks,
)
from repositories.vector_repository import count_chunks


pytestmark = pytest.mark.skipif(
    os.getenv("DOCUMIND_RUN_DB_TESTS") != "1",
    reason="Set DOCUMIND_RUN_DB_TESTS=1 to run database tests.",
)


@pytest.fixture
def configure_cli(monkeypatch, tmp_path):
    """Configure CLI tests without creating application log files."""

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        main_module,
        "setup_logging",
        lambda output_dir: None,
    )


def test_cli_rejects_empty_txt(
    configure_cli,
    monkeypatch,
    capsys,
    tmp_path,
):
    """Empty TXT files must not be inserted into PostgreSQL."""

    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    file_hash = calculate_file_hash(file_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", str(file_path)],
    )

    main_module.main()

    output = capsys.readouterr().out

    assert "Document ingestion failed" in output
    assert "No extractable content" in output

    assert get_document_by_hash(file_hash) is None


def test_cli_rejects_malformed_pdf(
    configure_cli,
    monkeypatch,
    capsys,
    tmp_path,
):
    """Malformed PDF files must not be inserted into PostgreSQL."""

    file_path = tmp_path / "malformed.pdf"

    file_path.write_bytes(
        b"This is not a valid PDF document."
    )

    file_hash = calculate_file_hash(file_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", str(file_path)],
    )

    main_module.main()

    output = capsys.readouterr().out

    assert "Document ingestion failed" in output
    assert "Traceback" not in output

    assert get_document_by_hash(file_hash) is None


def test_cli_skips_duplicate_document(
    configure_cli,
    monkeypatch,
    capsys,
    tmp_path,
):
    """Duplicate CLI ingestion must not generate embeddings or insert chunks."""

    file_path = tmp_path / "duplicate_test.txt"

    file_path.write_text(
        "This is a PostgreSQL duplicate detection test.",
        encoding="utf-8",
    )

    file_hash = calculate_file_hash(file_path)

    document_id = f"doc_cli_duplicate_{uuid4().hex}"

    document = {
        "document_id": document_id,
        "filename": file_path.name,
        "file_type": "txt",
        "file_hash": file_hash,
        "total_chunks": 1,
        "metadata": {"purpose": "duplicate CLI testing"},
    }

    embedding = [1.0] + [0.0] * 383

    chunk = {
        "chunk_id": f"{document_id}_chunk_0000",
        "text": "Existing duplicate test content.",
        "embedding": embedding,
        "metadata": {
            "document_id": document_id,
            "source": file_path.name,
            "file_type": "txt",
            "chunk_index": 0,
            "page_number": None,
        },
    }

    try:
        persist_document_with_chunks(
            document,
            [chunk],
        )

        # The embedding model must not be loaded for a duplicate.
        def unexpected_model_load():
            pytest.fail(
                "Embedding model loaded for an existing document."
            )

        monkeypatch.setattr(
            main_module,
            "load_embedding_model",
            unexpected_model_load,
        )

        monkeypatch.setattr(
            sys,
            "argv",
            ["main.py", str(file_path)],
        )

        main_module.main()

        output = capsys.readouterr().out

        assert "Document already exists in PostgreSQL" in output
        assert document_id in output

        # Verify that the original record and chunk remain unchanged.
        stored_document = get_document_by_hash(file_hash)

        assert stored_document is not None
        assert stored_document["document_id"] == document_id
        assert count_chunks(document_id) == 1

    finally:
        delete_document(document_id)
