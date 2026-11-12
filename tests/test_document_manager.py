
import pytest

from processor.document_manager import (
    load_registry,
    save_registry,
    register_document,
    remove_document,
    get_document,
    list_documents,
)


@pytest.fixture
def sample_document():
    return {
        "document_id": "doc_test123",
        "filename": "sample.pdf",
        "file_type": "pdf",
        "chunks_file": "documents/doc_test123_chunks.json",
        "embeddings_file": "documents/doc_test123_embeddings.json",
        "chunk_count": 10,
    }


def test_load_registry_creates_empty_registry(tmp_path):
    registry_path = tmp_path / "document_registry.json"

    registry = load_registry(registry_path)

    assert registry == {"documents": {}}


def test_register_document(sample_document):
    registry = {"documents": {}}

    register_document(registry, sample_document)

    assert "doc_test123" in registry["documents"]
    assert registry["documents"]["doc_test123"] == sample_document


def test_register_duplicate_document_raises_error(sample_document):
    registry = {"documents": {}}

    register_document(registry, sample_document)

    with pytest.raises(ValueError):
        register_document(registry, sample_document)


def test_register_document_requires_id():
    registry = {"documents": {}}

    with pytest.raises(ValueError):
        register_document(registry, {"filename": "sample.pdf"})


def test_save_and_load_registry(tmp_path, sample_document):
    registry_path = tmp_path / "output" / "document_registry.json"
    registry = {"documents": {}}

    register_document(registry, sample_document)
    save_registry(registry, registry_path)

    loaded_registry = load_registry(registry_path)

    assert loaded_registry == registry


def test_get_document(sample_document):
    registry = {"documents": {"doc_test123": sample_document}}

    result = get_document(registry, "doc_test123")

    assert result == sample_document


def test_get_missing_document():
    registry = {"documents": {}}

    assert get_document(registry, "missing_id") is None


def test_list_documents(sample_document):
    registry = {"documents": {"doc_test123": sample_document}}

    documents = list_documents(registry)

    assert len(documents) == 1
    assert documents[0] == sample_document


def test_remove_document(sample_document):
    registry = {"documents": {"doc_test123": sample_document}}

    removed = remove_document(registry, "doc_test123")

    assert removed == sample_document
    assert "doc_test123" not in registry["documents"]


def test_remove_missing_document():
    registry = {"documents": {}}

    assert remove_document(registry, "missing_id") is None


from processor.document_manager import (
    save_document_data,
    load_document_data,
)


def test_save_document_data(tmp_path):
    document_id = "doc_test123"

    chunks = [
        {
            "chunk_id": "doc_test123_chunk_0000",
            "text": "Example content",
            "metadata": {
                "document_id": document_id,
                "chunk_index": 0,
            },
        }
    ]

    embedded_chunks = [
        {
            **chunks[0],
            "embedding": [0.1, 0.2, 0.3],
        }
    ]

    paths = save_document_data(
        document_id,
        chunks,
        embedded_chunks,
        tmp_path,
    )

    assert (tmp_path / paths["chunks_file"]).is_file()
    assert (tmp_path / paths["embeddings_file"]).is_file()


def test_load_document_data(tmp_path):
    document_id = "doc_test123"

    chunks = [
        {
            "chunk_id": "doc_test123_chunk_0000",
            "text": "Example content",
            "metadata": {
                "document_id": document_id,
                "chunk_index": 0,
            },
        }
    ]

    embedded_chunks = [
        {
            **chunks[0],
            "embedding": [0.1, 0.2, 0.3],
        }
    ]

    paths = save_document_data(
        document_id,
        chunks,
        embedded_chunks,
        tmp_path,
    )

    document = {
        "document_id": document_id,
        **paths,
    }

    loaded_chunks, loaded_embeddings = load_document_data(
        document,
        tmp_path,
    )

    assert loaded_chunks == chunks
    assert loaded_embeddings == embedded_chunks


from processor.document_manager import (
    load_all_embeddings,
    rebuild_shared_index,
)


def test_load_all_embeddings_from_multiple_documents(tmp_path):
    registry = {"documents": {}}

    for document_id, filename, content in [
        ("doc_one", "first.pdf", "First document"),
        ("doc_two", "second.docx", "Second document"),
    ]:
        chunks = [
            {
                "chunk_id": f"{document_id}_chunk_0000",
                "text": content,
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": 0,
                },
            }
        ]

        embedded_chunks = [
            {
                **chunks[0],
                "embedding": [0.1, 0.2, 0.3],
            }
        ]

        paths = save_document_data(
            document_id,
            chunks,
            embedded_chunks,
            tmp_path,
        )

        register_document(
            registry,
            {
                "document_id": document_id,
                "filename": filename,
                "file_type": "pdf" if filename.endswith(".pdf") else "docx",
                **paths,
                "chunk_count": 1,
            },
        )

    all_embeddings = load_all_embeddings(registry, tmp_path)

    assert len(all_embeddings) == 2
    assert {
        chunk["metadata"]["document_id"]
        for chunk in all_embeddings
    } == {"doc_one", "doc_two"}


def test_load_all_embeddings_empty_registry(tmp_path):
    registry = {"documents": {}}

    result = load_all_embeddings(registry, tmp_path)

    assert result == []


def test_rebuild_shared_index_from_multiple_documents(tmp_path):
    registry = {"documents": {}}

    for document_id, embedding in [
        ("doc_one", [1.0, 0.0, 0.0]),
        ("doc_two", [0.0, 1.0, 0.0]),
    ]:
        chunks = [
            {
                "chunk_id": f"{document_id}_chunk_0000",
                "text": f"Content from {document_id}",
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": 0,
                },
            }
        ]

        embedded_chunks = [
            {
                **chunks[0],
                "embedding": embedding,
            }
        ]

        paths = save_document_data(
            document_id,
            chunks,
            embedded_chunks,
            tmp_path,
        )

        register_document(
            registry,
            {
                "document_id": document_id,
                "filename": f"{document_id}.txt",
                "file_type": "txt",
                **paths,
                "chunk_count": 1,
            },
        )

    index, all_embeddings = rebuild_shared_index(
        registry,
        tmp_path,
    )

    assert index is not None
    assert index.ntotal == 2
    assert len(all_embeddings) == 2


def test_rebuild_shared_index_empty_registry(tmp_path):
    registry = {"documents": {}}

    index, all_embeddings = rebuild_shared_index(
        registry,
        tmp_path,
    )

    assert index is None
    assert all_embeddings == []


from processor.document_manager import (
    calculate_file_hash,
    find_document_by_hash,
)


def test_calculate_file_hash(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello DocuMind", encoding="utf-8")

    first_hash = calculate_file_hash(file_path)
    second_hash = calculate_file_hash(file_path)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_different_files_have_different_hashes(tmp_path):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text("Document A", encoding="utf-8")
    second_file.write_text("Document B", encoding="utf-8")

    assert calculate_file_hash(first_file) != calculate_file_hash(second_file)


def test_find_document_by_hash():
    registry = {
        "documents": {
            "doc_one": {
                "document_id": "doc_one",
                "filename": "sample.pdf",
                "file_hash": "abc123",
            }
        }
    }

    result = find_document_by_hash(registry, "abc123")

    assert result["document_id"] == "doc_one"


def test_find_missing_document_by_hash():
    registry = {"documents": {}}

    result = find_document_by_hash(registry, "missing_hash")

    assert result is None
