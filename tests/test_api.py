"""Basic FastAPI route tests using mocked application services."""

from fastapi.testclient import TestClient

from api.main import app
from api.routes import documents, health, rag, search

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_database_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        health,
        "check_connection",
        lambda: {"database": "documind", "version": "PostgreSQL test"},
    )
    response = client.get("/health/database")
    assert response.status_code == 200
    assert response.json()["database"] == "documind"


def test_list_documents(monkeypatch):
    monkeypatch.setattr(
        documents,
        "list_all_documents",
        lambda: [{
            "document_id": "doc_test",
            "filename": "sample.pdf",
            "file_type": "pdf",
            "file_hash": "abc123",
            "total_chunks": 2,
            "metadata": {},
            "created_at": None,
        }],
    )
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert response.json()[0]["document_id"] == "doc_test"


def test_get_missing_document(monkeypatch):
    monkeypatch.setattr(documents, "get_document_by_id", lambda _id: None)
    response = client.get("/api/v1/documents/does-not-exist")
    assert response.status_code == 404


def test_delete_document(monkeypatch):
    monkeypatch.setattr(documents, "remove_document", lambda _id: True)
    response = client.delete("/api/v1/documents/doc_test")
    assert response.status_code == 200
    assert response.json()["deleted"] is True


def test_search_endpoint(monkeypatch):
    monkeypatch.setattr(
        search,
        "search_documents",
        lambda question, top_k: [{
            "chunk_id": "chunk_1",
            "text": "BERT uses a bidirectional encoder.",
            "metadata": {"source": "bert.pdf", "page_number": 3},
            "score": 0.91,
        }],
    )
    response = client.post(
        "/api/v1/search",
        json={"question": "What is BERT?", "top_k": 5},
    )
    assert response.status_code == 200
    assert response.json()["total_results"] == 1
    assert response.json()["results"][0]["score"] == 0.91


def test_ask_endpoint(monkeypatch):
    monkeypatch.setattr(
        rag,
        "answer_question",
        lambda question, top_k: {
            "question": question,
            "answer": "BERT is a bidirectional encoder [S1].",
            "citations": [{"citation": "S1", "source": "bert.pdf", "page_number": 3}],
            "valid_citations": ["S1"],
            "invalid_citations": [],
            "retrieval_count": 2,
        },
    )
    response = client.post(
        "/api/v1/ask",
        json={"question": "What is BERT?", "top_k": 5},
    )
    assert response.status_code == 200
    assert response.json()["valid_citations"] == ["S1"]


def test_search_rejects_invalid_top_k():
    response = client.post(
        "/api/v1/search",
        json={"question": "What is BERT?", "top_k": 0},
    )
    assert response.status_code == 422


def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/api/v1/documents",
        files={"file": ("script.exe", b"not a document")},
    )
    assert response.status_code == 415


def test_upload_document(monkeypatch):
    monkeypatch.setattr(
        documents,
        "ingest_file",
        lambda _path: {
            "inserted": True,
            "document_id": "doc_upload_test",
            "filename": "upload.txt",
            "total_chunks": 1,
            "stats": {"words": 4},
        },
    )
    response = client.post(
        "/api/v1/documents",
        files={"file": ("upload.txt", b"A small test document.", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["inserted"] is True
    assert body["document_id"] == "doc_upload_test"
    assert body["file_type"] == "txt"
