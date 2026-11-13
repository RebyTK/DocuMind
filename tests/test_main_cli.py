
import sys

import pytest

import main as main_module


class FakeIndex:
    ntotal = 1


class FakeVector:
    def tolist(self):
        return [0.1, 0.2, 0.3]


class FakeEmbeddingModel:
    def encode(self, *args, **kwargs):
        return [FakeVector()]


@pytest.fixture
def cli_environment(monkeypatch, tmp_path):
    """Prepare a mocked environment for CLI integration tests."""

    monkeypatch.chdir(tmp_path)

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    document_path = documents_dir / "bert.pdf"
    document_path.write_bytes(b"fake PDF content")

    document = {
        "document_id": "doc_bert",
        "filename": "bert.pdf",
        "file_type": "pdf",
    }

    registry = {
        "documents": {
            "doc_bert": document,
        }
    }

    retrieved_results = [
        {
            "chunk_id": "bert_chunk_001",
            "text": "BERT uses a bidirectional Transformer encoder.",
            "metadata": {
                "document_id": "doc_bert",
                "source": "bert.pdf",
                "chunk_index": 0,
                "page_number": 3,
            },
            "score": 0.85,
        }
    ]

    state = {
        "search_calls": [],
        "search_results": retrieved_results,
    }

    monkeypatch.setattr(
        main_module,
        "setup_logging",
        lambda output_dir: None,
    )

    monkeypatch.setattr(
        main_module,
        "load_registry",
        lambda path: registry,
    )

    monkeypatch.setattr(
        main_module,
        "calculate_file_hash",
        lambda path: "fake_hash",
    )

    monkeypatch.setattr(
        main_module,
        "find_document_by_hash",
        lambda registry, file_hash: document,
    )

    monkeypatch.setattr(
        main_module,
        "rebuild_shared_index",
        lambda registry, output_dir: (
            FakeIndex(),
            retrieved_results,
        ),
    )

    monkeypatch.setattr(
        main_module,
        "load_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    def fake_search_index(**kwargs):
        state["search_calls"].append(kwargs)
        return state["search_results"]

    monkeypatch.setattr(
        main_module,
        "search_index",
        fake_search_index,
    )

    return {
        "document_path": document_path,
        "retrieved_results": retrieved_results,
        "state": state,
    }


def test_cli_semantic_search_mode(
    cli_environment,
    monkeypatch,
    capsys,
):
    """Test semantic search without LLM generation."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(cli_environment["document_path"]),
            "--query",
            "What is BERT?",
            "--top-k",
            "3",
        ],
    )

    main_module.main()

    output = capsys.readouterr().out

    assert "DOCUMIND SEMANTIC SEARCH" in output
    assert "BERT uses a bidirectional Transformer encoder." in output
    assert "DOCUMIND GENERATED ANSWER" not in output

    assert len(cli_environment["state"]["search_calls"]) == 1

    assert (
        cli_environment["state"]["search_calls"][0]["top_k"]
        == 3
    )


def test_cli_answer_mode(
    cli_environment,
    monkeypatch,
    capsys,
):
    """Test RAG answer generation through the CLI."""

    generated_result = {
        "answer": "BERT uses a bidirectional Transformer encoder [S1].",
        "sources": [
            {
                "citation": "S1",
                "source": "bert.pdf",
                "page_number": 3,
                "chunk_id": "bert_chunk_001",
            }
        ],
        "cited_sources": [
            {
                "citation": "S1",
                "source": "bert.pdf",
                "page_number": 3,
                "chunk_id": "bert_chunk_001",
            }
        ],
        "valid_citations": ["S1"],
        "invalid_citations": [],
    }

    received = {}

    def fake_generate_answer(question, retrieved_chunks):
        received["question"] = question
        received["chunks"] = retrieved_chunks
        return generated_result

    monkeypatch.setattr(
        main_module,
        "generate_answer",
        fake_generate_answer,
    )

    def unexpected_search_display(*args, **kwargs):
        pytest.fail(
            "Raw semantic search output should not appear in answer mode."
        )

    monkeypatch.setattr(
        main_module,
        "display_search_results",
        unexpected_search_display,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(cli_environment["document_path"]),
            "--query",
            "What is BERT's architecture?",
            "--top-k",
            "3",
            "--answer",
        ],
    )

    main_module.main()

    output = capsys.readouterr().out

    assert "DOCUMIND GENERATED ANSWER" in output
    assert "BERT uses a bidirectional Transformer encoder [S1]." in output
    assert "[S1][bert.pdf, Page3]" in output
    assert "DOCUMIND SEMANTIC SEARCH" not in output

    assert received["question"] == "What is BERT's architecture?"
    assert received["chunks"] == cli_environment["retrieved_results"]
