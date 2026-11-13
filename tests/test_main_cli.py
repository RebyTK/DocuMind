import sys
import pytest
import main as main_module


class FakeVector:
    def tolist(self):
        return [0.1, 0.2, 0.3]


class FakeEmbeddingModel:
    def encode(self, *args, **kwargs):
        return [FakeVector()]


@pytest.fixture
def cli_environment(monkeypatch, tmp_path):
    """Prepare a mocked environment for PostgreSQL CLI tests."""

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        main_module,
        "setup_logging",
        lambda output_dir: None,
    )

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

    state = {"retrieval_calls": []}

    def fake_retrieve_chunks(**kwargs):
        state["retrieval_calls"].append(kwargs)
        return retrieved_results

    monkeypatch.setattr(
        main_module,
        "load_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    monkeypatch.setattr(
        main_module,
        "retrieve_chunks",
        fake_retrieve_chunks,
    )

    return {
        "retrieved_results": retrieved_results,
        "state": state,
    }


def test_cli_semantic_search_mode(
    cli_environment,
    monkeypatch,
    capsys,
):
    """Test PostgreSQL semantic search without LLM generation."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
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

    calls = cli_environment["state"]["retrieval_calls"]

    assert len(calls) == 1
    assert calls[0]["backend"] == "postgres"
    assert calls[0]["top_k"] == 3


def test_cli_answer_mode(
    cli_environment,
    monkeypatch,
    capsys,
):
    """Test PostgreSQL retrieval followed by RAG generation."""

    generated_result = {
        "answer": (
            "BERT uses a bidirectional Transformer encoder [S1]."
        ),
        "sources": [],
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
            "Raw search output should not appear in answer mode."
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
    assert "[S1] bert.pdf, Page 3" in output
    assert "DOCUMIND SEMANTIC SEARCH" not in output

    assert received["question"] == "What is BERT's architecture?"
    assert received["chunks"] == cli_environment["retrieved_results"]

    calls = cli_environment["state"]["retrieval_calls"]

    assert calls[0]["backend"] == "postgres"
