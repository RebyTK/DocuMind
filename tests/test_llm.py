import pytest
import requests

from processor.llm import (
    build_prompt,
    generate_response,
    generate_answer,
    extract_citations,
    validate_citations,
    OllamaConnectionError,
    OllamaResponseError,
)

SAMPLE_CHUNKS = [
    {
        "chunk_id": "bert_chunk_001",
        "text": "BERT uses a bidirectional Transformer encoder.",
        "metadata": {
            "source": "bert.pdf",
            "page_number": 3,
            "document_id": "bert_document",
        },
        "score": 0.85,
    }
]


class MockResponse:
    def __init__(self, data, ok=True):
        self.data = data
        self.ok = ok

    def json(self):
        return self.data


def test_build_prompt_includes_question_and_context():
    messages = build_prompt(
        "What architecture does BERT use?",
        SAMPLE_CHUNKS,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What architecture does BERT use?" in messages[1]["content"]
    assert "BERT uses a bidirectional Transformer encoder." in messages[1]["content"]
    assert "[S1]" in messages[1]["content"]


def test_build_prompt_rejects_empty_question():
    with pytest.raises(ValueError):
        build_prompt("", SAMPLE_CHUNKS)


def test_build_prompt_rejects_empty_chunks():
    with pytest.raises(ValueError):
        build_prompt("Explain BERT", [])


def test_build_prompt_rejects_empty_chunk_text():
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": " ",
            "metadata": {"source": "test.pdf"},
        }
    ]

    with pytest.raises(ValueError):
        build_prompt("Explain BERT", chunks)


def test_generate_response_success(monkeypatch):
    def mock_post(*args, **kwargs):
        return MockResponse({
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "BERT uses a bidirectional Transformer encoder.",
            },
        })

    monkeypatch.setattr(
        "processor.llm.requests.post",
        mock_post,
    )

    result = generate_response([
        {"role": "user", "content": "Explain BERT."}
    ])

    assert result == "BERT uses a bidirectional Transformer encoder."


def test_generate_response_extracts_final_answer(monkeypatch):
    def mock_post(*args, **kwargs):
        return MockResponse({
            "done_reason": "stop",
            "message": {
                "content": (
                    "Internal preamble </think>"
                    "Final answer [S1]."
                ),
            },
        })

    monkeypatch.setattr(
        "processor.llm.requests.post",
        mock_post,
    )

    result = generate_response([
        {"role": "user", "content": "Explain BERT."}
    ])

    assert result == "Final answer [S1]."


def test_generate_response_rejects_incomplete_generation(monkeypatch):
    def mock_post(*args, **kwargs):
        return MockResponse({
            "done_reason": "length",
            "message": {"content": "Incomplete answer"},
        })

    monkeypatch.setattr(
        "processor.llm.requests.post",
        mock_post,
    )

    with pytest.raises(OllamaResponseError):
        generate_response([
            {"role": "user", "content": "Explain BERT."}
        ])


def test_generate_response_rejects_empty_response(monkeypatch):
    def mock_post(*args, **kwargs):
        return MockResponse({
            "done_reason": "stop",
            "message": {"content": "   "},
        })

    monkeypatch.setattr(
        "processor.llm.requests.post",
        mock_post,
    )

    with pytest.raises(OllamaResponseError):
        generate_response([
            {"role": "user", "content": "Explain BERT."}
        ])


def test_generate_response_handles_timeout(monkeypatch):
    def mock_post(*args, **kwargs):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(
        "processor.llm.requests.post",
        mock_post,
    )

    with pytest.raises(OllamaConnectionError):
        generate_response([
            {"role": "user", "content": "Explain BERT."}
        ])


def test_generate_response_handles_connection_error(monkeypatch):
    def mock_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(
        "processor.llm.requests.post",
        mock_post,
    )

    with pytest.raises(OllamaConnectionError):
        generate_response([
            {"role": "user", "content": "Explain BERT."}
        ])


def test_generate_answer_without_context():
    result = generate_answer(
        question="Explain BERT.",
        retrieved_chunks=[],
    )

    assert "don't have enough information" in result["answer"].lower()
    assert result["sources"] == []


def test_generate_answer_preserves_sources(monkeypatch):
    def mock_generate_response(messages, model):
        assert "[S1]" in messages[1]["content"]
        return "BERT uses a bidirectional encoder [S1]."

    monkeypatch.setattr(
        "processor.llm.generate_response",
        mock_generate_response,
    )

    result = generate_answer(
        question="What architecture does BERT use?",
        retrieved_chunks=SAMPLE_CHUNKS,
    )

    assert result["answer"] == "BERT uses a bidirectional encoder [S1]."
    assert result["sources"][0]["citation"] == "S1"
    assert result["sources"][0]["source"] == "bert.pdf"
    assert result["sources"][0]["page_number"] == 3
    assert result["sources"][0]["chunk_id"] == "bert_chunk_001"
    assert result["cited_sources"] == result["sources"]
    assert result["valid_citations"] == ["S1"]
    assert result["invalid_citations"] == []

def test_extract_citations_returns_unique_labels():
    answer = "BERT uses an encoder [S1]. It uses bidirectional attention [S2]. See [S1]."

    result = extract_citations(answer)

    assert result == ["S1", "S2"]


def test_extract_citations_handles_no_citations():
    result = extract_citations("BERT uses a Transformer encoder.")

    assert result == []


def test_extract_citations_ignores_invalid_formats():
    answer = "Valid [S1], invalid [Sx], and lowercase [s2]."

    result = extract_citations(answer)

    assert result == ["S1"]


def test_validate_citations_separates_valid_and_invalid():
    answer = "BERT uses an encoder [S1] and bidirectional attention [S3]."

    sources = [
        {"citation": "S1", "source": "bert.pdf"},
        {"citation": "S2", "source": "attention.pdf"},
    ]

    result = validate_citations(answer, sources)

    assert result["valid_citations"] == ["S1"]
    assert result["invalid_citations"] == ["S3"]


def test_validate_citations_handles_no_citations():
    result = validate_citations(
        "BERT uses a bidirectional encoder.",
        [{"citation": "S1", "source": "bert.pdf"}],
    )

    assert result["valid_citations"] == []
    assert result["invalid_citations"] == []