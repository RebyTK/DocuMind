import pytest
import json
import numpy as np

from unittest.mock import Mock, patch

from evaluation.evaluate_retrieval import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    load_evaluation_queries,
    retrieve_chunk_ids,
    run_retrieval_evaluation
)


def test_precision_at_k():
    assert precision_at_k(
        [17, 12, 20],
        [17, 35],
        3
    ) == pytest.approx(1 / 3)


def test_precision_all_relevant():
    assert precision_at_k(
        [17, 35, 20],
        [17, 35, 20],
        3
    ) == 1.0


def test_precision_no_relevant():
    assert precision_at_k(
        [12, 20, 51],
        [17, 35],
        3
    ) == 0.0


def test_precision_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k([17, 12], [17], 0)


def test_recall_at_k():
    assert recall_at_k(
        [17, 12, 20],
        [17, 35],
        3
    ) == 0.5


def test_recall_all_relevant():
    assert recall_at_k(
        [17, 35, 20],
        [17, 35],
        3
    ) == 1.0


def test_recall_empty_relevant_chunks():
    with pytest.raises(ValueError):
        recall_at_k([17, 12], [], 2)


def test_reciprocal_rank_first():
    assert reciprocal_rank(
        [17, 12, 20],
        [17, 35]
    ) == 1.0


def test_reciprocal_rank_second():
    assert reciprocal_rank(
        [12, 17, 20],
        [17, 35]
    ) == 0.5


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(
        [12, 20, 51],
        [17, 35]
    ) == 0.0

def test_mrr_all_first_rank():
    retrieved = [[17, 12], [35, 20]]
    relevant = [[17], [35]]

    assert mean_reciprocal_rank(retrieved, relevant) == 1.0


def test_mrr_different_ranks():
    retrieved = [[12, 17], [35, 20]]
    relevant = [[17], [35]]

    assert mean_reciprocal_rank(retrieved, relevant) == 0.75


def test_mrr_no_relevant_results():
    retrieved = [[12, 20], [51, 35]]
    relevant = [[17], [99]]

    assert mean_reciprocal_rank(retrieved, relevant) == 0.0


def test_mrr_empty_input():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([], [])


def test_mrr_different_input_lengths():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([[17]], [[17], [35]])


# Helper function to create temporary JSON files
def create_json_file(tmp_path, data):
    file_path = tmp_path / "queries.json"

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file)

    return file_path


# 1. Valid dataset
def test_load_evaluation_queries_success(tmp_path):
    data = [
        {
            "query": "What is ISO 27001?",
            "relevant_chunks": [17, 20]
        }
    ]

    file_path = create_json_file(tmp_path, data)

    result = load_evaluation_queries(file_path)

    assert result == data


# 2. Missing file
def test_load_evaluation_queries_missing_file(tmp_path):
    file_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_evaluation_queries(file_path)


# 3. Invalid JSON
def test_load_evaluation_queries_invalid_json(tmp_path):
    file_path = tmp_path / "queries.json"
    file_path.write_text("{ invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_evaluation_queries(file_path)


# 4. Empty dataset
def test_load_evaluation_queries_empty_dataset(tmp_path):
    file_path = create_json_file(tmp_path, [])

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)


# 5. Root is not a list
def test_load_evaluation_queries_invalid_root(tmp_path):
    file_path = create_json_file(
        tmp_path,
        {"query": "What is ISO 27001?"}
    )

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)


# 6. Entry is not a dictionary
def test_load_evaluation_queries_invalid_entry(tmp_path):
    file_path = create_json_file(tmp_path, ["invalid entry"])

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)


# 7. Missing required key
def test_load_evaluation_queries_missing_key(tmp_path):
    data = [
        {"query": "What is ISO 27001?"}
    ]

    file_path = create_json_file(tmp_path, data)

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)


# 8. Empty query
def test_load_evaluation_queries_empty_query(tmp_path):
    data = [
        {"query": "   ", "relevant_chunks": [17]}
    ]

    file_path = create_json_file(tmp_path, data)

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)


# 9. Query is not a string
def test_load_evaluation_queries_invalid_query_type(tmp_path):
    data = [
        {"query": 123, "relevant_chunks": [17]}
    ]

    file_path = create_json_file(tmp_path, data)

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)


# 10. Relevant chunks are empty
def test_load_evaluation_queries_empty_chunks(tmp_path):
    data = [
        {"query": "What is ISO 27001?", "relevant_chunks": []}
    ]

    file_path = create_json_file(tmp_path, data)

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)


# 11. Chunk ID is not an integer
def test_load_evaluation_queries_invalid_chunk_id(tmp_path):
    data = [
        {"query": "What is ISO 27001?", "relevant_chunks": ["17"]}
    ]

    file_path = create_json_file(tmp_path, data)

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)


# 12. Negative chunk ID
def test_load_evaluation_queries_negative_chunk_id(tmp_path):
    data = [
        {"query": "What is ISO 27001?", "relevant_chunks": [-1]}
    ]

    file_path = create_json_file(tmp_path, data)

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)


# 13. Boolean chunk ID
def test_load_evaluation_queries_boolean_chunk_id(tmp_path):
    data = [
        {"query": "What is ISO 27001?", "relevant_chunks": [True]}
    ]

    file_path = create_json_file(tmp_path, data)

    with pytest.raises(ValueError):
        load_evaluation_queries(file_path)

def test_retrieve_chunk_ids_success():
    model = Mock()
    model.encode.return_value = np.array(
        [[0.1, 0.2, 0.3]],
        dtype=np.float32
    )

    index = Mock()

    embedded_chunks = [
        {"text": "Chunk A", "metadata": {"chunk_index": 10}},
        {"text": "Chunk B", "metadata": {"chunk_index": 11}},
    ]

    mock_results = [
        {"text": "Chunk B", "metadata": {"chunk_index": 11}},
        {"text": "Chunk A", "metadata": {"chunk_index": 10}},
    ]

    with patch(
        "evaluation.evaluate_retrieval.search_index",
        return_value=mock_results
    ):
        result = retrieve_chunk_ids(
            "What is ISO 27001?",
            model,
            index,
            embedded_chunks,
            k=2
        )

    assert result == [11, 10]
    model.encode.assert_called_once()


def test_retrieve_chunk_ids_empty_query():
    with pytest.raises(ValueError):
        retrieve_chunk_ids("", Mock(), Mock(), [], k=3)


def test_retrieve_chunk_ids_invalid_k():
    with pytest.raises(ValueError):
        retrieve_chunk_ids("What is ISO 27001?", Mock(), Mock(), [], k=0)


def test_retrieve_chunk_ids_empty_index():
    result = retrieve_chunk_ids(
        "What is ISO 27001?",
        Mock(),
        None,
        [],
        k=3
    )

    assert result == []

def test_run_retrieval_evaluation_success():
    queries = [
        {
            "query": "What is ISO 27001?",
            "relevant_chunks": [17]
        },
        {
            "query": "What is document control?",
            "relevant_chunks": [35]
        }
    ]

    mock_retrieval_results = [
        [17, 12, 20],
        [12, 35, 51]
    ]

    with patch(
        "evaluation.evaluate_retrieval.retrieve_chunk_ids",
        side_effect=mock_retrieval_results
    ):
        report = run_retrieval_evaluation(
            queries=queries,
            model=None,
            index=None,
            embedded_chunks=[],
            k=3
        )

    assert len(report["results"]) == 2

    assert report["results"][0]["precision_at_k"] == pytest.approx(1 / 3)
    assert report["results"][0]["recall_at_k"] == 1.0
    assert report["results"][0]["reciprocal_rank"] == 1.0

    assert report["results"][1]["precision_at_k"] == pytest.approx(1 / 3)
    assert report["results"][1]["recall_at_k"] == 1.0
    assert report["results"][1]["reciprocal_rank"] == 0.5

    assert report["summary"]["precision_at_k"] == pytest.approx(1 / 3)
    assert report["summary"]["recall_at_k"] == 1.0
    assert report["summary"]["mrr"] == pytest.approx(0.75)


def test_run_retrieval_evaluation_empty_queries():
    with pytest.raises(ValueError):
        run_retrieval_evaluation(
            queries=[],
            model=None,
            index=None,
            embedded_chunks=[],
            k=3
        )


def test_run_retrieval_evaluation_invalid_k():
    with pytest.raises(ValueError):
        run_retrieval_evaluation(
            queries=[{
                "query": "What is ISO 27001?",
                "relevant_chunks": [17]
            }],
            model=None,
            index=None,
            embedded_chunks=[],
            k=0
        )
