import json
from pathlib import Path
from processor.vector_store import search_index


def load_evaluation_queries(file_path: str | Path) -> list[dict]:
    """Load and validate retrieval evaluation queries."""

    # 1. Validate file existence
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 2. Load JSON data
    with file_path.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    # 3. Validate root structure
    if not isinstance(queries, list):
        raise ValueError("The JSON file must contain a list of queries.")

    if not queries:
        raise ValueError("Evaluation dataset cannot be empty.")

    # 4. Validate individual query entries
    for index, query in enumerate(queries):

        if not isinstance(query, dict):
            raise ValueError(
                f"Query at index {index} must be a dictionary."
            )

        if "query" not in query or "relevant_chunks" not in query:
            raise ValueError(
                f"Query at index {index} must contain "
                "'query' and 'relevant_chunks'."
            )

        # Validate query text
        query_text = query["query"]

        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(
                f"Query text at index {index} must be a non-empty string."
            )

        # Validate relevant chunk IDs
        relevant_chunks = query["relevant_chunks"]

        if not isinstance(relevant_chunks, list) or not relevant_chunks:
            raise ValueError(
                f"Relevant chunks at index {index} must be a non-empty list."
            )

        if any(type(chunk_id) is not int for chunk_id in relevant_chunks):
            raise ValueError(
                f"All relevant chunk IDs at index {index} must be integers."
            )

        if any(chunk_id < 0 for chunk_id in relevant_chunks):
            raise ValueError(
                f"Chunk IDs at index {index} cannot be negative."
            )

    return queries



def precision_at_k(
    retrieved_chunks: list[int],
    relevant_chunks: list[int],
    k: int
) -> float:
    """Calculate Precision@K."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Consider only the first k retrieved chunks.
    top_k = retrieved_chunks[:k]

    # Count how many retrieved chunks are relevant.
    relevant_count = sum(
        chunk in relevant_chunks
        for chunk in top_k
    )

    return relevant_count / k

def recall_at_k(
    retrieved_chunks: list[int],
    relevant_chunks: list[int],
    k: int
) -> float:
    """Calculate Recall@K."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Consider only the first k retrieved chunks.
    top_k = retrieved_chunks[:k]

    # Count how many relevant chunks are retrieved.
    relevant_count = sum(
        chunk in relevant_chunks
        for chunk in top_k
    )

    # Avoid division by zero if there are no relevant chunks.
    if not relevant_chunks:
        raise ValueError("relevant_chunks cannot be empty")

    return relevant_count / len(relevant_chunks)

def reciprocal_rank(
    retrieved_chunks: list[int],
    relevant_chunks: list[int]
) -> float:
    """Calculate Reciprocal Rank."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk in relevant_chunks:
            return 1.0 / rank
    return 0.0

def mean_reciprocal_rank(
    retrieved_chunks_list: list[list[int]],
    relevant_chunks_list: list[list[int]]
) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""

    if not retrieved_chunks_list or not relevant_chunks_list:
        raise ValueError("Input lists cannot be empty")

    if len(retrieved_chunks_list) != len(relevant_chunks_list):
        raise ValueError(
            "retrieved_chunks_list and relevant_chunks_list "
            "must have the same length"
        )

    reciprocal_ranks = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(
            retrieved_chunks_list,
            relevant_chunks_list
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def retrieve_chunk_ids(
    query: str,
    model,
    index,
    embedded_chunks: list[dict],
    k: int = 3
) -> list[int]:
    """Retrieve IDs of the most relevant document chunks."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if index is None or not embedded_chunks:
        return []

    # Generate query embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )[0].tolist()

    # Retrieve relevant chunks
    results = search_index(
        query_embedding=query_embedding,
        index=index,
        embedded_chunks=embedded_chunks,
        top_k=k
    )

    # Extract original chunk IDs
    return [
        result["metadata"]["chunk_index"]
        for result in results
    ]


def run_retrieval_evaluation(
    queries: list[dict],
    model,
    index,
    embedded_chunks: list[dict],
    k: int = 3
) -> dict:
    """Evaluate retrieval performance across multiple queries."""

    # Step 1: Validate inputs
    if not queries:
        raise ValueError("Evaluation queries cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    # Step 2: Store individual query results
    results = []

    retrieved_chunks_list = []
    relevant_chunks_list = []

    for query_data in queries:

        query = query_data["query"]
        relevant_chunks = query_data["relevant_chunks"]

        # Retrieve relevant chunk IDs
        retrieved_chunks = retrieve_chunk_ids(
            query=query,
            model=model,
            index=index,
            embedded_chunks=embedded_chunks,
            k=k
        )

        # Calculate metrics for this query
        precision = precision_at_k(
            retrieved_chunks,
            relevant_chunks,
            k
        )

        recall = recall_at_k(
            retrieved_chunks,
            relevant_chunks,
            k
        )

        rr = reciprocal_rank(
            retrieved_chunks,
            relevant_chunks
        )

        # Store individual results
        results.append({
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "relevant_chunks": relevant_chunks,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "reciprocal_rank": rr
        })

        retrieved_chunks_list.append(retrieved_chunks)
        relevant_chunks_list.append(relevant_chunks)

    # Step 3: Calculate aggregate metrics
    average_precision = sum(
        result["precision_at_k"] for result in results
    ) / len(results)

    average_recall = sum(
        result["recall_at_k"] for result in results
    ) / len(results)

    mrr = mean_reciprocal_rank(
        retrieved_chunks_list,
        relevant_chunks_list
    )

    # Step 4: Return evaluation report
    return {
        "results": results,
        "summary": {
            "precision_at_k": average_precision,
            "recall_at_k": average_recall,
            "mrr": mrr
        }
    }
