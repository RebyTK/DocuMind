from processor.vector_store import search_index
from repositories.vector_repository import search_similar_chunks


def diversify_results(
    candidates: list[dict],
    top_k: int,
    max_per_document: int = 2,
) -> list[dict]:
    """Limit chunks per document while preserving retrieval ranking."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if max_per_document < 1:
        raise ValueError("max_per_document must be at least 1.")

    results = []
    deferred = []
    document_counts = {}

    for candidate in candidates:
        metadata = candidate.get("metadata", {})

        document_key = (
            metadata.get("document_id")
            or metadata.get("source")
            or "unknown"
        )

        current_count = document_counts.get(document_key, 0)

        if current_count < max_per_document:
            results.append(candidate)
            document_counts[document_key] = current_count + 1
        else:
            deferred.append(candidate)

        if len(results) == top_k:
            break

    # Fill remaining positions when there are not enough documents.
    if len(results) < top_k:
        remaining = top_k - len(results)
        results.extend(deferred[:remaining])

    return results


def retrieve_chunks(
    query_embedding: list,
    top_k: int = 5,
    backend: str = "faiss",
    index=None,
    embedded_chunks: list[dict] | None = None,
    max_per_document: int = 2,
) -> list[dict]:
    """
    Retrieve chunks through FAISS or PostgreSQL.

    Both backends return the same result structure and use
    the same document-diversity policy.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if max_per_document < 1:
        raise ValueError("max_per_document must be at least 1.")

    if not isinstance(query_embedding, list):
        raise ValueError("Query embedding must be a list.")

    backend = backend.lower()

    # Retrieve a larger candidate pool before applying diversity.
    candidate_k = max(top_k * 10, top_k)

    if backend == "faiss":

        if index is None or not embedded_chunks:
            return []

        candidates = search_index(
            query_embedding=query_embedding,
            index=index,
            embedded_chunks=embedded_chunks,
            top_k=candidate_k,
            max_per_document=candidate_k,
        )

    elif backend == "postgres":

        candidates = search_similar_chunks(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

    else:
        raise ValueError(
            f"Unsupported retrieval backend: {backend}"
        )

    return diversify_results(
        candidates=candidates,
        top_k=top_k,
        max_per_document=max_per_document,
    )
