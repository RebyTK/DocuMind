import numpy as np
import faiss

def create_index(embedded_chunks: list[dict]):
    """Create a FAISS index from embedded document chunks."""

    if not embedded_chunks:
        return None
    embeddings = [
        chunk["embedding"]
        for chunk in embedded_chunks
    ]

    vectors = np.array(embeddings, dtype = np.float32)
    dimension = vectors.shape[1]
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    return index


def search_index(
    query_embedding: list,
    index,
    embedded_chunks: list[dict],
    top_k: int = 3,
    max_per_document: int = 2,
) -> list[dict]:
    """
    Search FAISS and diversify results across source documents.

    First pass prioritizes source diversity. If there are not enough
    distinct sources, remaining results are filled by similarity score.
    """

    if index is None or not embedded_chunks:
        return []

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if max_per_document < 1:
        raise ValueError("max_per_document must be at least 1.")

    query_vector = np.array(
        [query_embedding],
        dtype=np.float32,
    )

    if query_vector.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension doesn't match the index"
        )

    faiss.normalize_L2(query_vector)

    # Retrieve more candidates than we ultimately need.
    candidate_k = min(index.ntotal, max(top_k * 10, top_k))

    scores, indices = index.search(
        query_vector,
        candidate_k,
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(embedded_chunks):
            chunk = embedded_chunks[idx]

            candidates.append({
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(score),
            })

    # First pass: prioritize different source documents.
    results = []
    deferred = []
    document_counts = {}

    for result in candidates:
        metadata = result["metadata"]

        # Prefer document_id because filenames may not be unique.
        document_key = (
            metadata.get("document_id")
            or metadata.get("source")
            or "unknown"
        )

        current_count = document_counts.get(document_key, 0)

        if current_count < max_per_document:
            results.append(result)
            document_counts[document_key] = current_count + 1
        else:
            deferred.append(result)

        if len(results) == top_k:
            break

    # Second pass: fill unused slots with the best remaining chunks.
    if len(results) < top_k:
        remaining = top_k - len(results)
        results.extend(deferred[:remaining])

    return results
