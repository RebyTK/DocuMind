"""Embedding and PostgreSQL retrieval services."""

from functools import lru_cache

from processor.embeddings import load_embedding_model
from processor.retrieval import retrieve_chunks


@lru_cache(maxsize=1)
def get_query_embedding_model():
    """Load the sentence-transformer model once per API process."""
    return load_embedding_model()


def search_documents(question: str, top_k: int = 5) -> list[dict]:
    """Embed a question and retrieve relevant chunks from pgvector."""
    model = get_query_embedding_model()
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].tolist()

    return retrieve_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        backend="postgres",
        max_per_document=5,
    )
