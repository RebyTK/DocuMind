"""Retrieval-augmented answer generation service."""

from processor.llm import generate_answer
from services.retrieval_service import search_documents


def answer_question(question: str, top_k: int = 5) -> dict:
    """Retrieve context and generate a grounded answer with citations."""
    retrieved_chunks = search_documents(question, top_k=top_k)
    generated = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    return {
        "question": question,
        "answer": generated.get("answer", ""),
        "citations": generated.get("cited_sources", []),
        "valid_citations": generated.get("valid_citations", []),
        "invalid_citations": generated.get("invalid_citations", []),
        "retrieval_count": len(retrieved_chunks),
    }
