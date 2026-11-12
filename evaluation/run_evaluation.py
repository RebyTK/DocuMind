import json
from pathlib import Path

from processor.embeddings import load_embedding_model
from processor.vector_store import create_index

from evaluation.evaluate_retrieval import (
    load_evaluation_queries,
    run_retrieval_evaluation,
)


def main():
    """Run retrieval evaluation on the ISO 27001 document."""

    # Resolve project paths
    project_root = Path(__file__).resolve().parents[1]

    embeddings_path = project_root / "output" / "sample_embeddings.json"
    queries_path = project_root / "evaluation" / "queries.json"
    report_path = project_root / "output" / "evaluation_report.json"

    # Load embedded chunks
    with embeddings_path.open("r", encoding="utf-8") as file:
        embedded_chunks = json.load(file)

    if not isinstance(embedded_chunks, list) or not embedded_chunks:
        raise ValueError("No valid embedded chunks found.")

    print(f"Loaded {len(embedded_chunks)} embedded chunks.")

    # Create FAISS index
    index = create_index(embedded_chunks)

    if index is None:
        raise ValueError("Could not create FAISS index.")

    print(f"FAISS index created with {index.ntotal} vectors.")

    # Load embedding model
    model = load_embedding_model()

    # Load evaluation queries
    queries = load_evaluation_queries(queries_path)

    print(f"Loaded {len(queries)} evaluation queries.")

    # Run retrieval evaluation
    report = run_retrieval_evaluation(
        queries=queries,
        model=model,
        index=index,
        embedded_chunks=embedded_chunks,
        k=3
    )

    # Save evaluation report
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n========== RETRIEVAL EVALUATION ==========")

    for result in report["results"]:
        print(f"\nQuery: {result['query']}")
        print(f"Retrieved chunks: {result['retrieved_chunks']}")
        print(f"Precision@3: {result['precision_at_k']:.4f}")
        print(f"Recall@3: {result['recall_at_k']:.4f}")
        print(f"Reciprocal Rank: {result['reciprocal_rank']:.4f}")

    print("\n========== SUMMARY ==========")

    summary = report["summary"]

    print(f"Average Precision@3: {summary['precision_at_k']:.4f}")
    print(f"Average Recall@3: {summary['recall_at_k']:.4f}")
    print(f"Mean Reciprocal Rank: {summary['mrr']:.4f}")

    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
