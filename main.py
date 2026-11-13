import argparse
import json
import logging
from pathlib import Path

from processor.cleaner import clean_text
from processor.analyser import analyze_text
from processor.pdf_reader import extract_pdf_data
from processor.docx_reader import extract_text_from_docx
from processor.formatter import format_elements_as_markdown
from processor.pdf_table_reader import extract_tables_from_pdf
from processor.chunker import (
    chunk_text_with_metadata,
    generate_document_id,
)
from processor.embeddings import (
    load_embedding_model,
    embed_chunks,
)
from processor.vector_store import search_index
from processor.document_manager import (
    load_registry,
    save_registry,
    register_document,
    calculate_file_hash,
    find_document_by_hash,
    save_document_data,
    rebuild_shared_index,
)
from processor.llm import generate_answer


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def setup_logging(output_dir: Path) -> None:
    """Configure application logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                output_dir / "documind.log",
                encoding="utf-8",
            ),
        ],
    )


def extract_pdf_content(
    file_path: Path,
    document_id: str,
) -> tuple[str, list[dict], dict]:
    """Extract PDF text and tables page by page."""

    pdf_data = extract_pdf_data(file_path)
    pdf_tables = extract_tables_from_pdf(file_path)

    tables_by_page = {}

    for table in pdf_tables:
        tables_by_page.setdefault(
            table["page"], []
        ).append(table)

    chunks = []
    cleaned_pages = []
    chunk_index = 0

    for page_data in pdf_data["pages"]:
        page_number = page_data["page"]
        page_text = page_data["text"]

        page_sections = []

        if page_text.strip():
            page_sections.append(clean_text(page_text))

        # Add tables to their original pages.
        for table_data in tables_by_page.get(page_number, []):
            rows = table_data["rows"]

            if not rows:
                continue

            column_count = max(len(row) for row in rows)

            normalized_rows = [
                row + [""] * (column_count - len(row))
                for row in rows
            ]

            header = normalized_rows[0]
            separator = ["---"] * column_count

            markdown_rows = [
                "| " + " | ".join(row) + " |"
                for row in [
                    header,
                    separator,
                    *normalized_rows[1:],
                ]
            ]

            table_markdown = "\n".join(markdown_rows)

            page_sections.append(
                f"## Table {table_data['table_number']}\n\n"
                f"{table_markdown}"
            )

        page_content = "\n\n".join(page_sections)

        if not page_content.strip():
            continue

        cleaned_pages.append(
            f"## Page {page_number}\n\n{page_content}"
        )

        page_chunks = chunk_text_with_metadata(
            text=page_content,
            source=file_path.name,
            document_id=document_id,
            file_type="pdf",
            page_number=page_number,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            chunk_index_start=chunk_index,
        )

        chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    cleaned_content = "\n\n".join(cleaned_pages)

    extra_metadata = {
        "page_count": pdf_data["page_count"],
        "pages_without_extractable_text": (
            pdf_data["pages_without_extractable_text"]
        ),
        "tables": pdf_tables,
    }

    return cleaned_content, chunks, extra_metadata


def extract_document(
    file_path: Path,
    document_id: str,
) -> tuple[str, list[dict], dict]:
    """Extract and chunk a supported document."""

    file_type = file_path.suffix.lower().lstrip(".")

    extra_metadata = {}

    if file_type == "pdf":
        return extract_pdf_content(file_path, document_id)

    elif file_type == "docx":
        docx_data = extract_text_from_docx(file_path)

        structured_elements = docx_data["elements"]

        cleaned_content = format_elements_as_markdown(
            structured_elements
        )

        extra_metadata["elements"] = structured_elements

    elif file_type == "txt":
        content = file_path.read_text(encoding="utf-8")
        cleaned_content = clean_text(content)

    else:
        raise ValueError(
            f"Unsupported file type: {file_type}"
        )

    chunks = chunk_text_with_metadata(
        text=cleaned_content,
        source=file_path.name,
        document_id=document_id,
        file_type=file_type,
        page_number=None,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    return cleaned_content, chunks, extra_metadata


def save_document_outputs(
    output_dir: Path,
    document_id: str,
    file_path: Path,
    cleaned_content: str,
    chunks: list[dict],
    embedded_chunks: list[dict],
    extra_metadata: dict,
    file_hash: str,
) -> dict:
    """Save document content and metadata using its document ID."""

    documents_dir = output_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    output_extension = (
        ".md" if file_path.suffix.lower() == ".docx" else ".txt"
    )

    cleaned_filename = (
        f"{document_id}_cleaned{output_extension}"
    )

    metadata_filename = f"{document_id}_metadata.json"

    cleaned_path = documents_dir / cleaned_filename
    metadata_path = documents_dir / metadata_filename

    stats = analyze_text(cleaned_content)

    metadata = {
        "document_id": document_id,
        "filename": file_path.name,
        "file_type": file_path.suffix.lower().lstrip("."),
        "file_hash": file_hash,
        "total_chunks": len(chunks),
        "output_file": cleaned_filename,
        "characters": stats["characters"],
        "words": stats["words"],
        "lines": stats["lines"],
        **extra_metadata,
    }

    cleaned_path.write_text(
        cleaned_content,
        encoding="utf-8",
    )

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return {
        "cleaned_file": f"documents/{cleaned_filename}",
        "metadata_file": f"documents/{metadata_filename}",
        "stats": stats,
    }


def ingest_document(
    file_path: Path,
    output_dir: Path,
    registry: dict,
    file_hash: str,
    model,
) -> tuple[str, object]:
    """Process, embed, persist and register a new document."""

    document_id = generate_document_id()

    logging.info("Processing new document: %s", file_path.name)

    cleaned_content, chunks, extra_metadata = extract_document(
        file_path,
        document_id,
    )

    logging.info(
        "Document chunked: %s (%d chunks)",
        file_path.name,
        len(chunks),
    )

    embedded_chunks = embed_chunks(chunks, model)

    logging.info(
        "Embeddings generated: %s (%d chunks)",
        file_path.name,
        len(embedded_chunks),
    )

    # Persist chunks and embeddings.
    storage_paths = save_document_data(
        document_id=document_id,
        chunks=chunks,
        embedded_chunks=embedded_chunks,
        output_dir=output_dir,
    )

    # Save cleaned document and metadata.
    output_paths = save_document_outputs(
        output_dir=output_dir,
        document_id=document_id,
        file_path=file_path,
        cleaned_content=cleaned_content,
        chunks=chunks,
        embedded_chunks=embedded_chunks,
        extra_metadata=extra_metadata,
        file_hash=file_hash,
    )

    stats = output_paths["stats"]

    document_record = {
        "document_id": document_id,
        "filename": file_path.name,
        "file_type": file_path.suffix.lower().lstrip("."),
        "file_hash": file_hash,
        "chunks_file": storage_paths["chunks_file"],
        "embeddings_file": storage_paths["embeddings_file"],
        "cleaned_file": output_paths["cleaned_file"],
        "metadata_file": output_paths["metadata_file"],
        "chunk_count": len(chunks),
    }

    # Register only after document data has been persisted.
    register_document(registry, document_record)
    save_registry(registry, output_dir / "document_registry.json")

    logging.info(
        "Document registered successfully: %s",
        document_id,
    )

    return document_id, stats


def display_search_results(
    query: str,
    results: list[dict],
) -> None:
    """Display semantic search results with source metadata."""

    print("\n================================")
    print("DOCUMIND SEMANTIC SEARCH")
    print("================================")
    print("Query:", query)

    if not results:
        print("\nNo relevant chunks found.")
        return

    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\nResult {i}")
        print("Source:", metadata.get("source", "Unknown"))
        print("Document ID:", metadata.get("document_id", "Unknown"))
        print("Chunk ID:", result.get("chunk_id"))
        print("Chunk:", metadata.get("chunk_index"))

        page_number = metadata.get("page_number")

        if page_number is not None:
            print("Page:", page_number)

        print("Similarity:", round(result["score"], 4))
        print("\nContent:")
        print(result["text"])
        print("-" * 60)


def main():
    """Run the DocuMind document processing CLI."""

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    logging.info("DocuMind Started")

    parser = argparse.ArgumentParser(
        description="DocuMind - AI Document Processor"
    )

    parser.add_argument(
        "file",
        help="Path to the document to process",
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Search across all registered documents",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of relevant chunks to retrieve",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help = "Generate a grounded answer using local LLM",
    )

    args = parser.parse_args()

    if args.answer and not args.query:
        parser.error("--answer requires --query.")

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    file_path = Path(args.file)

    if not file_path.is_file():
        logging.error("File not found: %s", file_path)
        return

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logging.error(
            "Unsupported file type: %s",
            file_path.suffix or "(no extension)",
        )
        return

    # Load the persistent document registry.
    registry_path = output_dir / "document_registry.json"

    try:
        registry = load_registry(registry_path)

    except (OSError, ValueError, json.JSONDecodeError):
        logging.exception("Unable to load document registry")
        return

    # Calculate the file hash for duplicate detection.
    try:
        file_hash = calculate_file_hash(file_path)

    except OSError:
        logging.exception("Unable to calculate document hash")
        return

    existing_document = find_document_by_hash(
        registry,
        file_hash,
    )

    model = None
    stats = None

    if existing_document:
        document_id = existing_document["document_id"]

        print(
            f"\nDocument already registered: "
            f"{existing_document['filename']}"
        )

        logging.info(
            "Skipping duplicate document: %s",
            existing_document["filename"],
        )

    else:
        try:
            # Load the model only when a new document needs embeddings.
            model = load_embedding_model()

            document_id, stats = ingest_document(
                file_path=file_path,
                output_dir=output_dir,
                registry=registry,
                file_hash=file_hash,
                model=model,
            )

        except Exception:
            logging.exception(
                "Failed to ingest document: %s",
                file_path.name,
            )
            return

    # Rebuild the shared FAISS index from persisted embeddings.
    try:
        index, all_embedded_chunks = rebuild_shared_index(
            registry,
            output_dir,
        )

    except Exception:
        logging.exception("Unable to rebuild shared FAISS index")
        return

    if index is None:
        logging.warning("No document chunks available for searching")
    else:
        logging.info(
            "Shared FAISS index rebuilt: %d vectors across %d documents",
            index.ntotal,
            len(registry["documents"]),
        )

    # Search across all registered documents.
    if args.query:
        if index is None:
            print("\nNo documents are available for searching.")
            return

        try:
            # Reuse the model if it was loaded during ingestion.
            if model is None:
                model = load_embedding_model()

            query_embedding = model.encode(
                [args.query],
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0].tolist()

            #Retrieve relevant chunks from the shared FAISS index.
            results = search_index(
                query_embedding=query_embedding,
                index=index,
                embedded_chunks=all_embedded_chunks,
                top_k=args.top_k,
            )
            if args.answer:
                answer_result = generate_answer(
                    question=args.query,
                    retrieved_chunks=results,
                )
                print("\n================================")
                print("DOCUMIND GENERATED ANSWER")
                print("================================")
                print(answer_result["answer"])

                print("\nCited Sources:")

                cited_sources = answer_result["cited_sources"]
                invalid_citations = answer_result["invalid_citations"]
                if cited_sources:
                    for source in cited_sources:
                        page = source["page_number"]

                        page_info = (
                            f", Page{page}"
                            if page is not None
                            else ""
                        )
                        print(
                            f"[{source["citation"]}]"
                            f"[{source["source"]}{page_info}]"
                        )
                else:
                    print("No valid source citations were identified")
                if invalid_citations:
                    print("\nWARNING: Invalid citation references detected: ")
                    print(", ".join(invalid_citations))

            else:
                #Display raw retrieval results when --answer is not specified.
                display_search_results(args.query, results)

            

        except Exception:
            logging.exception("Semantic search failed")
            return

    # Display statistics for newly ingested documents.
    if stats is not None:
        print("\n================================")
        print("DOCUMIND DOCUMENT PROCESSOR")
        print("================================")
        print("File:", file_path.name)
        print("Document ID:", document_id)
        print("Type:", file_path.suffix)

        print("\nDocument Statistics")
        print("Characters:", stats["characters"])
        print("Words:", stats["words"])
        print("Lines:", stats["lines"])

    print("\nRegistered documents:", len(registry["documents"]))


if __name__ == "__main__":
    main()
