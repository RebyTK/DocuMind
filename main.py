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
from repositories.document_repository import (
    get_document_by_hash,
    list_documents,
    delete_document,
)
from processor.retrieval import retrieve_chunks

from repositories.ingestion_repository import (
    persist_document_with_chunks,
)


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
def ingest_document_to_postgres(
    file_path: Path,
    embedding_model=None,
) -> dict:
    """Extract, embed and persist a document directly in PostgreSQL."""

    logging.info(
        "Starting PostgreSQL ingestion: %s",
        file_path.name,
    )

    # Calculate SHA-256 hash.
    file_hash = calculate_file_hash(file_path)

    # Check whether the document already exists.
    existing_document = get_document_by_hash(file_hash)

    if existing_document:
        logging.info(
            "Document already exists in PostgreSQL: %s",
            existing_document["filename"],
        )

        return {
            "inserted": False,
            **existing_document,
        }

    # Generate a new document ID.
    document_id = generate_document_id()

    # Extract and chunk the document.
    cleaned_content, chunks, extra_metadata = extract_document(
        file_path=file_path,
        document_id=document_id,
    )

    if not chunks:
        raise ValueError(
            "No extractable content found in the document."
        )

    logging.info(
        "Document chunked: %d chunks",
        len(chunks),
    )

    # Generate embeddings. Reuse a model supplied by the CLI when available.
    if embedding_model is None:
        embedding_model = load_embedding_model()

    embedded_chunks = embed_chunks(
        chunks,
        embedding_model,
    )

    logging.info(
        "Generated embeddings: %d",
        len(embedded_chunks),
    )

    # Calculate document statistics.
    stats = analyze_text(cleaned_content)

    # Store useful summary metadata in PostgreSQL.
    document_metadata = {
        **stats,
        "document_id": document_id,
    }

    if "page_count" in extra_metadata:
        document_metadata["page_count"] = (
            extra_metadata["page_count"]
        )

    if "pages_without_extractable_text" in extra_metadata:
        document_metadata["pages_without_extractable_text"] = (
            extra_metadata["pages_without_extractable_text"]
        )

    if "tables" in extra_metadata:
        document_metadata["table_count"] = len(
            extra_metadata["tables"]
        )

    if "elements" in extra_metadata:
        document_metadata["element_count"] = len(
            extra_metadata["elements"]
        )

    # Build the database document record.
    document_record = {
        "document_id": document_id,
        "filename": file_path.name,
        "file_type": file_path.suffix.lower().lstrip("."),
        "file_hash": file_hash,
        "total_chunks": len(embedded_chunks),
        "metadata": document_metadata,
    }

    # Atomically persist document and chunks.
    result = persist_document_with_chunks(
        document=document_record,
        embedded_chunks=embedded_chunks,
    )

    if not result["inserted"]:
        logging.info(
            "Document was registered concurrently: %s",
            result["filename"],
        )
        return result

    logging.info(
        "Document successfully stored in PostgreSQL: %s",
        document_id,
    )

    result["stats"] = stats

    return result


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
    """Run the PostgreSQL-backed DocuMind CLI."""

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    parser = argparse.ArgumentParser(
        description="DocuMind - Multi-document RAG application"
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="Optional PDF, DOCX, or TXT document to ingest",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search all documents stored in PostgreSQL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Generate a grounded answer using the local LLM",
    )
    parser.add_argument(
        "--list-documents",
        action="store_true",
        help="List documents stored in PostgreSQL",
    )
    parser.add_argument(
        "--delete",
        metavar="DOCUMENT_ID",
        help="Delete a document and its chunks from PostgreSQL",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if args.answer and not args.query:
        parser.error("--answer requires --query")

    management_operation = args.list_documents or args.delete is not None
    if management_operation and (args.file or args.query or args.answer):
        parser.error(
            "--list-documents and --delete cannot be combined with "
            "document ingestion or search."
        )

    if args.list_documents and args.delete is not None:
        parser.error("Choose either --list-documents or --delete.")

    if not any((args.file, args.query, args.list_documents, args.delete)):
        parser.error(
            "Provide a document to ingest, --query, --list-documents, "
            "or --delete DOCUMENT_ID."
        )

    logging.info("DocuMind started")

    # Management operation: list all PostgreSQL documents.
    if args.list_documents:
        try:
            documents = list_documents()
        except Exception:
            logging.exception("Unable to list PostgreSQL documents")
            return

        print("\n================================")
        print("DOCUMIND DOCUMENT COLLECTION")
        print("================================")

        if not documents:
            print("No documents are registered.")
            return

        for document in documents:
            print(f"\nFilename: {document['filename']}")
            print(f"Document ID: {document['document_id']}")
            print(f"Type: {document['file_type']}")
            print(f"Chunks: {document['total_chunks']}")
            print(f"Created: {document['created_at']}")

        print(f"\nTotal documents: {len(documents)}")
        return

    # Management operation: delete one document. The schema's ON DELETE
    # CASCADE constraint removes its chunks in the same database operation.
    if args.delete is not None:
        try:
            deleted = delete_document(args.delete)
        except Exception:
            logging.exception("Unable to delete document from PostgreSQL")
            return

        if deleted:
            print(f"Document deleted successfully: {args.delete}")
        else:
            print(f"Document not found: {args.delete}")
        return

    file_path = None
    embedding_model = None

    # Ingest a document directly into PostgreSQL.
    if args.file:
        file_path = Path(args.file)

        if not file_path.is_file():
            parser.error(f"Document file not found: {file_path}")

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            parser.error(
                f"Unsupported file type: "
                f"{file_path.suffix or '(no extension)'}"
            )

        try:
            # If ingestion and search are requested together, load the
            # embedding model once and reuse it for both operations.
            if args.query:
                embedding_model = load_embedding_model()

            ingestion_result = ingest_document_to_postgres(
                file_path=file_path,
                embedding_model=embedding_model,
            )

        except Exception as error:
            # Keep expected document failures readable at the CLI boundary.
            # The ingestion repository handles database transaction rollback.
            logging.error(
                "PostgreSQL document ingestion failed: %s",
                error,
            )
            print(f"\nDocument ingestion failed: {error}")
            return

        if ingestion_result.get("inserted"):
            print("\n================================")
            print("DOCUMIND DOCUMENT INGESTION")
            print("================================")
            print("Status: Successfully ingested")
            print("Filename:", ingestion_result["filename"])
            print("Document ID:", ingestion_result["document_id"])
            print("Chunks:", ingestion_result["total_chunks"])

            stats = ingestion_result.get("stats", {})
            if stats:
                print("Characters:", stats.get("characters"))
                print("Words:", stats.get("words"))
                print("Lines:", stats.get("lines"))
        else:
            print(
                "\nDocument already exists in PostgreSQL: "
                f"{ingestion_result['filename']}"
            )
            print("Document ID:", ingestion_result["document_id"])
            print("Stored chunks:", ingestion_result["total_chunks"])

    # Search across the complete PostgreSQL document collection.
    if args.query:
        try:
            if embedding_model is None:
                embedding_model = load_embedding_model()

            query_embedding = embedding_model.encode(
                [args.query],
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0].tolist()

            results = retrieve_chunks(
                query_embedding=query_embedding,
                top_k=args.top_k,
                backend="postgres",
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
                cited_sources = answer_result.get("cited_sources", [])
                invalid_citations = answer_result.get("invalid_citations", [])

                if cited_sources:
                    for source in cited_sources:
                        page = source.get("page_number")
                        page_info = f", Page {page}" if page is not None else ""
                        print(
                            f"[{source['citation']}] "
                            f"{source['source']}{page_info}"
                        )
                else:
                    print("No valid source citations were identified.")

                if invalid_citations:
                    print("\nWARNING: Invalid citation references detected:")
                    print(", ".join(invalid_citations))
            else:
                display_search_results(args.query, results)

        except Exception:
            logging.exception("PostgreSQL query processing failed")
            return


if __name__ == "__main__":
    main()
