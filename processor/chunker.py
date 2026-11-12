from uuid import uuid4

def generate_document_id() -> str:
    """Generate a unique ID for an uploaded document."""
    return f"doc_{uuid4().hex}"


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def chunk_text_with_metadata(
    text: str,
    source: str,
    document_id: str | None = None,
    file_type: str | None = None,
    page_number: int | None = None,
    chunk_size: int = 1000,
    overlap: int = 150,
    chunk_index_start: int = 0,
) -> list[dict]:
    """Split text into chunks with document and source metadata."""

    if not source.strip():
        raise ValueError("source cannot be empty")

    if chunk_index_start < 0:
        raise ValueError("chunk_index_start cannot be negative")

    if page_number is not None and page_number < 1:
        raise ValueError("page_number must be at least 1")

    document_id = document_id or generate_document_id()

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    results = []

    for index, chunk in enumerate(chunks, start=chunk_index_start):
        results.append({
            "chunk_id": f"{document_id}_chunk_{index:04d}",
            "text": chunk,
            "metadata": {
                "document_id": document_id,
                "filename": source,
                "source": source,
                "file_type": file_type,
                "chunk_index": index,
                "page_number": page_number,
            },
        })

    return results

