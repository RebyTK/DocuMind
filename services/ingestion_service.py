"""Document-ingestion service facade.

The current, tested ingestion implementation lives in main.py. This facade
lets the API reuse it without duplicating extraction, chunking, embedding,
or PostgreSQL transaction logic. A later refactor can move that implementation
here without changing the API routes.
"""

from pathlib import Path


def ingest_file(file_path: Path) -> dict:
    """Ingest a PDF, DOCX, or TXT file into PostgreSQL."""
    # Import lazily to avoid importing CLI code until ingestion is requested.
    from main import ingest_document_to_postgres

    return ingest_document_to_postgres(file_path=file_path)
