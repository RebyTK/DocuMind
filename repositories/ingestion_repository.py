from pgvector import Vector
from psycopg.types.json import Jsonb

from database.connection import get_connection
from repositories.vector_repository import validate_embedding


def persist_document_with_chunks(
    document: dict,
    embedded_chunks: list[dict],
) -> dict:
    """
    Persist a document and all its embedded chunks atomically.

    Returns whether a new document was inserted or an existing
    document with the same SHA-256 hash was found.
    """

    required_fields = [
        "document_id",
        "filename",
        "file_type",
        "file_hash",
    ]

    for field in required_fields:
        value = document.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid document field: {field}")

    file_type = document["file_type"].lower()

    if file_type not in {"pdf", "docx", "txt"}:
        raise ValueError("Unsupported document file type.")

    if not isinstance(embedded_chunks, list):
        raise ValueError("Embedded chunks must be a list.")

    metadata = document.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError("Document metadata must be a dictionary.")

    total_chunks = document.get(
        "total_chunks",
        len(embedded_chunks),
    )

    if type(total_chunks) is not int or total_chunks < 0:
        raise ValueError("Invalid total chunk count.")

    if total_chunks != len(embedded_chunks):
        raise ValueError(
            "Document chunk count does not match embedded chunks."
        )

    document_id = document["document_id"]

    document_query = """
        INSERT INTO documents (
            document_id,
            filename,
            file_type,
            file_hash,
            total_chunks,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s)

        ON CONFLICT (file_hash) DO NOTHING

        RETURNING document_id;
    """

    chunk_query = """
        INSERT INTO document_chunks (
            chunk_id,
            document_id,
            chunk_index,
            page_number,
            content,
            embedding,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """

    rows = []
    seen_chunk_ids = set()
    seen_chunk_indexes = set()

    for chunk in embedded_chunks:

        if not isinstance(chunk, dict):
            raise ValueError("Each chunk must be a dictionary.")

        chunk_id = chunk.get("chunk_id")
        content = chunk.get("text")
        embedding = chunk.get("embedding")
        chunk_metadata = chunk.get("metadata")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Invalid chunk ID.")

        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")

        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty chunk content: {chunk_id}")

        if not isinstance(chunk_metadata, dict):
            raise ValueError(f"Invalid metadata: {chunk_id}")

        if chunk_metadata.get("document_id") != document_id:
            raise ValueError(
                f"Document ID mismatch: {chunk_id}"
            )

        chunk_index = chunk_metadata.get("chunk_index")
        page_number = chunk_metadata.get("page_number")

        if type(chunk_index) is not int or chunk_index < 0:
            raise ValueError(f"Invalid chunk index: {chunk_id}")

        if chunk_index in seen_chunk_indexes:
            raise ValueError(
                f"Duplicate chunk index: {chunk_index}"
            )

        if page_number is not None:
            if type(page_number) is not int or page_number < 1:
                raise ValueError(f"Invalid page number: {chunk_id}")

        validate_embedding(embedding)

        seen_chunk_ids.add(chunk_id)
        seen_chunk_indexes.add(chunk_index)

        rows.append((
            chunk_id,
            document_id,
            chunk_index,
            page_number,
            content,
            Vector([float(value) for value in embedding]),
            Jsonb(chunk_metadata),
        ))

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                document_query,
                (
                    document_id,
                    document["filename"],
                    file_type,
                    document["file_hash"],
                    total_chunks,
                    Jsonb(metadata),
                ),
            )

            inserted = cursor.fetchone()

            if inserted is None:
                cursor.execute(
                    """
                    SELECT document_id, filename, total_chunks
                    FROM documents
                    WHERE file_hash = %s;
                    """,
                    (document["file_hash"],),
                )

                existing = cursor.fetchone()

                if existing is None:
                    raise RuntimeError(
                        "Unable to identify existing document."
                    )

                return {
                    "inserted": False,
                    "document_id": existing[0],
                    "filename": existing[1],
                    "total_chunks": existing[2],
                }

            if rows:
                cursor.executemany(chunk_query, rows)

    return {
        "inserted": True,
        "document_id": document_id,
        "filename": document["filename"],
        "total_chunks": total_chunks,
    }
