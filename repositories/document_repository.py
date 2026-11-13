from psycopg.types.json import Jsonb

from database.connection import get_connection


def insert_document(document: dict) -> dict:
    """Insert a document into the PostgreSQL documents table."""

    required_fields = [
        "document_id",
        "filename",
        "file_type",
        "file_hash",
    ]

    for field in required_fields:
        if not document.get(field):
            raise ValueError(f"Missing required field: {field}")

    metadata = document.get("metadata", {})
    total_chunks = document.get("total_chunks", 0)

    if not isinstance(metadata, dict):
        raise ValueError("Document metadata must be a dictionary.")

    if not isinstance(total_chunks, int) or total_chunks < 0:
        raise ValueError("total_chunks must be a non-negative integer.")

    query = """
        INSERT INTO documents (
            document_id,
            filename,
            file_type,
            file_hash,
            total_chunks,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING
            document_id,
            filename,
            file_type,
            file_hash,
            total_chunks,
            metadata,
            created_at;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    document["document_id"],
                    document["filename"],
                    document["file_type"].lower(),
                    document["file_hash"],
                    total_chunks,
                    Jsonb(metadata),
                ),
            )

            columns = [column.name for column in cursor.description]
            result = cursor.fetchone()

            return dict(zip(columns, result))


def get_document(document_id: str) -> dict | None:
    """Retrieve a document using its document ID."""

    query = """
        SELECT
            document_id,
            filename,
            file_type,
            file_hash,
            total_chunks,
            metadata,
            created_at
        FROM documents
        WHERE document_id = %s;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (document_id,))

            result = cursor.fetchone()

            if result is None:
                return None

            columns = [column.name for column in cursor.description]

            return dict(zip(columns, result))


def list_documents() -> list[dict]:
    """Return all registered documents."""

    query = """
        SELECT
            document_id,
            filename,
            file_type,
            file_hash,
            total_chunks,
            metadata,
            created_at
        FROM documents
        ORDER BY created_at DESC;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)

            columns = [column.name for column in cursor.description]

            return [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]


def delete_document(document_id: str) -> bool:
    """Delete a document and its associated chunks."""

    query = """
        DELETE FROM documents
        WHERE document_id = %s
        RETURNING document_id;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (document_id,))

            return cursor.fetchone() is not None

def get_document_by_hash(file_hash: str) -> dict | None:
    """Find an existing document using its SHA-256 hash."""

    if not isinstance(file_hash, str) or not file_hash.strip():
        raise ValueError("File hash must be a non-empty string.")

    query = """
        SELECT
            document_id,
            filename,
            file_type,
            file_hash,
            total_chunks,
            metadata,
            created_at
        FROM documents
        WHERE file_hash = %s;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (file_hash,))
            result = cursor.fetchone()

            if result is None:
                return None

            columns = [
                column.name
                for column in cursor.description
            ]

            return dict(zip(columns, result))
