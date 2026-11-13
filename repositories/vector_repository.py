import math

from pgvector import Vector
from psycopg.types.json import Jsonb

from database.connection import get_connection


EMBEDDING_DIMENSIONS = 384


def validate_embedding(embedding: list) -> None:
    """Validate the dimensions and values of an embedding."""

    if not isinstance(embedding, (list, tuple)):
        raise ValueError("Embedding must be a list or tuple.")

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding must contain exactly "
            f"{EMBEDDING_DIMENSIONS} dimensions."
        )

    try:
        values = [float(value) for value in embedding]
    except (TypeError, ValueError) as error:
        raise ValueError("Embedding must contain numeric values.") from error

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding cannot contain NaN or infinite values.")


def insert_chunks(embedded_chunks: list[dict]) -> int:
    """Insert or update document chunks and their embeddings."""

    if not isinstance(embedded_chunks, list):
        raise ValueError("Embedded chunks must be a list.")

    if not embedded_chunks:
        return 0

    query = """
        INSERT INTO document_chunks (
            chunk_id,
            document_id,
            chunk_index,
            page_number,
            content,
            embedding,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)

        ON CONFLICT (chunk_id)
        DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata;
    """

    rows = []

    for chunk in embedded_chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Each embedded chunk must be a dictionary.")

        chunk_id = chunk.get("chunk_id")
        content = chunk.get("text")
        metadata = chunk.get("metadata")
        embedding = chunk.get("embedding")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Chunk ID must be a non-empty string.")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Chunk content cannot be empty.")

        if not isinstance(metadata, dict):
            raise ValueError("Chunk metadata must be a dictionary.")

        document_id = metadata.get("document_id")
        chunk_index = metadata.get("chunk_index")
        page_number = metadata.get("page_number")

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID is required in chunk metadata.")

        if type(chunk_index) is not int or chunk_index < 0:
            raise ValueError("Chunk index must be a non-negative integer.")

        if page_number is not None:
            if type(page_number) is not int or page_number < 1:
                raise ValueError("Page number must be a positive integer.")

        validate_embedding(embedding)

        rows.append(
            (
                chunk_id,
                document_id,
                chunk_index,
                page_number,
                content,
                Vector([float(value) for value in embedding]),
                Jsonb(metadata),
            )
        )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, rows)

    return len(rows)


def search_similar_chunks(
    query_embedding: list,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:
    """Retrieve chunks ordered by cosine similarity."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    validate_embedding(query_embedding)

    query_vector = Vector(
        [float(value) for value in query_embedding]
    )

    query = """
        SELECT
            chunk_id,
            document_id,
            chunk_index,
            page_number,
            content,
            metadata,
            embedding <=> %s AS distance

        FROM document_chunks
    """

    parameters = [query_vector]

    if document_id is not None:
        query += " WHERE document_id = %s"
        parameters.append(document_id)

    query += """
        ORDER BY embedding <=> %s
        LIMIT %s;
    """

    parameters.extend([query_vector, top_k])

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)

            columns = [
                column.name
                for column in cursor.description
            ]

            rows = [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]

    results = []

    for row in rows:
        metadata = dict(row["metadata"] or {})

        # Use the database columns as the authoritative metadata.
        metadata["document_id"] = row["document_id"]
        metadata["chunk_index"] = row["chunk_index"]
        metadata["page_number"] = row["page_number"]

        distance = float(row["distance"])

        results.append({
            "chunk_id": row["chunk_id"],
            "text": row["content"],
            "metadata": metadata,
            "score": 1.0 - distance,
        })

    return results


def count_chunks(document_id: str | None = None) -> int:
    """Return the number of stored chunks."""

    query = "SELECT COUNT(*) FROM document_chunks"
    parameters = []

    if document_id is not None:
        query += " WHERE document_id = %s"
        parameters.append(document_id)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            return cursor.fetchone()[0]