import argparse
import json
import math
from pathlib import Path

from pgvector import Vector
from psycopg.types.json import Jsonb

from database.connection import get_connection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
EXPECTED_DIMENSIONS = 384


def read_json(path: Path):
    """Read a JSON file."""

    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_output_path(
    output_dir: Path,
    relative_path: str,
) -> Path:
    """Resolve a stored path and prevent escaping the output directory."""

    base = output_dir.resolve()
    resolved = (base / relative_path).resolve()

    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(
            f"Invalid path outside output directory: {relative_path}"
        ) from error

    return resolved


def validate_embedding(embedding, chunk_id: str) -> None:
    """Validate a stored embedding before migration."""

    if not isinstance(embedding, list):
        raise ValueError(
            f"Embedding for {chunk_id} must be a list."
        )

    if len(embedding) != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Embedding for {chunk_id} has "
            f"{len(embedding)} dimensions; expected "
            f"{EXPECTED_DIMENSIONS}."
        )

    for value in embedding:
        if type(value) not in (int, float):
            raise ValueError(
                f"Non-numeric embedding value in {chunk_id}."
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite embedding value in {chunk_id}."
            )


def load_migration_data(output_dir: Path) -> list[dict]:
    """
    Load and validate all documents, chunks and embeddings.

    This function only reads local JSON files.
    It does not modify the database.
    """

    registry_path = output_dir / "document_registry.json"
    registry = read_json(registry_path)

    documents = registry.get("documents")

    if not isinstance(documents, dict) or not documents:
        raise ValueError("Document registry is empty or invalid.")

    migration_data = []
    seen_chunk_ids = set()
    total_chunks = 0

    for document_id, record in documents.items():

        print(f"\nValidating: {record.get('filename')}")

        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid registry record: {document_id}"
            )

        if record.get("document_id") != document_id:
            raise ValueError(
                f"Document ID mismatch for {document_id}."
            )

        filename = record.get("filename")
        file_type = record.get("file_type")
        file_hash = record.get("file_hash")

        if not all([filename, file_type, file_hash]):
            raise ValueError(
                f"Missing required document fields: {document_id}"
            )

        if file_type not in {"pdf", "docx", "txt"}:
            raise ValueError(
                f"Unsupported file type for {document_id}: {file_type}"
            )

        chunks_path = resolve_output_path(
            output_dir,
            record["chunks_file"],
        )

        embeddings_path = resolve_output_path(
            output_dir,
            record["embeddings_file"],
        )

        chunks = read_json(chunks_path)
        embedded_chunks = read_json(embeddings_path)

        if not isinstance(chunks, list):
            raise ValueError(
                f"Invalid chunks data for {document_id}."
            )

        if not isinstance(embedded_chunks, list):
            raise ValueError(
                f"Invalid embeddings data for {document_id}."
            )

        if len(chunks) != len(embedded_chunks):
            raise ValueError(
                f"Chunk and embedding count mismatch: {document_id}"
            )

        if len(chunks) != record.get("chunk_count"):
            raise ValueError(
                f"Registered chunk count mismatch: {document_id}"
            )

        rows = []
        document_chunk_indexes = set()

        for chunk, embedded in zip(chunks, embedded_chunks):

            chunk_id = chunk.get("chunk_id")
            embedded_id = embedded.get("chunk_id")

            if not chunk_id or chunk_id != embedded_id:
                raise ValueError(
                    f"Chunk ID mismatch in {document_id}."
                )

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID found: {chunk_id}"
                )

            seen_chunk_ids.add(chunk_id)

            if chunk.get("text") != embedded.get("text"):
                raise ValueError(
                    f"Chunk text mismatch: {chunk_id}"
                )

            metadata = embedded.get("metadata")

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Invalid metadata: {chunk_id}"
                )

            if metadata.get("document_id") != document_id:
                raise ValueError(
                    f"Document ID mismatch in metadata: {chunk_id}"
                )

            chunk_index = metadata.get("chunk_index")
            page_number = metadata.get("page_number")

            if type(chunk_index) is not int or chunk_index < 0:
                raise ValueError(
                    f"Invalid chunk index: {chunk_id}"
                )

            if chunk_index in document_chunk_indexes:
                raise ValueError(
                    f"Duplicate chunk index: {document_id}, {chunk_index}"
                )

            document_chunk_indexes.add(chunk_index)

            if page_number is not None:
                if type(page_number) is not int or page_number < 1:
                    raise ValueError(
                        f"Invalid page number: {chunk_id}"
                    )

            content = embedded.get("text")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Empty chunk content: {chunk_id}"
                )

            embedding = embedded.get("embedding")

            validate_embedding(embedding, chunk_id)

            rows.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "page_number": page_number,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
            })

        if len(rows) != len(chunks):
            raise ValueError(
                f"Not all chunks validated: {document_id}"
            )

        metadata_path = None

        if record.get("metadata_file"):
            metadata_path = resolve_output_path(
                output_dir,
                record["metadata_file"],
            )

        document_metadata = {}

        if metadata_path is not None and metadata_path.is_file():
            document_metadata = read_json(metadata_path)

            if not isinstance(document_metadata, dict):
                raise ValueError(
                    f"Invalid document metadata: {document_id}"
                )

        # Avoid duplicating full extracted text and table contents.
        document_metadata.pop("elements", None)
        document_metadata.pop("tables", None)

        migration_data.append({
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "file_hash": file_hash,
            "total_chunks": len(rows),
            "metadata": document_metadata,
            "chunks": rows,
        })

        total_chunks += len(rows)

        print(f"  Chunks validated: {len(rows)}")
        print("  Embeddings validated: 384 dimensions")

    print("\n" + "=" * 55)
    print("MIGRATION VALIDATION SUMMARY")
    print("=" * 55)
    print("Documents validated:", len(migration_data))
    print("Chunks validated:", total_chunks)
    print("Unique chunk IDs:", len(seen_chunk_ids))

    return migration_data


def migrate_to_postgres(migration_data: list[dict]) -> None:
    """Migrate all documents and chunks in one transaction."""

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

        ON CONFLICT (document_id)
        DO UPDATE SET
            filename = EXCLUDED.filename,
            file_type = EXCLUDED.file_type,
            total_chunks = EXCLUDED.total_chunks,
            metadata = EXCLUDED.metadata

        WHERE documents.file_hash = EXCLUDED.file_hash

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
        VALUES (%s, %s, %s, %s, %s, %s, %s)

        ON CONFLICT (chunk_id)
        DO UPDATE SET
            document_id = EXCLUDED.document_id,
            chunk_index = EXCLUDED.chunk_index,
            page_number = EXCLUDED.page_number,
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata;
    """

    document_ids = [
        document["document_id"]
        for document in migration_data
    ]

    total_migrated_chunks = 0

    # A single connection transaction covers the entire migration.
    # Any uncaught error causes the transaction to roll back.
    with get_connection() as connection:
        with connection.cursor() as cursor:

            for document in migration_data:

                cursor.execute(
                    document_query,
                    (
                        document["document_id"],
                        document["filename"],
                        document["file_type"],
                        document["file_hash"],
                        document["total_chunks"],
                        Jsonb(document["metadata"]),
                    ),
                )

                inserted_document = cursor.fetchone()

                if inserted_document is None:
                    raise ValueError(
                        "Existing document ID has a different "
                        f"file hash: {document['document_id']}"
                    )

                chunk_rows = []

                for chunk in document["chunks"]:
                    chunk_rows.append((
                        chunk["chunk_id"],
                        chunk["document_id"],
                        chunk["chunk_index"],
                        chunk["page_number"],
                        chunk["content"],
                        Vector(chunk["embedding"]),
                        Jsonb(chunk["metadata"]),
                    ))

                if chunk_rows:
                    cursor.executemany(
                        chunk_query,
                        chunk_rows,
                    )

                total_migrated_chunks += len(chunk_rows)

            # Verify persisted counts before committing.
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM documents
                WHERE document_id = ANY(%s);
                """,
                (document_ids,),
            )

            persisted_documents = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM document_chunks
                WHERE document_id = ANY(%s);
                """,
                (document_ids,),
            )

            persisted_chunks = cursor.fetchone()[0]

            expected_chunks = sum(
                document["total_chunks"]
                for document in migration_data
            )

            if persisted_documents != len(migration_data):
                raise RuntimeError(
                    "Document count verification failed."
                )

            if persisted_chunks != expected_chunks:
                raise RuntimeError(
                    "Chunk count verification failed."
                )

    # Reaching this point means the transaction committed successfully.
    print("\n" + "=" * 55)
    print("MIGRATION COMPLETED SUCCESSFULLY")
    print("=" * 55)
    print("Documents migrated:", len(migration_data))
    print("Chunks migrated:", total_migrated_chunks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate DocuMind JSON data into PostgreSQL."
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag, only validate.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to the existing DocuMind output directory.",
    )

    args = parser.parse_args()

    try:
        migration_data = load_migration_data(args.output_dir)

        if not args.apply:
            print("\nDRY RUN: No database changes were made.")
            print(
                "Run again with --apply to migrate the validated data."
            )
            return 0

        migrate_to_postgres(migration_data)
        return 0

    except Exception as error:
        print(f"\nMigration aborted: {error}")
        print("No partial migration was committed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
