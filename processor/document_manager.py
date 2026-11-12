import json
from pathlib import Path
import tempfile
import hashlib


def load_registry(registry_path: Path) -> dict:
    """Load the document registry or create an empty registry."""

    if not registry_path.exists():
        return {"documents": {}}

    with registry_path.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    if "documents" not in registry:
        raise ValueError("Invalid document registry format.")

    return registry


def save_registry(registry: dict, registry_path: Path) -> None:
    """Save the document registry to JSON."""

    registry_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = registry_path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            registry,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temp_path.replace(registry_path)


def register_document(
    registry: dict,
    document: dict,
) -> None:
    """Add a document to the registry."""

    document_id = document.get("document_id")

    if not document_id:
        raise ValueError("Document ID is required.")

    if document_id in registry["documents"]:
        raise ValueError(
            f"Document already registered: {document_id}"
        )

    required_fields = [
        "filename",
        "file_type",
        "chunks_file",
        "embeddings_file",
        "chunk_count",
    ]

    for field in required_fields:
        if field not in document:
            raise ValueError(f"Missing document field: {field}")

    registry["documents"][document_id] = document


def remove_document(registry: dict, document_id: str):
    """Remove a document from the registry."""

    return registry["documents"].pop(document_id, None)


def get_document(registry: dict, document_id: str):
    """Retrieve a document by its ID."""

    return registry["documents"].get(document_id)


def list_documents(registry: dict) -> list[dict]:
    """Return all registered documents."""

    return list(registry["documents"].values())


def save_document_data(
    document_id: str,
    chunks: list[dict],
    embedded_chunks: list[dict],
    output_dir: Path,
) -> dict:
    """Persist a document's chunks and embeddings."""

    if not document_id or Path(document_id).name != document_id:
        raise ValueError("Invalid document ID.")

    documents_dir = output_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    chunks_filename = f"{document_id}_chunks.json"
    embeddings_filename = f"{document_id}_embeddings.json"

    chunks_path = documents_dir / chunks_filename
    embeddings_path = documents_dir / embeddings_filename

    def write_json(data: list[dict], path: Path) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                data,
                temp_file,
                indent=4,
                ensure_ascii=False,
            )

        temp_path.replace(path)

    write_json(chunks, chunks_path)

    try:
        write_json(embedded_chunks, embeddings_path)
    except Exception:
        chunks_path.unlink(missing_ok=True)
        raise

    return {
        "chunks_file": f"documents/{chunks_filename}",
        "embeddings_file": f"documents/{embeddings_filename}",
    }


def load_document_data(
    document: dict,
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Load persisted chunks and embeddings for a document."""

    chunks_path = output_dir / document["chunks_file"]
    embeddings_path = output_dir / document["embeddings_file"]

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    with embeddings_path.open("r", encoding="utf-8") as file:
        embedded_chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("Invalid chunks file format.")

    if not isinstance(embedded_chunks, list):
        raise ValueError("Invalid embeddings file format.")

    if len(chunks) != len(embedded_chunks):
        raise ValueError("Chunks and embeddings count do not match.")

    return chunks, embedded_chunks


def load_all_embeddings(
    registry: dict,
    output_dir: Path,
) -> list[dict]:
    """Load and combine embeddings from all registered documents."""

    all_embedded_chunks = []

    for document in list_documents(registry):
        _, embedded_chunks = load_document_data(
            document,
            output_dir,
        )

        all_embedded_chunks.extend(embedded_chunks)

    return all_embedded_chunks


def rebuild_shared_index(
    registry: dict,
    output_dir: Path,
):
    """Rebuild one FAISS index from all persisted document embeddings."""

    from processor.vector_store import create_index

    embedded_chunks = load_all_embeddings(
        registry,
        output_dir,
    )

    index = create_index(embedded_chunks)

    return index, embedded_chunks

def calculate_file_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


def find_document_by_hash(
    registry: dict,
    file_hash: str,
) -> dict | None:
    """Find an already registered document by its file hash."""

    for document in list_documents(registry):
        if document.get("file_hash") == file_hash:
            return document

    return None
