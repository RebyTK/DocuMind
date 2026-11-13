"""Document-management application services."""

from repositories.document_repository import (
    delete_document,
    get_document,
    list_documents,
)


def list_all_documents() -> list[dict]:
    return list_documents()


def get_document_by_id(document_id: str) -> dict | None:
    return get_document(document_id)


def remove_document(document_id: str) -> bool:
    return delete_document(document_id)
