"""Document upload and management endpoints."""

import logging
import tempfile
from pathlib import Path
from zipfile import BadZipFile

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pypdf.errors import PdfReadError
from starlette.concurrency import run_in_threadpool

from api.schemas import DeleteResponse, DocumentSummary, IngestionResponse
from services.document_service import (
    get_document_by_id,
    list_all_documents,
    remove_document,
)
from services.ingestion_service import ingest_file

router = APIRouter(prefix="/documents", tags=["documents"])
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("", response_model=IngestionResponse)
async def upload_document(file: UploadFile = File(...)) -> IngestionResponse:
    """Upload and ingest one PDF, DOCX, or TXT document."""
    original_name = (file.filename or "").replace("\\", "/")
    filename = Path(original_name).name
    suffix = Path(filename).suffix.lower()

    if not filename or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Supported file types are PDF, DOCX, and TXT.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 50 MB upload limit.",
        )
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        # Keep the temporary file alive until synchronous ingestion completes.
        with tempfile.TemporaryDirectory(prefix="documind_upload_") as temp_dir:
            temporary_path = Path(temp_dir) / filename
            temporary_path.write_bytes(content)
            result = await run_in_threadpool(ingest_file, temporary_path)
    except (ValueError, PdfReadError, BadZipFile, UnicodeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("API document ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check the API logs.",
        ) from error
    finally:
        await file.close()

    inserted = bool(result.get("inserted"))
    return IngestionResponse(
        inserted=inserted,
        message=(
            "Document ingested successfully."
            if inserted
            else "This document is already registered."
        ),
        document_id=result["document_id"],
        filename=result["filename"],
        file_type=suffix.lstrip("."),
        total_chunks=result["total_chunks"],
        stats=result.get("stats"),
    )


@router.get("", response_model=list[DocumentSummary])
def list_documents_endpoint() -> list[dict]:
    try:
        return list_all_documents()
    except Exception as error:
        logging.exception("Unable to list documents")
        raise HTTPException(
            status_code=500,
            detail="Unable to list documents.",
        ) from error


@router.get("/{document_id}", response_model=DocumentSummary)
def get_document_endpoint(document_id: str) -> dict:
    try:
        document = get_document_by_id(document_id)
    except Exception as error:
        logging.exception("Unable to retrieve document")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve document.",
        ) from error

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document_endpoint(document_id: str) -> DeleteResponse:
    try:
        deleted = remove_document(document_id)
    except Exception as error:
        logging.exception("Unable to delete document")
        raise HTTPException(
            status_code=500,
            detail="Unable to delete document.",
        ) from error

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DeleteResponse(
        document_id=document_id,
        deleted=True,
        message="Document and associated chunks deleted.",
    )
