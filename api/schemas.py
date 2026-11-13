"""Pydantic request and response schemas for the DocuMind API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    filename: str
    file_type: str
    file_hash: str
    total_chunks: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class IngestionResponse(BaseModel):
    inserted: bool
    message: str
    document_id: str
    filename: str
    file_type: str
    total_chunks: int
    stats: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class SearchHit(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float


class SearchResponse(BaseModel):
    question: str
    results: list[SearchHit]
    total_results: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    valid_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    retrieval_count: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str
