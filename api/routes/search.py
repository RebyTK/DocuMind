"""Semantic-search endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import SearchRequest, SearchResponse
from services.retrieval_service import search_documents

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def semantic_search(request: SearchRequest) -> SearchResponse:
    try:
        results = search_documents(
            question=request.question,
            top_k=request.top_k,
        )
    except Exception as error:
        logging.exception("Semantic search failed")
        raise HTTPException(
            status_code=500,
            detail="Semantic search failed. Check the API logs.",
        ) from error

    return SearchResponse(
        question=request.question,
        results=results,
        total_results=len(results),
    )
