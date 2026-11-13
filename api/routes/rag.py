"""RAG question-answering endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import AskRequest, AskResponse
from services.rag_service import answer_question

router = APIRouter(prefix="/ask", tags=["rag"])


@router.post("", response_model=AskResponse)
def ask_documents(request: AskRequest) -> AskResponse:
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
        )
    except Exception as error:
        logging.exception("RAG answer generation failed")
        raise HTTPException(
            status_code=500,
            detail="Answer generation failed. Check the API logs.",
        ) from error

    return AskResponse(**result)
