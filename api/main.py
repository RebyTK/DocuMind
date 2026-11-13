"""FastAPI application entry point for DocuMind."""

from fastapi import FastAPI

from api.routes import documents, health, rag, search

app = FastAPI(
    title="DocuMind API",
    description=(
        "Document ingestion, PostgreSQL/pgvector semantic search, "
        "and local-LLM retrieval-augmented question answering."
    ),
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(documents.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")


@app.get("/")
def root() -> dict:
    return {
        "name": "DocuMind API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }
