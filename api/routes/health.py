"""Health-check endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from database.connection import check_connection

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "documind-api"}


@router.get("/health/database")
def database_health_check() -> dict:
    try:
        database_info = check_connection()
    except Exception as error:
        logging.exception("Database health check failed")
        raise HTTPException(
            status_code=503,
            detail="Database is unavailable.",
        ) from error

    return {
        "status": "ok",
        "database": database_info["database"],
        "database_version": database_info["version"],
    }
