import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

# Locate the project root independently of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load credentials from the project's .env file.
load_dotenv(PROJECT_ROOT / ".env")


def get_connection() -> psycopg.Connection:
    """
    Create and return a PostgreSQL database connection.

    Connection settings are loaded from environment variables.
    """

    required_variables = [
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing_variables:
        raise RuntimeError(
            "Missing database environment variables: "
            + ", ".join(missing_variables)
        )

    connection = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=5,
    )
    register_vector(connection)

    return connection


def check_connection() -> dict:
    """Check the database connection and return database information."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    current_database(),
                    current_user,
                    version()
                """
            )

            database, user, version = cursor.fetchone()

    return {
        "database": database,
        "user": user,
        "version": version,
    }
