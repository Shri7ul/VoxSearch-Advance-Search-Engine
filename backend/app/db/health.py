"""Read-only database connectivity checks and the ``/health/db`` endpoint.

Goal 05 is connection-only: no tables, migrations, schemas, or application data
are created or modified. Both SQL statements below are strictly read-only. No
connection string, password, host, or stack trace is ever exposed.
"""
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db.session import get_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_READINESS_QUERY = "SELECT 1;"
_VECTOR_EXTENSION_QUERY = (
    "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
)

_HEALTHY: dict[str, str] = {
    "status": "healthy",
    "database": "connected",
    "vector_extension": "available",
}
_UNHEALTHY: dict[str, str] = {
    "status": "unhealthy",
    "database": "disconnected",
    "vector_extension": "unavailable",
}


class VectorExtensionMissingError(RuntimeError):
    """Raised when the required pgvector ('vector') extension is absent."""


def check_database() -> dict[str, str]:
    """Run the two read-only checks and return the healthy payload.

    Any failure raises so the endpoint can translate it into a safe, credential-
    free unhealthy response. The engine is obtained lazily; no details are logged.
    """
    engine = get_engine()
    with engine.connect() as connection:
        connection.execute(text(_READINESS_QUERY))
        version = connection.execute(text(_VECTOR_EXTENSION_QUERY)).scalar()

    if version is None:
        raise VectorExtensionMissingError(
            "The pgvector ('vector') extension is not installed."
        )
    return dict(_HEALTHY)


@router.get("/health/db", response_model=None, summary="Database readiness probe")
def db_health_check():
    """Report Supabase PostgreSQL connectivity and pgvector availability.

    Returns HTTP 200 with the healthy payload when both checks pass; otherwise a
    safe HTTP 503 response that never includes connection strings, credentials,
    host details, or stack traces.
    """
    try:
        return check_database()
    except Exception as exc:  # noqa: BLE001 - a health probe must never leak details
        # Log only the exception class name: str(exc) may embed the DSN/password.
        logger.warning("Database health check failed (%s)", type(exc).__name__)
        return JSONResponse(status_code=503, content=dict(_UNHEALTHY))
