"""VoxSearch FastAPI application entry point.

Run from the ``backend/`` directory::

    uvicorn app.main:app --reload
"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import health, search
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db import health as db_health
from app.db.session import DatabaseNotConfiguredError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log startup and shutdown; a hook for future resource setup/teardown."""
    settings = get_settings()
    logger.info(
        "Starting %s v%s (env=%s, api_prefix=%s)",
        settings.app_name,
        settings.app_version,
        settings.app_env,
        settings.api_prefix,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    setup_logging(settings)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "VoxSearch backend API. Exposes a liveness probe (GET /health), a "
            "read-only database readiness probe (GET /health/db), and the Goal 18 "
            "transcript search endpoint (POST /api/v1/search, aliased as "
            "POST /search). Query-text embedding (Goal 19) is not implemented yet."
        ),
        lifespan=lifespan,
    )

    # Both probes are mounted at the root so GET /health and GET /health/db work
    # without a prefix. The versioned search router is mounted under API_PREFIX;
    # an unversioned alias is kept for the tracker's literal "POST /search".
    application.include_router(health.router)
    application.include_router(db_health.router)
    application.include_router(search.router, prefix=settings.api_prefix)
    application.include_router(search.router, include_in_schema=False)
    application.state.api_prefix = settings.api_prefix

    @application.exception_handler(RequestValidationError)
    async def _request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return 422 details without echoing request data back to the client.

        Pydantic v2 includes the offending ``input`` (for ``/search`` that is the
        entire 1024-float query vector) and raw ``ctx`` values in each error.
        Those fields are not needed to explain a validation failure, inflate the
        response, and can expose derived query data, so they are stripped here.
        """
        errors = [
            {key: value for key, value in error.items() if key not in {"input", "ctx"}}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": errors},
        )

    @application.exception_handler(DatabaseNotConfiguredError)
    async def _database_not_configured(
        request: Request, exc: DatabaseNotConfiguredError
    ) -> JSONResponse:
        # The message is deliberately generic: never echo a connection string.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    return application


app = create_app()
