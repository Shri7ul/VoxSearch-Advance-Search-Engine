"""VoxSearch FastAPI application entry point.

Run from the ``backend/`` directory::

    uvicorn app.main:app --reload
"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db import health as db_health

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
            "VoxSearch backend API. Exposes a basic liveness probe (GET /health) "
            "and a read-only database readiness probe (GET /health/db). Search, "
            "ingestion, embeddings, and application data are planned for later "
            "goals."
        ),
        lifespan=lifespan,
    )

    # Both probes are mounted at the root so GET /health and GET /health/db work
    # without a prefix. API_PREFIX is reserved for future versioned routers. The
    # basic liveness probe never touches the database.
    application.include_router(health.router)
    application.include_router(db_health.router)
    application.state.api_prefix = settings.api_prefix

    return application


app = create_app()
