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
            "VoxSearch backend API — Goal 02 foundation. Currently exposes only "
            "a health check; search, ingestion, and database features are planned "
            "for later goals."
        ),
        lifespan=lifespan,
    )

    # The liveness probe is mounted at the root so GET /health works without a
    # prefix. API_PREFIX is reserved for future versioned feature routers.
    application.include_router(health.router)
    application.state.api_prefix = settings.api_prefix

    return application


app = create_app()
