"""Liveness/readiness health check endpoint."""
from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", status_code=200, summary="Liveness probe")
def health_check() -> dict[str, str]:
    """Return service health information.

    No external dependencies are touched (no database, Supabase, or ML models),
    so this endpoint stays reliable while the rest of the platform is unbuilt.
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }
