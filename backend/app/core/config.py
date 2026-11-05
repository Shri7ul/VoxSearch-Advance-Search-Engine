"""Centralized application settings.

Values are read from environment variables (case-insensitive) and, when present,
from a local `.env` file. Safe defaults are provided so the service runs locally
with no configuration at all. ``DATABASE_URL`` is optional and is only required
when a database connection is actually requested (see :mod:`app.db.session`).
"""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the VoxSearch backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "voxsearch-backend"
    app_env: str = "development"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # Supabase PostgreSQL connection string (Goal 05, connection-only).
    # Optional so the app and GET /health work with no database configured; a
    # clear, credential-free error is raised only when a DB connection is
    # actually requested. SecretStr masks the value in repr/str.
    database_url: SecretStr | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()
