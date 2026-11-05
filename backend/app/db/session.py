"""Synchronous SQLAlchemy engine + session dependency for Supabase PostgreSQL.

Goal 05 is a *connection-only* foundation: no ORM models, tables, schemas, or
migrations are defined here. The engine is created lazily (and cached) so that
importing this module — and importing the FastAPI app — never opens a database
connection. Basic liveness (``/health``) never needs the database.
"""
from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when ``DATABASE_URL`` is missing or empty.

    The message is deliberately generic and never contains the connection
    string, password, host, or any other credential.
    """


def database_url(settings: Settings) -> str:
    """Return the configured connection string or raise a clear, safe error.

    A small pure function (no engine, no network) so configuration handling is
    easy to unit-test in isolation.
    """
    secret = settings.database_url
    if secret is None:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL is not configured; database access is unavailable."
        )
    url = secret.get_secret_value().strip()
    if not url:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL is empty; database access is unavailable."
        )
    return url


def _psycopg_url(url: str) -> str:
    """Force the psycopg (v3) SQLAlchemy dialect.

    A plain ``postgresql://`` URL defaults to the psycopg2 driver, which is
    intentionally NOT installed. Rewriting these common schemes guarantees
    Psycopg 3 is used without requiring users to alter their Supabase string.
    """
    if url.startswith("postgresql+psycopg://"):
        return url
    for scheme in ("postgresql+psycopg2://", "postgresql://", "postgres://"):
        if url.startswith(scheme):
            return "postgresql+psycopg://" + url[len(scheme):]
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (once) a synchronous SQLAlchemy engine backed by Psycopg 3.

    ``pool_pre_ping=True`` validates pooled connections before use, important
    for Supabase's network path and idle-connection recycling. ``create_engine``
    does NOT open a connection — that happens on first use. The URL/credentials
    are never logged or echoed (``echo`` stays off).
    """
    url = _psycopg_url(database_url(get_settings()))
    return create_engine(url, pool_pre_ping=True)


# A single reusable, unbound session factory. The concrete engine is bound per
# call in get_db() so importing this module requires no engine and no DATABASE_URL.
SessionFactory = sessionmaker(
    autocommit=False, autoflush=False, expire_on_commit=False
)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yield a session and always close it.

    Intended for future data endpoints (no route uses it in Goal 05 — there is
    no application data). The ``finally`` block guarantees cleanup even if the
    consumer raises, returning the connection to the pool.
    """
    session: Session = SessionFactory(bind=get_engine())
    try:
        yield session
    finally:
        session.close()
