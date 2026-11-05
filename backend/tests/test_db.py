"""Mocked unit tests for the Goal 05 database connection layer.

These tests NEVER open a real network/Supabase connection. Live Supabase
connectivity is verified separately (see the Goal 05 validation report).
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings
from app.db.health import VectorExtensionMissingError, check_database
from app.db.session import DatabaseNotConfiguredError, database_url, get_db
from app.main import app

HEALTHY = {
    "status": "healthy",
    "database": "connected",
    "vector_extension": "available",
}
UNHEALTHY = {
    "status": "unhealthy",
    "database": "disconnected",
    "vector_extension": "unavailable",
}


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


# --- 1. Database configuration handling -------------------------------------

def test_database_url_returns_configured_value() -> None:
    settings = Settings(
        _env_file=None,
        database_url=SecretStr("postgresql://placeholder-host/placeholder-db"),
    )
    assert database_url(settings) == "postgresql://placeholder-host/placeholder-db"


def test_database_url_missing_raises_clear_safe_error() -> None:
    settings = Settings(_env_file=None, database_url=None)
    with pytest.raises(DatabaseNotConfiguredError) as exc:
        database_url(settings)
    message = str(exc.value)
    assert "DATABASE_URL" in message
    # A clear, safe error must not echo any DSN / credential material.
    assert "postgresql" not in message
    assert "password" not in message.lower()


def test_database_url_empty_raises_clear_safe_error() -> None:
    settings = Settings(_env_file=None, database_url=SecretStr("   "))
    with pytest.raises(DatabaseNotConfiguredError):
        database_url(settings)


# --- 2. Database dependency / session behavior -------------------------------

def test_get_db_yields_and_closes_session() -> None:
    mock_session = MagicMock()
    mock_engine = MagicMock()
    with (
        patch("app.db.session.get_engine", return_value=mock_engine),
        patch("app.db.session.SessionFactory", return_value=mock_session) as factory,
    ):
        gen = get_db()
        yielded = next(gen)
        assert yielded is mock_session
        factory.assert_called_once_with(bind=mock_engine)
        mock_session.close.assert_not_called()
        with pytest.raises(StopIteration):
            next(gen)
    mock_session.close.assert_called_once()


def test_get_db_closes_session_on_consumer_error() -> None:
    mock_session = MagicMock()
    with (
        patch("app.db.session.get_engine", return_value=MagicMock()),
        patch("app.db.session.SessionFactory", return_value=mock_session),
    ):
        gen = get_db()
        next(gen)
        with pytest.raises(RuntimeError):
            gen.throw(RuntimeError("consumer failed"))
    mock_session.close.assert_called_once()


# --- 3. Database health check logic + endpoint -------------------------------

def test_check_database_runs_readonly_queries() -> None:
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.scalar.return_value = "0.8.0"
    with patch("app.db.health.get_engine", return_value=mock_engine):
        result = check_database()
    assert result == HEALTHY
    # Two read-only statements were executed, and the extension version was read.
    assert mock_conn.execute.call_count == 2
    mock_conn.execute.return_value.scalar.assert_called_once_with()


def test_check_database_raises_when_vector_extension_missing() -> None:
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.scalar.return_value = None
    with (
        patch("app.db.health.get_engine", return_value=mock_engine),
        pytest.raises(VectorExtensionMissingError),
    ):
        check_database()


def test_db_health_endpoint_healthy(client: TestClient) -> None:
    with patch("app.db.health.check_database", return_value=dict(HEALTHY)):
        response = client.get("/health/db")
    assert response.status_code == 200
    assert response.json() == HEALTHY


def test_db_health_endpoint_safe_failure(client: TestClient) -> None:
    # The exception message deliberately contains credential-like tokens; the
    # response must never leak them, a DSN, host, or a stack trace.
    failure = SQLAlchemyError(
        "connection failed: password=hunter2 host=secret.supabase.co dsn=postgresql://x"
    )
    with patch("app.db.health.check_database", side_effect=failure):
        response = client.get("/health/db")
    assert response.status_code == 503
    assert response.json() == UNHEALTHY
    leaked = response.text.lower()
    for forbidden in ("hunter2", "password", "supabase.co", "postgresql", "traceback"):
        assert forbidden not in leaked


# --- 5. Existing application health endpoint regression ----------------------

def test_basic_health_regression(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "voxsearch-backend",
        "version": "0.1.0",
    }
