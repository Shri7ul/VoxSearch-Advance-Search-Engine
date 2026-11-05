"""Tests for the health endpoint and application metadata."""
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_http_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_fields(client: TestClient) -> None:
    payload = client.get("/health").json()
    settings = get_settings()
    assert set(payload) == {"status", "service", "version"}
    assert payload["status"] == "healthy"
    assert payload["service"] == settings.app_name
    assert payload["version"] == settings.app_version


def test_health_expected_payload(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "healthy",
        "service": "voxsearch-backend",
        "version": "0.1.0",
    }


def test_application_metadata() -> None:
    settings = get_settings()
    assert app.title == settings.app_name
    assert app.version == settings.app_version
    assert app.description
    assert app.state.api_prefix == settings.api_prefix
