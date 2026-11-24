"""Tests for Goal 34 — YouTube API authentication setup (no live network)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.youtube import (
    DEFAULT_YOUTUBE_API_BASE_URL,
    YouTubeAccessRestrictedError,
    YouTubeApiClient,
    YouTubeAuthError,
    YouTubeAuthFailedError,
    YouTubeAuthStatus,
    YouTubeCredentials,
    YouTubeNotConfiguredError,
    YouTubeQuotaError,
    YouTubeRequestError,
)

KEY = "AIzaSyDUMMYKEY-1234567890-abcdefghijklmno"  # test-only placeholder


def _client(
    status: int = 200,
    *,
    body: bytes = b"",
    raise_error: Exception | None = None,
):
    captured: dict[str, object] = {}

    def transport(url: str, timeout: float):
        captured["url"] = url
        captured["timeout"] = timeout
        if raise_error is not None:
            raise raise_error
        return SimpleNamespace(status=status, body=body)

    credentials = YouTubeCredentials(api_key=SecretStr(KEY))
    return YouTubeApiClient(credentials, transport=transport), captured


# --- Credentials and configuration -----------------------------------------

def test_api_key_is_masked_in_representations() -> None:
    credentials = YouTubeCredentials(api_key=SecretStr(KEY))
    assert KEY not in repr(credentials)
    assert KEY not in str(credentials)


def test_empty_key_and_non_allowlisted_base_are_rejected() -> None:
    with pytest.raises(YouTubeNotConfiguredError):
        YouTubeCredentials(api_key=SecretStr("   "))
    with pytest.raises(YouTubeAuthError):
        YouTubeCredentials(
            api_key=SecretStr(KEY), base_url="https://evil.example/youtube/v3"
        )


def test_from_settings_requires_key() -> None:
    with pytest.raises(YouTubeNotConfiguredError):
        YouTubeApiClient.from_settings(Settings(youtube_api_key=None))


def test_from_settings_uses_configured_key() -> None:
    client = YouTubeApiClient.from_settings(
        Settings(
            youtube_api_key=SecretStr(KEY),
            youtube_api_base_url=DEFAULT_YOUTUBE_API_BASE_URL,
        ),
        transport=lambda url, timeout: SimpleNamespace(status=200),
    )
    assert client.is_configured() is True
    assert client.verify_authentication() == YouTubeAuthStatus()


def test_unconfigured_client_cannot_build_or_probe() -> None:
    client = YouTubeApiClient(transport=lambda url, timeout: None)
    assert client.is_configured() is False
    with pytest.raises(YouTubeNotConfiguredError):
        client.authenticated_url("channels", {"part": "id"})
    with pytest.raises(YouTubeNotConfiguredError):
        client.verify_authentication()


@pytest.mark.parametrize("bad", [0, -1, 61, True, "10"])
def test_timeout_validation(bad) -> None:
    with pytest.raises(YouTubeAuthError):
        YouTubeApiClient(
            YouTubeCredentials(api_key=SecretStr(KEY)), timeout_seconds=bad
        )


# --- Authenticated URL construction ----------------------------------------

def test_authenticated_url_is_https_encoded_and_keyed() -> None:
    client, _ = _client()
    url = client.authenticated_url("i18nLanguages", {"part": "snippet"})
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "www.googleapis.com"
    assert parsed.path == "/youtube/v3/i18nLanguages"
    query = parse_qs(parsed.query)
    assert query["key"] == [KEY]
    assert query["part"] == ["snippet"]


def test_url_builder_rejects_non_allowlisted_resources_and_bad_params() -> None:
    client, _ = _client()
    with pytest.raises(YouTubeAuthError):
        client.authenticated_url("playlists", {"part": "id"})
    with pytest.raises(YouTubeAuthError):
        client.authenticated_url("channels", {"part": "id"})
    with pytest.raises(YouTubeAuthError):
        client.authenticated_url("i18nLanguages", {"part": 123})


def test_injected_malicious_value_is_urlencoded_not_concatenated() -> None:
    client, _ = _client()
    url = client.authenticated_url("i18nLanguages", {"part": "snippet&key=evil"})
    assert parse_qs(urlparse(url).query)["part"] == ["snippet&key=evil"]


# --- Authentication probe ---------------------------------------------------

def test_successful_probe_consumes_no_video_data() -> None:
    client, captured = _client(200)
    assert client.verify_authentication() == YouTubeAuthStatus(
        authenticated=True, resource="i18nLanguages"
    )
    assert captured["timeout"] == 10.0
    assert "part=snippet" in str(captured["url"])


@pytest.mark.parametrize("status", [400, 401, 403])
def test_rejected_request_maps_to_closed_error(status: int) -> None:
    client, _ = _client(status)
    with pytest.raises(YouTubeAuthFailedError) as exc:
        client.verify_authentication()
    assert KEY not in str(exc.value)
    assert "HTTP" in str(exc.value)
    assert "credentials" not in str(exc.value)


@pytest.mark.parametrize(
    "reason",
    ["quotaExceeded", "dailyLimitExceeded", "userRateLimitExceeded", "rateLimitExceeded"],
)
def test_quota_403_is_not_misclassified_as_bad_credentials(reason: str) -> None:
    body = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()
    client, _ = _client(403, body=body)
    with pytest.raises(YouTubeQuotaError) as exc:
        client.verify_authentication()
    assert "quota" in str(exc.value).lower()
    assert KEY not in str(exc.value)


@pytest.mark.parametrize(
    "reason", ["apiNotEnabled", "ipRefererBlocked", "accessNotConfigured"]
)
def test_restriction_403_is_not_misclassified_as_bad_credentials(reason: str) -> None:
    body = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()
    client, _ = _client(403, body=body)
    with pytest.raises(YouTubeAccessRestrictedError) as exc:
        client.verify_authentication()
    assert "restricted" in str(exc.value)
    assert KEY not in str(exc.value)


def test_bad_credentials_reason_stays_auth_failure() -> None:
    body = json.dumps({"error": {"errors": [{"reason": "badCredentials"}]}}).encode()
    client, _ = _client(401, body=body)
    with pytest.raises(YouTubeAuthFailedError):
        client.verify_authentication()


def test_malformed_error_body_falls_back_to_neutral_auth_failure() -> None:
    client, _ = _client(403, body=b"not-json")
    with pytest.raises(YouTubeAuthFailedError) as exc:
        client.verify_authentication()
    assert "credentials" not in str(exc.value)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transport_failures_map_to_request_error(status: int) -> None:
    client, _ = _client(status)
    with pytest.raises(YouTubeRequestError) as exc:
        client.verify_authentication()
    assert KEY not in str(exc.value)


def test_network_failure_is_typed_and_secret_free() -> None:
    client, _ = _client(raise_error=TimeoutError("timed out"))
    with pytest.raises(YouTubeRequestError) as exc:
        client.verify_authentication()
    assert KEY not in str(exc.value)
    assert "TimeoutError" in str(exc.value)


def test_malformed_transport_response_is_rejected() -> None:
    credentials = YouTubeCredentials(api_key=SecretStr(KEY))
    client = YouTubeApiClient(
        credentials, transport=MagicMock(return_value=SimpleNamespace(status=None))
    )
    with pytest.raises(YouTubeRequestError):
        client.verify_authentication()
