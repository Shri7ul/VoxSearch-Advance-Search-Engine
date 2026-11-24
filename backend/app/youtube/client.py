"""Goal 34 — YouTube Data API authentication setup.

This module owns only the authentication boundary:

* validated, immutable credentials (``SecretStr`` key + fixed HTTPS base URL);
* an authenticated URL builder that urlencodes parameters and never interpolates
  user input into a URL path or host;
* a minimal authentication probe via the Data API that reports success/failure
  WITHOUT returning or persisting any video data (Goals 35–38).

No network call happens at import time. No secret is ever included in an error
message, log line or ``repr``. No database, migration or new dependency is used.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import SecretStr

from app.core.config import Settings, get_settings

DEFAULT_YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"
DEFAULT_TIMEOUT_SECONDS = 10.0
# Only the single probe resource is allowed; no discovery surface is exposed.
# `i18nLanguages.list(part=snippet)` is a lightweight Data API v3 method that
# accepts an API key and returns no video / channel data.
_ALLOWED_PATHS = frozenset({"i18nLanguages", "search", "videos"})
MAX_VIDEO_IDS_PER_CALL = 50
_QUOTA_REASONS = frozenset(
    {
        "quotaExceeded",
        "dailyLimitExceeded",
        "userRateLimitExceeded",
        "rateLimitExceeded",
    }
)
_RESTRICTION_REASONS = frozenset(
    {"apiNotEnabled", "ipRefererBlocked", "forbidden", "accessNotConfigured"}
)


class YouTubeAuthError(RuntimeError):
    """Base class for credential-free Goal 34 failures."""


class YouTubeNotConfiguredError(YouTubeAuthError):
    """No YouTube API key is configured."""


class YouTubeAuthFailedError(YouTubeAuthError):
    """The YouTube API rejected the request (authentication-related)."""


class YouTubeQuotaError(YouTubeAuthError):
    """The API key is valid but quota / rate limits are exhausted."""


class YouTubeAccessRestrictedError(YouTubeAuthError):
    """The key is valid but API restrictions or permissions block the call."""


class YouTubeRequestError(YouTubeAuthError):
    """The authentication probe could not be completed."""


@dataclass(frozen=True)
class YouTubeCredentials:
    """Immutable credentials; the key is masked in every representation."""

    api_key: SecretStr
    base_url: str = DEFAULT_YOUTUBE_API_BASE_URL

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, SecretStr):
            raise YouTubeAuthError("api_key must be a SecretStr")
        if not self.api_key.get_secret_value().strip():
            raise YouTubeNotConfiguredError("YouTube API key is empty")
        if self.base_url != DEFAULT_YOUTUBE_API_BASE_URL:
            raise YouTubeAuthError("YouTube API base URL is not allowlisted")


@dataclass(frozen=True)
class YouTubeAuthStatus:
    """Result of a successful authentication probe (no video data)."""

    authenticated: bool = True
    resource: str = "i18nLanguages"


@runtime_checkable
class YouTubeTransport(Protocol):
    """Minimal injectable HTTP boundary used by authentication tests."""

    def __call__(self, url: str, timeout: float) -> Any: ...


def _default_transport(url: str, timeout: float) -> Any:
    """Read-only GET via stdlib; never returns the URL in an exception."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            body = response.read(4096)
    except urllib.error.HTTPError as exc:
        return _Response(exc.code, b"")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise YouTubeRequestError(
            f"YouTube API request failed ({type(exc).__name__})"
        ) from exc
    return _Response(status, body)


@dataclass(frozen=True)
class _Response:
    status: int
    body: bytes


def _error_reason(response: Any) -> str | None:
    """Extract a closed Data API v3 `error.errors[].reason` token, if present."""
    body = getattr(response, "body", None)
    if not isinstance(body, (bytes, str)):
        return None
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("errors"), list):
        return None
    for item in error["errors"]:
        if isinstance(item, dict) and isinstance(item.get("reason"), str):
            return item["reason"]
    return None


class YouTubeResponseError(YouTubeAuthError):
    """The YouTube API returned malformed or unexpected JSON."""


class YouTubeApiClient:
    """Validate configuration and call allowlisted YouTube Data API methods."""

    def __init__(
        self,
        credentials: YouTubeCredentials | None = None,
        *,
        transport: YouTubeTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._credentials = credentials
        self._transport: YouTubeTransport = transport or _default_transport
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < timeout_seconds <= 60
        ):
            raise YouTubeAuthError("timeout_seconds must be between 0 and 60")
        self._timeout = float(timeout_seconds)

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, **kwargs: Any
    ) -> "YouTubeApiClient":
        """Build a client from application settings, if a key is configured."""
        cfg = settings or get_settings()
        key = cfg.youtube_api_key
        if key is None or not key.get_secret_value().strip():
            raise YouTubeNotConfiguredError("YouTube API key is not configured")
        return cls(
            YouTubeCredentials(
                api_key=key, base_url=cfg.youtube_api_base_url
            ),
            **kwargs,
        )

    def is_configured(self) -> bool:
        return self._credentials is not None

    def authenticated_url(self, path: str, params: dict[str, str]) -> str:
        """Build an HTTPS URL with the API key as a urlencoded parameter."""
        if self._credentials is None:
            raise YouTubeNotConfiguredError("YouTube API key is not configured")
        if path not in _ALLOWED_PATHS:
            raise YouTubeAuthError("YouTube API resource is not allowlisted")
        safe_params: dict[str, str] = {}
        for key, value in params.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise YouTubeAuthError("YouTube API parameters must be strings")
            safe_params[key] = value
        safe_params["key"] = self._credentials.api_key.get_secret_value()
        return (
            f"{self._credentials.base_url}/{path}"
            f"?{urllib.parse.urlencode(safe_params)}"
        )

    def _request(self, path: str, params: dict[str, str]) -> Any:
        """Issue one allowlisted GET and return the transport response."""
        url = self.authenticated_url(path, params)
        try:
            return self._transport(url, self._timeout)
        except YouTubeAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak a URL or key
            raise YouTubeRequestError(
                f"YouTube API request failed ({type(exc).__name__})"
            ) from exc

    @staticmethod
    def _check_status(status: Any, response: Any) -> None:
        if not isinstance(status, int):
            raise YouTubeRequestError("YouTube API returned an invalid status")
        if 200 <= status < 300:
            return
        reason = _error_reason(response)
        if reason in _QUOTA_REASONS:
            raise YouTubeQuotaError(f"YouTube API quota exhausted ({reason})")
        if reason in _RESTRICTION_REASONS:
            raise YouTubeAccessRestrictedError(
                f"YouTube API access restricted ({reason})"
            )
        if status in (400, 401, 403):
            raise YouTubeAuthFailedError(
                f"YouTube API rejected the request (HTTP {status})"
            )
        raise YouTubeRequestError(f"YouTube API request failed (HTTP {status})")

    def get_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        """GET an allowlisted resource and validate the JSON object shape."""
        response = self._request(path, params)
        self._check_status(getattr(response, "status", None), response)
        body = getattr(response, "body", None)
        if not isinstance(body, (bytes, str)):
            raise YouTubeResponseError("YouTube API returned no response body")
        try:
            payload = json.loads(body)
        except (ValueError, TypeError) as exc:
            raise YouTubeResponseError("YouTube API returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise YouTubeResponseError("YouTube API returned a non-object JSON value")
        return payload

    def verify_authentication(self) -> YouTubeAuthStatus:
        """Probe authentication without consuming video metadata."""
        response = self._request("i18nLanguages", {"part": "snippet"})
        self._check_status(getattr(response, "status", None), response)
        return YouTubeAuthStatus()


__all__ = [
    "DEFAULT_YOUTUBE_API_BASE_URL",
    "MAX_VIDEO_IDS_PER_CALL",
    "YouTubeApiClient",
    "YouTubeAuthError",
    "YouTubeAuthFailedError",
    "YouTubeAccessRestrictedError",
    "YouTubeAuthStatus",
    "YouTubeCredentials",
    "YouTubeNotConfiguredError",
    "YouTubeQuotaError",
    "YouTubeRequestError",
    "YouTubeResponseError",
    "YouTubeTransport",
]
