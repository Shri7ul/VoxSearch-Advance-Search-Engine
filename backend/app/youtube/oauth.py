"""Goal 37 — creator authorization via Google OAuth 2.0 (web-server flow).

Scope is limited to the authorization workflow itself:

* validated OAuth client configuration (secrets masked, redirect URI fixed);
* cryptographically random, single-use CSRF `state` with TTL and a hard bound;
* an authorization URL for Google's official HTTPS endpoint;
* authorization-code exchange at Google's official HTTPS token endpoint;
* strict token-response validation and secret-safe error taxonomy.

No HTTP routes, token persistence, transcript ingestion (Goal 38), import-time
network calls, extra dependencies or database migrations are introduced.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import SecretStr

from app.core.config import Settings, get_settings

GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
YOUTUBE_READONLY_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_STATE_ENTRIES = 128
STATE_TTL_SECONDS = 600.0
MAX_CODE_CHARACTERS = 512


class OAuthError(RuntimeError):
    """Base class for credential-free Goal 37 failures."""


class OAuthConfigError(OAuthError, ValueError):
    """OAuth client configuration is invalid or incomplete."""


class OAuthStateError(OAuthError):
    """Missing, unknown, expired or reused OAuth state."""


class OAuthAuthorizationError(OAuthError):
    """Google denied or could not complete the authorization."""


class OAuthTransportError(OAuthError):
    """The OAuth endpoint could not be reached or returned an invalid response."""


class OAuthTokenError(OAuthError):
    """The token response was invalid or the code exchange failed."""


@dataclass(frozen=True)
class OAuthConfig:
    """Validated, immutable client configuration."""

    client_id: str
    client_secret: SecretStr
    redirect_uri: str

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, str) or not self.client_id.strip():
            raise OAuthConfigError("OAuth client id is required")
        if len(self.client_id) > 256:
            raise OAuthConfigError("OAuth client id is too long")
        if not isinstance(self.client_secret, SecretStr):
            raise OAuthConfigError("OAuth client secret must be a SecretStr")
        if not self.client_secret.get_secret_value().strip():
            raise OAuthConfigError("OAuth client secret is required")
        if not isinstance(self.redirect_uri, str) or not self.redirect_uri.strip():
            raise OAuthConfigError("OAuth redirect URI is required")
        parsed = urllib.parse.urlparse(self.redirect_uri)
        local = parsed.hostname in ("localhost", "127.0.0.1")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise OAuthConfigError("OAuth redirect URI must be HTTPS or local HTTP")
        if parsed.query or parsed.fragment:
            raise OAuthConfigError("OAuth redirect URI cannot contain query/fragment")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "OAuthConfig":
        cfg = settings or get_settings()
        if not cfg.oauth_client_id or not cfg.oauth_client_secret or not cfg.oauth_redirect_uri:
            raise OAuthConfigError("OAuth client configuration is incomplete")
        return cls(
            client_id=cfg.oauth_client_id,
            client_secret=cfg.oauth_client_secret,
            redirect_uri=cfg.oauth_redirect_uri,
        )


@dataclass(frozen=True)
class OAuthTokens:
    """Validated tokens; secret values are masked in every representation."""

    access_token: SecretStr
    expires_in: int
    token_type: str
    scope: str
    refresh_token: SecretStr | None = None
    issued_at: float = 0.0

    def is_expired(self, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        return current >= self.issued_at + self.expires_in


@runtime_checkable
class OAuthTransport(Protocol):
    """Injectable HTTPS POST boundary for the token endpoint."""

    def __call__(self, url: str, body: bytes, timeout: float) -> Any: ...


def _default_transport(url: str, body: bytes, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _Response(getattr(response, "status", 200), response.read(8192))
    except urllib.error.HTTPError as exc:
        return _Response(exc.code, b"")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OAuthTransportError(
            f"OAuth token endpoint unreachable ({type(exc).__name__})"
        ) from exc


@dataclass(frozen=True)
class _Response:
    status: int
    body: bytes


class OAuthStateStore:
    """Bounded, TTL-based, single-use CSRF state store (in-memory)."""

    def __init__(self, *, ttl_seconds: float = STATE_TTL_SECONDS) -> None:
        if not 1 <= ttl_seconds <= 3600:
            raise OAuthConfigError("state TTL is out of bounds")
        self._ttl = ttl_seconds
        self._states: dict[str, float] = {}
        self._lock = threading.Lock()

    def issue(self) -> str:
        state = secrets.token_urlsafe(32)
        with self._lock:
            now = time.monotonic()
            self._states = {
                key: expires for key, expires in self._states.items() if expires > now
            }
            while len(self._states) >= MAX_STATE_ENTRIES:
                oldest = min(self._states, key=self._states.get)
                del self._states[oldest]
            self._states[state] = now + self._ttl
        return state

    def accept(self, state: Any) -> str:
        """Register a caller-supplied state so its callback can be consumed."""
        if not isinstance(state, str) or not 16 <= len(state) <= 128:
            raise OAuthStateError("OAuth state is out of bounds")
        with self._lock:
            now = time.monotonic()
            self._states = {
                key: expires for key, expires in self._states.items() if expires > now
            }
            while len(self._states) >= MAX_STATE_ENTRIES:
                oldest = min(self._states, key=self._states.get)
                del self._states[oldest]
            self._states[state] = now + self._ttl
        return state

    def consume(self, state: Any) -> None:
        if not isinstance(state, str) or not state:
            raise OAuthStateError("invalid OAuth state")
        with self._lock:
            expires = self._states.pop(state, None)
        if expires is None or expires <= time.monotonic():
            raise OAuthStateError("OAuth state is unknown, expired or already used")


class OAuthWorkflow:
    """Build authorization URLs and exchange codes with strong state checks."""

    def __init__(
        self,
        config: OAuthConfig | None = None,
        *,
        state_store: OAuthStateStore | None = None,
        transport: OAuthTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._config = config
        self._states = state_store or OAuthStateStore()
        self._transport: OAuthTransport = transport or _default_transport
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < timeout_seconds <= 60
        ):
            raise OAuthConfigError("timeout_seconds is out of bounds")
        self._timeout = float(timeout_seconds)

    def _require_config(self) -> OAuthConfig:
        if self._config is None:
            self._config = OAuthConfig.from_settings()
        return self._config

    def authorization_url(self, state: str | None = None) -> tuple[str, str]:
        """Return ``(url, state)``; the state is generated when omitted."""
        config = self._require_config()
        active_state = (
            self._states.issue() if state is None else self._states.accept(state)
        )
        query = urllib.parse.urlencode(
            {
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "scope": YOUTUBE_READONLY_SCOPE,
                "access_type": "offline",
                "include_granted_scopes": "true",
                "state": active_state,
            }
        )
        return f"{GOOGLE_AUTHORIZE_URL}?{query}", active_state

    def handle_callback(
        self, *, code: str | None, state: str, error: str | None = None
    ) -> OAuthTokens:
        """Validate callback inputs, then exchange the authorization code."""
        # Consume state FIRST for every callback attempt (denial, invalid code
        # or success), so a state can never be replayed or probed repeatedly.
        self._states.consume(state)
        if error is not None:
            raise OAuthAuthorizationError("authorization was denied or failed")
        if not isinstance(code, str) or not code.strip():
            raise OAuthAuthorizationError("authorization code is missing")
        if len(code) > MAX_CODE_CHARACTERS:
            raise OAuthAuthorizationError("authorization code is too long")
        return self._exchange(code)

    def _exchange(self, code: str) -> OAuthTokens:
        config = self._require_config()
        body = urllib.parse.urlencode(
            {
                "code": code,
                "client_id": config.client_id,
                "client_secret": config.client_secret.get_secret_value(),
                "redirect_uri": config.redirect_uri,
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")
        try:
            response = self._transport(GOOGLE_TOKEN_URL, body, self._timeout)
        except OAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak a token or body
            raise OAuthTransportError(
                f"OAuth token request failed ({type(exc).__name__})"
            ) from exc
        status = getattr(response, "status", None)
        if not isinstance(status, int):
            raise OAuthTransportError("OAuth token endpoint returned an invalid status")
        raw_body = getattr(response, "body", None)
        if not isinstance(raw_body, (bytes, str)):
            raise OAuthTransportError("OAuth token endpoint returned no body")
        if status == 400:
            raise OAuthTokenError("Google rejected the authorization code")
        if status in (401, 403):
            raise OAuthAuthorizationError("Google rejected the OAuth client")
        if not 200 <= status < 300:
            raise OAuthTransportError(f"OAuth token request failed (HTTP {status})")
        try:
            payload = json.loads(raw_body)
        except (ValueError, TypeError) as exc:
            raise OAuthTokenError("OAuth token response is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise OAuthTokenError("OAuth token response is not an object")

        access = payload.get("access_token")
        expires = payload.get("expires_in")
        token_type = payload.get("token_type")
        scope = payload.get("scope")
        refresh = payload.get("refresh_token")
        if not isinstance(access, str) or not access.strip():
            raise OAuthTokenError("OAuth token response has no access token")
        if (
            isinstance(expires, bool)
            or not isinstance(expires, int)
            or not 1 <= expires <= 604800
        ):
            raise OAuthTokenError("OAuth token response has invalid expiry")
        if token_type != "Bearer":
            raise OAuthTokenError("OAuth token response has an unsupported token type")
        if not isinstance(scope, str) or YOUTUBE_READONLY_SCOPE not in scope.split():
            raise OAuthTokenError("required YouTube scope was not granted")
        if refresh is not None and (not isinstance(refresh, str) or not refresh.strip()):
            raise OAuthTokenError("OAuth token response has an invalid refresh token")
        return OAuthTokens(
            access_token=SecretStr(access),
            expires_in=expires,
            token_type=token_type,
            scope=scope,
            refresh_token=SecretStr(refresh) if refresh else None,
            issued_at=time.time(),
        )


__all__ = [
    "GOOGLE_AUTHORIZE_URL",
    "GOOGLE_TOKEN_URL",
    "YOUTUBE_READONLY_SCOPE",
    "OAuthAuthorizationError",
    "OAuthConfig",
    "OAuthConfigError",
    "OAuthError",
    "OAuthStateError",
    "OAuthStateStore",
    "OAuthTokenError",
    "OAuthTokens",
    "OAuthTransport",
    "OAuthTransportError",
    "OAuthWorkflow",
]
