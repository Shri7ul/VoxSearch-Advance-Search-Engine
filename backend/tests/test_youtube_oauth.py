"""Tests for Goal 37 — OAuth workflow with mocked HTTPS transport."""
from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.youtube import (
    GOOGLE_AUTHORIZE_URL,
    GOOGLE_TOKEN_URL,
    YOUTUBE_READONLY_SCOPE,
    OAuthAuthorizationError,
    OAuthConfig,
    OAuthConfigError,
    OAuthStateError,
    OAuthStateStore,
    OAuthTokenError,
    OAuthTokens,
    OAuthTransportError,
    OAuthWorkflow,
)

CLIENT_ID = "test-client-id.apps.googleusercontent.com"
CLIENT_SECRET = "super-secret-client-value"
REDIRECT = "https://app.example.com/api/v1/auth/youtube/callback"
ACCESS = "ya29.access-token-value"
REFRESH = "1//refresh-token-value"


def _config() -> OAuthConfig:
    return OAuthConfig(
        client_id=CLIENT_ID,
        client_secret=SecretStr(CLIENT_SECRET),
        redirect_uri=REDIRECT,
    )


def _tokens(**overrides):
    payload = {
        "access_token": ACCESS,
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": YOUTUBE_READONLY_SCOPE,
        "refresh_token": REFRESH,
    }
    payload.update(overrides)
    return payload


def _workflow(status: int = 200, payload=None, *, raise_error=None):
    captured: dict = {}

    def transport(url: str, body: bytes, timeout: float):
        captured.update(url=url, body=body, timeout=timeout)
        if raise_error is not None:
            raise raise_error
        if payload is None:
            return SimpleNamespace(status=status, body=b"")
        return SimpleNamespace(
            status=status, body=json.dumps(payload).encode("utf-8")
        )

    return OAuthWorkflow(_config(), transport=transport), captured


def _callback(wf: OAuthWorkflow, code: str = "4/code"):
    url, state = wf.authorization_url()
    return wf.handle_callback(code=code, state=state)


# --- Configuration ----------------------------------------------------------

def test_config_masks_client_secret() -> None:
    assert CLIENT_SECRET not in repr(_config())
    assert CLIENT_SECRET not in str(_config())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"client_id": ""},
        {"client_secret": "plain"},
        {"redirect_uri": "http://evil.example/callback"},
        {"redirect_uri": "https://app.example/cb?x=1"},
        {"redirect_uri": "https://app.example/cb#frag"},
    ],
)
def test_invalid_config_is_rejected(kwargs) -> None:
    base = {
        "client_id": CLIENT_ID,
        "client_secret": SecretStr(CLIENT_SECRET),
        "redirect_uri": REDIRECT,
    }
    base.update(kwargs)
    with pytest.raises(OAuthConfigError):
        OAuthConfig(**base)


def test_local_http_redirect_is_allowed_for_development() -> None:
    config = OAuthConfig(
        client_id=CLIENT_ID,
        client_secret=SecretStr(CLIENT_SECRET),
        redirect_uri="http://localhost:8080/callback",
    )
    assert config.redirect_uri.endswith("/callback")


def test_from_settings_requires_all_fields() -> None:
    with pytest.raises(OAuthConfigError):
        OAuthConfig.from_settings(Settings())
    config = OAuthConfig.from_settings(
        Settings(
            oauth_client_id=CLIENT_ID,
            oauth_client_secret=SecretStr(CLIENT_SECRET),
            oauth_redirect_uri=REDIRECT,
        )
    )
    assert config.client_id == CLIENT_ID


# --- State and authorization URL -------------------------------------------

def test_state_is_random_bounded_and_single_use() -> None:
    store = OAuthStateStore()
    first = store.issue()
    second = store.issue()
    assert first != second
    assert 16 <= len(first) <= 128
    store.consume(first)
    with pytest.raises(OAuthStateError):
        store.consume(first)
    with pytest.raises(OAuthStateError):
        store.consume("unknown-state-value-1234")


def test_state_store_is_bounded() -> None:
    store = OAuthStateStore()
    states = [store.issue() for _ in range(200)]
    with pytest.raises(OAuthStateError):
        store.consume(states[0])  # oldest evicted


def test_expired_state_is_rejected() -> None:
    import time

    store = OAuthStateStore()
    state = store.issue()
    store._states[state] = time.monotonic() - 1  # simulate elapsed TTL
    with pytest.raises(OAuthStateError):
        store.consume(state)


def test_authorization_url_uses_official_endpoint_and_params() -> None:
    wf, _ = _workflow()
    url, state = wf.authorization_url()
    parsed = urlparse(url)
    assert url.startswith(GOOGLE_AUTHORIZE_URL)
    assert parsed.scheme == "https"
    query = parse_qs(parsed.query)
    assert query["response_type"] == ["code"]
    assert query["client_id"] == [CLIENT_ID]
    assert query["redirect_uri"] == [REDIRECT]
    assert query["scope"] == [YOUTUBE_READONLY_SCOPE]
    assert query["access_type"] == ["offline"]
    assert query["state"] == [state]
    assert CLIENT_SECRET not in url


def test_caller_supplied_state_is_registered_and_consumable() -> None:
    wf, _ = _workflow(payload=_tokens())
    supplied = "caller-supplied-state-value-1234"
    url, state = wf.authorization_url(state=supplied)
    assert state == supplied
    assert parse_qs(urlparse(url).query)["state"] == [supplied]
    tokens = wf.handle_callback(code="4/code", state=supplied)
    assert tokens.access_token.get_secret_value() == ACCESS


def test_state_mismatch_blocks_exchange() -> None:
    wf, captured = _workflow(payload=_tokens())
    wf.authorization_url()
    with pytest.raises(OAuthStateError):
        wf.handle_callback(code="4/code", state="wrong-state-value-1234")
    assert captured == {}


def test_denial_consumes_state_and_raises() -> None:
    wf, captured = _workflow(payload=_tokens())
    _, state = wf.authorization_url()
    with pytest.raises(OAuthAuthorizationError):
        wf.handle_callback(code=None, state=state, error="access_denied")
    with pytest.raises(OAuthStateError):
        wf.handle_callback(code="4/code", state=state)
    assert captured == {}


@pytest.mark.parametrize("code", ["", "   ", None, "x" * 513, 123])
def test_invalid_code_is_rejected_and_consumes_state(code) -> None:
    wf, captured = _workflow(payload=_tokens())
    _, state = wf.authorization_url()
    with pytest.raises(OAuthAuthorizationError):
        wf.handle_callback(code=code, state=state)
    with pytest.raises(OAuthStateError):
        wf.handle_callback(code="4/valid-code", state=state)
    assert captured == {}


# --- Token exchange ---------------------------------------------------------

def test_successful_exchange_returns_masked_tokens() -> None:
    wf, captured = _workflow(payload=_tokens())
    tokens = _callback(wf)
    assert isinstance(tokens, OAuthTokens)
    assert tokens.access_token.get_secret_value() == ACCESS
    assert tokens.refresh_token is not None
    assert tokens.refresh_token.get_secret_value() == REFRESH
    assert ACCESS not in repr(tokens)
    assert REFRESH not in repr(tokens)
    assert CLIENT_SECRET not in repr(tokens)
    assert captured["url"] == GOOGLE_TOKEN_URL
    assert captured["timeout"] == 10.0
    body = parse_qs(captured["body"].decode())
    assert body["grant_type"] == ["authorization_code"]
    assert body["redirect_uri"] == [REDIRECT]
    assert body["client_secret"] == [CLIENT_SECRET]


def test_tokens_expiry_is_deterministic() -> None:
    tokens = OAuthTokens(
        access_token=SecretStr(ACCESS),
        expires_in=100,
        token_type="Bearer",
        scope=YOUTUBE_READONLY_SCOPE,
        issued_at=1000.0,
    )
    assert tokens.is_expired(now=1099.0) is False
    assert tokens.is_expired(now=1100.0) is True


def test_retrying_callback_with_same_state_is_blocked() -> None:
    wf, captured = _workflow(payload=_tokens())
    url, state = wf.authorization_url()
    wf.handle_callback(code="4/code", state=state)
    with pytest.raises(OAuthStateError):
        wf.handle_callback(code="4/code", state=state)
    assert len(captured) == 3  # only one transport call


@pytest.mark.parametrize(
    "payload,exc",
    [
        ({**_tokens(), "access_token": ""}, OAuthTokenError),
        ({**_tokens(), "expires_in": 0}, OAuthTokenError),
        ({**_tokens(), "expires_in": True}, OAuthTokenError),
        ({**_tokens(), "token_type": "MAC"}, OAuthTokenError),
        ({**_tokens(), "scope": "other.scope"}, OAuthTokenError),
        ({**_tokens(), "refresh_token": 123}, OAuthTokenError),
        ({"access_token": ACCESS}, OAuthTokenError),
    ],
)
def test_malformed_token_payloads_are_rejected(payload, exc) -> None:
    wf, _ = _workflow(payload=payload)
    with pytest.raises(exc):
        _callback(wf)


def test_invalid_json_and_missing_fields_are_rejected() -> None:
    wf, _ = _workflow(status=200, payload=None)
    with pytest.raises(OAuthTokenError):
        _callback(wf)


@pytest.mark.parametrize(
    "status,exc",
    [(400, OAuthTokenError), (401, OAuthAuthorizationError), (403, OAuthAuthorizationError), (500, OAuthTransportError)],
)
def test_http_failures_are_typed(status: int, exc) -> None:
    wf, _ = _workflow(status=status, payload=_tokens())
    with pytest.raises(exc) as caught:
        _callback(wf)
    assert ACCESS not in str(caught.value)
    assert CLIENT_SECRET not in str(caught.value)


def test_transport_failure_is_typed_and_secret_free() -> None:
    wf, _ = _workflow(raise_error=TimeoutError("timed out"))
    with pytest.raises(OAuthTransportError) as caught:
        _callback(wf)
    assert "TimeoutError" in str(caught.value)
    assert CLIENT_SECRET not in str(caught.value)


def test_timeout_validation() -> None:
    with pytest.raises(OAuthConfigError):
        OAuthWorkflow(_config(), timeout_seconds=0)
    with pytest.raises(OAuthConfigError):
        OAuthWorkflow(_config(), timeout_seconds=True)
