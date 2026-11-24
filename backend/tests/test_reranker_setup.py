"""Offline tests for Goal 26 — isolated BGE reranker setup / model load.

No real model, torch import or network is used: the subprocess and the lazy
FlagEmbedding import are replaced by controlled doubles.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from typing import Any

import pytest
from pydantic import ValidationError

from app.reranking.contracts import (
    DEFAULT_RERANKER_MODEL_ID,
    RERANKER_PYTHON_ENV,
    RerankerConfig,
    RerankerError,
    RerankerModelLoadError,
    RerankerSetup,
    RerankerStatus,
    RerankerUnavailableError,
)


class FakeProvider:
    def __init__(self, status: RerankerStatus | None = None) -> None:
        self.calls: list[RerankerConfig] = []
        self.status = status or RerankerStatus(model_id=DEFAULT_RERANKER_MODEL_ID)

    def load(self, config: RerankerConfig) -> RerankerStatus:
        self.calls.append(config)
        return self.status


# --- Configuration ----------------------------------------------------------

def test_default_config_is_valid_and_immutable() -> None:
    config = RerankerConfig()
    assert config.model_id == DEFAULT_RERANKER_MODEL_ID
    assert config.model_path is None
    with pytest.raises(ValidationError):
        config.model_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": ""},
        {"model_id": "   "},
        {"model_path": "https://example.com/model"},
        {"model_path": "   "},
        {"timeout_seconds": 0},
        {"timeout_seconds": 4000.0},
        {"timeout_seconds": True},
        {"timeout_seconds": False},
        {"timeout_seconds": "10"},
        {"unexpected": 1},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RerankerConfig(**kwargs)


def test_integer_timeout_is_accepted_without_coercion() -> None:
    config = RerankerConfig(timeout_seconds=30)
    assert config.timeout_seconds == 30
    assert isinstance(config.timeout_seconds, int)


def test_local_model_path_is_accepted() -> None:
    assert RerankerConfig(model_path="C:/models/bge-reranker").model_path is not None


# --- Setup orchestration ----------------------------------------------------

def test_setup_uses_injected_provider_and_returns_status() -> None:
    provider = FakeProvider()
    status = RerankerSetup(provider).load_model()
    assert status.loaded is True
    assert provider.calls == [RerankerConfig()]


@pytest.mark.parametrize(
    "status",
    [
        RerankerStatus(model_id=DEFAULT_RERANKER_MODEL_ID, loaded=False),
        RerankerStatus(model_id="   ", loaded=True),
    ],
)
def test_setup_fails_closed_on_invalid_status(status: RerankerStatus) -> None:
    with pytest.raises(RerankerModelLoadError, match="did not report a loaded model"):
        RerankerSetup(FakeProvider(status)).load_model()


def test_setup_uses_config_when_loading() -> None:
    provider = FakeProvider()
    config = RerankerConfig(model_id="custom-reranker", timeout_seconds=30.0)
    RerankerSetup(provider, config=config).load_model()
    assert provider.calls == [config]


def test_missing_interpreter_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RERANKER_PYTHON_ENV, raising=False)
    with pytest.raises(RerankerUnavailableError):
        RerankerSetup().load_model()


def test_explicit_interpreter_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reranker_worker import client as client_module

    monkeypatch.setenv(RERANKER_PYTHON_ENV, "wrong/python.exe")
    captured: dict[str, Any] = {}

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        payload = json.dumps(
            {
                "loaded": True,
                "model_id": DEFAULT_RERANKER_MODEL_ID,
                "provider_error": None,
            }
        ).encode("utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=payload)

    monkeypatch.setattr(client_module.subprocess, "run", fake_run)
    status = RerankerSetup(worker_python="right/python.exe").load_model()
    assert status.loaded is True
    assert captured["argv"] == ["right/python.exe", "-m", "reranker_worker.worker"]


# --- Subprocess contract ----------------------------------------------------

def test_subprocess_contract_is_isolated_and_credential_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reranker_worker import client as client_module

    monkeypatch.setenv("DATABASE_URL", "secret-must-not-be-forwarded")
    captured: dict[str, Any] = {}

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        payload = json.dumps(
            {
                "loaded": True,
                "model_id": DEFAULT_RERANKER_MODEL_ID,
                "provider_error": None,
            }
        ).encode("utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=payload)

    monkeypatch.setattr(client_module.subprocess, "run", fake_run)
    status = RerankerSetup(
        worker_python="isolated/python.exe",
        config=RerankerConfig(timeout_seconds=12.0),
    ).load_model()

    assert status == RerankerStatus(model_id=DEFAULT_RERANKER_MODEL_ID, loaded=True)
    kwargs = captured["kwargs"]
    assert kwargs["shell"] is False
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["timeout"] == 12.0
    assert kwargs["cwd"].replace("\\", "/").endswith("backend")
    assert "DATABASE_URL" not in kwargs["env"]
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"


@pytest.mark.parametrize(
    "payload,returncode,expected",
    [
        ({"loaded": False, "provider_error": "model_load_error"}, 0, RerankerModelLoadError),
        ({"loaded": False, "provider_error": "invalid_request"}, 0, RerankerError),
        ({"loaded": True, "model_id": "x"}, 1, RerankerModelLoadError),
        ({"loaded": False, "provider_error": None}, 0, RerankerModelLoadError),
        ({"loaded": "yes", "model_id": "x", "provider_error": None}, 0, RerankerModelLoadError),
    ],
)
def test_worker_failures_map_to_closed_errors(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    returncode: int,
    expected: type[Exception],
) -> None:
    from reranker_worker import client as client_module

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        data = json.dumps(payload).encode("utf-8")
        return subprocess.CompletedProcess(argv, returncode, stdout=data)

    monkeypatch.setattr(client_module.subprocess, "run", fake_run)
    with pytest.raises(expected):
        RerankerSetup(worker_python="python").load_model()


def test_worker_timeout_and_launch_failure_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reranker_worker import client as client_module

    def timeout(argv: Any, **kwargs: Any):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(client_module.subprocess, "run", timeout)
    with pytest.raises(RerankerModelLoadError, match="timed out"):
        RerankerSetup(worker_python="python").load_model()

    def missing(argv: Any, **kwargs: Any):
        raise OSError("missing")

    monkeypatch.setattr(client_module.subprocess, "run", missing)
    with pytest.raises(RerankerModelLoadError, match="could not be launched"):
        RerankerSetup(worker_python="python").load_model()


def test_worker_emits_closed_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker itself must never print exception text or paths."""
    from reranker_worker import worker

    fake_stdin = io.BytesIO(b'{"model_id": "BAAI/bge-reranker-v2-m3"}')
    fake_stdout = io.StringIO()
    monkeypatch.setattr(worker.sys, "stdin", types.SimpleNamespace(buffer=fake_stdin))
    monkeypatch.setattr(worker.sys, "stdout", fake_stdout)

    def broken_load(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("C:/secret/path and DSN must not leak")

    monkeypatch.setitem(sys.modules, "FlagEmbedding", types.ModuleType("FlagEmbedding"))
    monkeypatch.setattr(
        sys.modules["FlagEmbedding"], "FlagReranker", broken_load, raising=False
    )
    assert worker.main() == 0
    result = json.loads(fake_stdout.getvalue())
    assert result["provider_error"] == "model_load_error"
    assert "secret" not in fake_stdout.getvalue()
