"""Tests for Goal 19 — natural-language query embedding generation.

The tests inject a controlled Goal 15 ``EmbeddingProvider``; no model, no torch,
no isolated interpreter and no network is used. The subprocess path is verified
with a controlled ``subprocess.run`` double.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any, Sequence

import pytest

from app.embeddings.contracts import (
    DENSE_DIMENSION,
    EmbeddingConfig,
    EmbeddingInferenceError,
    EmbeddingProvider,
    InvalidEmbeddingInputError,
    InvalidEmbeddingOutputError,
)
from app.embeddings.query import (
    QUERY_EMBEDDING_PYTHON_ENV,
    QueryEmbeddingService,
    QueryEmbeddingUnavailableError,
)

UNIT = [1.0 / math.sqrt(DENSE_DIMENSION)] * DENSE_DIMENSION


class FakeProvider:
    """Controlled Goal 15 provider: records calls, returns a fixed row."""

    def __init__(self, rows: Sequence[Sequence[float]] | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], EmbeddingConfig]] = []
        self.rows = rows if rows is not None else [UNIT]

    def embed(
        self, texts: Sequence[str], *, config: EmbeddingConfig
    ) -> Sequence[Sequence[float]]:
        self.calls.append((tuple(texts), config))
        return self.rows


def _service(provider: EmbeddingProvider | None = None) -> QueryEmbeddingService:
    return QueryEmbeddingService(provider if provider is not None else FakeProvider())


# --- 1. Happy path ----------------------------------------------------------

def test_embed_returns_validated_unit_vector() -> None:
    vector = _service().embed("what did the speaker say about latency?")
    assert len(vector.values) == DENSE_DIMENSION
    assert vector.dimension == DENSE_DIMENSION
    assert vector.values == pytest.approx(UNIT)
    assert math.isclose(sum(v * v for v in vector.values), 1.0, rel_tol=1e-6)


def test_embed_passes_verbatim_query_and_config_to_provider() -> None:
    provider = FakeProvider()
    config = EmbeddingConfig(model_id="BAAI/bge-m3", batch_size=1, max_length=128)
    QueryEmbeddingService(provider, config=config).embed("  keep  spacing  ")
    assert provider.calls == [(("  keep  spacing  ",), config)]


def test_query_embedding_service_exposes_config() -> None:
    config = EmbeddingConfig()
    assert QueryEmbeddingService(FakeProvider(), config=config).config is config


# --- 2. Input validation (Goal 15 contract, no silent repair) ---------------

@pytest.mark.parametrize(
    "bad_query",
    ["", "   \t ", 123, None, b"bytes", ["text"], "x" * 8193, " ".join(["w"] * 257)],
)
def test_invalid_query_is_rejected_without_provider_call(bad_query: Any) -> None:
    provider = FakeProvider()
    with pytest.raises(InvalidEmbeddingInputError):
        QueryEmbeddingService(provider).embed(bad_query)
    assert provider.calls == []


# --- 3. Output validation (no silent rescaling) -----------------------------

@pytest.mark.parametrize(
    "bad_rows",
    [
        [[0.03125] * 1023],                       # wrong dimension
        [[float("nan")] + [0.0] * 1023],           # non-finite
        [[0.5] + [0.0] * 1023],                    # not unit norm
    ],
)
def test_invalid_provider_output_is_rejected(bad_rows: Sequence[Sequence[float]]) -> None:
    with pytest.raises(InvalidEmbeddingOutputError):
        QueryEmbeddingService(FakeProvider(bad_rows)).embed("query")


def test_falsy_injected_provider_is_not_replaced_by_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An injected provider is used even when it defines a falsy __len__."""

    class FalsyProvider:
        def __init__(self) -> None:
            self.calls = 0

        def __len__(self) -> int:
            return 0

        def embed(self, texts: Sequence[str], *, config: EmbeddingConfig) -> Any:
            self.calls += 1
            return [UNIT]

    provider = FalsyProvider()
    monkeypatch.delenv(QUERY_EMBEDDING_PYTHON_ENV, raising=False)
    vector = QueryEmbeddingService(provider).embed("query")
    assert provider.calls == 1
    assert vector.values == pytest.approx(UNIT)


def test_provider_failure_is_wrapped_as_typed_error() -> None:
    class BrokenProvider:
        def embed(self, texts: Sequence[str], *, config: EmbeddingConfig) -> Any:
            raise RuntimeError("boom")

    with pytest.raises(EmbeddingInferenceError):
        QueryEmbeddingService(BrokenProvider()).embed("query")


# --- 4. Isolated subprocess boundary ----------------------------------------

def test_missing_worker_interpreter_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(QUERY_EMBEDDING_PYTHON_ENV, raising=False)
    with pytest.raises(QueryEmbeddingUnavailableError):
        QueryEmbeddingService().embed("query")


def test_default_provider_uses_isolated_worker_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default path launches ONLY the isolated interpreter, with no shell."""
    interpreter = "C:/isolated/python.exe"
    monkeypatch.setenv(QUERY_EMBEDDING_PYTHON_ENV, interpreter)
    captured: dict[str, Any] = {}

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        payload = json.dumps(
            {"vectors": [UNIT], "provider_error": None}
        ).encode("utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=payload)

    from embedding_worker import client as worker_client

    monkeypatch.setattr(worker_client.subprocess, "run", fake_run)
    vector = QueryEmbeddingService().embed("query text")
    assert vector.values == pytest.approx(UNIT)
    assert captured["argv"] == [interpreter, "-m", "embedding_worker.worker"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"].endswith("backend")
    assert captured["kwargs"]["stderr"] is subprocess.DEVNULL
    # Credentials from the parent process must never be forwarded.
    env = captured["kwargs"]["env"]
    assert "DATABASE_URL" not in env
    assert "HF_HUB_OFFLINE" in env


def test_explicit_worker_python_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(QUERY_EMBEDDING_PYTHON_ENV, "wrong/python.exe")
    captured: dict[str, Any] = {}

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = argv
        payload = json.dumps(
            {"vectors": [UNIT], "provider_error": None}
        ).encode("utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=payload)

    from embedding_worker import client as worker_client

    monkeypatch.setattr(worker_client.subprocess, "run", fake_run)
    QueryEmbeddingService(worker_python="right/python.exe").embed("query")
    assert captured["argv"][0] == "right/python.exe"
