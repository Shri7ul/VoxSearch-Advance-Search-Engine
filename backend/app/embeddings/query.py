"""Goal 19 — natural-language query embedding generation.

This module turns ONE query string into the same validated, unit-normalized
1024-dimensional dense vector contract used for indexed chunks (Goal 15). It is
a thin orchestration layer over :func:`app.embeddings.contracts.generate_embeddings`:

* input validation, finiteness, dimension and normalization checks are delegated
  to the Goal 15 contract — invalid output is REJECTED, never silently rescaled;
* inference runs through :class:`embedding_worker.client.SubprocessEmbeddingProvider`
  in the OPTIONAL isolated interpreter, so ``torch`` / ``FlagEmbedding`` / BGE-M3
  are NEVER imported or loaded in the core API process;
* no embedding is written anywhere, no database is touched, and no raw vector is
  exposed by this module's callers except through the validated value object.

The isolated interpreter path is read from ``VOXSEARCH_EMBEDDING_PYTHON`` only
when a provider is not injected. That is a process environment variable — this
module never reads a ``.env`` file and never modifies PATH or global Python.
"""
from __future__ import annotations

import os
from typing import Any

from app.embeddings.contracts import (
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingVector,
    generate_embeddings,
)

QUERY_EMBEDDING_PYTHON_ENV = "VOXSEARCH_EMBEDDING_PYTHON"


class QueryEmbeddingError(EmbeddingError):
    """Base class for Goal 19 query-embedding failures."""


class QueryEmbeddingUnavailableError(QueryEmbeddingError, RuntimeError):
    """No isolated embedding worker interpreter is configured."""


class QueryEmbeddingService:
    """Generate one validated query vector from one natural-language query.

    Parameters
    ----------
    provider:
        Optional Goal 15 ``EmbeddingProvider``. Injecting a provider keeps the
        service fully testable with no model, no subprocess and no heavy imports.
        When omitted, the isolated subprocess provider is built lazily on first
        use from ``worker_python`` or ``VOXSEARCH_EMBEDDING_PYTHON``.
    config:
        Goal 15 embedding configuration (model id / path / bounds).
    worker_python:
        Explicit path to the optional isolated interpreter. Takes precedence
        over the environment variable.
    """

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        *,
        config: EmbeddingConfig | None = None,
        worker_python: str | None = None,
    ) -> None:
        self._provider = provider
        self._config = config if config is not None else EmbeddingConfig()
        self._worker_python = worker_python

    @property
    def config(self) -> EmbeddingConfig:
        """The immutable Goal 15 configuration used for every query."""
        return self._config

    def embed(self, query: Any) -> EmbeddingVector:
        """Return the validated query embedding for ``query``.

        Raises Goal 15's typed ``EmbeddingError`` subclasses for invalid input,
        failed inference or invalid output, and
        :class:`QueryEmbeddingUnavailableError` if the isolated worker is not
        configured. Nothing is normalized or repaired here.
        """
        # Explicit ``None`` check: a provider with a falsy __bool__/__len__ is
        # still a valid provider and must never be silently replaced by the
        # subprocess fallback.
        provider = self._provider if self._provider is not None else self._build_provider()
        batch = generate_embeddings((query,), provider, config=self._config)
        # generate_embeddings enforces exact 1:1 ordering/count; this assertion
        # is a defensive invariant, not a fallback that could hide bad output.
        if batch.count != 1:
            raise QueryEmbeddingError(
                "query embedding produced an unexpected number of vectors"
            )
        return batch.vectors[0]

    def _build_provider(self) -> EmbeddingProvider:
        """Lazily construct the isolated subprocess provider (no heavy imports)."""
        from embedding_worker.client import SubprocessEmbeddingProvider

        python = self._worker_python
        if not python:
            python = os.environ.get(QUERY_EMBEDDING_PYTHON_ENV, "").strip()
        if not python:
            raise QueryEmbeddingUnavailableError(
                "query embedding worker is not configured"
            )
        return SubprocessEmbeddingProvider(worker_python=python)


__all__ = [
    "QUERY_EMBEDDING_PYTHON_ENV",
    "QueryEmbeddingError",
    "QueryEmbeddingService",
    "QueryEmbeddingUnavailableError",
]
