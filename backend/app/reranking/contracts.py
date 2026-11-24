"""Goal 26 — BGE reranker setup (model-load) contract for the core backend.

This module is the torch-free application boundary for the OPTIONAL isolated
BGE reranker environment. It owns validated configuration, a closed error
taxonomy and a read-only ``load_model`` operation that proves the isolated
model can be constructed. It intentionally does NOT score or reorder candidates
— that is Goal 27 — and it never imports torch, transformers or FlagEmbedding.

The heavy model lives only in the separate ``reranker_worker`` process, mirroring
Goal 14/15's proven isolation boundary.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
)

DEFAULT_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
MAX_MODEL_ID_CHARACTERS = 256
MAX_MODEL_PATH_CHARACTERS = 4096
DEFAULT_TIMEOUT_SECONDS = 300.0
RERANKER_PYTHON_ENV = "VOXSEARCH_RERANKER_PYTHON"


class RerankerError(Exception):
    """Base class for credential-free Goal 26 reranker setup failures."""


class InvalidRerankerConfigError(RerankerError, ValueError):
    """Reranker setup configuration is malformed."""


class RerankerUnavailableError(RerankerError, RuntimeError):
    """No isolated reranker interpreter is configured."""


class RerankerModelLoadError(RerankerError, RuntimeError):
    """The isolated BGE reranker model could not be loaded."""


class RerankerConfig(BaseModel):
    """Validated, immutable reranker setup parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = DEFAULT_RERANKER_MODEL_ID
    model_path: str | None = None
    # Strict scalars: never coerce bool or strings into a timeout value.
    timeout_seconds: StrictFloat | StrictInt = Field(
        default=DEFAULT_TIMEOUT_SECONDS, gt=0.0, le=3600.0
    )

    @field_validator("model_id")
    @classmethod
    def _model_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model_id must be a non-empty str")
        if len(value) > MAX_MODEL_ID_CHARACTERS:
            raise ValueError(
                f"model_id exceeds {MAX_MODEL_ID_CHARACTERS} characters"
            )
        return value

    @field_validator("model_path")
    @classmethod
    def _model_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip() or "://" in value:
            raise ValueError("model_path must be a LOCAL path, not a URL")
        if len(value) > MAX_MODEL_PATH_CHARACTERS:
            raise ValueError(
                f"model_path exceeds {MAX_MODEL_PATH_CHARACTERS} characters"
            )
        return value


@dataclass(frozen=True)
class RerankerStatus:
    """Result of a successful isolated model-load check."""

    model_id: str
    loaded: bool = True


@runtime_checkable
class RerankerProvider(Protocol):
    """Owns the isolated model lifecycle; setup only, no scoring."""

    def load(self, config: RerankerConfig) -> RerankerStatus: ...


class RerankerSetup:
    """Verify that the isolated BGE reranker model can be loaded."""

    def __init__(
        self,
        provider: RerankerProvider | None = None,
        *,
        config: RerankerConfig | None = None,
        worker_python: str | None = None,
    ) -> None:
        self._provider = provider
        self._config = config if config is not None else RerankerConfig()
        self._worker_python = worker_python

    @property
    def config(self) -> RerankerConfig:
        return self._config

    def load_model(self) -> RerankerStatus:
        """Load the model once in the isolated process and report success.

        Fails closed if a provider reports ``loaded=False`` or an empty model id:
        Goal 26's achieve condition is that the model actually loads.
        """
        provider = self._provider
        if provider is None:
            provider = self._build_provider()
        status = provider.load(self._config)
        if not status.loaded or not status.model_id.strip():
            raise RerankerModelLoadError("isolated reranker did not report a loaded model")
        return status

    def _build_provider(self) -> RerankerProvider:
        from reranker_worker.client import SubprocessRerankerProvider

        python = self._worker_python
        if not python:
            python = os.environ.get(RERANKER_PYTHON_ENV, "").strip()
        if not python:
            raise RerankerUnavailableError("reranker worker is not configured")
        return SubprocessRerankerProvider(
            worker_python=python,
            timeout_seconds=self._config.timeout_seconds,
        )


__all__ = [
    "DEFAULT_RERANKER_MODEL_ID",
    "DEFAULT_TIMEOUT_SECONDS",
    "RERANKER_PYTHON_ENV",
    "InvalidRerankerConfigError",
    "RerankerConfig",
    "RerankerError",
    "RerankerModelLoadError",
    "RerankerProvider",
    "RerankerSetup",
    "RerankerStatus",
    "RerankerUnavailableError",
]
