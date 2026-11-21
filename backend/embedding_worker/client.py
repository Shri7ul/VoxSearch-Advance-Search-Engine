"""Stdlib process launcher for the isolated BGE-M3 worker (Goal 15).

:class:`SubprocessEmbeddingProvider` satisfies the
:class:`app.embeddings.contracts.EmbeddingProvider` protocol: given validated
texts + an :class:`~app.embeddings.contracts.EmbeddingConfig`, it launches
``python -m embedding_worker.worker`` in a SEPARATE process (explicit argv,
``shell=False``) and returns the ordered dense vectors. It has NO heavy ML
import — the model lives only in the isolated worker process.

Mirrors Goal 12's proven boundary:

* bounded JSON on stdin (``MAX_REQUEST_BYTES``) and bounded stdout
  (``MAX_RESPONSE_BYTES``), a hard timeout, and ``stderr=DEVNULL`` (no path /
  credential / traceback can leak into logs),
* a MINIMAL, controlled environment built from a small allowlist — the parent
  process's ``DATABASE_URL`` / Supabase / ``*_KEY`` / ``*_TOKEN`` / ``*_SECRET``
  env vars are NEVER forwarded,
* explicit HF-offline flags and conservative CPU thread caps (float32 CPU),
* closed, credential-free error reasons mapped to the typed
  :class:`~app.embeddings.contracts.EmbeddingError` subclasses so
  :func:`app.embeddings.contracts.generate_embeddings` re-raises them intact.

The model is loaded ONCE per :meth:`embed` call inside the worker — NOT once per
chunk. Live real-weights execution is opt-in and OUTSTANDING for the default
test suite (which injects a controlled provider double instead).
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from typing import Any

from app.embeddings.contracts import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    EmbeddingConfig,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    InvalidEmbeddingInputError,
    InvalidEmbeddingOutputError,
)

# MAX_REQUEST_BYTES / MAX_RESPONSE_BYTES are imported from the authoritative
# ``app.embeddings.contracts`` single source of truth — there is NO hidden local
# copy here that could drift from the core or the isolated worker.
DEFAULT_TIMEOUT_SECONDS = 300.0

# Only these env var NAMES are forwarded from the parent. Anything
# credential-like (DATABASE_URL, SUPABASE_*, *_KEY, *_TOKEN, *_SECRET, ...) is
# deliberately NOT in this list and therefore cannot reach the isolated process.
_ENV_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",
    "COMSPEC",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "PYTHONIOENCODING",
    "PYTHONPATH",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
)

# Explicitly SET (never inherited): force offline cache use and keep CPU
# inference small / deterministic. No network, no GPU, no fp16.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _controlled_env() -> dict[str, str]:
    """Minimal env: allowlisted parent names + forced offline/CPU flags."""
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env.update(_OFFLINE_ENV)
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        _BACKEND_DIR
        if not inherited
        else _BACKEND_DIR + os.pathsep + inherited
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class SubprocessEmbeddingProvider:
    """Launch the isolated worker ONCE per call and return ordered vectors.

    Parameters
    ----------
    worker_python:
        Path to the OPTIONAL isolated interpreter that has
        ``requirements-embedding.txt`` installed. Passed as argv[0] verbatim
        (explicit, no shell).
    timeout_seconds:
        Hard wall-clock cap on the subprocess (model load + bounded encode).
    """

    def __init__(
        self,
        *,
        worker_python: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._worker_python = worker_python
        self._timeout = timeout_seconds

    def embed(
        self, texts: Sequence[str], *, config: EmbeddingConfig
    ) -> Sequence[Sequence[float]]:
        payload: dict[str, Any] = {
            "texts": list(texts),
            "model_id": config.model_id,
            "model_path": config.model_path,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
        }
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise InvalidEmbeddingInputError(
                "embedding request is not JSON-serializable"
            ) from exc
        if len(data) > MAX_REQUEST_BYTES:
            raise InvalidEmbeddingInputError("embedding request is too large")

        argv = [self._worker_python, "-m", "embedding_worker.worker"]
        try:
            completed = subprocess.run(
                argv,
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=self._timeout,
                env=_controlled_env(),
                cwd=_BACKEND_DIR,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise EmbeddingInferenceError("embedding worker timed out") from exc
        except OSError as exc:
            raise EmbeddingModelLoadError(
                "embedding worker could not be launched"
            ) from exc

        if completed.returncode != 0:
            raise EmbeddingInferenceError("embedding worker exited with an error")
        stdout = completed.stdout or b""
        if len(stdout) > MAX_RESPONSE_BYTES:
            raise EmbeddingInferenceError("embedding worker output is too large")
        try:
            raw = json.loads(stdout.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise EmbeddingInferenceError(
                "embedding worker returned invalid JSON"
            ) from exc
        if not isinstance(raw, dict):
            raise EmbeddingInferenceError("embedding worker returned a non-object")

        error = raw.get("provider_error")
        if error:
            self._raise_for(str(error))

        vectors = raw.get("vectors")
        if not isinstance(vectors, list):
            raise InvalidEmbeddingOutputError(
                "embedding worker returned no vector list"
            )
        if len(vectors) != len(texts):
            raise InvalidEmbeddingOutputError(
                f"embedding worker returned {len(vectors)} vectors for "
                f"{len(texts)} texts"
            )
        return vectors

    @staticmethod
    def _raise_for(reason: str) -> None:
        if reason == "model_load_error":
            raise EmbeddingModelLoadError("isolated model could not be loaded")
        if reason == "invalid_output":
            raise InvalidEmbeddingOutputError(
                "isolated model produced invalid dense output"
            )
        if reason in ("invalid_request", "text_too_long"):
            raise InvalidEmbeddingInputError(
                f"isolated worker rejected the request ({reason})"
            )
        raise EmbeddingInferenceError("isolated worker inference failed")


__all__ = ["SubprocessEmbeddingProvider"]
