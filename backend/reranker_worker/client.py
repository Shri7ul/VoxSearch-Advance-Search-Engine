"""Stdlib process launcher for the isolated BGE reranker worker (Goal 26).

Mirrors Goal 15's ``embedding_worker.client`` boundary: bounded JSON on stdin /
stdout, a hard timeout, ``stderr=DEVNULL``, ``shell=False``, and a MINIMAL
allowlisted environment. Credential-like parent variables (DATABASE_URL,
SUPABASE_*, *_KEY / *_TOKEN / *_SECRET) are NEVER forwarded. Errors are mapped
to the closed :mod:`app.reranking.contracts` taxonomy with credential-free
messages.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Sequence
from typing import Any

from app.reranking.contracts import (
    RerankerConfig,
    RerankerError,
    RerankerModelLoadError,
    RerankerStatus,
)

MAX_REQUEST_BYTES = 4_000_000
MAX_RESPONSE_BYTES = 4_000_000

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

_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _controlled_env() -> dict[str, str]:
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env.update(_OFFLINE_ENV)
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        _BACKEND_DIR if not inherited else _BACKEND_DIR + os.pathsep + inherited
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class SubprocessRerankerProvider:
    """Load the isolated BGE reranker model once per :meth:`load` call."""

    def __init__(self, *, worker_python: str, timeout_seconds: float) -> None:
        self._worker_python = worker_python
        self._timeout = timeout_seconds

    def load(self, config: RerankerConfig) -> RerankerStatus:
        payload = {"model_id": config.model_id, "model_path": config.model_path}
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RerankerError("reranker request is not JSON-serializable") from exc
        if len(data) > MAX_REQUEST_BYTES:
            raise RerankerError("reranker request is too large")

        argv = [self._worker_python, "-m", "reranker_worker.worker"]
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
            raise RerankerModelLoadError("reranker worker timed out") from exc
        except OSError as exc:
            raise RerankerModelLoadError("reranker worker could not be launched") from exc

        if completed.returncode != 0:
            raise RerankerModelLoadError("reranker worker exited with an error")
        stdout = completed.stdout or b""
        if len(stdout) > MAX_RESPONSE_BYTES:
            raise RerankerModelLoadError("reranker worker output is too large")
        try:
            raw = json.loads(stdout.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise RerankerModelLoadError("reranker worker returned invalid JSON") from exc
        if not isinstance(raw, dict):
            raise RerankerModelLoadError("reranker worker returned a non-object")

        reason = raw.get("provider_error")
        if reason:
            if reason == "invalid_request":
                raise RerankerError("reranker worker rejected the request")
            raise RerankerModelLoadError("isolated reranker could not be loaded")
        if raw.get("loaded") is not True or not isinstance(raw.get("model_id"), str):
            raise RerankerModelLoadError("reranker worker reported no loaded model")
        return RerankerStatus(model_id=raw["model_id"], loaded=True)

    def score(
        self, pairs: Sequence[Sequence[str]], config: RerankerConfig
    ) -> list[float]:
        """Score ``[query, text]`` pairs in the isolated process (Goal 27)."""
        payload: dict[str, Any] = {
            "model_id": config.model_id,
            "model_path": config.model_path,
            "pairs": [list(pair) for pair in pairs],
        }
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RerankerError("reranker scoring request is not JSON-serializable") from exc
        if len(data) > MAX_REQUEST_BYTES:
            raise RerankerError("reranker scoring request is too large")

        argv = [self._worker_python, "-m", "reranker_worker.worker"]
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
            raise RerankerModelLoadError("reranker scoring worker timed out") from exc
        except OSError as exc:
            raise RerankerModelLoadError("reranker scoring worker could not be launched") from exc
        if completed.returncode != 0:
            raise RerankerModelLoadError("reranker scoring worker exited with an error")
        stdout = completed.stdout or b""
        if len(stdout) > MAX_RESPONSE_BYTES:
            raise RerankerModelLoadError("reranker scoring output is too large")
        try:
            raw = json.loads(stdout.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise RerankerModelLoadError("reranker scoring returned invalid JSON") from exc
        if not isinstance(raw, dict):
            raise RerankerModelLoadError("reranker scoring returned a non-object")
        reason = raw.get("provider_error")
        if reason == "invalid_request":
            raise RerankerError("reranker worker rejected the scoring request")
        if reason:
            raise RerankerModelLoadError(f"reranker scoring failed ({reason})")
        scores = raw.get("scores")
        if not isinstance(scores, list) or len(scores) != len(pairs):
            raise RerankerModelLoadError("reranker returned an invalid score list")
        cleaned: list[float] = []
        for value in scores:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RerankerModelLoadError("reranker returned a non-numeric score")
            number = float(value)
            if not math.isfinite(number):
                raise RerankerModelLoadError("reranker returned a non-finite score")
            cleaned.append(number)
        return cleaned


__all__ = ["SubprocessRerankerProvider"]
