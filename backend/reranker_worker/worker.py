"""BGE reranker MODEL-LOAD worker — ISOLATED process entry point (Goal 26).

Run ONLY as a separate process::

    <isolated-python> -m reranker_worker.worker

Reads ONE bounded JSON request from stdin and writes ONE bounded JSON result to
stdout. It deliberately does NOT import :mod:`app`, so it runs in the OPTIONAL
isolated ``backend/.venv-embedding`` interpreter without the core API deps.

Scope: verify that the BGE reranker model can be loaded. There is NO passage /
query scoring, no ranking and no candidate reordering — those belong to Goal 27.
The heavy FlagEmbedding import is LAZY inside :func:`_load_model`.
"""
from __future__ import annotations

import json
import sys
from typing import Any

MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_BYTES = 1_000_000
DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
MAX_MODEL_ID_CHARACTERS = 256
MAX_MODEL_PATH_CHARACTERS = 4096

_REASON_INVALID_REQUEST = "invalid_request"
_REASON_MODEL_LOAD_ERROR = "model_load_error"


def _emit(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        text = json.dumps({"loaded": False, "provider_error": _REASON_MODEL_LOAD_ERROR})
    sys.stdout.write(text)
    sys.stdout.flush()


def _failure(reason: str) -> None:
    _emit({"loaded": False, "provider_error": reason})


def _read_request() -> dict[str, Any] | None:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _validate(req: dict[str, Any]) -> dict[str, Any] | None:
    model_id = req.get("model_id", DEFAULT_MODEL_ID)
    if (
        not isinstance(model_id, str)
        or not model_id.strip()
        or len(model_id) > MAX_MODEL_ID_CHARACTERS
    ):
        return None
    model_path = req.get("model_path")
    if model_path is not None:
        if (
            not isinstance(model_path, str)
            or not model_path.strip()
            or "://" in model_path
            or len(model_path) > MAX_MODEL_PATH_CHARACTERS
        ):
            return None
    return {"model_id": model_id, "model_path": model_path}


def _load_model(req: dict[str, Any]) -> Any:
    # Heavy import is local and lazy: importing this module never loads torch.
    from FlagEmbedding import FlagReranker  # type: ignore[import-not-found]

    name = req["model_path"] or req["model_id"]
    return FlagReranker(name, use_fp16=False)


def main() -> int:
    req = _read_request()
    if req is None:
        _failure(_REASON_INVALID_REQUEST)
        return 0
    normalized = _validate(req)
    if normalized is None:
        _failure(_REASON_INVALID_REQUEST)
        return 0
    try:
        model = _load_model(normalized)
    except Exception:  # noqa: BLE001 - never leak a traceback / path / secret
        _failure(_REASON_MODEL_LOAD_ERROR)
        return 0
    if model is None:
        _failure(_REASON_MODEL_LOAD_ERROR)
        return 0
    _emit(
        {
            "loaded": True,
            "model_id": normalized["model_id"],
            "provider_error": None,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
