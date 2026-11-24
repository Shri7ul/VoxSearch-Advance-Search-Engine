"""BGE reranker MODEL-LOAD worker — ISOLATED process entry point (Goal 26).

Run ONLY as a separate process::

    <isolated-python> -m reranker_worker.worker

Reads ONE bounded JSON request from stdin and writes ONE bounded JSON result to
stdout. It deliberately does NOT import :mod:`app`, so it runs in the OPTIONAL
isolated ``backend/.venv-embedding`` interpreter without the core API deps.

Scope: Goal 26 verifies that the BGE reranker model can be loaded; Goal 27
scores bounded ``(query, text)`` pairs when a ``pairs`` array is supplied. The
service layer owns ranking/reordering — this worker only returns scores. The
heavy FlagEmbedding import is LAZY inside :func:`_load_model`.
"""
from __future__ import annotations

import json
import sys
from typing import Any

MAX_REQUEST_BYTES = 4_000_000
MAX_RESPONSE_BYTES = 4_000_000
DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
MAX_MODEL_ID_CHARACTERS = 256
MAX_MODEL_PATH_CHARACTERS = 4096
MAX_PAIRS = 50
MAX_TEXT_CHARACTERS = 8192

_REASON_INVALID_REQUEST = "invalid_request"
_REASON_MODEL_LOAD_ERROR = "model_load_error"
_REASON_SCORING_ERROR = "scoring_error"
_REASON_INVALID_OUTPUT = "invalid_output"


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
    pairs = req.get("pairs")
    if pairs is not None:
        if not isinstance(pairs, list) or not (1 <= len(pairs) <= MAX_PAIRS):
            return None
        clean_pairs: list[list[str]] = []
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                return None
            if any(
                not isinstance(text, str)
                or not text.strip()
                or len(text) > MAX_TEXT_CHARACTERS
                for text in pair
            ):
                return None
            clean_pairs.append([pair[0], pair[1]])  # verbatim, never normalized
        return {"model_id": model_id, "model_path": model_path, "pairs": clean_pairs}
    return {"model_id": model_id, "model_path": model_path}


def _to_scores(raw: Any, expected: int) -> list[float]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        values = [raw]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        raise ValueError("unexpected score shape")
    if len(values) != expected:
        raise ValueError("score count mismatch")
    scores: list[float] = []
    import math

    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("non-numeric score")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite score")
        scores.append(number)
    return scores


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

    pairs = normalized.get("pairs")
    if pairs is None:
        _emit(
            {
                "loaded": True,
                "model_id": normalized["model_id"],
                "provider_error": None,
            }
        )
        return 0

    try:
        raw_scores = model.compute_score(pairs)
    except Exception:  # noqa: BLE001 - never leak a traceback / path / secret
        _failure(_REASON_SCORING_ERROR)
        return 0
    try:
        scores = _to_scores(raw_scores, len(pairs))
    except ValueError:
        _failure(_REASON_INVALID_OUTPUT)
        return 0
    _emit({"scores": scores, "provider_error": None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
