"""BGE-M3 dense embedding worker — ISOLATED process entry point (Goal 15).

Run ONLY as a separate process::

    <isolated-python> -m embedding_worker.worker

Reads ONE JSON request from stdin and writes ONE JSON result to stdout. It is
deliberately SELF-CONTAINED (stdlib + a lazy ``FlagEmbedding`` import); it does
NOT import :mod:`app`, so it runs in the OPTIONAL isolated
``backend/.venv-embedding`` interpreter without the core API dependencies
(pydantic / fastapi / sqlalchemy are NOT required here).

Goal 15 scope: dense embedding generation ONLY. No DB, no persistence, no HTTP,
no search, no sparse / ColBERT retrieval, no credentials, no network.

Safety / fail-closed properties
-------------------------------
* The heavy ``FlagEmbedding`` import is LAZY inside :func:`_load_model`, so even
  importing / introspecting this module never loads torch / transformers.
* Model initialization is separate from inference and happens ONCE per call
  (never once per chunk). The model is CPU-only fp32: ``use_fp16=False``,
  ``devices=["cpu"]``, dense head ONLY (sparse / ColBERT disabled).
* The request is validated fail-closed. A LOCAL ``model_path`` (cached snapshot)
  is honored; a URL is rejected. With ``HF_HUB_OFFLINE=1`` (set by the client)
  the model resolves from the local Hugging Face cache — NO download.
* Batching is bounded (small ``batch_size``); input order is preserved exactly.
* Exact subword-token length is checked BEFORE ``encode`` so the tokenizer can
  NEVER silently truncate a text (explicit ``text_too_long`` rejection).
* Output is validated (count, exactly 1024 dims, all finite — no NaN/Inf) and
  bounded. On ANY failure the worker emits ONE JSON object with a closed
  ``provider_error`` and exit code 0 — it never prints a traceback, exception
  text, path, or credential to stdout/stderr.

Result protocol (stdout JSON)
-----------------------------

    {"vectors": [[f, ... x1024], ...], "provider_error": null}

On failure the ``vectors`` list is empty and ``provider_error`` is one of:
``invalid_request`` / ``model_load_error`` / ``inference_error`` /
``invalid_output`` / ``text_too_long`` (all closed, credential-free strings).

NOTE: live real-weights inference is a separately approved, opt-in step. The
default test suite injects a controlled ``FlagEmbedding`` double on PYTHONPATH
and does NOT load the multi-GB model. See
``backend/tests/test_embedding_worker.py``.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any

# --- Local safety bounds (kept local so this process needs NO app import) ----
# These are LOCAL COPIES of the ``app.embeddings.contracts`` constants. The
# worker MUST NOT import ``app`` (it runs in the optional ``.venv-embedding``
# interpreter without pydantic / fastapi / sqlalchemy). To prevent silent drift,
# ``tests/test_embedding_worker.py::TestLimitConsistency`` asserts every value
# below equals its authoritative counterpart in ``contracts.py``.
MAX_REQUEST_BYTES = 4_000_000
MAX_RESPONSE_BYTES = 16_000_000
DENSE_DIMENSION = 1024
MAX_TEXTS = 64
MAX_BATCH_SIZE = 8
MAX_MAX_LENGTH = 512
MAX_TEXT_CHARACTERS = 8192
MAX_MODEL_ID_CHARACTERS = 256
MAX_MODEL_PATH_CHARACTERS = 4096
DEFAULT_MODEL_ID = "BAAI/bge-m3"

_REASON_INVALID_REQUEST = "invalid_request"
_REASON_MODEL_LOAD_ERROR = "model_load_error"
_REASON_INFERENCE_ERROR = "inference_error"
_REASON_INVALID_OUTPUT = "invalid_output"
_REASON_TEXT_TOO_LONG = "text_too_long"


class _WorkerTextTooLong(Exception):
    """A text exceeds the tokenizer max_length; reject, never truncate."""


class _WorkerInvalidOutput(Exception):
    """Provider dense output has a bad shape/count or a NaN/Inf value."""


def _emit(payload: dict[str, Any]) -> None:
    """Write ONE bounded JSON object to stdout. Never log secrets/paths."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        text = json.dumps({"vectors": [], "provider_error": _REASON_INVALID_OUTPUT})
    sys.stdout.write(text)
    sys.stdout.flush()


def _failure(reason: str) -> None:
    _emit({"vectors": [], "provider_error": reason})


def _read_request() -> dict[str, Any] | None:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_request(
    req: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate shape + safety bounds fail-closed. Verbatim text is preserved."""
    texts = req.get("texts")
    if not isinstance(texts, list) or not (1 <= len(texts) <= MAX_TEXTS):
        return None, _REASON_INVALID_REQUEST
    clean_texts: list[str] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            return None, _REASON_INVALID_REQUEST
        if len(text) > MAX_TEXT_CHARACTERS:
            return None, _REASON_INVALID_REQUEST
        clean_texts.append(text)  # VERBATIM — never stripped / normalized

    model_id = req.get("model_id", DEFAULT_MODEL_ID)
    if (
        not isinstance(model_id, str)
        or not model_id.strip()
        or len(model_id) > MAX_MODEL_ID_CHARACTERS
    ):
        return None, _REASON_INVALID_REQUEST

    model_path = req.get("model_path")
    if model_path is not None:
        if (
            not isinstance(model_path, str)
            or not model_path.strip()
            or "://" in model_path
            or len(model_path) > MAX_MODEL_PATH_CHARACTERS
        ):
            return None, _REASON_INVALID_REQUEST  # local path only, never a URL

    batch_size = req.get("batch_size")
    if not _is_int(batch_size) or not (1 <= batch_size <= MAX_BATCH_SIZE):
        return None, _REASON_INVALID_REQUEST

    max_length = req.get("max_length")
    if not _is_int(max_length) or not (1 <= max_length <= MAX_MAX_LENGTH):
        return None, _REASON_INVALID_REQUEST

    return (
        {
            "texts": clean_texts,
            "model_id": model_id,
            "model_path": model_path,
            "batch_size": batch_size,
            "max_length": max_length,
        },
        None,
    )


def _load_model(req: dict[str, Any]) -> Any:
    """Load the model ONCE. Heavy import is local (lazy). CPU-only, dense-only."""
    from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]

    name = req["model_path"] or req["model_id"]
    return BGEM3FlagModel(
        name,
        normalize_embeddings=True,
        use_fp16=False,            # CPU-only; never enable fp16 on CPU
        devices=["cpu"],
        return_dense=True,
        return_sparse=False,       # dense ONLY — no sparse / ColBERT heads
        return_colbert_vecs=False,
        query_max_length=req["max_length"],
        passage_max_length=req["max_length"],
        batch_size=req["batch_size"],
    )


def _assert_within_length(model: Any, batch: list[str], max_length: int) -> None:
    """Reject (do NOT silently truncate) text longer than max_length tokens.

    Fail CLOSED: if the exact token length cannot be measured (no tokenizer,
    or an unexpected ``encode`` result), the batch is rejected too. Otherwise
    ``encode`` would run with ``truncation=True`` and silently truncate it.
    """
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise _WorkerTextTooLong
    for text in batch:
        try:
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        except TypeError:
            ids = tokenizer.encode(text)
        if isinstance(ids, dict):
            ids = ids.get("input_ids", [])
        if not isinstance(ids, (list, tuple)):
            raise _WorkerTextTooLong
        if len(ids) > max_length:
            raise _WorkerTextTooLong


def _to_rows(dense: Any) -> list[list[float]]:
    """Convert dense output to plain Python rows; validate shape + finiteness."""
    if dense is None:
        raise _WorkerInvalidOutput
    # numpy ndarray -> .tolist(); a plain list-of-lists -> copied. No numpy import
    # is needed here, so the controlled fake provider (pure stdlib) also works.
    rows = dense.tolist() if hasattr(dense, "tolist") else [list(r) for r in dense]
    if not isinstance(rows, list):
        raise _WorkerInvalidOutput
    cleaned: list[list[float]] = []
    for row in rows:
        if isinstance(row, (str, bytes)) or not isinstance(row, (list, tuple)):
            raise _WorkerInvalidOutput
        if len(row) != DENSE_DIMENSION:
            raise _WorkerInvalidOutput
        new_row: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _WorkerInvalidOutput
            number = float(value)
            if not math.isfinite(number):
                raise _WorkerInvalidOutput
            new_row.append(number)
        cleaned.append(new_row)
    return cleaned


# Deliberate duplication (kept by design): this bounded, ordered batch loop
# mirrors ``app.embeddings.contracts.iter_batches`` / ``InProcessEmbeddingProvider``.
# It is re-implemented HERE because this process MUST NOT import ``app``; the
# shared bounds are pinned by ``TestLimitConsistency`` so the two copies cannot
# drift. See ``backend/README.md`` §12.
def _encode_batches(model: Any, req: dict[str, Any]) -> list[list[float]]:
    """Encode in bounded, ORDERED batches over ONE already-loaded model."""
    texts = req["texts"]
    batch_size = req["batch_size"]
    max_length = req["max_length"]
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        _assert_within_length(model, batch, max_length)
        result = model.encode(
            list(batch),
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = result.get("dense_vecs") if isinstance(result, dict) else None
        rows = _to_rows(dense)
        if len(rows) != len(batch):
            raise _WorkerInvalidOutput
        out.extend(rows)
    return out


def main() -> int:
    """Entry point: validate -> load ONCE -> encode -> emit. Always exits 0."""
    req = _read_request()
    if req is None:
        _failure(_REASON_INVALID_REQUEST)
        return 0
    normalized, reason = _validate_request(req)
    if normalized is None or reason is not None:
        _failure(reason or _REASON_INVALID_REQUEST)
        return 0
    try:
        model = _load_model(normalized)
    except Exception:  # noqa: BLE001 — never leak a traceback / path / secret
        _failure(_REASON_MODEL_LOAD_ERROR)
        return 0
    try:
        vectors = _encode_batches(model, normalized)
    except _WorkerTextTooLong:
        _failure(_REASON_TEXT_TOO_LONG)
        return 0
    except _WorkerInvalidOutput:
        _failure(_REASON_INVALID_OUTPUT)
        return 0
    except Exception:  # noqa: BLE001
        _failure(_REASON_INFERENCE_ERROR)
        return 0
    _emit({"vectors": vectors, "provider_error": None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
