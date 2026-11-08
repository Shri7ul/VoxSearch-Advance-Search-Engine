"""WhisperX 3.8.6 forced-alignment worker — ISOLATED process entry point.

Run ONLY as a separate process::

    <isolated-python> -m alignment_worker.worker

It reads ONE JSON request from stdin and writes ONE JSON result to stdout. It
is deliberately SELF-CONTAINED (stdlib + lazy ``whisperx``/``torch`` imports);
it does NOT import :mod:`app`, so it can run in the OPTIONAL isolated
``requirements-alignment.txt`` interpreter without the core API dependencies.

Safety / fail-closed properties
-------------------------------
* Heavy imports (``torch`` / ``whisperx``) are performed LAZILY inside
  :func:`main`, so importing this module (e.g. for introspection/tests) never
  loads the model stack.
* The mandated adapter flags are RE-VALIDATED and pinned here:
  ``interpolate_method="ignore"`` (the default ``"nearest"`` would backfill NaN
  timestamps with INTERPOLATED values, which Goal 12 forbids persisting) and
  ``return_char_alignments=True``. A request with any other value is REJECTED
  with a closed ``provider_error`` and zero model calls.
* The language is gated against the EXACT ``whisperx==3.8.6`` allowlist (Bengali
  ``bn`` / ``ben`` is unsupported in this goal). Unsupported languages produce
  a structured error with ZERO model calls.
* ``audio_path`` must be a LOCAL file path (never a URL, never a signed URL,
  never a query string) and is NEVER logged. Only the opaque ``storage_reference``
  lives in the database; resolving it to a local path is the operator's job and
  happens OUTSIDE this process.
* Output is bounded. On ANY failure the worker emits a single JSON object with
  a closed ``provider_error`` and exit code 0 — it never prints a traceback,
  exception text, path, or credential to stdout/stderr.

Result protocol (RAW WhisperX output, normalized by the client)
--------------------------------------------------------------
Stdout JSON::

    {
      "segments": [
        {"segment_id": "<uuid>",
         "words": [{"word": "Hello", "start": 1.0, "end": 1.4, "score": 0.9}, ...],
         "chars": [{"char": "H", "start": 1.0, "end": 1.1, "score": 0.9}, ...],
         "error": null},
        ...
      ],
      "provider_error": null          # or a closed reason string
    }

Word dicts are passed THROUGH VERBATIM: when ``interpolate_method="ignore"``
leaves a timing missing (NaN), WhisperX omits the ``start`` / ``end`` keys. The
worker MUST NOT add, interpolate, or backfill them. The client merges
per-word char corroboration (via the pure
:func:`app.transcripts.word_alignment.reconstruct_word_chars`) and the service's
:func:`validate_provider_segment` rejects any missing timing.

NOTE (outstanding provider-integration verification): the exact runtime shape
of ``v3.8.6`` char dicts (final output omits ``word-idx``) cannot be confirmed
here without loading the real model / weights. The pure reconstruction fails
closed when the assumption does not hold. Live execution is a separately
approved step and is NOT performed by this goal's default test suite.
"""
from __future__ import annotations

import json
import sys
from typing import Any

MAX_REQUEST_BYTES = 2_000_000
MAX_RESPONSE_BYTES = 8_000_000

# Exact whisperx==3.8.6 DEFAULT_ALIGN_MODELS_TORCH + DEFAULT_ALIGN_MODELS_HF.
# Mirrors app.transcripts.word_contract.SUPPORTED_ALIGNMENT_LANGUAGES; kept local
# so this process needs NO app import. A test asserts both sets are identical.
SUPPORTED_LANGUAGES = frozenset({
    "en", "fr", "de", "es", "it",
    "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi", "fa",
    "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml", "no",
    "nn", "sk", "sl", "hr", "ro", "eu", "gl", "ka", "lv", "tl", "sv", "id",
})
BENGALI_CODES = frozenset({"bn", "ben", "bn-bd", "bn-in"})
ALLOWED_DEVICES = frozenset({"cpu", "cuda"})

_REASON_PROVIDER_ERROR = "provider_error"
_REASON_UNSUPPORTED_LANGUAGE = "unsupported_language"


def _emit(payload: dict[str, Any]) -> None:
    """Write ONE bounded JSON object to stdout. Never log secrets/paths."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        text = json.dumps(
            {"segments": [], "provider_error": _REASON_PROVIDER_ERROR}
        )
    sys.stdout.write(text)
    sys.stdout.flush()


def _failure(reason: str) -> None:
    _emit({"segments": [], "provider_error": reason})


def _read_request() -> dict[str, Any] | None:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _validate_request(
    req: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the request; return ``(normalized, error_reason)``.

    Enforces the pinned adapter flags, the language gate, a local (non-URL)
    audio path, and a bounded, well-formed segment list. Everything is
    fail-closed with a closed reason string (never an exception message).
    """
    if req.get("interpolate_method") != "ignore":
        return None, _REASON_PROVIDER_ERROR
    if req.get("return_char_alignments") is not True:
        return None, _REASON_PROVIDER_ERROR

    language = req.get("language")
    if not isinstance(language, str) or language.strip().lower() not in SUPPORTED_LANGUAGES:
        reason = (
            _REASON_UNSUPPORTED_LANGUAGE
            if isinstance(language, str)
            and language.strip().lower() in BENGALI_CODES
            else _REASON_PROVIDER_ERROR
        )
        return None, reason

    audio_path = req.get("audio_path")
    if (
        not isinstance(audio_path, str)
        or not audio_path.strip()
        or "://" in audio_path
        or "?" in audio_path
    ):
        return None, _REASON_PROVIDER_ERROR  # never a URL / signed URL

    device = req.get("device", "cpu")
    if device not in ALLOWED_DEVICES:
        return None, _REASON_PROVIDER_ERROR

    segments = req.get("segments")
    if not isinstance(segments, list) or len(segments) > 10_000:
        return None, _REASON_PROVIDER_ERROR
    normalized_segments: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            return None, _REASON_PROVIDER_ERROR
        seg_id = seg.get("segment_id")
        text = seg.get("text")
        start = seg.get("start")
        end = seg.get("end")
        if (
            not isinstance(seg_id, str)
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
        ):
            return None, _REASON_PROVIDER_ERROR
        normalized_segments.append(
            {"segment_id": seg_id, "text": text, "start": float(start), "end": float(end)}
        )
    return (
        {
            "language": language.strip().lower(),
            "device": device,
            "audio_path": audio_path,
            "segments": normalized_segments,
        },
        None,
    )


def _run_alignment(req: dict[str, Any]) -> dict[str, Any]:
    """Load the model + audio, call WhisperX align, return RAW output JSON.

    Heavy imports are local to this function. The WhisperX adapter call uses
    the mandated flags; missing timings (NaN with ``interpolate_method="ignore"``)
    are passed through verbatim and NEVER filled here.
    """
    import os  # noqa: F401  (path existence check only; path is never logged)

    if not os.path.isfile(req["audio_path"]):
        return {"segments": [], "provider_error": _REASON_PROVIDER_ERROR}

    # Lazy heavy imports: importing this module does NOT load torch/whisperx.
    import whisperx  # type: ignore[import-not-found]

    language = req["language"]
    device = req["device"]

    model, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    audio = whisperx.load_audio(req["audio_path"])

    # Build WhisperX segments from VERBATIM original text + AUDIO-timeline
    # seconds. WhisperX align() produces the word/char timings; our tokens are
    # NOT sent (the service validates strict 1:1 ordinal mapping afterwards).
    transcript = [
        {"text": seg["text"], "start": seg["start"], "end": seg["end"]}
        for seg in req["segments"]
    ]

    result = whisperx.align(
        transcript,
        model,
        metadata,
        audio,
        device,
        interpolate_method="ignore",
        return_char_alignments=True,
    )

    aligned_segments = result.get("segments") if isinstance(result, dict) else None
    if not isinstance(aligned_segments, list) or len(aligned_segments) != len(
        req["segments"]
    ):
        return {"segments": [], "provider_error": _REASON_PROVIDER_ERROR}

    out_segments: list[dict[str, Any]] = []
    for req_seg, aligned in zip(req["segments"], aligned_segments):
        if not isinstance(aligned, dict):
            out_segments.append(
                {"segment_id": req_seg["segment_id"], "words": [], "chars": [], "error": _REASON_PROVIDER_ERROR}
            )
            continue
        words = aligned.get("words")
        chars = aligned.get("chars")
        out_segments.append(
            {
                "segment_id": req_seg["segment_id"],
                "words": words if isinstance(words, list) else [],
                "chars": chars if isinstance(chars, list) else [],
                "error": None,
            }
        )
    return {"segments": out_segments, "provider_error": None}


def main() -> int:
    """Entry point: validate -> align -> emit. Always exits 0 with JSON."""
    req = _read_request()
    if req is None:
        _failure(_REASON_PROVIDER_ERROR)
        return 0
    normalized, reason = _validate_request(req)
    if normalized is None or reason is not None:
        _failure(reason or _REASON_PROVIDER_ERROR)
        return 0
    try:
        payload = _run_alignment(normalized)
    except ValueError as exc:
        # WhisperX's own unsupported-language error is a ValueError. Classify
        # by the documented, credential-free message WITHOUT leaking it.
        lowered = str(exc).lower()
        reason = (
            _REASON_UNSUPPORTED_LANGUAGE
            if "align-model" in lowered or "language" in lowered
            else _REASON_PROVIDER_ERROR
        )
        _failure(reason)
        return 0
    except Exception:  # noqa: BLE001 — never leak a traceback / path / secret
        _failure(_REASON_PROVIDER_ERROR)
        return 0
    _emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
