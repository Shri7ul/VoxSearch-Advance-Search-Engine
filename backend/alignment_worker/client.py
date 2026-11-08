"""Stdlib process launcher for the isolated alignment worker (Goal 12).

:class:`SubprocessAlignmentProvider` satisfies the service's
``ProviderCallable`` protocol: it takes a request dict (from
:meth:`app.transcripts.word_alignment.AlignmentRequest.to_dict`) and returns the
service-shaped result mapping. It launches ``python -m alignment_worker.worker``
in a SEPARATE process using an EXPLICIT argv list and ``shell=False``, with:

* bounded JSON on stdin (``MAX_REQUEST_BYTES``) and bounded stdout
  (``MAX_RESPONSE_BYTES``),
* a hard timeout,
* a MINIMAL, controlled environment built from a small allowlist — the parent
  process's ``DATABASE_URL`` / Supabase / secret env vars are NEVER forwarded,
* ``stderr=DEVNULL`` and NO logging of the audio path / request / response, so
  no secret or signed URL can leak into logs.

It imports NO torch / whisperx. It normalizes the worker's RAW WhisperX output
into the service protocol by merging per-word char corroboration via the pure
:func:`app.transcripts.word_alignment.reconstruct_word_chars` (a single shared
implementation — the worker itself stays free of any :mod:`app` import).

Live execution is opt-in and OUTSTANDING: this goal installs no WhisperX,
downloads no model weights, and fetches no audio, so the default test suite
uses a FAKE provider instead of this launcher.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from typing import Any

from app.transcripts.word_alignment import reconstruct_word_chars

MAX_REQUEST_BYTES = 2_000_000
MAX_RESPONSE_BYTES = 8_000_000
DEFAULT_TIMEOUT_SECONDS = 600.0

# Only these env var NAMES are forwarded to the worker. Anything credential-like
# (DATABASE_URL, SUPABASE_*, *_KEY, *_TOKEN, *_SECRET, ...) is deliberately NOT
# in this list and therefore cannot reach the isolated process.
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

_REASON_PROVIDER_ERROR = "provider_error"
_REASON_UNSUPPORTED_LANGUAGE = "unsupported_language"

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _controlled_env() -> dict[str, str]:
    """Build a minimal env: allowlisted names + PYTHONPATH for the worker pkg."""
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        _BACKEND_DIR
        if not inherited
        else _BACKEND_DIR + os.pathsep + inherited
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _error(reason: str = _REASON_PROVIDER_ERROR) -> dict[str, Any]:
    return {"segments": [], "provider_error": reason}


class SubprocessAlignmentProvider:
    """Launch the isolated worker and normalize its RAW output.

    Parameters
    ----------
    worker_python:
        Path to the OPTIONAL isolated interpreter that has
        ``requirements-alignment.txt`` installed. Passed as argv[0] verbatim
        (explicit, no shell).
    audio_path:
        LOCAL path to the authorized audio file (resolved from the opaque
        ``storage_reference`` by the operator). Never a URL; never logged.
    timeout_seconds:
        Hard wall-clock cap on the subprocess.
    """

    def __init__(
        self,
        *,
        worker_python: str,
        audio_path: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._worker_python = worker_python
        self._audio_path = audio_path
        self._timeout = timeout_seconds

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        payload["audio_path"] = self._audio_path
        payload.setdefault("device", "cpu")

        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError):
            return _error()
        if len(data) > MAX_REQUEST_BYTES:
            return _error()

        argv = [self._worker_python, "-m", "alignment_worker.worker"]
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
        except (subprocess.TimeoutExpired, OSError):
            return _error()

        if completed.returncode != 0:
            return _error()
        stdout = completed.stdout or b""
        if len(stdout) > MAX_RESPONSE_BYTES:
            return _error()
        try:
            raw = json.loads(stdout.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return _error()
        if not isinstance(raw, dict):
            return _error()
        return self._normalize(raw)

    @staticmethod
    def _normalize(raw: Mapping[str, Any]) -> dict[str, Any]:
        """Turn RAW worker output into the service's provider-result protocol.

        For each segment, reconstruct per-word char corroboration from the
        ``chars`` array and merge ``char_start`` / ``char_end`` /
        ``char_corroborated`` into each word dict. Word timings themselves are
        passed through VERBATIM (missing keys stay missing -> the service
        rejects them). A ``provider_error`` is propagated unchanged.
        """
        provider_error = raw.get("provider_error")
        if provider_error:
            text = str(provider_error)
            reason = (
                _REASON_UNSUPPORTED_LANGUAGE
                if "language" in text.lower() or "align-model" in text.lower()
                else _REASON_PROVIDER_ERROR
            )
            return _error(reason)

        segments = raw.get("segments")
        if not isinstance(segments, list):
            return _error()

        out_segments: list[dict[str, Any]] = []
        for seg in segments:
            if not isinstance(seg, Mapping):
                continue
            words = seg.get("words")
            chars = seg.get("chars")
            words_list = words if isinstance(words, list) else []
            chars_list = chars if isinstance(chars, list) else []
            evidence = reconstruct_word_chars(words_list, chars_list)

            merged: list[dict[str, Any]] = []
            for word, extra in zip(words_list, evidence):
                item = dict(word) if isinstance(word, Mapping) else {}
                item["char_start"] = extra.get("char_start")
                item["char_end"] = extra.get("char_end")
                item["char_corroborated"] = extra.get(
                    "char_corroborated", False
                )
                merged.append(item)
            out_segments.append(
                {
                    "segment_id": seg.get("segment_id"),
                    "words": merged,
                    "error": seg.get("error"),
                }
            )
        return {"segments": out_segments, "provider_error": None}


__all__ = ["SubprocessAlignmentProvider"]
