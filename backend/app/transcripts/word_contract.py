"""Word-level alignment contract primitives (Goal 12 — Timestamp preservation).

This module is the PURE, dependency-light foundation shared by the alignment
orchestration (:mod:`app.transcripts.word_alignment`), the sentence-timing
derivation (:mod:`app.transcripts.sentence_timestamps`), the persistence layer,
and the isolated provider worker. It deliberately contains:

* NO database access, NO ORM, NO SQLAlchemy,
* NO torch / whisperx imports (the core API runtime must stay free of them),
* NO clock, NO I/O, NO credentials, NO network,
* NO interpolation, NO backfill, NO proportional / character-offset timing.

Units (critical, approved decision #6/#7)
-----------------------------------------
Existing ``transcript_segments`` timestamps are SECONDS (float) and their
contracts are UNCHANGED. New Goal 12 word timestamps are stored as INTEGER
MILLISECONDS. The two units meet ONLY at the explicit conversion boundaries
below. Integer milliseconds are a REPRESENTATION CHOICE, not a claim of 1 ms
measurement accuracy: converting a seconds float to ``int`` quantizes it.

Timeline (approved decision: ``timeline_offset_ms``)
----------------------------------------------------
The authorized audio asset may be a clip whose own timeline is offset from the
video timeline. ``video_ms = audio_ms + timeline_offset_ms`` and, conversely,
``audio_ms = video_ms - timeline_offset_ms``. WhisperX receives audio-timeline
timestamps; persisted word timestamps are in the VIDEO timeline. Nothing here
shifts, clamps, or guesses a timestamp — those are service-layer, fail-closed
decisions.

Language gate (approved decisions #1/#2/#10)
--------------------------------------------
WhisperX wav2vec2 forced alignment is the single primary provider. The language
allowlist below is transcribed from the EXACT ``whisperx==3.8.6`` model maps
(``DEFAULT_ALIGN_MODELS_TORCH`` + ``DEFAULT_ALIGN_MODELS_HF``). Bengali
(``bn`` / ``ben`` and regional variants) is NOT supported in this goal and is
rejected with zero provider calls; it remains ``SEGMENT_LEVEL``.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.transcripts.contracts import _finite_float

MS_PER_SECOND = Decimal(1000)

_CHECKSUM_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# The pinned, isolated alignment subsystem (requirements-alignment.txt). These
# constants are recorded on every alignment run; the injected FAKE provider used
# by the default test suite does NOT change them (they describe the CONFIGURED
# provider/version, not the test double). Live whisperx==3.8.6 execution is
# opt-in / outstanding.
ALIGNMENT_PROVIDER = "whisperx"
ALIGNMENT_PROVIDER_VERSION = "3.8.6"


class SourceAudioAssetInput(BaseModel):
    """Validated input contract for registering an AUTHORIZED source audio asset.

    Spec-compliant field set (design-freeze §3.1 / full task §5.A / §7):

    * ``storage_uri``: OPAQUE, NON-SECRET object reference (bucket/key or
      internal handle). Deliberately NOT a URL and never carrying credentials
      or signed-URL query strings (which would leak into logs / the isolated
      worker boundary).
    * ``checksum_sha256``: 64 lowercase hex chars binding the row to exact
      bytes (mirrors the DB CHECK ``checksum_sha256 ~ '^[0-9a-f]{64}$'``).
    * ``mime_type`` / ``rights_basis``: format + authorization provenance.
    * ``sample_rate_hz``: OPTIONAL positive int (NULL = unknown, never guessed).
    * ``duration_ms``: length of the AVAILABLE audio window (audio-timeline
      ``[0, duration_ms]``) used to reject segments outside it. The DB column is
      nullable per spec, but THIS authorized provisioning path requires a known
      positive window so alignment can fail closed on out-of-window segments
      (full task §7) rather than guess.
    * ``timeline_offset_ms``: approved audio->video offset (``video_ms =
      audio_ms + offset``); may be negative; a finite integer.
    """

    model_config = ConfigDict(extra="forbid")

    video_id: uuid.UUID
    content_right_id: uuid.UUID
    storage_uri: str = Field(min_length=1)
    checksum_sha256: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    rights_basis: str = Field(min_length=1)
    # Strict numeric fields: an input contract must NOT silently coerce a
    # bool / str / float into a count/offset. Strict int validation rejects
    # those at the type level so the guards below (and the fail-closed window
    # math) see only real integers.
    sample_rate_hz: int | None = Field(default=None, strict=True)
    timeline_offset_ms: int = Field(strict=True)
    duration_ms: int = Field(strict=True)

    @field_validator("storage_uri", "mime_type", "rights_basis")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be blank")
        return value

    @field_validator("checksum_sha256")
    @classmethod
    def _checksum_is_sha256_hex(cls, value: str) -> str:
        if _CHECKSUM_SHA256_RE.fullmatch(value) is None:
            raise ValueError(
                "checksum_sha256 must be exactly 64 lowercase hex characters"
            )
        return value

    @field_validator("sample_rate_hz")
    @classmethod
    def _positive_sample_rate(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("sample_rate_hz must be a positive int or None")
        return value

    @field_validator("timeline_offset_ms")
    @classmethod
    def _finite_offset(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("timeline_offset_ms must be an int")
        return value

    @field_validator("duration_ms")
    @classmethod
    def _nonnegative_duration(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("duration_ms must be an int")
        if value < 0:
            raise ValueError("duration_ms must be >= 0")
        return value

    @model_validator(mode="after")
    def _window(self) -> "SourceAudioAssetInput":
        # The audio window in the audio timeline is [0, duration_ms]; fail
        # closed if it is empty (nothing could be aligned inside it).
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be > 0 (empty audio window)")
        return self

# --- Alignment-run status vocabulary (approved closed set) -------------------
# Stored as TEXT with a CHECK constraint (see app.db.models and the Goal 12
# migration). ``pending`` is reserved for a future queue that has not yet picked
# up an attempt; the orchestrator in this goal records an attempt as RUNNING.
PENDING_STATUS = "pending"
RUNNING_STATUS = "running"
COMPLETED_STATUS = "completed"
PARTIAL_STATUS = "partial"
FAILED_STATUS = "failed"
UNSUPPORTED_LANGUAGE_STATUS = "unsupported_language"
SUPERSEDED_STATUS = "superseded"

ALIGNMENT_RUN_STATUSES = (
    PENDING_STATUS,
    RUNNING_STATUS,
    COMPLETED_STATUS,
    PARTIAL_STATUS,
    FAILED_STATUS,
    UNSUPPORTED_LANGUAGE_STATUS,
    SUPERSEDED_STATUS,
)

# Statuses that carry a valid, current word set. Backs the partial UNIQUE index
# that permits at most ONE current successful run per
# (transcript_asset_id, source_audio_asset_id).
CURRENT_SUCCESS_STATUSES = (COMPLETED_STATUS, PARTIAL_STATUS)

# Terminal failure / skip reasons (closed, credential-free).
REASON_RIGHTS_REVOKED = "rights_revoked"
REASON_PROVIDER_ERROR = "provider_error"
REASON_UNSUPPORTED_LANGUAGE = "unsupported_language"
REASON_ALL_SEGMENTS_SKIPPED = "all_segments_skipped"
REASON_NO_ALIGNABLE_SEGMENTS = "no_alignable_segments"
REASON_SUPERSEDED_BY_CONCURRENT_RUN = "superseded_by_concurrent_run"

REASON_NULL_ORIGINAL_TEXT = "null_original_text"
REASON_EMPTY_ORIGINAL_TEXT = "empty_original_text"
REASON_OUTSIDE_AUDIO_WINDOW = "outside_audio_window"
REASON_INVALID_WINDOW = "invalid_window"
REASON_PROVIDER_SEGMENT_ERROR = "provider_segment_error"
REASON_TOKEN_COUNT_MISMATCH = "token_count_mismatch"
REASON_TOKEN_TEXT_MISMATCH = "token_text_mismatch"
REASON_WORD_TIMING_MISSING = "word_timing_missing"
REASON_WORD_TIMING_INVALID = "word_timing_invalid"
REASON_WORD_OUTSIDE_WINDOW = "word_outside_window"
REASON_NON_MONOTONIC_TIMING = "non_monotonic_timing"
REASON_CHAR_CORROBORATION_FAILED = "char_corroboration_failed"

# --- WhisperX 3.8.6 language allowlist ---------------------------------------
# DEFAULT_ALIGN_MODELS_TORCH: en, fr, de, es, it
# DEFAULT_ALIGN_MODELS_HF: ja, zh, nl, uk, pt, ar, cs, ru, pl, hu, fi, fa, el,
# tr, da, he, vi, ko, ur, te, hi, ca, ml, no, nn, sk, sl, hr, ro, eu, gl, ka,
# lv, tl, sv, id
SUPPORTED_ALIGNMENT_LANGUAGES = frozenset({
    "en", "fr", "de", "es", "it",
    "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi", "fa",
    "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml", "no",
    "nn", "sk", "sl", "hr", "ro", "eu", "gl", "ka", "lv", "tl", "sv", "id",
})

# Explicit Bengali / Bengali-regional codes rejected in THIS goal even though a
# future goal may add a Bengali alignment model + evaluation dataset.
BENGALI_LANGUAGE_CODES = frozenset({"bn", "ben", "bn-bd", "bn-in"})


def is_alignment_language_supported(language: str) -> bool:
    """Return ``True`` only for an EXACT WhisperX 3.8.6 supported code.

    Fail closed: anything unknown, region-tagged, blank, or unsupported returns
    ``False``. Comparison is case-insensitive over the stripped code. This is a
    pure function — no provider call, no model load.
    """
    if not isinstance(language, str):
        return False
    return language.strip().lower() in SUPPORTED_ALIGNMENT_LANGUAGES


def language_gate_reason(language: str) -> str | None:
    """Return ``None`` when supported, else a closed, credential-free reason.

    Distinguishes the explicitly-deferred Bengali case from generic unsupported
    / unknown codes so callers can report honestly without guessing.
    """
    if not isinstance(language, str) or not language.strip():
        return "invalid_language"
    normalized = language.strip().lower()
    if normalized in SUPPORTED_ALIGNMENT_LANGUAGES:
        return None
    if normalized in BENGALI_LANGUAGE_CODES:
        return REASON_UNSUPPORTED_LANGUAGE  # Bengali: deferred (decision #2/#10)
    return "unsupported_language"


# --- Strict, deterministic unit conversion -----------------------------------

def seconds_to_ms(seconds: float) -> int:
    """Convert a finite, non-negative seconds value to integer milliseconds.

    Uses ``Decimal(str(seconds))`` and ``ROUND_HALF_UP`` so binary-float
    artifacts (e.g. 1.2345) are quantized deterministically, WITHOUT relying on
    ``round()``'s banker's rounding. Rejects (raises ``ValueError``):

    * bool, str, ``None``, or any non int/float,
    * NaN / +/-Infinity,
    * negative values (timestamps must be >= 0),
    * values that overflow / cannot be represented.

    This is a REPRESENTATION conversion only; it does NOT add measurement
    accuracy. The existing seconds-based contracts are unchanged.
    """
    value = _finite_float(seconds, field="seconds")  # strict: no bool/str/NaN/Inf
    if value < 0:
        raise ValueError("seconds must be >= 0 (negative timestamps rejected)")
    try:
        scaled = Decimal(str(value)) * MS_PER_SECOND
        result = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, OverflowError) as exc:  # pragma: no cover
        raise ValueError("seconds cannot be converted to milliseconds") from exc
    if result < 0:
        raise ValueError("converted milliseconds must be >= 0")
    return result


def ms_to_seconds(ms: int) -> float:
    """Convert a finite, non-negative integer millisecond value to seconds.

    Strict: rejects bool, str, float, ``None``, negatives. Only real ``int``
    milliseconds are accepted. Round-tripping ``ms_to_seconds(seconds_to_ms(x))``
    reflects the integer quantization (it is NOT guaranteed to equal ``x``).
    """
    if isinstance(ms, bool) or not isinstance(ms, int):
        raise ValueError(
            f"ms must be an int; got {type(ms).__name__}."
        )
    if ms < 0:
        raise ValueError("ms must be >= 0 (negative timestamps rejected)")
    return float(Decimal(ms) / MS_PER_SECOND)


# --- Deterministic original-text tokenizer -----------------------------------
# Matches Goal 10's ``_iter_tokens`` maximal non-whitespace semantics EXACTLY.
# ``str.isspace()`` is Unicode-aware, so newlines/tabs/NBSP separate tokens and
# whitespace between tokens is preserved in the source (it belongs to the
# FOLLOWING token span). Offsets are Python code-point indices into the
# VERBATIM ``original_text`` — never into a normalized string.

@dataclass(frozen=True)
class Token:
    """One maximal non-whitespace token of the verbatim original text."""

    text: str
    char_start: int
    char_end: int
    ordinal: int


def tokenize_original(original_text: str) -> tuple[Token, ...]:
    """Tokenize ``original_text`` into ordered :class:`Token` values.

    Pure and deterministic. Offsets index the untransformed input; ``text`` is
    exactly ``original_text[char_start:char_end]``. Empty / whitespace-only
    input yields an empty tuple (the caller decides to skip such a segment).
    """
    if not isinstance(original_text, str):
        raise ValueError("original_text must be a str")
    tokens: list[Token] = []
    i, n = 0, len(original_text)
    ordinal = 0
    while i < n:
        if original_text[i].isspace():
            i += 1
            continue
        j = i + 1
        while j < n and not original_text[j].isspace():
            j += 1
        tokens.append(
            Token(
                text=original_text[i:j],
                char_start=i,
                char_end=j,
                ordinal=ordinal,
            )
        )
        ordinal += 1
        i = j
    return tuple(tokens)


# --- A fully validated, video-timeline aligned word ---------------------------

@dataclass(frozen=True)
class AlignedWord:
    """One validated word with VIDEO-timeline integer-ms timestamps.

    ``ordinal`` is the 0-based position in the segment's original token stream;
    ``char_start`` / ``char_end`` index the verbatim ``original_text``;
    ``start_ms`` / ``end_ms`` are in the VIDEO timeline (audio ms +
    ``timeline_offset_ms``). Only words that passed strict validation, window,
    monotonicity, and char-level corroboration ever become ``AlignedWord``.
    """

    segment_id: uuid.UUID
    ordinal: int
    text: str
    char_start: int
    char_end: int
    start_ms: int
    end_ms: int


# --- Low-level numeric validators ---------------------------------------------

def is_finite_number(value: Any) -> bool:
    """True only for a real, finite int/float (not bool, not NaN/Inf)."""
    return _is_finite_number(value)


def _is_finite_number(value: Any) -> bool:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def validate_provider_second(value: Any) -> float | None:
    """Return a validated audio-timeline seconds value, or ``None``.

    A provider word/char timing is accepted only when it is a real, finite,
    non-negative number. Missing/NaN/Inf/bool/str/negative all return ``None``
    (fail closed) — this is what prevents interpolated or backfilled values
    (WhisperX ``interpolate_method=\"ignore\"`` leaves such timings as NaN ->
    omitted keys -> ``None`` here) from ever being persisted.
    """
    if not _is_finite_number(value):
        return None
    result = float(value)
    if result < 0:
        return None
    return result


__all__ = [
    "ALIGNMENT_RUN_STATUSES",
    "AlignedWord",
    "BENGALI_LANGUAGE_CODES",
    "COMPLETED_STATUS",
    "CURRENT_SUCCESS_STATUSES",
    "FAILED_STATUS",
    "MS_PER_SECOND",
    "PARTIAL_STATUS",
    "PENDING_STATUS",
    "REASON_ALL_SEGMENTS_SKIPPED",
    "REASON_CHAR_CORROBORATION_FAILED",
    "REASON_EMPTY_ORIGINAL_TEXT",
    "REASON_INVALID_WINDOW",
    "REASON_NO_ALIGNABLE_SEGMENTS",
    "REASON_NON_MONOTONIC_TIMING",
    "REASON_NULL_ORIGINAL_TEXT",
    "REASON_OUTSIDE_AUDIO_WINDOW",
    "REASON_PROVIDER_ERROR",
    "REASON_PROVIDER_SEGMENT_ERROR",
    "REASON_RIGHTS_REVOKED",
    "REASON_SUPERSEDED_BY_CONCURRENT_RUN",
    "REASON_TOKEN_COUNT_MISMATCH",
    "REASON_TOKEN_TEXT_MISMATCH",
    "REASON_UNSUPPORTED_LANGUAGE",
    "REASON_WORD_OUTSIDE_WINDOW",
    "REASON_WORD_TIMING_INVALID",
    "REASON_WORD_TIMING_MISSING",
    "RUNNING_STATUS",
    "SUPERSEDED_STATUS",
    "SUPPORTED_ALIGNMENT_LANGUAGES",
    "SourceAudioAssetInput",
    "Token",
    "UNSUPPORTED_LANGUAGE_STATUS",
    "is_alignment_language_supported",
    "is_finite_number",
    "language_gate_reason",
    "ms_to_seconds",
    "seconds_to_ms",
    "tokenize_original",
    "validate_provider_second",
]
