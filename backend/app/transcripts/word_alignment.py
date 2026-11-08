"""Deterministic word-alignment domain logic (Goal 12 — Timestamp preservation).

PURE layer: no DB, no ORM, no torch / whisperx, no I/O, no clock, no
credentials. It sits between the WordAlignmentService (orchestration +
persistence) and the isolated provider worker (WhisperX). It owns the
fail-closed rules that guarantee NO fabricated / interpolated / clamped
timestamps ever become persisted word rows.

Provider protocol (JSON-shaped, deliberately version-agnostic)
--------------------------------------------------------------
A provider callable receives an :class:`AlignmentRequest` and returns a
``Mapping`` shaped as::

    {
      "segments": [
        {
          "segment_id": "<uuid str>",
          "words": [
            {"word": "Hello",                 # token text, exact-match required
             "start": 1.0, "end": 1.4,       # audio-timeline SECONDS (finite)
             "score": 0.9,                    # optional
             "char_start": 1.02,             # corroborating char timing (audio s)
             "char_end": 1.38,
             "char_corroborated": true},     # REQUIRED True, else fail closed
            ...
          ],
          "error": null                       # or a closed reason string
        },
        ...
      ],
      "provider_error": null                 # or a closed reason string
    }

The isolated worker builds ``char_start``/``char_end``/``char_corroborated``
from WhisperX ``return_char_alignments=True`` output via
:func:`reconstruct_word_chars` (the exact ``v3.8.6`` char dicts omit
``word-idx``, so per-word char evidence must be reconstructed deterministically
or fail closed). Missing word/char timings (``interpolate_method="ignore"``)
stay ``None`` and are REJECTED here — never backfilled or interpolated.

Strict ordinal mapping
----------------------
The ORIGINAL token stream of a segment (from ``original_text``) must map 1:1,
in order, onto the provider word list: equal count and exact text per position.
Any mismatch skips the ENTIRE segment (zero words persisted), never a partial
or re-indexed subset.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.transcripts.word_contract import (
    AlignedWord,
    REASON_CHAR_CORROBORATION_FAILED,
    REASON_EMPTY_ORIGINAL_TEXT,
    REASON_INVALID_WINDOW,
    REASON_NON_MONOTONIC_TIMING,
    REASON_NULL_ORIGINAL_TEXT,
    REASON_OUTSIDE_AUDIO_WINDOW,
    REASON_PROVIDER_SEGMENT_ERROR,
    REASON_TOKEN_COUNT_MISMATCH,
    REASON_TOKEN_TEXT_MISMATCH,
    REASON_UNSUPPORTED_LANGUAGE,
    REASON_WORD_OUTSIDE_WINDOW,
    REASON_WORD_TIMING_INVALID,
    REASON_WORD_TIMING_MISSING,
    Token,
    ms_to_seconds,
    seconds_to_ms,
    tokenize_original,
    validate_provider_second,
)


# --- Value objects -----------------------------------------------------------

@dataclass(frozen=True)
class SegmentData:
    """A transcript segment VIEW used to prepare alignment (pure, DB-free)."""

    segment_id: uuid.UUID
    segment_ref: str
    start_seconds: float
    end_seconds: float
    original_text: str | None


@dataclass(frozen=True)
class PreparedSegment:
    """A segment cleared for alignment, with computed audio/video windows."""

    segment_id: uuid.UUID
    segment_ref: str
    original_text: str
    tokens: tuple[Token, ...]
    video_start_ms: int
    video_end_ms: int
    audio_start_ms: int
    audio_end_ms: int


@dataclass(frozen=True)
class SkippedSegment:
    """A segment deliberately NOT aligned, with a closed, honest reason."""

    segment_ref: str
    reason: str


@dataclass(frozen=True)
class SegmentPreparation:
    prepared: tuple[PreparedSegment, ...]
    skipped: tuple[SkippedSegment, ...]


@dataclass(frozen=True)
class AlignmentSegmentTask:
    """One segment as sent to the provider (audio-timeline timestamps)."""

    segment_id: str
    text: str                 # verbatim original_text (never normalized)
    start: float              # audio-timeline seconds
    end: float                # audio-timeline seconds
    tokens: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AlignmentRequest:
    """The complete, bounded request handed to the alignment provider.

    ``interpolate_method="ignore"`` and ``return_char_alignments=True`` are the
    MANDATED ``whisperx==3.8.6`` adapter settings — the default
    ``interpolate_method="nearest"`` MUST NOT be used because it backfills NaN
    timestamps with interpolated values that this goal forbids persisting.
    """

    language: str
    segments: tuple[AlignmentSegmentTask, ...]
    interpolate_method: str = "ignore"
    return_char_alignments: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form (UUIDs as strings) for the isolated worker."""
        return {
            "language": self.language,
            "interpolate_method": self.interpolate_method,
            "return_char_alignments": self.return_char_alignments,
            "segments": [
                {
                    "segment_id": task.segment_id,
                    "text": task.text,
                    "start": task.start,
                    "end": task.end,
                    "tokens": list(task.tokens),
                }
                for task in self.segments
            ],
        }


@dataclass(frozen=True)
class ValidatedSegment:
    """A prepared segment whose words all passed strict validation."""

    prepared: PreparedSegment
    words: tuple[AlignedWord, ...]


# --- Phase A preparation (pure) ----------------------------------------------

def prepare_segments(
    segments: Iterable[SegmentData],
    *,
    timeline_offset_ms: int,
    audio_duration_ms: int,
) -> SegmentPreparation:
    """Compute audio/video windows and tokenize each segment's original text.

    A segment is SKIPPED (never sent to the provider) with a closed reason when:

    * ``original_text IS NULL`` (never fabricate an original from normalized
      text — approved decision),
    * tokenization yields no tokens,
    * the segment's video timestamps cannot be converted / are inverted,
    * its audio-timeline window ``[video_ms - offset, ...]`` falls outside the
      available audio window ``[0, audio_duration_ms]``.

    These are fail-closed decisions: nothing is shifted, clamped, or guessed.
    """
    if isinstance(timeline_offset_ms, bool) or not isinstance(timeline_offset_ms, int):
        raise ValueError("timeline_offset_ms must be an int")
    if isinstance(audio_duration_ms, bool) or not isinstance(audio_duration_ms, int):
        raise ValueError("audio_duration_ms must be an int")
    if audio_duration_ms < 0:
        raise ValueError("audio_duration_ms must be >= 0")

    prepared: list[PreparedSegment] = []
    skipped: list[SkippedSegment] = []
    for seg in segments:
        if seg.original_text is None:
            skipped.append(SkippedSegment(seg.segment_ref, REASON_NULL_ORIGINAL_TEXT))
            continue
        tokens = tokenize_original(seg.original_text)
        if not tokens:
            skipped.append(SkippedSegment(seg.segment_ref, REASON_EMPTY_ORIGINAL_TEXT))
            continue
        try:
            video_start_ms = seconds_to_ms(seg.start_seconds)
            video_end_ms = seconds_to_ms(seg.end_seconds)
        except ValueError:
            skipped.append(SkippedSegment(seg.segment_ref, REASON_INVALID_WINDOW))
            continue
        if video_end_ms <= video_start_ms:
            skipped.append(SkippedSegment(seg.segment_ref, REASON_INVALID_WINDOW))
            continue

        audio_start_ms = video_start_ms - timeline_offset_ms
        audio_end_ms = video_end_ms - timeline_offset_ms
        if audio_start_ms < 0 or audio_end_ms > audio_duration_ms:
            skipped.append(
                SkippedSegment(seg.segment_ref, REASON_OUTSIDE_AUDIO_WINDOW)
            )
            continue

        prepared.append(
            PreparedSegment(
                segment_id=seg.segment_id,
                segment_ref=seg.segment_ref,
                original_text=seg.original_text,
                tokens=tokens,
                video_start_ms=video_start_ms,
                video_end_ms=video_end_ms,
                audio_start_ms=audio_start_ms,
                audio_end_ms=audio_end_ms,
            )
        )
    return SegmentPreparation(tuple(prepared), tuple(skipped))


def build_request(language: str, preparation: SegmentPreparation) -> AlignmentRequest:
    """Build the provider request from prepared segments.

    WhisperX receives AUDIO-timeline timestamps (video ms - offset, converted
    back to seconds) and the VERBATIM original text. The mandated adapter flags
    are pinned here so no caller can silently enable interpolation.
    """
    tasks: list[AlignmentSegmentTask] = []
    for seg in preparation.prepared:
        tasks.append(
            AlignmentSegmentTask(
                segment_id=str(seg.segment_id),
                text=seg.original_text,
                start=ms_to_seconds(seg.audio_start_ms),
                end=ms_to_seconds(seg.audio_end_ms),
                tokens=tuple(
                    {
                        "text": t.text,
                        "char_start": t.char_start,
                        "char_end": t.char_end,
                        "ordinal": t.ordinal,
                    }
                    for t in seg.tokens
                ),
            )
        )
    return AlignmentRequest(
        language=language,
        segments=tuple(tasks),
        interpolate_method="ignore",
        return_char_alignments=True,
    )


# --- Phase B validation (pure, fail closed) ----------------------------------

def _classify_language_error(text: str) -> bool:
    lowered = text.lower()
    return "align-model" in lowered or "language" in lowered


def validate_provider_segment(
    prepared: PreparedSegment,
    provider_segment: Mapping[str, Any] | None,
    *,
    timeline_offset_ms: int,
) -> tuple[AlignedWord, ...] | SkippedSegment:
    """Validate ONE provider segment against its prepared original tokens.

    Returns the ordered :class:`AlignedWord` tuple (VIDEO-timeline ms) on full
    success, or a :class:`SkippedSegment` with a closed reason. On ANY word
    failure the WHOLE segment is skipped with ZERO words — never a partial set.
    """
    if provider_segment is None:
        return SkippedSegment(prepared.segment_ref, REASON_PROVIDER_SEGMENT_ERROR)

    error = provider_segment.get("error")
    if error:
        reason = (
            REASON_UNSUPPORTED_LANGUAGE
            if _classify_language_error(str(error))
            else REASON_PROVIDER_SEGMENT_ERROR
        )
        return SkippedSegment(prepared.segment_ref, reason)

    words = provider_segment.get("words")
    if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
        return SkippedSegment(prepared.segment_ref, REASON_TOKEN_COUNT_MISMATCH)

    # 1. Strict ordinal mapping: equal count, exact token text per position.
    if len(words) != len(prepared.tokens):
        return SkippedSegment(prepared.segment_ref, REASON_TOKEN_COUNT_MISMATCH)

    aligned: list[AlignedWord] = []
    prev_start_ms = -1
    for i, raw in enumerate(words):
        token = prepared.tokens[i]
        if not isinstance(raw, Mapping):
            return SkippedSegment(prepared.segment_ref, REASON_TOKEN_TEXT_MISMATCH)
        if raw.get("word") != token.text:
            return SkippedSegment(prepared.segment_ref, REASON_TOKEN_TEXT_MISMATCH)

        # 2. Both word timings MUST be present and finite (no interpolation).
        if "start" not in raw or "end" not in raw:
            return SkippedSegment(prepared.segment_ref, REASON_WORD_TIMING_MISSING)
        start_s = validate_provider_second(raw.get("start"))
        end_s = validate_provider_second(raw.get("end"))
        if start_s is None or end_s is None or end_s <= start_s:
            return SkippedSegment(prepared.segment_ref, REASON_WORD_TIMING_INVALID)

        # 3. Char-level corroboration (worker reconstructed from v3.8.6 chars).
        if raw.get("char_corroborated") is not True:
            return SkippedSegment(
                prepared.segment_ref, REASON_CHAR_CORROBORATION_FAILED
            )
        char_start_s = validate_provider_second(raw.get("char_start"))
        char_end_s = validate_provider_second(raw.get("char_end"))
        if (
            char_start_s is None
            or char_end_s is None
            or char_start_s < start_s
            or char_end_s > end_s
            or char_start_s > char_end_s
        ):
            return SkippedSegment(
                prepared.segment_ref, REASON_CHAR_CORROBORATION_FAILED
            )

        # 4. Convert audio-timeline seconds -> integer ms; enforce the segment's
        #    audio window with NO clamping/guessing.
        start_ms = seconds_to_ms(start_s)
        end_ms = seconds_to_ms(end_s)
        if start_ms >= end_ms:
            return SkippedSegment(prepared.segment_ref, REASON_WORD_TIMING_INVALID)
        if (
            start_ms < prepared.audio_start_ms
            or end_ms > prepared.audio_end_ms
        ):
            return SkippedSegment(prepared.segment_ref, REASON_WORD_OUTSIDE_WINDOW)
        if start_ms < prev_start_ms:
            return SkippedSegment(
                prepared.segment_ref, REASON_NON_MONOTONIC_TIMING
            )
        prev_start_ms = start_ms

        # 5. Persist in VIDEO timeline (audio ms + offset). Negative video ms
        #    is impossible for a valid clip; fail closed rather than clamp.
        video_start_ms = start_ms + timeline_offset_ms
        video_end_ms = end_ms + timeline_offset_ms
        if video_start_ms < 0 or video_end_ms <= video_start_ms:
            return SkippedSegment(prepared.segment_ref, REASON_WORD_OUTSIDE_WINDOW)

        aligned.append(
            AlignedWord(
                segment_id=prepared.segment_id,
                ordinal=token.ordinal,
                text=token.text,
                char_start=token.char_start,
                char_end=token.char_end,
                start_ms=video_start_ms,
                end_ms=video_end_ms,
            )
        )
    return tuple(aligned)


# --- Deterministic char reconstruction (for the isolated worker) ------------

def reconstruct_word_chars(
    words: Sequence[Mapping[str, Any]],
    segment_chars: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct per-word char corroboration from WhisperX ``v3.8.6`` chars.

    ``v3.8.6`` final char dicts contain ``char`` and OPTIONAL ``start``/``end``/
    ``score`` but NO ``word-idx``. Therefore per-word char evidence must be
    rebuilt deterministically:

    * Flatten all NON-SPACE characters (with their dict's timing) in order.
      WhisperX inserts space characters between words; those separators carry
      no word timing and are ignored here.
    * For each word, consume exactly ``len(non_space_chars_of_word_text)`` flat
      chars; they MUST spell the word's non-space characters exactly. Any
      mismatch, shortage, or ambiguity fails closed for that word.
    * Among the consumed chars, use ONLY those with REAL (finite) timings to
      compute ``char_start`` = min start, ``char_end`` = max end. If NONE have
      real timings, corroboration FAILS — no backfill / interpolation / ffill.

    The returned list (one entry per input word) carries ``char_start`` /
    ``char_end`` (audio-timeline seconds) and ``char_corroborated`` (bool) and
    is meant to be merged into the provider word dicts by the worker.

    NOTE (outstanding provider-integration verification): the exact runtime
    shape of ``v3.8.6`` char dicts cannot be confirmed here without loading the
    real model / weights. This reconstruction documents the deterministic
    assumption and fails closed when it does not hold; live verification is a
    separately-approved step.
    """
    flat: list[tuple[str, float | None, float | None]] = []
    for entry in segment_chars:
        if not isinstance(entry, Mapping):
            continue
        text = entry.get("char")
        if not isinstance(text, str):
            continue
        c_start = validate_provider_second(entry.get("start"))
        c_end = validate_provider_second(entry.get("end"))
        for ch in text:
            if ch.isspace():
                continue
            flat.append((ch, c_start, c_end))

    result: list[dict[str, Any]] = []
    cursor = 0
    for word in words:
        word_text = word.get("word") if isinstance(word, Mapping) else None
        target = (
            "".join(c for c in word_text if not c.isspace())
            if isinstance(word_text, str)
            else ""
        )
        n = len(target)
        chunk = flat[cursor : cursor + n] if n else []
        cursor += n

        corroborated = (
            n > 0
            and len(chunk) == n
            and "".join(c for c, _, _ in chunk) == target
        )
        real_starts = [s for _, s, _ in chunk if s is not None]
        real_ends = [e for _, _, e in chunk if e is not None]
        if corroborated and real_starts and real_ends:
            result.append(
                {
                    "char_start": min(real_starts),
                    "char_end": max(real_ends),
                    "char_corroborated": True,
                }
            )
        else:
            # Fail closed: no corroboration, no fabricated timing.
            result.append(
                {
                    "char_start": None,
                    "char_end": None,
                    "char_corroborated": False,
                }
            )
    return result


__all__ = [
    "AlignmentRequest",
    "AlignmentSegmentTask",
    "PreparedSegment",
    "SegmentData",
    "SegmentPreparation",
    "SkippedSegment",
    "ValidatedSegment",
    "build_request",
    "prepare_segments",
    "reconstruct_word_chars",
    "validate_provider_segment",
]
