"""Cross-segment / contextual transcript validation (Goal 09).

Responsibilities are deliberately split:

* **Structural** validation (field types, finite timestamps, ``start >= 0``,
  ``end > start``, non-blank text, closed source set, ``>= 1`` segment, no
  unknown fields) lives in the Pydantic contract (:mod:`app.transcripts.contracts`).
* **Cross-segment / contextual** validation lives HERE because it needs the
  whole segment list and/or external context (the known video duration).

Overlap policy (examined against the alignment requirements)
-----------------------------------------------------------
The architecture (§7, "Word-level timestamp alignment" via a validated
aligner such as WhisperX) and ASR / creator caption exports commonly produce
**overlapping** windows: a segment's ``end_time`` may exceed the next segment's
``start_time`` (speaker turns, word-level timings, or sentence windows that
intentionally overlap to preserve context). Rejecting such overlap would
discard legitimate alignment data, so overlapping segments are **PRESERVED
verbatim** — no trimming, no merging, no timestamp repair, no silent repair.

What IS required is a **strictly increasing** ``start_time`` ("chronologically
ordered"). Strictly increasing (not merely non-decreasing) is deliberate: the
existing ``uq_transcript_segments_asset_start`` unique constraint forbids two
segments of the same asset sharing a start offset, so equal starts cannot be
persisted. Overlap between segments with DISTINCT starts is allowed.

No silent repair
----------------
Invalid data raises :class:`TranscriptValidationError` with the offending
0-based segment index and a clear, credential-free message. Nothing is dropped,
reordered, clamped, or rewritten. Validation never mutates the payload.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.transcripts.contracts import TranscriptInput


class TranscriptValidationError(ValueError):
    """A clear, credential-free, index-aware transcript validation failure."""


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of contextual validation (no mutation of the payload).

    ``overlapping_pair_count`` records how many adjacent pairs overlap. It is
    informational: overlap is ALLOWED and preserved, never rejected. Surfacing
    the count makes the documented policy explicit and testable.
    """

    segment_count: int
    overlapping_pair_count: int
    max_end_time: float


def validate_transcript(
    payload: TranscriptInput,
    *,
    video_duration_seconds: int | float | None = None,
) -> ValidationReport:
    """Validate chronological ordering and the optional known-duration bound.

    Parameters
    ----------
    payload:
        A structurally valid :class:`TranscriptInput`.
    video_duration_seconds:
        The persisted ``Video.duration_seconds`` (seconds). When ``None``
        (unknown duration) the duration bound is skipped. When known, EVERY
        segment's ``end_time`` must be ``<=`` it.

    Raises
    ------
    TranscriptValidationError
        On non-chronological / duplicate starts, a negative known duration, or
        a segment end exceeding the known duration.
    """
    segments = payload.segments  # >= 1 is guaranteed by the contract
    overlapping = 0
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        cur = segments[i]
        if cur.start_time <= prev.start_time:
            raise TranscriptValidationError(
                f"segment {i} start_time ({cur.start_time}) must be strictly "
                f"greater than segment {i - 1} start_time ({prev.start_time}); "
                "segments must be chronologically ordered with distinct starts."
            )
        if prev.end_time > cur.start_time:
            overlapping += 1  # allowed and preserved; documented policy

    max_end = max(seg.end_time for seg in segments)

    if video_duration_seconds is not None:
        if video_duration_seconds < 0:
            raise TranscriptValidationError(
                "known video duration must be >= 0."
            )
        bound = float(video_duration_seconds)
        for i, seg in enumerate(segments):
            if seg.end_time > bound:
                raise TranscriptValidationError(
                    f"segment {i} end_time ({seg.end_time}) exceeds the known "
                    f"video duration ({video_duration_seconds}s)."
                )

    return ValidationReport(
        segment_count=len(segments),
        overlapping_pair_count=overlapping,
        max_end_time=max_end,
    )


__all__ = [
    "TranscriptValidationError",
    "ValidationReport",
    "validate_transcript",
]
