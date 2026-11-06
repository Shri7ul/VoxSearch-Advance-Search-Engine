"""Transcript input contract (Goal 09).

Defines the single, explicit, validated shape of transcript data accepted by
the ingestion pipeline. This is a *library-level* contract — there are no HTTP
routes, ASR calls, or caption downloads here. Callers (future workers / ASR /
upload adapters) construct :class:`TranscriptInput` from authorized data.

Design decisions
----------------
* **Pydantic 2** ``BaseModel`` with ``extra="forbid"``: unknown fields produce a
  clear ``ValidationError`` instead of being silently discarded (explicit
  "no silent repair / no silent discard" policy).
* **Timestamps are strict**: a ``field_validator(mode="before")`` accepts only
  real ``int``/``float`` (rejects strings, bools, ``None``) and rejects NaN /
  +/-Inf via :func:`math.isfinite`. Nothing is coerced or repaired.
* Unit = **seconds** (float), matching the existing DB columns
  ``start_seconds`` / ``end_seconds`` and ``Video.duration_seconds``. (The
  architecture doc sketches integer milliseconds for a future normalized
  format; the persisted schema here is in seconds.)
* **Original text is preserved verbatim**: the contract only *rejects* blank
  text; it never trims or rewrites it. Trimming / whitespace collapsing belongs
  to the separate normalization layer, which returns a distinct value. The
  ingestion service persists BOTH: the verbatim original in ``original_text``
  and the normalized form in the existing ``"text"`` column.
* ``speaker`` and ``metadata`` are OPTIONAL input fields (justified by the
  architecture's ``speaker_id`` and ASR quality/config metadata, §7.2/§7.4).
  Persistence policy: the current ``transcript_segments`` table has NO columns
  for them, so they are accepted and surfaced in the ingestion result but NOT
  persisted. Persisting them would require a schema migration, explicitly out
  of scope for this foundation goal (see the ingestion service docstring).

Compliance
----------
:class:`TranscriptSource` is a CLOSED set mapping 1:1 to the four authorized
source priorities in the architecture (§7.1). There is intentionally NO
``youtube_captions`` / scraping source: VoxSearch does not claim arbitrary
public YouTube captions are accessible and does not circumvent access
restrictions.
"""
from __future__ import annotations

import math
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TranscriptSource(str, Enum):
    """Explicit, closed set of permitted transcript provenance values.

    * ``creator_provided``    — creator-authorized caption / transcript source.
    * ``licensed_transcript`` — third-party licensed transcript provider.
    * ``user_uploaded``       — transcript supplied by a user with rights.
    * ``asr_generated``       — ASR on audio the system is authorized to process.
    """

    CREATOR_PROVIDED = "creator_provided"
    LICENSED_TRANSCRIPT = "licensed_transcript"
    USER_UPLOADED = "user_uploaded"
    ASR_GENERATED = "asr_generated"


def _finite_float(value: Any, *, field: str) -> float:
    """Strict numeric validation: no coercion, no NaN/Inf, no bool, no repair."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field} must be numeric (int or float); got {type(value).__name__}."
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{field} must be finite; NaN and Infinity are not allowed."
        )
    return result


class TranscriptSegmentInput(BaseModel):
    """One timestamped spoken-content segment.

    Fields
    ------
    start_time, end_time:
        Offsets in seconds. Numeric, finite, ``start_time >= 0``,
        ``end_time > start_time``.
    text:
        Original transcript text, preserved verbatim; must be non-blank after
        trimming (but is NOT trimmed here).
    speaker, metadata:
        Optional, accepted but NOT persisted (no schema columns exist).
    """

    model_config = ConfigDict(extra="forbid")

    start_time: float
    end_time: float
    text: str
    speaker: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _times(cls, value: Any, info) -> float:
        result = _finite_float(value, field=info.field_name)
        if info.field_name == "start_time" and result < 0:
            raise ValueError("start_time must be >= 0")
        return result

    @field_validator("text")
    @classmethod
    def _text_nonblank(cls, value: str) -> str:
        # Do NOT strip: the original text (including surrounding whitespace) is
        # preserved verbatim. We only reject text that is entirely whitespace.
        if not value or not value.strip():
            raise ValueError("segment text must not be empty after trimming")
        return value

    @model_validator(mode="after")
    def _end_after_start(self) -> TranscriptSegmentInput:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        return self


class TranscriptInput(BaseModel):
    """A validated transcript payload for ONE video / ONE transcript asset."""

    model_config = ConfigDict(extra="forbid")

    video_id: uuid.UUID
    content_right_id: uuid.UUID
    source_type: TranscriptSource
    language: str = Field(min_length=1)
    segments: list[TranscriptSegmentInput] = Field(min_length=1)

    @field_validator("language")
    @classmethod
    def _language_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("language must not be blank")
        return value


__all__ = [
    "TranscriptInput",
    "TranscriptSegmentInput",
    "TranscriptSource",
]
