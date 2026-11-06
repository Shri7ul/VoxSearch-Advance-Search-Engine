"""Deterministic, multilingual sentence segmentation (Goal 10).

Purpose & scope
---------------
Goal 09 produced validated, timestamped transcript *segments* with both verbatim
and normalized text persisted. Goal 10 adds a **library-level** layer that splits
one segment's source text into ordered sentence units, preserving source
provenance and character offsets. Its output is an INPUT for the future
sentence-aware chunking / semantic-indexing pipeline (architecture §8, §23 step
2). This module does NOT chunk, embed, align, rerank, search, download, or run
workers. It is a PURE function: no DB, no I/O, no clock, no credentials, no
external NLP service, and no ML model.

Timestamp precision strategy (critical)
---------------------------------------
Goal 09 stores **segment-level** timestamps only. A sentence splitter must NOT
fabricate precise sentence timestamps from character positions or sentence
lengths. Therefore every :class:`Sentence` carries:

* the parent segment reference (``source_segment_ref``),
* the parent's start/end timestamps VERBATIM
  (``source_segment_start_seconds`` / ``source_segment_end_seconds``), and
* an explicit ``timestamp_precision = TimestampPrecision.SEGMENT_LEVEL``.

There are NO ``sentence_start_seconds`` / ``sentence_end_seconds`` fields: the
inherited window is never presented as an exact sentence boundary. Text offsets
(``char_start`` / ``char_end``) and temporal offsets are deliberately SEPARATE
concepts. ``exact_start_seconds`` / ``exact_end_seconds`` are reserved, ALWAYS
``None`` in this goal, and can only be populated by a FUTURE word-level
alignment goal (architecture §7.5) that upgrades the precision to
``WORD_ALIGNED``. Nothing here interpolates, scales, or proportionally divides
time by character count.

Text preservation
-----------------
* Segmentation operates on the ORIGINAL source string. Offsets are Python code
  point indices into that untransformed string — NEVER into a normalized one.
* ``Sentence.text`` is exactly ``source_text[char_start:char_end]``. Punctuation,
  Bangla Unicode, English technical terms, casing, and whitespace are preserved;
  no lowercasing, no punctuation removal, no rewriting.
* ``normalized_text`` is a SEPARATE derived value produced by the EXISTING
  :func:`app.transcripts.normalization.normalize_text` (reused, not duplicated).
  It never replaces ``text`` and never affects the offsets.
* Spans TILE the source: ``spans[0].char_start == 0``,
  ``spans[-1].char_end == len(source_text)``, and each span starts exactly where
  the previous ended. Hence ``''.join(s.text for s in sentences) == source_text``
  — no character is silently lost and no sentence silently disappears.

Segmentation behaviour / ambiguous boundaries
---------------------------------------------
* Sentence-ending punctuation: English ``. ? !`` and Bangla danda ``।`` (plus
  double danda ``॥``). Mixed punctuation in one transcript is handled uniformly.
* A boundary is recognised only when the punctuation is in a token's TRAILING
  run. Periods INSIDE a token (``3.14``, ``example.com``, ``https://x.y``,
  ``Dr.Rahman``, ``v1.2.3``) are never candidates — this is the deterministic
  rule that avoids "split every period blindly".
* Closing quotes/brackets after a boundary belong to the SAME sentence
  (``He said "Hi." Then left.``).
* Repeated punctuation (``!!!``, ``?!``, ``...``) is ONE boundary; a
  standalone punctuation-only token creates NO boundary (no empty sentence).
* Known abbreviations, dotted acronyms (``U.S.``, ``e.g.``, ``Ph.D.``), and
  single-letter initials in any script (``J.``, ``ড.``) are NOT boundaries.
  Consequence: an abbreviation that genuinely ends a sentence causes
  UNDER-splitting (a documented trade-off, never a silent text loss).
* Text without terminal punctuation becomes a SINGLE sentence spanning to the
  end. Whitespace between sentences attaches to the FOLLOWING sentence;
  trailing whitespace merges into the last sentence. Sentences are never
  reordered.
* Empty / whitespace-only input is REJECTED with :class:`EmptySourceTextError`
  (explicit policy, mirroring Goal 09's blank-text rejection).

Known limitations
-----------------
Perfect sentence segmentation is NOT guaranteed. Rule-based detection cannot
resolve every ambiguity: an abbreviation at a true sentence end under-splits;
missing whitespace after a period (``Hello.World``) under-splits; ellipsis used
as a pause (``Wait... what?``) over-splits; quoted/direct-speech boundaries and
language-specific orthography are heuristic. There is NO language detection,
NO ML model, and NO external NLP service. Only authorized transcript content
should be passed to this layer.

No persistence / no migration
-----------------------------
The architecture (§9.1) lists ``transcript_segments`` as the timestamped
sentence/segment table and ``transcript_words`` as OPTIONAL word-level
alignment; it does NOT define a dedicated sentence table. Sentence records are
therefore NOT persisted in this goal, no new table is created, and the Goal 09
ingestion transaction is left untouched. Persisting sentences (or adding exact
timestamps) belongs to a later, separately-approved goal.
"""
from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.transcripts.contracts import _finite_float
from app.transcripts.normalization import normalize_text


# --- Public vocabulary -------------------------------------------------------


class TimestampPrecision(str, Enum):
    """How much temporal meaning a sentence's timestamps carry.

    * ``SEGMENT_LEVEL`` — the ONLY value emitted in Goal 10. The sentence
      inherits its parent segment's start/end timestamps verbatim; these are
      NOT exact sentence timestamps. Marking this explicitly stops downstream
      code from mistaking an inherited window for a measured sentence boundary.
    * ``WORD_ALIGNED`` — RESERVED for a future goal. Once word-level alignment
      exists, exact sentence start/end can be derived from word timestamps and
      the precision upgraded. The deterministic splitter NEVER emits it.
    """

    SEGMENT_LEVEL = "segment_level"
    WORD_ALIGNED = "word_aligned"


class EmptySourceTextError(ValueError):
    """Raised for empty / whitespace-only source text (explicit policy).

    Blank input is REJECTED, not silently turned into an empty sentence list.
    This mirrors Goal 09's ``TranscriptSegmentInput`` and keeps the
    "no silent discard" rule: a caller must explicitly handle blank input.
    """


@dataclass(frozen=True)
class SentenceSpan:
    """A half-open ``[char_start, char_end)`` slice of the ORIGINAL source text.

    ``text`` is exactly ``source_text[char_start:char_end]``. Offsets are
    Python string indices (code points) into the untransformed input.
    """

    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class Sentence:
    """One sentence unit with full source provenance.

    ``text`` is verbatim; ``ordinal`` is a stable 0-based index in source order;
    ``char_start`` / ``char_end`` index the ORIGINAL source string;
    ``source_segment_ref`` identifies the parent transcript segment;
    ``source_segment_start_seconds`` / ``source_segment_end_seconds`` are the
    parent's segment-level timestamps (NOT exact sentence times);
    ``timestamp_precision`` records that fact; ``exact_start_seconds`` /
    ``exact_end_seconds`` are reserved for future word alignment and are ALWAYS
    ``None`` here; ``normalized_text`` is a separate derived value.
    """

    text: str
    ordinal: int
    char_start: int
    char_end: int
    source_segment_ref: str
    source_segment_start_seconds: float
    source_segment_end_seconds: float
    language: str
    timestamp_precision: TimestampPrecision
    normalized_text: str
    exact_start_seconds: float | None = None
    exact_end_seconds: float | None = None


class SentenceSegmentationInput(BaseModel):
    """Input contract for segmenting ONE source transcript segment.

    Reuses Goal 09's ``_finite_float`` (numeric validation is NOT duplicated)
    and ``ConfigDict(extra="forbid")`` (unknown fields are rejected, not
    discarded). ``source_text`` is preserved verbatim; only blank text is
    rejected. Timestamps are in SECONDS (float), matching the persisted
    ``start_seconds`` / ``end_seconds`` columns.
    """

    model_config = ConfigDict(extra="forbid")

    source_text: str
    source_segment_ref: str = Field(min_length=1)
    start_seconds: float
    end_seconds: float
    language: str = Field(min_length=1)

    @field_validator("source_text")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source_text must not be blank")
        return value  # verbatim; NEVER trimmed

    @field_validator("source_segment_ref", "language")
    @classmethod
    def _nonblank_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("start_seconds", "end_seconds", mode="before")
    @classmethod
    def _times(cls, value: Any, info) -> float:
        return _finite_float(value, field=info.field_name)

    @model_validator(mode="after")
    def _order(self) -> SentenceSegmentationInput:
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be >= 0")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


# --- Deterministic boundary engine -------------------------------------------


# Sentence-ending punctuation: English (. ? !) and Bangla danda (।) / double
# danda (॥). Bangla also uses ? and !, which are already included.
_BOUNDARY_CHARS = frozenset(".?!।॥")

# Quotes / brackets that may wrap a sentence. A boundary followed by a closing
# one stays in the SAME sentence (e.g. ``He said "Hi." Then left.``).
_OPENING = frozenset("\"'“‘([{«‹（「『〈【")
_CLOSING = frozenset("\"'”’)]}»›）」』〉】")
_QUOTE_STRIP = "".join(_OPENING | _CLOSING)  # str.strip() argument
_TRAILING_CHARS = _BOUNDARY_CHARS | _CLOSING  # membership test

# Dotted acronyms whose trailing period is NOT a sentence end: U.S., e.g.,
# i.e., Ph.D., B.Sc. Groups are 1-3 letters so real domains (example.com) stay
# OUT of this rule and can still terminate a genuine sentence.
_ACRONYM = re.compile(r"(?:[A-Za-z]{1,3}\.){2,}")

# Common abbreviations (case-insensitive), including English abbreviations that
# commonly appear inside Bangla sentences (Dr., Prof., etc.).
_ABBREVIATIONS = frozenset({
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "st.", "mt.",
    "vs.", "v.", "etc.", "fig.", "no.", "inc.", "ltd.", "co.", "corp.",
    "dept.", "univ.", "approx.", "est.", "cf.", "al.", "resp.", "ibid.",
    "e.g.", "i.e.", "ph.d.", "b.sc.", "m.sc.", "u.s.", "u.k.", "u.n.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.",
    "sept.", "oct.", "nov.", "dec.", "a.m.", "p.m.", "ft.", "vol.",
    "chap.", "ed.", "eds.", "p.", "pp.", "sec.", "min.", "max.", "sq.",
})


def _iter_tokens(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` for each maximal non-whitespace run.

    ``str.isspace()`` is Unicode-aware, so newlines, tabs, NBSP, and other
    Unicode spaces separate tokens. Whitespace between tokens is preserved in
    the source and belongs to the FOLLOWING sentence span.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i + 1
        while j < n and not text[j].isspace():
            j += 1
        yield i, j
        i = j


def _is_false_boundary(token: str) -> bool:
    """True when a trailing period belongs to an abbreviation / acronym.

    Only periods are ambiguous; ``?`` / ``!`` / danda are always strong
    boundaries, so this check is deliberately period-oriented.
    """
    bare = token.strip(_QUOTE_STRIP)
    if bare.lower() in _ABBREVIATIONS:
        return True
    if _ACRONYM.fullmatch(bare) is not None:
        return True
    # Single-letter initial in ANY script (J., A., Bangla ড., ...).
    if len(bare) == 2 and bare[0].isalpha() and bare[1] == ".":
        return True
    return False


def _boundary_cuts(text: str) -> list[int]:
    """Return absolute, strictly increasing cut positions (exclusive ends).

    A cut is placed at the end of a token whose TRAILING run contains boundary
    punctuation. Periods *inside* a token (``3.14``, ``example.com``,
    ``https://x.y``, ``Dr.Rahman``, ``v1.2.3``) are never candidates — the
    deterministic rule that avoids "split every period blindly".
    """
    cuts: list[int] = []
    for start, end in _iter_tokens(text):
        token = text[start:end]
        k = end
        while k > start and text[k - 1] in _TRAILING_CHARS:
            k -= 1
        trailing = text[k:end]
        if not any(ch in _BOUNDARY_CHARS for ch in trailing):
            continue
        core = text[start:k].strip(_QUOTE_STRIP)
        if not core:
            # Pure-punctuation token (a standalone "..."): no content to end a
            # sentence, so it does not create a boundary or an empty sentence.
            continue
        if _is_false_boundary(token):
            continue
        cuts.append(end)
    return cuts


def find_sentence_spans(source_text: str) -> tuple[SentenceSpan, ...]:
    """Pure, deterministic sentence-span detection (independently testable).

    Spans TILE the entire source string: ``spans[0].char_start == 0``,
    ``spans[-1].char_end == len(source_text)``, and each span starts exactly
    where the previous ended. Therefore no character is dropped, punctuation /
    Unicode / casing are preserved, and every ``span.text`` equals
    ``source_text[span.char_start:span.char_end]``.

    Whitespace between sentences is attached to the FOLLOWING span (the next
    sentence may carry leading whitespace); trailing whitespace is merged into
    the last span so no whitespace-only "sentence" is fabricated.

    Raises
    ------
    EmptySourceTextError
        For empty or whitespace-only input (explicit rejection policy).
    """
    if not source_text or not source_text.strip():
        raise EmptySourceTextError(
            "source text is empty or whitespace-only; no sentences to segment "
            "(explicit rejection, not a silent empty result)."
        )

    spans: list[tuple[int, int]] = []
    start = 0
    for cut in _boundary_cuts(source_text):
        if cut > start:
            spans.append((start, cut))
            start = cut

    if start < len(source_text):
        tail_blank = not source_text[start:].strip()
        if tail_blank and spans:
            # Trailing whitespace belongs to the last sentence (no phantom row).
            spans[-1] = (spans[-1][0], len(source_text))
        else:
            spans.append((start, len(source_text)))

    return tuple(
        SentenceSpan(char_start=s, char_end=e, text=source_text[s:e])
        for s, e in spans
    )


def segment_sentences(
    payload: SentenceSegmentationInput | Mapping[str, Any],
) -> tuple[Sentence, ...]:
    """Segment ONE source transcript segment into ordered :class:`Sentence` objects.

    Accepts an already-built contract or a raw mapping (validated here via
    Pydantic, mirroring ``TranscriptIngestionService.ingest``). PURE library
    function: no DB, no I/O, no credentials, no clock, no persistence.
    """
    contract = (
        payload
        if isinstance(payload, SentenceSegmentationInput)
        else SentenceSegmentationInput.model_validate(payload)
    )
    spans = find_sentence_spans(contract.source_text)
    return tuple(
        Sentence(
            text=span.text,
            ordinal=ordinal,
            char_start=span.char_start,
            char_end=span.char_end,
            source_segment_ref=contract.source_segment_ref,
            source_segment_start_seconds=contract.start_seconds,
            source_segment_end_seconds=contract.end_seconds,
            language=contract.language,
            timestamp_precision=TimestampPrecision.SEGMENT_LEVEL,
            normalized_text=normalize_text(span.text).normalized,
            exact_start_seconds=None,
            exact_end_seconds=None,
        )
        for ordinal, span in enumerate(spans)
    )


__all__ = [
    "EmptySourceTextError",
    "Sentence",
    "SentenceSegmentationInput",
    "SentenceSpan",
    "TimestampPrecision",
    "find_sentence_spans",
    "segment_sentences",
]
