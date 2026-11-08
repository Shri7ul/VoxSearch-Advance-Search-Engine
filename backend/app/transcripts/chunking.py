"""Deterministic, sentence-aware transcript chunking (Goal 13).

Purpose & scope
---------------
Goal 13 groups consecutive, already-segmented sentences into **retrieval
chunks**. It is a PURE library-level component: no DB, no I/O, no clock, no
credentials, no network, no ML model, no embeddings, no rewrite of source text.
The ONLY input is a sequence of :class:`~app.transcripts.sentence_timestamps.AlignedSentence`
— the Goal 12 composition of Goal 10 :class:`~app.transcripts.sentence_segmentation.Sentence`
objects and validated aligned words (see
:meth:`app.services.word_alignment_service.WordAlignmentService.get_aligned_sentences`).
Goal 13 does NOT run alignment, does NOT build a second timestamp contract, and
does NOT populate Goal 10's ``exact_start_seconds`` / ``exact_end_seconds``.
An OPTIONAL :class:`ChunkParentContext` may carry REAL ``video_id`` /
``transcript_asset_id`` / ``content_right_id`` from the existing Goal 09/12
parent context (never fabricated); :func:`build_production_chunks` requires it.

Timestamp precision (critical, approved design)
----------------------------------------------
An ``AlignedSentence`` is either:

* ``WORD_ALIGNED`` — exact, integer-ms, VIDEO-timeline ``exact_start_ms`` /
  ``exact_end_ms`` are present and trustworthy; or
* ``SEGMENT_LEVEL`` — only the parent segment's seconds window is known; the
  exact fields are ``None``. Multiple sentences from one segment SHARE that
  window; the window is NOT an exact sentence boundary.

Therefore:

* A chunk's precision is the **minimum** (weakest) precision across ALL its
  sentences. A chunk NEVER claims ``WORD_ALIGNED`` if any sentence is coarse.
* A chunk's temporal range is ``[min(reliable start), max(reliable end)]`` over
  its sentences — the union of the RELIABLE source windows. It is NEVER a
  proportional split, interpolation, or character-offset estimate.
* When ANY sentence is ``SEGMENT_LEVEL`` (a "coarse" chunk), the 30s / 10s /
  60s / 5s DURATION thresholds **cannot be measured accurately** (the parent
  window does not grow as more of its sentences are added). The algorithm then
  falls back to WORD COUNT as the reliable size signal, records the
  ``coarse_timestamp_precision`` diagnostic, and treats duration targets as
  best-effort ONLY. Duration thresholds are enforced precisely ONLY for a fully
  ``WORD_ALIGNED`` chunk. This implements the mandated "do not fabricate timing"
  / "document where thresholds cannot be enforced" rules.

Boundary-selection policy (deterministic, documented verbatim)
-------------------------------------------------------------
Starting at the next unassigned (evidence) sentence ``i``:

1. Append consecutive sentences; track cumulative word count ``w`` and reliable
   extent duration ``d`` (ms). Track whether the group is fully ``WORD_ALIGNED``.
2. A single sentence that (by itself) exceeds ``target_max_words`` (or, when
   fully exact, ``max_duration``) is kept INTACT as a one-sentence chunk
   (``single_sentence_exceeds_limits``) — never split, never truncated.
3. While accumulating (before the target is reached), if adding the next
   sentence would push a FULLY EXACT group beyond ``max_duration`` (60s), stop
   at the current boundary (``max_duration``). Word count has no hard cap
   (150 is a target, not a limit); exceeding it is recorded as a diagnostic.
4. The target is reached at the FIRST boundary where ``w >= target_min_words``
   (80) OR (the group is fully exact AND ``d >= target_duration`` (30s)).
5. At that boundary, a ONE-STEP lookahead decides whether to also include the
   next sentence: it is included only when it (a) keeps a fully-exact group
   fully exact, (b) stays within ``target_max_words`` and ``max_duration``, and
   (c) lowers a combined "distance to targets" score (closeness to 30s AND the
   80-150 word band centre). Ties stop at the earlier boundary. Result reason
   ``target_reached``.
6. The final group is the chunk's EVIDENCE. The next chunk starts at the first
   sentence AFTER that group. Step 5's lookahead is the ONLY permitted lookahead;
   otherwise accumulation is greedy and strictly forward.

Overlap policy (deterministic, documented verbatim)
--------------------------------------------------
Consecutive chunks share ~5s of overlap, built from COMPLETE sentences only:

* Overlap = a SUFFIX of the previous chunk's evidence sentences, re-tagged
  ``role="context"`` in the next chunk. Evidence sentences are disjoint and each
  appears as evidence in exactly one chunk. Context is NEVER presented as new
  evidence (evidence vs context separation).
* Under FULLY EXACT precision, candidate suffixes are the trailing 1..N
  sentences of the previous evidence group. Each suffix's overlap duration is
  its reliable extent; the suffix CLOSEST to ``overlap`` (5s) is chosen (ties ->
  SHORTER suffix). A suffix whose duration would exceed ``max_duration`` is not
  eligible; if none is eligible overlap is omitted (``overlap_unavailable``).
* Under COARSE precision, 5s CANNOT be measured (inherited parent windows are
  not sentence durations and do not shrink), so the honest, bounded fallback is
  the SMALLEST complete-sentence overlap — ONE trailing sentence — flagged
  ``overlap_coarse_precision``.
* When context + evidence would push a FULLY EXACT chunk beyond
  ``max_duration``, leading context sentences are trimmed (``overlap_trimmed``):
  overlap NEVER inflates a chunk past the maximum.
* Overlap reuses the SAME immutable source data; it never mutates or duplicates
  a source record, and preserves global sentence indices, IDs, and timestamps.

Provenance & honesty
--------------------
Each sentence ref preserves verbatim text, language, Goal 10 ordinal/char
offsets, parent segment ref + verbatim seconds window, per-sentence precision,
and exact ms (only when word-aligned). Chunk text is the VERBATIM concatenation
of its sentences (context first, then evidence) — no character is lost.
``source_segment_refs`` are the ordered, de-duplicated distinct segment refs of
the EVIDENCE sentences.

REAL parent identity is propagated, never invented: a frozen, strictly
validated :class:`ChunkParentContext` carries the genuine ``video_id`` /
``transcript_asset_id`` (and the real ``content_right_id`` only when already
present) from the existing Goal 09 ``IngestionResult`` / Goal 12
``TranscriptAlignmentRun`` parent context, and :func:`build_production_chunks`
stamps them onto EVERY production chunk. A stable ``sentence_ref`` is COMPOSED
deterministically from immutable ``source_segment_ref`` + ``ordinal`` +
``char_start`` + ``char_end`` — NO sentence UUID is fabricated. There is no
speaker identity and no persisted chunk identity: chunks have a 0-based
``index`` (sequence/order) but no database id — Goal 13 is library-level, adds
NO table, NO migration, and NO embeddings. The pure :func:`chunk_sentences`
keeps parent context OPTIONAL for backward compatibility; only
:func:`build_production_chunks` REQUIRES it, so a production chunk can never
silently omit ownership provenance.

Edge cases
----------
Empty input -> empty tuple. Repeated sentences, Unicode/Bangla punctuation,
URLs/decimals/dotted acronyms are already resolved by Goal 10 segmentation and
are grouped verbatim here. Duplicate source refs are de-duplicated in provenance
only. Non-monotonic / invalid timestamp input is REJECTED (never repaired) via
:func:`validate_aligned_sentences`.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.transcripts.contracts import _finite_float
from app.transcripts.sentence_segmentation import TimestampPrecision
from app.transcripts.sentence_timestamps import AlignedSentence
from app.transcripts.word_contract import (
    ms_to_seconds,
    seconds_to_ms,
    tokenize_original,
)

# --- Closed diagnostic / reason vocabulary (credential-free) -----------------

REASON_TARGET_REACHED = "target_reached"
REASON_MAX_DURATION = "max_duration"
REASON_END_OF_INPUT = "end_of_input"
REASON_SINGLE_SENTENCE = "single_sentence_exceeds_limits"

DIAG_COARSE_PRECISION = "coarse_timestamp_precision"
DIAG_MIN_DURATION_UNMET = "min_duration_unmet"
DIAG_SINGLE_EXCEEDS_MAX_DURATION = "single_sentence_exceeds_max_duration"
DIAG_SINGLE_EXCEEDS_MAX_WORDS = "single_sentence_exceeds_max_words"
DIAG_TARGET_MAX_WORDS_EXCEEDED = "target_max_words_exceeded"
DIAG_TARGET_MIN_WORDS_UNMET = "target_min_words_unmet"
DIAG_OVERLAP_UNAVAILABLE = "overlap_unavailable"
DIAG_OVERLAP_COARSE = "overlap_coarse_precision"
DIAG_OVERLAP_TRIMMED = "overlap_trimmed_for_max_duration"

ROLE_EVIDENCE = "evidence"
ROLE_CONTEXT = "context"

_VALID_PRECISIONS = (
    TimestampPrecision.SEGMENT_LEVEL,
    TimestampPrecision.WORD_ALIGNED,
)


class InvalidChunkInputError(ValueError):
    """Clear, index-aware, credential-free rejection of invalid chunk input.

    Raised instead of silently repairing, clamping, sorting, or dropping bad
    timestamp / precision data (the "no silent repair" rule).
    """


# --- Explicit, testable configuration ----------------------------------------


class ChunkingConfig(BaseModel):
    """Validated, immutable chunking parameters (single source of truth).

    All durations are in seconds (finite, > 0). No magic numbers are scattered
    through the algorithm — every threshold flows from here. Deterministic for a
    given config + input.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_duration_seconds: float = 30.0
    min_duration_seconds: float = 10.0
    max_duration_seconds: float = 60.0
    overlap_seconds: float = 5.0
    target_min_words: int = Field(default=80, strict=True, ge=1)
    target_max_words: int = Field(default=150, strict=True, ge=1)

    @field_validator(
        "target_duration_seconds",
        "min_duration_seconds",
        "max_duration_seconds",
        "overlap_seconds",
        mode="before",
    )
    @classmethod
    def _durations(cls, value: Any, info) -> float:
        # Reuse Goal 09's strict numeric validator: rejects bool / str / NaN /
        # +/-Inf and never coerces. Durations must be strictly positive.
        result = _finite_float(value, field=info.field_name)
        if result <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return result

    @model_validator(mode="after")
    def _order(self) -> "ChunkingConfig":
        if not (
            self.min_duration_seconds
            <= self.target_duration_seconds
            <= self.max_duration_seconds
        ):
            raise ValueError(
                "require min_duration_seconds <= target_duration_seconds "
                "<= max_duration_seconds"
            )
        if self.overlap_seconds >= self.max_duration_seconds:
            raise ValueError("overlap_seconds must be < max_duration_seconds")
        if self.target_min_words > self.target_max_words:
            raise ValueError("target_min_words must be <= target_max_words")
        return self


# --- Real parent provenance (never fabricated) -------------------------------


class ChunkParentContext(BaseModel):
    """Frozen, strictly validated REAL ownership provenance for chunking.

    Carries the genuine ``video_id`` / ``transcript_asset_id`` (and the real
    ``content_right_id`` ONLY when already present) from the existing Goal 09
    ``IngestionResult`` / Goal 12 ``TranscriptAlignmentRun`` parent context.
    All IDs are strict :class:`uuid.UUID` values: strings / ints / bools /
    unknown fields are REJECTED, never coerced or fabricated. No new rights
    lookup and no change to the rights architecture.

    Use :meth:`from_alignment_run` / :meth:`from_ingestion_result` to read the
    real fields verbatim from those parent objects without importing them into
    this pure module.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: uuid.UUID = Field(strict=True)
    transcript_asset_id: uuid.UUID = Field(strict=True)
    content_right_id: uuid.UUID | None = Field(default=None, strict=True)

    @classmethod
    def from_alignment_run(cls, run: Any) -> "ChunkParentContext":
        """Read real IDs from a Goal 12 ``TranscriptAlignmentRun`` row.

        ``video_id`` / ``transcript_asset_id`` / ``content_right_id`` are real
        persisted columns. ``content_right_id`` is read with a ``None`` default
        so it is propagated ONLY when present — there is NO new rights lookup.
        """
        return cls(
            video_id=run.video_id,
            transcript_asset_id=run.transcript_asset_id,
            content_right_id=getattr(run, "content_right_id", None),
        )

    @classmethod
    def from_ingestion_result(cls, result: Any) -> "ChunkParentContext":
        """Read real IDs from a Goal 09 ``IngestionResult``.

        ``result.asset_id`` IS the transcript asset id (the created
        ``TranscriptAsset.id``); it is mapped verbatim, NOT derived. Same real
        ``content_right_id`` handling as above.
        """
        return cls(
            video_id=result.video_id,
            transcript_asset_id=result.asset_id,
            content_right_id=getattr(result, "content_right_id", None),
        )


# --- Output data contract -----------------------------------------------------


@dataclass(frozen=True)
class ChunkSentence:
    """One sentence's preserved provenance inside a chunk (never rewritten)."""

    index: int  # global 0-based index in the input sequence
    role: str  # ROLE_EVIDENCE | ROLE_CONTEXT
    text: str  # verbatim sentence text
    language: str
    ordinal: int  # Goal 10 ordinal within its source segment
    char_start: int
    char_end: int
    source_segment_ref: str
    timestamp_precision: TimestampPrecision  # per-sentence authoritative precision
    source_segment_start_seconds: float  # verbatim parent window (seconds)
    source_segment_end_seconds: float
    exact_start_ms: int | None  # present only when WORD_ALIGNED
    exact_end_ms: int | None
    word_count: int  # tokens in sentence.text (reliable, always available)

    @property
    def sentence_ref(self) -> str:
        """Stable, deterministic reference built from immutable provenance.

        COMPOSED from ``source_segment_ref`` + Goal 10 ``ordinal`` +
        ``char_start`` + ``char_end`` — NO sentence UUID is invented. Because
        overlap/context occurrences are built from the SAME immutable sentence
        data (see ``_ref``), this value is identical whether a sentence appears
        as evidence or as overlap context in another chunk.
        """
        return (
            f"{self.source_segment_ref}:{self.ordinal}:"
            f"{self.char_start}:{self.char_end}"
        )


@dataclass(frozen=True)
class Chunk:
    """One deterministic retrieval chunk (library-level, NOT persisted)."""

    index: int  # 0-based chunk sequence/order
    text: str  # verbatim concatenation of all sentences (context + evidence)
    sentences: tuple[ChunkSentence, ...]
    start_ms: int  # reliable union extent start (integer ms, video timeline)
    end_ms: int
    duration_ms: int
    timestamp_precision: TimestampPrecision  # MIN precision across all sentences
    word_count: int  # words across ALL sentences (context + evidence)
    evidence_start: int  # index into `sentences` where evidence begins
    source_segment_refs: tuple[str, ...]  # ordered, de-duplicated (EVIDENCE only)
    boundary_reason: str
    diagnostics: tuple[str, ...]
    overlap_sentence_indices: tuple[int, ...]  # global indices reused as context

    # REAL parent provenance. ``None`` only on the pure optional-parent path;
    # build_production_chunks guarantees these are set for production chunks.
    # Never fabricated: they come from a validated ChunkParentContext.
    video_id: uuid.UUID | None = None
    transcript_asset_id: uuid.UUID | None = None
    content_right_id: uuid.UUID | None = None

    @property
    def start_seconds(self) -> float:
        return ms_to_seconds(self.start_ms)

    @property
    def end_seconds(self) -> float:
        return ms_to_seconds(self.end_ms)

    @property
    def duration_seconds(self) -> float:
        return ms_to_seconds(self.duration_ms)

    @property
    def has_context(self) -> bool:
        return self.evidence_start > 0

    @property
    def context_sentences(self) -> tuple[ChunkSentence, ...]:
        return self.sentences[: self.evidence_start]

    @property
    def evidence_sentences(self) -> tuple[ChunkSentence, ...]:
        return self.sentences[self.evidence_start :]


# --- Input validation (no silent repair) -------------------------------------


def _valid_ms(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_aligned_sentences(aligned: Sequence[AlignedSentence]) -> None:
    """Validate a chunking input sequence, raising on any invalid record.

    Enforced (with 0-based index in the error message):

    * Each element is an :class:`AlignedSentence` with a Goal 10 ``Sentence``.
    * Parent segment window is finite, ``start >= 0``, ``end > start``.
    * ``precision`` is one of the two closed :class:`TimestampPrecision` values.
    * ``WORD_ALIGNED`` carries valid integer-ms exact times (``0 <= start <= end``)
      and ``SEGMENT_LEVEL`` carries ``None`` exact times (no silent discard).
    * Parent segment starts are NON-DECREASING across the ordered sequence
      (equal starts are allowed for sentences sharing a segment; overlap between
      distinct segments is preserved, mirroring Goal 09's validation policy).

    Nothing is mutated, clamped, sorted, or repaired.
    """
    prev_start: float | None = None
    for idx, item in enumerate(aligned):
        if not isinstance(item, AlignedSentence):
            raise InvalidChunkInputError(
                f"sentence {idx}: expected AlignedSentence, got {type(item).__name__}"
            )
        sentence = item.sentence
        precision = item.precision
        if precision not in _VALID_PRECISIONS:
            raise InvalidChunkInputError(
                f"sentence {idx}: unknown timestamp precision {precision!r}"
            )
        try:
            seg_start = _finite_float(
                sentence.source_segment_start_seconds, field="start_seconds"
            )
            seg_end = _finite_float(
                sentence.source_segment_end_seconds, field="end_seconds"
            )
        except ValueError as exc:
            raise InvalidChunkInputError(
                f"sentence {idx}: invalid segment window: {exc}"
            ) from exc
        if seg_start < 0 or seg_end <= seg_start:
            raise InvalidChunkInputError(
                f"sentence {idx}: require 0 <= start_seconds < end_seconds "
                f"(got {seg_start}, {seg_end})"
            )
        if prev_start is not None and seg_start < prev_start:
            raise InvalidChunkInputError(
                f"sentence {idx}: non-monotonic segment start {seg_start} "
                f"< previous {prev_start}; input must be chronologically ordered"
            )
        prev_start = seg_start

        if not isinstance(sentence.text, str):
            raise InvalidChunkInputError(f"sentence {idx}: text must be a str")
        for attr in ("char_start", "char_end", "ordinal"):
            value = getattr(sentence, attr)
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidChunkInputError(
                    f"sentence {idx}: {attr} must be an int"
                )
        if precision == TimestampPrecision.WORD_ALIGNED:
            s, e = item.exact_start_ms, item.exact_end_ms
            # If either is invalid, ``not (...)`` short-circuits before ``s > e``.
            if not (_valid_ms(s) and _valid_ms(e)) or s > e:  # type: ignore[operator]
                raise InvalidChunkInputError(
                    f"sentence {idx}: WORD_ALIGNED requires integer-ms "
                    f"0 <= exact_start_ms <= exact_end_ms"
                )
        else:  # SEGMENT_LEVEL
            if item.exact_start_ms is not None or item.exact_end_ms is not None:
                raise InvalidChunkInputError(
                    f"sentence {idx}: SEGMENT_LEVEL must not carry exact ms"
                )


# --- Internal algorithm helpers ----------------------------------------------


def _word_count(sentence) -> int:
    """Reliable word count from verbatim text (Goals 10/12 tokenizer reused)."""
    return len(tokenize_original(sentence.text))


def _interval(item: AlignedSentence) -> tuple[int, int]:
    """RELIABLE (start_ms, end_ms) interval for one sentence (video timeline)."""
    if item.precision == TimestampPrecision.WORD_ALIGNED:
        # Validated: both are real ints.
        return item.exact_start_ms, item.exact_end_ms  # type: ignore[return-value]
    return (
        seconds_to_ms(item.sentence.source_segment_start_seconds),
        seconds_to_ms(item.sentence.source_segment_end_seconds),
    )


def _extent(indices: Sequence[int], intervals: Sequence[tuple[int, int]]) -> tuple[int, int, int]:
    """(start_ms, end_ms, duration_ms) union of reliable intervals."""
    start = min(intervals[k][0] for k in indices)
    end = max(intervals[k][1] for k in indices)
    return start, end, end - start


def _is_exact(indices: Sequence[int], exact_flags: Sequence[bool]) -> bool:
    return all(exact_flags[k] for k in indices)


def _score(
    words: int,
    duration_ms: int,
    exact: bool,
    target_ms: int,
    band_words: int,
) -> float:
    """Lower = closer to the configured duration + word-count targets.

    Duration is used ONLY for fully exact groups (it is unreliable under coarse
    precision). Word count is always reliable. Both terms are normalized to be
    comparable.
    """
    word_term = abs(words - band_words) / max(band_words, 1)
    if not exact:
        return word_term
    return word_term + abs(duration_ms - target_ms) / max(target_ms, 1)


def _select_boundary(
    i: int,
    n: int,
    words: Sequence[int],
    intervals: Sequence[tuple[int, int]],
    exact_flags: Sequence[bool],
    config: ChunkingConfig,
    target_ms: int,
    min_ms: int,
    max_ms: int,
) -> tuple[int, str, tuple[str, ...]]:
    """Return ``(end_index_inclusive, reason, diagnostics)`` for one evidence group."""
    band_words = (config.target_min_words + config.target_max_words) // 2
    b = i
    while True:
        group = range(i, b + 1)
        w = sum(words[k] for k in group)
        _, _, d = _extent(group, intervals)
        exact = _is_exact(group, exact_flags)

        # (2) Single oversized sentence — keep intact, never split/truncate.
        if b == i and (
            w > config.target_max_words or (exact and d > max_ms)
        ):
            diags: list[str] = []
            if exact and d > max_ms:
                diags.append(DIAG_SINGLE_EXCEEDS_MAX_DURATION)
            if w > config.target_max_words:
                diags.append(DIAG_SINGLE_EXCEEDS_MAX_WORDS)
            return b, REASON_SINGLE_SENTENCE, tuple(diags)

        # (3) End of input.
        if b + 1 >= n:
            return b, REASON_END_OF_INPUT, _final_diags(w, d, exact, config, min_ms)

        w2 = w + words[b + 1]
        _, _, d2 = _extent(range(i, b + 2), intervals)
        exact2 = _is_exact(range(i, b + 2), exact_flags)

        # (4/5) Target reached?
        reached = w >= config.target_min_words or (exact and d >= target_ms)
        if reached:
            # One-step lookahead: include next sentence only if it is valid AND
            # strictly improves the distance-to-targets score.
            cap_ok = (
                w2 <= config.target_max_words
                and not (exact2 and d2 > max_ms)
                and not (exact and not exact2)  # do not downgrade an exact group
            )
            if cap_ok:
                score_b = _score(w, d, exact, target_ms, band_words)
                score_2 = _score(w2, d2, exact2, target_ms, band_words)
                if score_2 < score_b:
                    b = b + 1
                    # Recompute final state at the extended boundary.
                    g2 = range(i, b + 1)
                    w = sum(words[k] for k in g2)
                    _, _, d = _extent(g2, intervals)
                    exact = _is_exact(g2, exact_flags)
            return b, REASON_TARGET_REACHED, _final_diags(w, d, exact, config, min_ms)

        # (3) Not reached yet: enforce the 60s hard cap for fully exact groups
        # before blindly accumulating. Word count has no hard cap.
        if exact2 and d2 > max_ms:
            return b, REASON_MAX_DURATION, _final_diags(w, d, exact, config, min_ms)
        b += 1


def _final_diags(
    words: int,
    duration_ms: int,
    exact: bool,
    config: ChunkingConfig,
    min_ms: int,
) -> tuple[str, ...]:
    diags: list[str] = []
    if not exact:
        diags.append(DIAG_COARSE_PRECISION)
    elif duration_ms < min_ms:
        diags.append(DIAG_MIN_DURATION_UNMET)
    if words > config.target_max_words:
        diags.append(DIAG_TARGET_MAX_WORDS_EXCEEDED)
    elif words < config.target_min_words:
        diags.append(DIAG_TARGET_MIN_WORDS_UNMET)
    return tuple(diags)


def _select_overlap(
    evidence: Sequence[int],
    intervals: Sequence[tuple[int, int]],
    exact_flags: Sequence[bool],
    overlap_ms: int,
    max_ms: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Choose the ~5s complete-sentence overlap suffix of the previous evidence.

    Under COARSE (any SEGMENT_LEVEL sentence) precision, 5 seconds CANNOT be
    measured: the inherited parent segment window is not a sentence duration
    and does not shrink as a suffix shortens. Attempting "closest to 5s" there
    would pick an arbitrarily large, unmeasurable suffix. The honest, bounded
    fallback is the SMALLEST complete-sentence overlap (one trailing sentence),
    flagged ``overlap_coarse_precision``. Under FULLY EXACT precision, the
    suffix whose reliable extent is closest to 5s is chosen (ties -> shorter).
    """
    if not evidence:
        return (), ()
    if not _is_exact(evidence, exact_flags):
        return (evidence[-1],), (DIAG_OVERLAP_COARSE,)

    candidates: list[tuple[int, int, tuple[int, ...]]] = []
    for k in range(1, len(evidence) + 1):
        suffix = tuple(evidence[-k:])
        start, end, _ = _extent(suffix, intervals)
        overlap_duration = end - start
        if overlap_duration > max_ms:  # do not exceed maximum merely for overlap
            continue
        candidates.append((abs(overlap_duration - overlap_ms), k, suffix))
    if not candidates:
        return (), (DIAG_OVERLAP_UNAVAILABLE,)
    # min by (distance, k): ties prefer the SHORTER suffix (less duplication).
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2], ()


def _ref(item: AlignedSentence, index: int, role: str, words: int) -> ChunkSentence:
    s = item.sentence
    return ChunkSentence(
        index=index,
        role=role,
        text=s.text,
        language=s.language,
        ordinal=s.ordinal,
        char_start=s.char_start,
        char_end=s.char_end,
        source_segment_ref=s.source_segment_ref,
        timestamp_precision=item.precision,
        source_segment_start_seconds=s.source_segment_start_seconds,
        source_segment_end_seconds=s.source_segment_end_seconds,
        exact_start_ms=item.exact_start_ms,
        exact_end_ms=item.exact_end_ms,
        word_count=words,
    )


# --- Public entry point -------------------------------------------------------


def chunk_sentences(
    aligned_sentences: Sequence[AlignedSentence],
    *,
    parent: ChunkParentContext | None = None,
    config: ChunkingConfig | None = None,
) -> tuple[Chunk, ...]:
    """Deterministically group ``AlignedSentence`` values into retrieval chunks.

    Pure: no DB, no I/O, no clock, no credentials, no mutation of inputs, no
    embedding. Empty input returns ``()``. See the module docstring for the full
    boundary-selection, overlap, precision, and provenance contracts.

    ``parent`` is OPTIONAL for backward compatibility: when supplied, its real
    ``video_id`` / ``transcript_asset_id`` / ``content_right_id`` are stamped on
    every chunk; when omitted those fields are ``None``. Use
    :func:`build_production_chunks` for the production path, which REQUIRES a
    valid parent so no production chunk silently omits ownership provenance.
    """
    if parent is not None and not isinstance(parent, ChunkParentContext):
        raise InvalidChunkInputError(
            "parent must be a ChunkParentContext instance "
            f"(got {type(parent).__name__})"
        )
    cfg = config or ChunkingConfig()
    items = tuple(aligned_sentences)
    validate_aligned_sentences(items)
    n = len(items)
    if n == 0:
        return ()

    video_id = parent.video_id if parent is not None else None
    transcript_asset_id = parent.transcript_asset_id if parent is not None else None
    content_right_id = parent.content_right_id if parent is not None else None

    target_ms = seconds_to_ms(cfg.target_duration_seconds)
    min_ms = seconds_to_ms(cfg.min_duration_seconds)
    max_ms = seconds_to_ms(cfg.max_duration_seconds)
    overlap_ms = seconds_to_ms(cfg.overlap_seconds)

    words = [_word_count(item.sentence) for item in items]
    intervals = [_interval(item) for item in items]
    exact_flags = [item.precision == TimestampPrecision.WORD_ALIGNED for item in items]

    chunks: list[Chunk] = []
    pending_overlap: tuple[int, ...] = ()
    pending_overlap_diags: tuple[str, ...] = ()
    i = 0
    chunk_index = 0

    while i < n:
        b, reason, diags = _select_boundary(
            i, n, words, intervals, exact_flags, cfg, target_ms, min_ms, max_ms
        )
        evidence = tuple(range(i, b + 1))

        # Context (overlap from previous chunk), with max-duration trimming.
        context = list(pending_overlap)
        all_idx = context + list(evidence)
        if all_idx:
            _, _, combined_d = _extent(all_idx, intervals)
            if _is_exact(all_idx, exact_flags) and combined_d > max_ms:
                trimmed = False
                while context:
                    context.pop(0)  # drop oldest (earliest) context sentence
                    trimmed = True
                    trial = context + list(evidence)
                    if _extent(trial, intervals)[2] <= max_ms:
                        break
                # Report trimming ONLY when a context sentence was actually
                # removed. A single oversized evidence sentence already exceeds
                # max_duration by itself and has NO overlap context; emitting
                # this diagnostic there would be a false "overlap trimmed" claim.
                if trimmed:
                    diags = diags + (DIAG_OVERLAP_TRIMMED,)

        context = tuple(context)
        sentence_refs = tuple(
            _ref(items[k], k, ROLE_CONTEXT, words[k]) for k in context
        ) + tuple(
            _ref(items[k], k, ROLE_EVIDENCE, words[k]) for k in evidence
        )
        start, end, duration = _extent(context + evidence, intervals)
        chunk_precision = (
            TimestampPrecision.WORD_ALIGNED
            if _is_exact(context + evidence, exact_flags)
            else TimestampPrecision.SEGMENT_LEVEL
        )
        # Provenance: ordered, de-duplicated distinct segment refs of EVIDENCE.
        refs: list[str] = []
        for k in evidence:
            ref = items[k].sentence.source_segment_ref
            if ref not in refs:
                refs.append(ref)
        text = "".join(r.text for r in sentence_refs)
        total_words = sum(ref.word_count for ref in sentence_refs)
        combined_diags = tuple(dict.fromkeys(diags + pending_overlap_diags))

        chunks.append(
            Chunk(
                index=chunk_index,
                text=text,
                sentences=sentence_refs,
                start_ms=start,
                end_ms=end,
                duration_ms=duration,
                timestamp_precision=chunk_precision,
                word_count=total_words,
                evidence_start=len(context),
                source_segment_refs=tuple(refs),
                boundary_reason=reason,
                diagnostics=combined_diags,
                overlap_sentence_indices=context,
                video_id=video_id,
                transcript_asset_id=transcript_asset_id,
                content_right_id=content_right_id,
            )
        )

        # Overlap to carry into the NEXT chunk (from this chunk's evidence).
        pending_overlap, pending_overlap_diags = _select_overlap(
            evidence, intervals, exact_flags, overlap_ms, max_ms
        )
        i = b + 1
        chunk_index += 1

    return tuple(chunks)


def build_production_chunks(
    aligned_sentences: Sequence[AlignedSentence],
    *,
    parent: ChunkParentContext,
    config: ChunkingConfig | None = None,
) -> tuple[Chunk, ...]:
    """Production composition wrapper: REQUIRES real parent provenance.

    Unlike the pure :func:`chunk_sentences`, ``parent`` is a REQUIRED
    keyword-only argument (no default) and MUST be a valid, frozen
    :class:`ChunkParentContext` built from the existing Goal 09
    ``IngestionResult`` / Goal 12 ``TranscriptAlignmentRun`` parent context
    (see ``from_ingestion_result`` / ``from_alignment_run``). No ID is
    fabricated and none is derived from an unrelated field. The wrapper
    supplies the parent to the pure primitive and then FAILS CLOSED if any
    produced chunk does not carry the exact parent IDs — production chunks can
    NEVER silently omit ``video_id`` / ``transcript_asset_id``.
    """
    if not isinstance(parent, ChunkParentContext):
        raise InvalidChunkInputError(
            "production chunking requires a valid ChunkParentContext "
            f"(got {type(parent).__name__})"
        )
    chunks = chunk_sentences(aligned_sentences, parent=parent, config=config)
    for chunk in chunks:
        if (
            chunk.video_id != parent.video_id
            or chunk.transcript_asset_id != parent.transcript_asset_id
            or chunk.content_right_id != parent.content_right_id
        ):
            raise InvalidChunkInputError(
                "production chunk missing required parent provenance"
            )
    return chunks


__all__ = [
    "DIAG_COARSE_PRECISION",
    "DIAG_MIN_DURATION_UNMET",
    "DIAG_OVERLAP_COARSE",
    "DIAG_OVERLAP_TRIMMED",
    "DIAG_OVERLAP_UNAVAILABLE",
    "DIAG_SINGLE_EXCEEDS_MAX_DURATION",
    "DIAG_SINGLE_EXCEEDS_MAX_WORDS",
    "DIAG_TARGET_MAX_WORDS_EXCEEDED",
    "DIAG_TARGET_MIN_WORDS_UNMET",
    "REASON_END_OF_INPUT",
    "REASON_MAX_DURATION",
    "REASON_SINGLE_SENTENCE",
    "REASON_TARGET_REACHED",
    "ROLE_CONTEXT",
    "ROLE_EVIDENCE",
    "Chunk",
    "ChunkParentContext",
    "ChunkSentence",
    "ChunkingConfig",
    "InvalidChunkInputError",
    "build_production_chunks",
    "chunk_sentences",
    "validate_aligned_sentences",
]
