"""Transcript-processing foundation for VoxSearch (Goal 09).

This package provides the *library-level* building blocks that later ASR,
alignment, embedding, and search components will build on:

* :mod:`app.transcripts.contracts` — the explicit, Pydantic 2 validated input
  contract (a closed set of authorized transcript sources).
* :mod:`app.transcripts.validation` — cross-segment / contextual validation
  (chronological order, documented overlap policy, known-duration bound).
* :mod:`app.transcripts.normalization` — deterministic, multilingual-safe text
  normalization that preserves the original text separately.
* :mod:`app.transcripts.sentence_segmentation` — deterministic, multilingual
  (English / Bangla / mixed) sentence segmentation with verbatim source text,
  character offsets, and segment-level timestamp provenance (Goal 10).

Persistence is deliberately NOT here: the ingestion orchestration lives in
:mod:`app.services.transcript_ingestion_service`, which composes the existing
repository layer inside a single atomic transaction.

Scope boundary (Goals 09–10): this is a transcript *processing foundation*
only. There is NO ASR, NO caption downloading / scraping, NO chunking, NO
embeddings, NO reranking, and NO search. Only authorized sources
(creator-provided, licensed, user-provided, or ASR on permitted audio) are
represented; nothing here implies that arbitrary public YouTube captions are
accessible.
"""
from app.transcripts.contracts import (
    TranscriptInput,
    TranscriptSegmentInput,
    TranscriptSource,
)
from app.transcripts.normalization import NormalizedText, normalize_text
from app.transcripts.sentence_segmentation import (
    EmptySourceTextError,
    Sentence,
    SentenceSegmentationInput,
    SentenceSpan,
    TimestampPrecision,
    find_sentence_spans,
    segment_sentences,
)
from app.transcripts.validation import (
    TranscriptValidationError,
    ValidationReport,
    validate_transcript,
)

__all__ = [
    "EmptySourceTextError",
    "NormalizedText",
    "Sentence",
    "SentenceSegmentationInput",
    "SentenceSpan",
    "TimestampPrecision",
    "TranscriptInput",
    "TranscriptSegmentInput",
    "TranscriptSource",
    "TranscriptValidationError",
    "ValidationReport",
    "find_sentence_spans",
    "normalize_text",
    "segment_sentences",
    "validate_transcript",
]
