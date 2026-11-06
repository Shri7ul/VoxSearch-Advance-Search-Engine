"""Transcript-processing foundation for VoxSearch (Goal 09).

This package provides the *library-level* building blocks that later ASR,
alignment, embedding, and search components will build on:

* :mod:`app.transcripts.contracts` — the explicit, Pydantic 2 validated input
  contract (a closed set of authorized transcript sources).
* :mod:`app.transcripts.validation` — cross-segment / contextual validation
  (chronological order, documented overlap policy, known-duration bound).
* :mod:`app.transcripts.normalization` — deterministic, multilingual-safe text
  normalization that preserves the original text separately.

Persistence is deliberately NOT here: the ingestion orchestration lives in
:mod:`app.services.transcript_ingestion_service`, which composes the existing
repository layer inside a single atomic transaction.

Scope boundary (Goal 09): this is a transcript *processing foundation* only.
There is NO ASR, NO caption downloading / scraping, NO embeddings, NO
reranking, and NO search. Only authorized sources (creator-provided, licensed,
user-provided, or ASR on permitted audio) are represented; nothing here implies
that arbitrary public YouTube captions are accessible.
"""
from app.transcripts.contracts import (
    TranscriptInput,
    TranscriptSegmentInput,
    TranscriptSource,
)
from app.transcripts.normalization import NormalizedText, normalize_text
from app.transcripts.validation import (
    TranscriptValidationError,
    ValidationReport,
    validate_transcript,
)

__all__ = [
    "NormalizedText",
    "TranscriptInput",
    "TranscriptSegmentInput",
    "TranscriptSource",
    "TranscriptValidationError",
    "ValidationReport",
    "normalize_text",
    "validate_transcript",
]
