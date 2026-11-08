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
* :mod:`app.transcripts.word_contract` — the pure Goal 12 word contract:
  strict seconds<->milliseconds conversion, deterministic original-text
  tokenizer, the WhisperX 3.8.6 language allowlist, alignment-run statuses,
  and the :class:`AlignedWord` value object.
* :mod:`app.transcripts.word_alignment` — pure Goal 12 preparation/validation:
  audio-timeline window preparation, the provider request/result protocol,
  strict ordinal mapping, fail-closed word/char validation, and deterministic
  char reconstruction for the isolated worker.
* :mod:`app.transcripts.sentence_timestamps` — pure derivation of
  :class:`AlignedSentence` (``WORD_ALIGNED`` only for fully covered sentences).
* :mod:`app.transcripts.chunking` — deterministic, sentence-aware chunking
  (Goal 13): sentence-integrity-first grouping with best-effort timestamp
  precision, complete-sentence overlap, and full source provenance.

Persistence is deliberately NOT here: the ingestion orchestration lives in
:mod:`app.services.transcript_ingestion_service` and word-alignment persistence
in :mod:`app.services.word_alignment_service`, which compose the repository
layer inside service-owned transactions.

Scope boundary (Goals 09–13): this is a transcript *processing foundation*.
Chunking (Goal 13) is a deterministic, library-level grouping of
:class:`AlignedSentence` values; it adds NO persistence, NO embeddings, NO
reranking, and NO search. Alignment provider dependencies are OPTIONAL and
isolated in the :mod:`alignment_worker` process — importing this package never
imports torch or whisperx. Only authorized sources are represented.
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
from app.transcripts.sentence_timestamps import (
    AlignedSentence,
    derive_aligned_sentences,
)
from app.transcripts.chunking import (
    Chunk,
    ChunkParentContext,
    ChunkSentence,
    ChunkingConfig,
    InvalidChunkInputError,
    build_production_chunks,
    chunk_sentences,
    validate_aligned_sentences,
)
from app.transcripts.validation import (
    TranscriptValidationError,
    ValidationReport,
    validate_transcript,
)
from app.transcripts.word_alignment import (
    AlignmentRequest,
    AlignmentSegmentTask,
    PreparedSegment,
    SegmentData,
    SegmentPreparation,
    SkippedSegment,
    ValidatedSegment,
    build_request,
    prepare_segments,
    reconstruct_word_chars,
    validate_provider_segment,
)
from app.transcripts.word_contract import (
    ALIGNMENT_RUN_STATUSES,
    AlignedWord,
    BENGALI_LANGUAGE_CODES,
    COMPLETED_STATUS,
    CURRENT_SUCCESS_STATUSES,
    FAILED_STATUS,
    PARTIAL_STATUS,
    PENDING_STATUS,
    RUNNING_STATUS,
    SUPERSEDED_STATUS,
    SUPPORTED_ALIGNMENT_LANGUAGES,
    SourceAudioAssetInput,
    Token,
    UNSUPPORTED_LANGUAGE_STATUS,
    is_alignment_language_supported,
    language_gate_reason,
    ms_to_seconds,
    seconds_to_ms,
    tokenize_original,
    validate_provider_second,
)

__all__ = [
    "ALIGNMENT_RUN_STATUSES",
    "AlignedSentence",
    "AlignedWord",
    "AlignmentRequest",
    "AlignmentSegmentTask",
    "BENGALI_LANGUAGE_CODES",
    "Chunk",
    "ChunkParentContext",
    "ChunkSentence",
    "ChunkingConfig",
    "COMPLETED_STATUS",
    "CURRENT_SUCCESS_STATUSES",
    "EmptySourceTextError",
    "FAILED_STATUS",
    "InvalidChunkInputError",
    "NormalizedText",
    "PARTIAL_STATUS",
    "PENDING_STATUS",
    "PreparedSegment",
    "RUNNING_STATUS",
    "SUPERSEDED_STATUS",
    "SUPPORTED_ALIGNMENT_LANGUAGES",
    "SegmentData",
    "SegmentPreparation",
    "Sentence",
    "SentenceSegmentationInput",
    "SentenceSpan",
    "SkippedSegment",
    "SourceAudioAssetInput",
    "TimestampPrecision",
    "Token",
    "TranscriptInput",
    "TranscriptSegmentInput",
    "TranscriptSource",
    "TranscriptValidationError",
    "UNSUPPORTED_LANGUAGE_STATUS",
    "ValidatedSegment",
    "ValidationReport",
    "build_production_chunks",
    "build_request",
    "chunk_sentences",
    "derive_aligned_sentences",
    "find_sentence_spans",
    "is_alignment_language_supported",
    "language_gate_reason",
    "ms_to_seconds",
    "normalize_text",
    "prepare_segments",
    "reconstruct_word_chars",
    "seconds_to_ms",
    "segment_sentences",
    "tokenize_original",
    "validate_aligned_sentences",
    "validate_provider_second",
    "validate_provider_segment",
    "validate_transcript",
]
