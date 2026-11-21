"""Goal 16 — persistence of Goal 13 chunks + Goal 15 dense embeddings.

This is the application/business boundary for vector persistence. It
orchestrates :class:`~app.repositories.video_chunks.VideoChunkRepository` and
owns the transaction through :func:`app.services.transaction`. It enforces the
fail-closed rules a DB constraint alone cannot express:

* every chunk must carry STRICT ``uuid.UUID`` parent provenance (never ``str``,
  ``None``, or a fabricated value), and a batch is bound to ONE lineage
  (identical ``video_id`` / ``transcript_asset_id`` / ``content_right_id``);
* the :class:`~app.embeddings.contracts.EmbeddingBatch` must be 1:1 aligned
  with the chunks (count / dimension / model / positional order);
* every vector is re-validated through Goal 15's
  :meth:`~app.embeddings.contracts.EmbeddingVector.from_sequence` — exact 1024
  dims, finite floats, L2 norm within the documented tolerance. Invalid output
  is REJECTED, never silently rescaled or repaired.

Transaction safety
------------------
ALL validation AND the single multi-row repository call happen INSIDE one
``with transaction(self._session):`` block. Any domain error or
``SQLAlchemyError`` therefore triggers ``session.rollback()`` and is re-raised
unchanged; ``session.commit()`` runs only on full success. The repository emits
ONE statement, so a batch is atomic — no partial persistence.

Boundaries (Goal 16)
--------------------
Insert/upsert only. NO cosine similarity / ANN search (Goal 17), NO search
endpoint (Goal 18), NO reranking / sparse / ColBERT, NO embedding generation
(Goal 15), NO ORM relationship objects. The pgvector extension is assumed
pre-installed; no migration is applied here.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.embeddings.contracts import (
    DENSE_DIMENSION,
    EmbeddingBatch,
    EmbeddingVector,
    InvalidEmbeddingInputError,
    InvalidEmbeddingOutputError,
    texts_from_chunks,
)
from app.repositories.video_chunks import VideoChunkRepository
from app.services import VoxSearchServiceError, transaction
from app.transcripts.chunking import Chunk


# --- Explicit, credential-free error taxonomy --------------------------------

class VectorPersistenceError(VoxSearchServiceError):
    """Base class for Goal 16 persistence failures."""


class InvalidChunkParentError(VectorPersistenceError):
    """A chunk lacks strict UUID parent provenance or the batch mixes lineage."""


class EmbeddingChunkAlignmentError(VectorPersistenceError):
    """Chunks and the embedding batch are not 1:1 aligned / consistent."""


class EmbeddingPersistenceValidationError(VectorPersistenceError):
    """A vector / model id failed the Goal 15 invariants on the way to the DB."""


class VideoChunkPersistenceService:
    """Persist production chunks + their dense embeddings (one transaction)."""

    def __init__(
        self,
        session: Session,
        repo: VideoChunkRepository | None = None,
    ) -> None:
        self._session = session
        self._repo = repo if repo is not None else VideoChunkRepository(session)

    def persist(
        self,
        chunks: Sequence[Chunk],
        embeddings: EmbeddingBatch,
        *,
        model_id: str | None = None,
    ) -> int:
        """Validate and upsert ``chunks`` + ``embeddings``; return row count.

        ``model_id`` is an OPTIONAL assertion: when supplied it MUST equal the
        batch's ``model_id`` (an override would store a vector under the wrong
        label). When omitted, the batch's own ``model_id`` is used. The model id
        is part of the idempotent-upsert key, so re-embedding with a different
        model never clobbers the previous vector.
        """
        with transaction(self._session):
            parents = self._validate_parents(chunks)
            texts = self._validated_texts(chunks)
            effective_model_id = self._validate_alignment(
                embeddings, n=len(chunks), model_id=model_id
            )
            rows = self._build_rows(
                chunks, texts, embeddings, parents, effective_model_id
            )
            # ONE multi-row statement -> atomic batch (empty -> no-op).
            self._repo.upsert_batch(rows)
        return len(rows)

    # --- validation (all called INSIDE the transaction) ---------------------

    @staticmethod
    def _validate_parents(
        chunks: Sequence[Chunk],
    ) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID] | None:
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise InvalidChunkParentError("chunks must be a sequence of Chunk")
        parents: tuple[uuid.UUID, uuid.UUID, uuid.UUID] | None = None
        seen_indices: set[int] = set()
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise InvalidChunkParentError(
                    f"chunk {idx}: expected Chunk, got {type(chunk).__name__}"
                )
            ids = (
                chunk.video_id,
                chunk.transcript_asset_id,
                chunk.content_right_id,
            )
            for label, value in zip(
                ("video_id", "transcript_asset_id", "content_right_id"), ids
            ):
                # Fail-closed: a str / None / anything that is not a real
                # uuid.UUID is rejected, never coerced or fabricated.
                if not isinstance(value, uuid.UUID):
                    raise InvalidChunkParentError(
                        f"chunk {idx}: {label} must be a strict uuid.UUID, "
                        f"got {type(value).__name__}"
                    )
            if parents is None:
                parents = ids  # type: ignore[assignment]
            elif ids != parents:
                raise InvalidChunkParentError(
                    "all chunks in a batch must share the identical parent "
                    "(video_id, transcript_asset_id, content_right_id) triple"
                )

            # Chunk index invariants: non-negative integer, unique within batch
            # (PostgreSQL ON CONFLICT DO UPDATE forbids affecting the same row twice).
            if (
                isinstance(chunk.index, bool)
                or not isinstance(chunk.index, int)
                or chunk.index < 0
            ):
                raise EmbeddingChunkAlignmentError(
                    f"chunk {idx}: chunk_index must be a non-negative int, "
                    f"got {chunk.index!r}"
                )
            if chunk.index in seen_indices:
                raise EmbeddingChunkAlignmentError(
                    f"chunk {idx}: duplicate chunk_index {chunk.index} in batch; "
                    "chunk indices must be unique within a batch"
                )
            seen_indices.add(chunk.index)

            # Timestamp invariants matching ck_video_chunks_time_window
            if (
                isinstance(chunk.start_ms, bool)
                or not isinstance(chunk.start_ms, int)
                or chunk.start_ms < 0
                or isinstance(chunk.end_ms, bool)
                or not isinstance(chunk.end_ms, int)
                or chunk.end_ms <= chunk.start_ms
                or isinstance(chunk.duration_ms, bool)
                or not isinstance(chunk.duration_ms, int)
                or chunk.duration_ms < 0
            ):
                raise EmbeddingChunkAlignmentError(
                    f"chunk {idx}: invalid time window (require start_ms >= 0, "
                    f"end_ms > start_ms, duration_ms >= 0; got start={chunk.start_ms}, "
                    f"end={chunk.end_ms}, duration={chunk.duration_ms})"
                )
        return parents

    @staticmethod
    def _validated_texts(chunks: Sequence[Chunk]) -> tuple[str, ...]:
        """Extract VERBATIM chunk text via the Goal 15 bridge.

        Guarantees the STORED text is byte-for-byte the text that was embedded.
        Goal 15 input errors are surfaced as alignment errors, never swallowed.
        """
        try:
            return texts_from_chunks(chunks)
        except InvalidEmbeddingInputError as exc:
            raise EmbeddingChunkAlignmentError(
                f"invalid chunk text: {exc}"
            ) from exc

    @staticmethod
    def _validate_alignment(
        embeddings: EmbeddingBatch,
        *,
        n: int,
        model_id: str | None,
    ) -> str:
        if not isinstance(embeddings, EmbeddingBatch):
            raise EmbeddingChunkAlignmentError(
                "embeddings must be a Goal 15 EmbeddingBatch"
            )
        if embeddings.count != n:
            raise EmbeddingChunkAlignmentError(
                f"chunk/embedding count mismatch: {n} chunks vs "
                f"{embeddings.count} vectors"
            )
        if embeddings.dimension != DENSE_DIMENSION:
            raise EmbeddingChunkAlignmentError(
                f"embedding dimension {embeddings.dimension} != "
                f"{DENSE_DIMENSION}"
            )
        batch_model = embeddings.model_id
        if not isinstance(batch_model, str) or not batch_model.strip():
            raise EmbeddingPersistenceValidationError(
                "embedding batch model_id must be a non-empty str"
            )
        if model_id is not None and model_id != batch_model:
            raise EmbeddingChunkAlignmentError(
                f"explicit model_id {model_id!r} does not match the embedding "
                f"batch model_id {batch_model!r}"
            )
        return batch_model

    @staticmethod
    def _build_rows(
        chunks: Sequence[Chunk],
        texts: tuple[str, ...],
        embeddings: EmbeddingBatch,
        parents: tuple[uuid.UUID, uuid.UUID, uuid.UUID] | None,
        model_id: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, (chunk, vector) in enumerate(
            zip(chunks, embeddings.vectors)
        ):
            # Re-validate each vector through the SAME Goal 15 gate used at
            # generation time. Accept both an EmbeddingVector and a raw
            # sequence so a malformed batch cannot slip through. Invalid
            # output (wrong dims / NaN / Inf / non-unit norm) is REJECTED,
            # never silently rescaled.
            raw_values = (
                vector.values if isinstance(vector, EmbeddingVector) else vector
            )
            try:
                validated = EmbeddingVector.from_sequence(
                    tuple(raw_values), index=idx
                )
            except InvalidEmbeddingOutputError as exc:
                raise EmbeddingPersistenceValidationError(str(exc)) from exc

            precision = chunk.timestamp_precision
            precision_value = (
                precision.value if isinstance(precision, Enum) else str(precision)
            )
            assert parents is not None  # guaranteed when chunks is non-empty
            video_id, asset_id, right_id = parents
            rows.append({
                "id": uuid.uuid4(),
                "video_id": video_id,
                "transcript_asset_id": asset_id,
                "content_right_id": right_id,
                "chunk_index": chunk.index,
                "text": texts[idx],
                "start_ms": chunk.start_ms,
                "end_ms": chunk.end_ms,
                "duration_ms": chunk.duration_ms,
                "timestamp_precision": precision_value,
                "word_count": chunk.word_count,
                "boundary_reason": chunk.boundary_reason,
                "source_segment_refs": list(chunk.source_segment_refs),
                "overlap_sentence_indices": list(chunk.overlap_sentence_indices),
                "diagnostics": list(chunk.diagnostics),
                "model_id": model_id,
                "embedding": validated.values,
            })
        return rows


__all__ = [
    "EmbeddingChunkAlignmentError",
    "EmbeddingPersistenceValidationError",
    "InvalidChunkParentError",
    "VectorPersistenceError",
    "VideoChunkPersistenceService",
]
