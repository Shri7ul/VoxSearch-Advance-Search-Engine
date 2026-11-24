"""Repository for ``voxsearch.video_chunks`` (Goals 16–17).

Persists Goal 13 chunk metadata + its Goal 15 1024-dimensional dense embedding,
and provides read-only cosine-distance retrieval (Goal 17). This is a THIN
data-access layer: persistence translates one validated operation into a single
PostgreSQL multi-row ``INSERT ... ON CONFLICT DO UPDATE``; retrieval is a single
parameterized ``SELECT`` using pgvector ``<=>``. It NEVER commits (transaction
ownership stays with the service).

Upsert / duplicate policy (approved design)
-------------------------------------------
The UNIQUE key ``(transcript_asset_id, chunk_index, model_id)`` is the
idempotent-upsert key. Re-inserting the same chunk/model refreshes the
embedding and chunk metadata IN PLACE instead of creating a duplicate. This is
a single statement, so a batch is atomic: either all rows land or none do.

Note on ``session.execute`` vs ORM ``flush``: ``onupdate=func.now()`` on
``updated_at`` fires only for ORM flushes, NOT for a Core INSERT, so the SET
clause sets ``updated_at = now()`` explicitly. The INSERT path relies on the
column's ``server_default``. No commit, no flush, no swallowing of
``SQLAlchemyError`` — failures propagate to the owning service, which rolls
back and re-raises.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import TranscriptAsset, Video, VideoChunk
from app.repositories.base import BaseRepository

# Goal 17 retrieval bounds. The repository owns the data-access bound so that a
# direct caller cannot issue an unbounded or nonsensical LIMIT; the service
# re-exports the same constants and maps violations to its domain error type.
DEFAULT_TOP_K = 10
MAX_TOP_K = 100


def validate_top_k(top_k: Any) -> int:
    """Return a valid ``top_k`` or raise ``ValueError`` (fail-closed).

    Rejects ``bool`` (an ``int`` subclass), non-integers, and values outside
    ``[1, MAX_TOP_K]``. The bound is enforced here, at the data-access boundary,
    so every caller is protected even if it bypasses the service layer.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(
            f"top_k must be an integer, got {type(top_k).__name__}"
        )
    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(
            f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}"
        )
    return top_k


@dataclass(frozen=True)
class ChunkSearchResult:
    """One candidate chunk retrieved by vector similarity (Goal 17).

    Contains complete chunk metadata, verbatim text, and the computed cosine
    distance from the query vector. The raw 1024-dimensional embedding column
    is intentionally omitted to prevent unnecessary payload transfer.
    """

    id: uuid.UUID
    video_id: uuid.UUID
    transcript_asset_id: uuid.UUID
    content_right_id: uuid.UUID
    chunk_index: int
    text: str
    start_ms: int
    end_ms: int
    duration_ms: int
    timestamp_precision: str
    word_count: int
    boundary_reason: str
    source_segment_refs: tuple[str, ...]
    overlap_sentence_indices: tuple[int, ...]
    diagnostics: tuple[str, ...]
    model_id: str
    distance: float
    # Existing videos.youtube_video_id, surfaced for Goal 32 timestamp links.
    youtube_video_id: str = ""

    @property
    def score(self) -> float:
        """Cosine similarity: ``1.0 - distance`` (pgvector ``<=>`` is 1 - cos)."""
        return 1.0 - self.distance

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

    @classmethod
    def from_row(cls, row: Any) -> ChunkSearchResult:
        return cls(
            id=row.id,
            video_id=row.video_id,
            transcript_asset_id=row.transcript_asset_id,
            content_right_id=row.content_right_id,
            chunk_index=row.chunk_index,
            text=row.text,
            start_ms=row.start_ms,
            end_ms=row.end_ms,
            duration_ms=row.duration_ms,
            timestamp_precision=row.timestamp_precision,
            word_count=row.word_count,
            boundary_reason=row.boundary_reason,
            source_segment_refs=tuple(row.source_segment_refs),
            overlap_sentence_indices=tuple(row.overlap_sentence_indices),
            diagnostics=tuple(row.diagnostics),
            model_id=row.model_id,
            distance=float(row.distance),
            youtube_video_id=getattr(row, "youtube_video_id", "") or "",
        )


# Columns refreshed on conflict. ``id``, ``transcript_asset_id``,
# ``chunk_index``, ``model_id`` and ``created_at`` are the stable identity and
# are deliberately NOT overwritten.
_REFRESHED_ON_CONFLICT = (
    "video_id",
    "content_right_id",
    "text",
    "start_ms",
    "end_ms",
    "duration_ms",
    "timestamp_precision",
    "word_count",
    "boundary_reason",
    "source_segment_refs",
    "overlap_sentence_indices",
    "diagnostics",
    "embedding",
)


class VideoChunkRepository(BaseRepository):
    """Data access for :class:`~app.db.models.VideoChunk` rows."""

    def upsert_batch(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Insert/refresh a batch of chunk+embedding rows in ONE statement.

        Each mapping must supply the columns of ``video_chunks`` except the
        timestamps (those come from server defaults / the SET clause). The
        service supplies an explicit ``id`` (a ``uuid.UUID``) so the multi-row
        INSERT is deterministic and does not depend on Core default evaluation;
        on conflict the proposed id is simply discarded (the existing row keeps
        its own PK). The ``embedding`` value is a sequence of 1024 floats; the
        dependency-free :class:`~app.db.types.VectorType` serializes it into a
        pgvector literal and wraps the bind in ``CAST(... AS vector(1024))``.

        NO commit and NO flush: ``session.execute`` emits the statement inside
        the caller's transaction. An empty batch emits nothing.
        """
        if not rows:
            return
        stmt = pg_insert(VideoChunk.__table__).values([dict(r) for r in rows])
        excluded = stmt.excluded
        stmt = stmt.on_conflict_do_update(
            index_elements=["transcript_asset_id", "chunk_index", "model_id"],
            set_={
                **{name: getattr(excluded, name) for name in _REFRESHED_ON_CONFLICT},
                "updated_at": func.now(),
            },
        )
        self.session.execute(stmt)

    def list_by_asset(
        self,
        transcript_asset_id: uuid.UUID,
        *,
        model_id: str | None = None,
    ) -> list[VideoChunk]:
        """List an asset's chunks in ``chunk_index`` order (read-only)."""
        stmt = select(VideoChunk).where(
            VideoChunk.transcript_asset_id == transcript_asset_id
        )
        if model_id is not None:
            stmt = stmt.where(VideoChunk.model_id == model_id)
        stmt = stmt.order_by(VideoChunk.chunk_index.asc(), VideoChunk.id.asc())
        return list(self.session.scalars(stmt).all())

    def list_context_chunks(
        self,
        *,
        transcript_asset_id: uuid.UUID,
        chunk_index: int,
        model_id: str | None = None,
        radius: int = 1,
    ) -> list[tuple[int, str, int, int]]:
        """Read-only neighbouring chunks for Goal 28 context expansion.

        Only rows from the SAME transcript asset (and, when supplied, the SAME
        embedding model) within ``chunk_index +/- radius`` are returned; the
        target chunk itself is excluded. Results are ordered by distance from
        the target (then chunk_index) and bounded to ``2 * radius`` rows. No
        embedding column is selected and no write is performed.
        """
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            raise ValueError("chunk_index must be an integer")
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
            raise ValueError("radius must be a positive integer")
        distance = func.abs(VideoChunk.chunk_index - chunk_index).label("distance")
        stmt = select(
            VideoChunk.chunk_index,
            VideoChunk.text,
            VideoChunk.start_ms,
            VideoChunk.end_ms,
        ).where(
            VideoChunk.transcript_asset_id == transcript_asset_id,
            VideoChunk.chunk_index != chunk_index,
            VideoChunk.chunk_index >= chunk_index - radius,
            VideoChunk.chunk_index <= chunk_index + radius,
        )
        if model_id is not None:
            stmt = stmt.where(VideoChunk.model_id == model_id)
        stmt = stmt.order_by(distance.asc(), VideoChunk.chunk_index.asc()).limit(
            2 * radius
        )
        return [tuple(row) for row in self.session.execute(stmt).all()]

    def search_similar(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = DEFAULT_TOP_K,
        video_id: uuid.UUID | None = None,
        transcript_asset_id: uuid.UUID | None = None,
        model_id: str | None = None,
        language: str | None = None,
        max_duration_seconds: int | None = None,
    ) -> list[ChunkSearchResult]:
        """Retrieve the top-k most similar chunks by cosine distance (Goal 17).

        Uses PostgreSQL pgvector's cosine distance operator ``<=>``. Ranking is
        strictly deterministic: ordered by distance ascending, then chunk_index
        ascending, then id ascending. Raw embeddings are not retrieved.

        Filtering by video_id, transcript_asset_id, and/or model_id is applied
        in the WHERE clause before ranking and limit. ``top_k`` is bounded to
        ``[1, MAX_TOP_K]`` at this boundary (``ValueError`` otherwise); an empty
        result set is returned as ``[]``.

        Dimension / finiteness of ``query_vector`` are enforced by the column's
        :class:`~app.db.types.VectorType` bind processor; callers that need the
        full L2-normalization contract should use
        :class:`~app.services.video_chunks.VideoChunkSearchService`.
        """
        top_k = validate_top_k(top_k)
        dist = VideoChunk.embedding.op("<=>")(query_vector).label("distance")
        stmt = select(
            VideoChunk.id,
            VideoChunk.video_id,
            VideoChunk.transcript_asset_id,
            VideoChunk.content_right_id,
            VideoChunk.chunk_index,
            VideoChunk.text,
            VideoChunk.start_ms,
            VideoChunk.end_ms,
            VideoChunk.duration_ms,
            VideoChunk.timestamp_precision,
            VideoChunk.word_count,
            VideoChunk.boundary_reason,
            VideoChunk.source_segment_refs,
            VideoChunk.overlap_sentence_indices,
            VideoChunk.diagnostics,
            VideoChunk.model_id,
            Video.youtube_video_id.label("youtube_video_id"),
            dist,
        ).join(Video, VideoChunk.video_id == Video.id)
        if video_id is not None:
            stmt = stmt.where(VideoChunk.video_id == video_id)
        if transcript_asset_id is not None:
            stmt = stmt.where(VideoChunk.transcript_asset_id == transcript_asset_id)
        if model_id is not None:
            stmt = stmt.where(VideoChunk.model_id == model_id)
        # Goal 33 refinement filters, backed by existing schema columns only.
        if language is not None:
            stmt = stmt.join(
                TranscriptAsset,
                VideoChunk.transcript_asset_id == TranscriptAsset.id,
            ).where(TranscriptAsset.language == language)
        if max_duration_seconds is not None:
            stmt = stmt.where(Video.duration_seconds <= max_duration_seconds)

        stmt = stmt.order_by(
            dist.asc(),
            VideoChunk.chunk_index.asc(),
            VideoChunk.id.asc(),
        ).limit(top_k)

        rows = self.session.execute(stmt).all()
        return [ChunkSearchResult.from_row(r) for r in rows]


__all__ = [
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "ChunkSearchResult",
    "VideoChunkRepository",
    "validate_top_k",
]
