"""Repository for ``voxsearch.video_chunks`` (Goal 16).

Persists Goal 13 chunk metadata + its Goal 15 1024-dimensional dense
embedding. This is a THIN data-access layer: it translates one validated
operation into a single PostgreSQL multi-row ``INSERT ... ON CONFLICT DO
UPDATE`` and NEVER commits (transaction ownership stays with the service).

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
from typing import Any

import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import VideoChunk
from app.repositories.base import BaseRepository

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
