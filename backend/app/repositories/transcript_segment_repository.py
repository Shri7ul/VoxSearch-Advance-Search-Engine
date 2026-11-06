"""Repository for ``voxsearch.transcript_segments`` (Goal 08).

Segment ordering is always ``start_seconds ASC, id ASC`` — the natural
timestamp order with a deterministic tiebreaker — which is what the search
feature needs for timestamped links.

Text search is a **basic case-insensitive substring filter** (``ILIKE``), not
PostgreSQL full-text search. The existing schema (Goals 06/07) has no
``tsvector`` column and no GIN index, and Goal 08 forbids migrations, so FTS
would require a schema change that is explicitly out of scope. There are no
embeddings or vector operations here either. The filter is a parameterized
SQLAlchemy expression (no SQL injection) and is documented so a later goal can
replace it with real FTS once a migration introduces the supporting column.
The separate ``original_text`` column (verbatim source) is intentionally NOT a
search target and NOT indexed; search continues to use the normalized ``"text"``
column so existing query semantics are unchanged.

This repository never commits.
"""
import uuid

from sqlalchemy import select

from app.db.models import TranscriptSegment
from app.repositories.base import BaseRepository


class TranscriptSegmentRepository(BaseRepository):
    """Data access for :class:`~app.db.models.TranscriptSegment` rows."""

    def create_segment(
        self,
        *,
        transcript_asset_id: uuid.UUID,
        video_id: uuid.UUID,
        start_seconds: float,
        end_seconds: float,
        segment_text: str,
        language: str | None = None,
        original_text: str | None = None,
    ) -> TranscriptSegment:
        """Insert a segment and flush (no commit).

        ``segment_text`` is the NORMALIZED text (physical ``"text"`` column);
        ``original_text`` is the optional verbatim source text (physical
        ``"original_text"`` column, nullable). Callers that do not have an
        original (e.g. the incremental API, or pre-migration data paths) leave
        it ``None``. The two values are never combined or overwritten.

        The timestamps/text are validated in Python as defense in depth for the
        existing DB check constraints (start >= 0, end > start, non-empty text),
        giving callers a clear ``ValueError`` before the flush. The composite FK
        ``fk_transcript_segments_transcript_asset_id_video_id`` guarantees the
        asset belongs to ``video_id``; a cross-video pair raises
        ``IntegrityError`` and is never swallowed here.
        """
        if start_seconds < 0:
            raise ValueError("start_seconds must be >= 0")
        if end_seconds <= start_seconds:
            raise ValueError("end_seconds must be > start_seconds")
        if not segment_text or not segment_text.strip():
            raise ValueError("segment_text must be non-empty")
        segment = TranscriptSegment(
            transcript_asset_id=transcript_asset_id,
            video_id=video_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            segment_text=segment_text,
            original_text=original_text,
            language=language,
        )
        self.session.add(segment)
        self.flush()
        return segment

    def get_segment_by_id(self, segment_id: uuid.UUID) -> TranscriptSegment | None:
        """Return a segment by primary key, or ``None``."""
        return self.session.get(TranscriptSegment, segment_id)

    def list_segments_by_video(
        self,
        video_id: uuid.UUID,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[TranscriptSegment]:
        """List a video's segments in timestamp order, with pagination."""
        stmt = (
            select(TranscriptSegment)
            .where(TranscriptSegment.video_id == video_id)
            .order_by(
                TranscriptSegment.start_seconds.asc(), TranscriptSegment.id.asc()
            )
        )
        stmt = self.paginate(stmt, limit=limit, offset=offset)
        return list(self.session.scalars(stmt).all())

    def list_segments_by_asset(
        self,
        asset_id: uuid.UUID,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[TranscriptSegment]:
        """List an asset's segments in timestamp order, with pagination."""
        stmt = (
            select(TranscriptSegment)
            .where(TranscriptSegment.transcript_asset_id == asset_id)
            .order_by(
                TranscriptSegment.start_seconds.asc(), TranscriptSegment.id.asc()
            )
        )
        stmt = self.paginate(stmt, limit=limit, offset=offset)
        return list(self.session.scalars(stmt).all())

    def search_segments_by_text(
        self,
        query: str,
        *,
        video_id: uuid.UUID | None = None,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[TranscriptSegment]:
        """Case-insensitive substring search over segment text.

        A blank query returns an empty list without hitting the database. When
        ``video_id`` is supplied the search is scoped to that video. Results are
        in timestamp order with ``id`` as tiebreaker, and paginated.
        """
        if not query or not query.strip():
            return []
        stmt = select(TranscriptSegment).where(
            TranscriptSegment.segment_text.ilike(f"%{query}%")
        )
        if video_id is not None:
            stmt = stmt.where(TranscriptSegment.video_id == video_id)
        stmt = stmt.order_by(
            TranscriptSegment.start_seconds.asc(), TranscriptSegment.id.asc()
        )
        stmt = self.paginate(stmt, limit=limit, offset=offset)
        return list(self.session.scalars(stmt).all())
