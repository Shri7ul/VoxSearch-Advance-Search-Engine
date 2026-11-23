"""Goal 22 — PostgreSQL full-text (exact keyword) retrieval over video chunks.

This repository is the lexical counterpart of Goal 17's dense vector retrieval.
It selects verbatim chunk text matching a keyword query using PostgreSQL's
``to_tsvector`` / ``websearch_to_tsquery`` / ``ts_rank`` and the ``@@`` match
operator, returning the top-k ranked chunks.

Safety and scope:

* The search text is an SQL **bind parameter**, never an interpolated string.
  The fixed regconfig constant is also bound, but explicitly cast to
  ``regconfig``: PostgreSQL requires ``to_tsvector(regconfig, text)` and
  ``websearch_to_tsquery(regconfig, text)`, and a bare Python ``str`` parameter
  would arrive as ``text`` and fail function resolution at runtime.
* ``websearch_to_tsquery`` is used instead of raw ``to_tsquery`` so user input
  cannot inject tsquery operators or cause syntax errors.
* Filtering is applied in ``WHERE`` before ranking and ``LIMIT``; ordering is
  deterministic (rank descending, then chunk_index, then id).
* The stored 1024-dimensional embedding column is never selected.
* Read-only: no write, commit, flush, DDL, index or migration. Goal 22 does not
  create a GIN index — on large tables this is a sequential scan by design.
  Dense+lexical combination (Goal 23), RRF (Goal 24) and dedup (Goal 25) are out
  of scope.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import cast, func, literal, select
from sqlalchemy.dialects.postgresql import REGCONFIG

from app.db.models import VideoChunk
from app.repositories.base import BaseRepository
from app.repositories.video_chunks import DEFAULT_TOP_K, validate_top_k

# Fixed regconfig. It is bound (never derived from user input) and cast to the
# PostgreSQL ``regconfig`` type so the two-argument FTS functions resolve.
FTS_CONFIGURATION = "english"


@dataclass(frozen=True)
class LexicalSearchResult:
    """One keyword-matched chunk with its PostgreSQL relevance rank."""

    id: uuid.UUID
    video_id: uuid.UUID
    transcript_asset_id: uuid.UUID
    content_right_id: uuid.UUID
    chunk_index: int
    text: str
    start_ms: int
    end_ms: int
    duration_ms: int
    model_id: str
    rank: float

    @classmethod
    def from_row(cls, row: Sequence[object]) -> "LexicalSearchResult":
        return cls(
            id=row[0],
            video_id=row[1],
            transcript_asset_id=row[2],
            content_right_id=row[3],
            chunk_index=row[4],
            text=row[5],
            start_ms=row[6],
            end_ms=row[7],
            duration_ms=row[8],
            model_id=row[9],
            rank=float(row[10]),
        )


class LexicalSearchRepository(BaseRepository):
    """Read-only PostgreSQL FTS access to :class:`~app.db.models.VideoChunk`."""

    def search_keywords(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        video_id: uuid.UUID | None = None,
        transcript_asset_id: uuid.UUID | None = None,
        model_id: str | None = None,
    ) -> list[LexicalSearchResult]:
        """Return the top-k chunks whose text matches ``query``.

        ``query`` is passed verbatim to ``websearch_to_tsquery`` as a bind
        parameter; empty / whitespace-only input is rejected by the service
        layer. ``top_k`` is bounded to ``[1, MAX_TOP_K]`` at this boundary.
        """
        top_k = validate_top_k(top_k)
        config = cast(literal(FTS_CONFIGURATION), REGCONFIG)
        document = func.to_tsvector(config, VideoChunk.text)
        tsquery = func.websearch_to_tsquery(config, query)
        rank = func.ts_rank(document, tsquery).label("rank")

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
            VideoChunk.model_id,
            rank,
        ).where(document.op("@@")(tsquery))
        if video_id is not None:
            stmt = stmt.where(VideoChunk.video_id == video_id)
        if transcript_asset_id is not None:
            stmt = stmt.where(VideoChunk.transcript_asset_id == transcript_asset_id)
        if model_id is not None:
            stmt = stmt.where(VideoChunk.model_id == model_id)

        stmt = stmt.order_by(
            rank.desc(),
            VideoChunk.chunk_index.asc(),
            VideoChunk.id.asc(),
        ).limit(top_k)
        return [LexicalSearchResult.from_row(row) for row in self.session.execute(stmt).all()]


__all__ = [
    "FTS_CONFIGURATION",
    "LexicalSearchRepository",
    "LexicalSearchResult",
]
