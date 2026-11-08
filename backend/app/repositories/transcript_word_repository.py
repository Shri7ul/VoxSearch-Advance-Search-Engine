"""Repository for ``voxsearch.transcript_words`` (Goal 12).

Rows are VALIDATED, VIDEO-timeline integer-ms words. Timestamps are never
interpolated / backfilled / clamped here — that guarantee is enforced upstream
in :mod:`app.transcripts.word_alignment`; this repository only persists what it
is given and never commits.

The pure :class:`AlignedWord` uses ``ordinal`` for the 0-based position in the
segment's original token stream; the spec-compliant DB column is ``word_index``
(design-freeze §3.3 / §7 step 8: ``word_index = ordinal``). The mapping happens
in :meth:`add_words`. ``word`` stores the VERBATIM original token slice, never
the provider-cleaned text.

Replacement semantics (approved retry policy): a successful replacement run
deletes the superseded run's word set and inserts its own inside ONE service
transaction. :meth:`delete_words_by_run` and :meth:`add_words` both flush so the
ordering inside that transaction is deterministic (supersede -> delete -> insert
-> promote), keeping the pair's partial UNIQUE index valid at every flush.
"""
import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select

from app.db.models import TranscriptWord
from app.repositories.base import BaseRepository
from app.transcripts.word_contract import AlignedWord


class TranscriptWordRepository(BaseRepository):
    """Data access for :class:`~app.db.models.TranscriptWord` rows."""

    def add_words(
        self,
        words: Sequence[AlignedWord],
        *,
        alignment_run_id: uuid.UUID,
        transcript_asset_id: uuid.UUID,
        video_id: uuid.UUID,
    ) -> int:
        """Bulk-insert validated words for ONE run and flush (no commit).

        Each :class:`AlignedWord` (pure dataclass) supplies the segment id,
        ordinal (mapped to the ``word_index`` DB column), verbatim text,
        original-text char span, and VIDEO-timeline ms timestamps; the run /
        asset / video ids are supplied by the service so the composite
        ownership FKs resolve to the SAME lineage. Returns the number of rows
        added.
        """
        for w in words:
            self.session.add(
                TranscriptWord(
                    alignment_run_id=alignment_run_id,
                    segment_id=w.segment_id,
                    transcript_asset_id=transcript_asset_id,
                    video_id=video_id,
                    word_index=w.ordinal,
                    word=w.text,
                    char_start=w.char_start,
                    char_end=w.char_end,
                    start_ms=w.start_ms,
                    end_ms=w.end_ms,
                )
            )
        self.flush()
        return len(words)

    def delete_words_by_run(self, alignment_run_id: uuid.UUID) -> int:
        """Delete a run's entire word set and flush (no commit).

        Used when a run is superseded. Returns the number of rows deleted.
        """
        stmt = delete(TranscriptWord).where(
            TranscriptWord.alignment_run_id == alignment_run_id
        )
        result = self.session.execute(stmt)
        self.flush()
        return result.rowcount or 0

    def count_words_by_run(self, alignment_run_id: uuid.UUID) -> int:
        """Count a run's word rows (read-only)."""
        stmt = select(func.count()).select_from(TranscriptWord).where(
            TranscriptWord.alignment_run_id == alignment_run_id
        )
        return int(self.session.scalar(stmt) or 0)

    def list_words_by_run(
        self, alignment_run_id: uuid.UUID
    ) -> list[TranscriptWord]:
        """List a run's words in ``(segment_id, word_index)`` order (read-only)."""
        stmt = (
            select(TranscriptWord)
            .where(TranscriptWord.alignment_run_id == alignment_run_id)
            .order_by(
                TranscriptWord.segment_id.asc(), TranscriptWord.word_index.asc()
            )
        )
        return list(self.session.scalars(stmt).all())

    def list_words_by_asset(
        self, transcript_asset_id: uuid.UUID
    ) -> list[TranscriptWord]:
        """List an asset's words in ``(segment_id, word_index)`` order (read-only)."""
        stmt = (
            select(TranscriptWord)
            .where(
                TranscriptWord.transcript_asset_id == transcript_asset_id
            )
            .order_by(
                TranscriptWord.segment_id.asc(), TranscriptWord.word_index.asc()
            )
        )
        return list(self.session.scalars(stmt).all())
