"""Repository for ``voxsearch.transcript_alignment_runs`` (Goal 12).

One row = one alignment ATTEMPT, tracked for retry / audit. The status lifecycle
is owned by :mod:`app.services.word_alignment_service`; this repository only
creates / reads / updates rows and NEVER commits.

Every attempt records the spec-required audit metadata: ``attempt_number``,
``provider`` / ``provider_version`` / ``align_model_name``, ``audio_duration_ms``,
segment/aligned/skipped/word counts, ``started_at`` / ``finished_at`` and a
closed, credential-free ``failure_reason``. Nullable count/timing columns stay
NULL when not yet known — they are never guessed.

Concurrency safety (approved decisions #4, #8, and the retry policy): the
partial UNIQUE index ``uq_alignment_runs_current_success`` permits at most
ONE row with a ``current success`` status (``completed`` / ``partial``) per
``(transcript_asset_id, source_audio_asset_id)``. :meth:`find_current_success`
locates that row so a successful replacement can supersede it atomically in the
SAME service transaction. Superseded / failed runs are excluded from the index
by its WHERE clause.

Status values below are kept as local literals so the repository layer stays
decoupled from :mod:`app.transcripts` (the same philosophy as
``app.db.models.ALIGNMENT_RUN_STATUS_VALUES``); a Goal 12 test asserts all three
sources agree.
"""
import uuid
from datetime import datetime

from sqlalchemy import func, select

from app.db.models import TranscriptAlignmentRun
from app.repositories.base import BaseRepository

RUNNING_STATUS = "running"
COMPLETED_STATUS = "completed"
PARTIAL_STATUS = "partial"
FAILED_STATUS = "failed"
UNSUPPORTED_LANGUAGE_STATUS = "unsupported_language"
SUPERSEDED_STATUS = "superseded"
CURRENT_SUCCESS_STATUSES = (COMPLETED_STATUS, PARTIAL_STATUS)


class TranscriptAlignmentRunRepository(BaseRepository):
    """Data access for :class:`~app.db.models.TranscriptAlignmentRun` rows."""

    def next_attempt_number(
        self,
        transcript_asset_id: uuid.UUID,
        source_audio_asset_id: uuid.UUID,
        *,
        provider: str,
        provider_version: str,
    ) -> int:
        """Return the next 1-based attempt number for this (asset, audio,
        provider, version) scope.

        Scoped to EXACTLY the columns of
        ``uq_alignment_runs_attempt`` so attempt numbers are stable per
        provider/version and never collide. Read-only; never commits.
        """
        stmt = select(func.max(TranscriptAlignmentRun.attempt_number)).where(
            TranscriptAlignmentRun.transcript_asset_id == transcript_asset_id,
            TranscriptAlignmentRun.source_audio_asset_id
            == source_audio_asset_id,
            TranscriptAlignmentRun.provider == provider,
            TranscriptAlignmentRun.provider_version == provider_version,
        )
        current = self.session.scalar(stmt)
        return int(current or 0) + 1

    def create_run(
        self,
        *,
        video_id: uuid.UUID,
        transcript_asset_id: uuid.UUID,
        source_audio_asset_id: uuid.UUID,
        content_right_id: uuid.UUID,
        language: str,
        status: str,
        attempt_number: int,
        provider: str,
        provider_version: str,
        audio_duration_ms: int | None = None,
        align_model_name: str | None = None,
        started_at: datetime | None = None,
    ) -> TranscriptAlignmentRun:
        """Insert a RUNNING attempt and flush (no commit).

        ``status`` is caller-supplied (the service passes ``RUNNING_STATUS`` in
        Phase A). The two composite FKs guarantee the asset and the audio asset
        are both authorized by the SAME ``content_right_id`` / ``video_id``; a
        mismatch raises ``IntegrityError`` at flush and is never swallowed here.
        NOT NULL audit columns (``attempt_number`` / ``provider`` /
        ``provider_version``) are supplied by the service; nullable
        count/timing columns start NULL (honest "not yet known").
        """
        run = TranscriptAlignmentRun(
            video_id=video_id,
            transcript_asset_id=transcript_asset_id,
            source_audio_asset_id=source_audio_asset_id,
            content_right_id=content_right_id,
            language=language,
            status=status,
            attempt_number=attempt_number,
            provider=provider,
            provider_version=provider_version,
            audio_duration_ms=audio_duration_ms,
            align_model_name=align_model_name,
            started_at=started_at,
        )
        self.session.add(run)
        self.flush()
        return run

    def get_run_by_id(self, run_id: uuid.UUID) -> TranscriptAlignmentRun | None:
        """Return a run by primary key, or ``None``."""
        return self.session.get(TranscriptAlignmentRun, run_id)

    def find_current_success(
        self,
        transcript_asset_id: uuid.UUID,
        source_audio_asset_id: uuid.UUID,
        *,
        exclude_run_id: uuid.UUID | None = None,
    ) -> TranscriptAlignmentRun | None:
        """Return the ONE current successful run for a pair, or ``None``.

        Scoped to ``CURRENT_SUCCESS_STATUSES`` (the same set the partial UNIQUE
        index covers). ``exclude_run_id`` lets the service omit the run it is
        currently finalizing. Deterministic tiebreaker on ``id``.
        """
        stmt = (
            select(TranscriptAlignmentRun)
            .where(
                TranscriptAlignmentRun.transcript_asset_id
                == transcript_asset_id
            )
            .where(
                TranscriptAlignmentRun.source_audio_asset_id
                == source_audio_asset_id
            )
            .where(TranscriptAlignmentRun.status.in_(CURRENT_SUCCESS_STATUSES))
        )
        if exclude_run_id is not None:
            stmt = stmt.where(TranscriptAlignmentRun.id != exclude_run_id)
        stmt = stmt.order_by(TranscriptAlignmentRun.id.asc())
        return self.session.scalars(stmt).first()

    def set_run_outcome(
        self,
        run_id: uuid.UUID,
        *,
        status: str,
        failure_reason: str | None = None,
        word_count: int | None = None,
        segment_count: int | None = None,
        aligned_segment_count: int | None = None,
        skipped_segment_count: int | None = None,
        finished_at: datetime | None = None,
    ) -> TranscriptAlignmentRun | None:
        """Set a run's terminal status + audit metadata; flush (no commit).

        Only the outcome columns are changed; ownership / identity columns are
        never touched. Optional metadata is written only when supplied (NULL =
        not known, never guessed). Returns ``None`` when the run does not exist.
        """
        run = self.get_run_by_id(run_id)
        if run is None:
            return None
        run.status = status
        run.failure_reason = failure_reason
        if word_count is not None:
            run.word_count = word_count
        if segment_count is not None:
            run.segment_count = segment_count
        if aligned_segment_count is not None:
            run.aligned_segment_count = aligned_segment_count
        if skipped_segment_count is not None:
            run.skipped_segment_count = skipped_segment_count
        if finished_at is not None:
            run.finished_at = finished_at
        self.flush()
        return run

    def supersede_run(self, run_id: uuid.UUID) -> TranscriptAlignmentRun | None:
        """Mark a previously-successful run ``superseded``; flush (no commit).

        Called ONLY inside the winning replacement's service transaction,
        BEFORE the new run is promoted to a current-success status. This removes
        the old row from the partial UNIQUE index so the pair is free for the
        new run. Returns ``None`` when the row does not exist.
        """
        run = self.get_run_by_id(run_id)
        if run is None:
            return None
        run.status = SUPERSEDED_STATUS
        self.flush()
        return run

    def list_runs_by_asset(
        self,
        transcript_asset_id: uuid.UUID,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[TranscriptAlignmentRun]:
        """List an asset's alignment attempts newest-first, paginated."""
        stmt = (
            select(TranscriptAlignmentRun)
            .where(
                TranscriptAlignmentRun.transcript_asset_id
                == transcript_asset_id
            )
            .order_by(
                TranscriptAlignmentRun.created_at.desc(),
                TranscriptAlignmentRun.id.desc(),
            )
        )
        stmt = self.paginate(stmt, limit=limit, offset=offset)
        return list(self.session.scalars(stmt).all())
