"""Transcript ingestion service (Goal 09): validated, atomic ingestion.

Orchestration boundary — why repositories are composed directly
--------------------------------------------------------------
This service composes the EXISTING repository abstractions inside ONE
:func:`app.services.transaction` block. It deliberately does NOT call the
existing ``TranscriptService.create_transcript_asset()`` / ``save_segments()``
methods: each of those owns its OWN ``transaction()`` (a separate commit), so
sequencing them would produce two commits and a failure in the second would
leave the asset persisted with no segments — violating "no partially persisted
transcript data". A single atomic transaction is required, so the repositories
are composed directly here (documented technical reason). Existing service
logic is still reused: ``ContentRightsService.is_active`` provides the
fail-closed authorization check, and ``transaction`` owns commit/rollback.
This duplicates NO functionality — those methods remain the incremental
operation API; ingestion is a distinct, all-or-nothing operation.

Authorization (fail closed)
---------------------------
Before ANY write: the video must exist; the content-right must exist, belong to
this video, and pass ``ContentRightsService.is_active`` (rejects revoked,
expired, not-yet-granted, inactive, or otherwise unauthorized records). No
record is ever auto-created or auto-authorized. The Goal 07 composite FK is a
second, database-level ownership guarantee.

Processing status lifecycle (existing ``status`` column; NO status migration)
-----------------------------------------------------------------------------
* ``processing`` — set when the asset is created at the start of the atomic
  unit ("processing started"). It is TRANSIENT: never observable at rest,
  because the unit commits only after completion.
* ``completed`` — set via the existing ``update_asset_status`` immediately
  before the single commit ("processing completed"). It is the ONLY status
  persisted on success.
* ``failed`` — NOT a persisted row, by design. Any failure (validation,
  authorization, or DB error) either happens before the transaction or rolls
  the whole transaction back, so the database is left completely unchanged
  (zero assets, zero segments). This is the strongest reading of "no partially
  persisted transcript data". Failures surface as raised exceptions;
  :data:`FAILED_STATUS` documents the outcome label only.

Repeated ingestion
------------------
The current schema has no checksum / dedup key and no unique constraint on
(video_id, source_type, language), so each ingestion creates a NEW independent
asset. Repeated ingestion is allowed and deterministic; it is NOT silently
deduplicated. (Architecture §7.3 notes checksum-based dedup as a future
concern requiring schema support — out of scope for this foundation.)

Original vs normalized text (Goal 09 corrective)
------------------------------------------------
Each segment is persisted with BOTH values in separate columns: the verbatim
original in ``original_text`` (never trimmed, NFC-composed, or rewritten) and
the deterministic normalized form in the existing ``"text"`` column. The
normalized column keeps its original meaning and remains the search target.
The single additive migration introduces only the nullable ``original_text``
column; it performs NO backfill (legacy rows keep NULL, an honest "not
captured" marker) and NEVER overwrites existing data.

Scope & safety
--------------
Nothing here downloads captions, calls ASR, embeds, reranks, or searches. No
credentials, API keys, or database URLs are logged or returned. Optional
``speaker`` / ``metadata`` input fields are surfaced in the result but NOT
persisted (no schema columns exist; adding them remains out of scope).
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.repositories.content_rights_repository import ContentRightsRepository
from app.repositories.transcript_asset_repository import TranscriptAssetRepository
from app.repositories.transcript_segment_repository import TranscriptSegmentRepository
from app.repositories.video_repository import VideoRepository
from app.services import (
    ContentRightsNotEligibleError,
    VoxSearchServiceError,
    transaction,
)
from app.services.content_rights_service import ContentRightsService
from app.transcripts.contracts import TranscriptInput
from app.transcripts.normalization import normalize_text
from app.transcripts.validation import validate_transcript

PROCESSING_STATUS = "processing"
COMPLETED_STATUS = "completed"
FAILED_STATUS = "failed"  # outcome label ONLY; never a persisted row


class VideoNotFoundError(VoxSearchServiceError):
    """The video referenced by the transcript does not exist."""


@dataclass(frozen=True)
class IngestedSegment:
    """One ingested segment, preserving BOTH original and normalized text.

    Both values are also persisted separately (``original_text`` / ``"text"``);
    this result object simply avoids a re-query for callers.
    """

    index: int
    start_time: float
    end_time: float
    original_text: str
    normalized_text: str
    speaker: str | None
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class IngestionResult:
    """Clear, credential-free outcome of a successful ingestion."""

    asset_id: uuid.UUID
    video_id: uuid.UUID
    content_right_id: uuid.UUID
    source_type: str
    language: str
    status: str
    segment_count: int
    overlapping_pair_count: int
    segments: tuple[IngestedSegment, ...]


class TranscriptIngestionService:
    """Atomic, validated transcript ingestion. Owns ONE transaction per call."""

    def __init__(
        self,
        session: Session,
        *,
        videos: VideoRepository | None = None,
        rights: ContentRightsRepository | None = None,
        assets: TranscriptAssetRepository | None = None,
        segments: TranscriptSegmentRepository | None = None,
    ) -> None:
        self._session = session
        self.videos = videos or VideoRepository(session)
        self.rights = rights or ContentRightsRepository(session)
        self.assets = assets or TranscriptAssetRepository(session)
        self.segments = segments or TranscriptSegmentRepository(session)

    def ingest(
        self,
        payload: TranscriptInput | Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> IngestionResult:
        """Validate, authorize, normalize, and atomically persist a transcript.

        Accepts an already-built :class:`TranscriptInput` or a raw mapping
        (validated here via Pydantic, producing a clear ``ValidationError`` on
        malformed input). Raises on any validation / authorization / DB failure
        with NO partial persistence.
        """
        contract = (
            payload
            if isinstance(payload, TranscriptInput)
            else TranscriptInput.model_validate(payload)
        )

        # 1. Video existence + known duration (read-only, before any write).
        video = self.videos.get_video_by_id(contract.video_id)
        if video is None:
            raise VideoNotFoundError(
                "The video referenced by this transcript does not exist."
            )

        # 2. Contextual validation (chronology, overlap policy, duration bound).
        report = validate_transcript(
            contract, video_duration_seconds=video.duration_seconds
        )

        # 3. Fail-closed rights ownership + authorization (read-only).
        rights = self.rights.get_rights_by_id(contract.content_right_id)
        if (
            rights is None
            or rights.video_id != contract.video_id
            or not ContentRightsService.is_active(rights, now=now)
        ):
            raise ContentRightsNotEligibleError(
                "No valid, currently active rights record for this video."
            )

        # 4. Deterministic normalization (pure); keep original + normalized.
        prepared: list[IngestedSegment] = []
        for i, seg in enumerate(contract.segments):
            norm = normalize_text(seg.text)
            prepared.append(
                IngestedSegment(
                    index=i,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    original_text=norm.original,
                    normalized_text=norm.normalized,
                    speaker=seg.speaker,
                    metadata=seg.metadata,
                )
            )

        # 5. ONE atomic transaction: asset(processing) + all segments +
        #    status(completed). Any failure rolls back ALL of it.
        with transaction(self._session):
            asset = self.assets.create_transcript_asset(
                video_id=contract.video_id,
                content_right_id=contract.content_right_id,
                source_type=contract.source_type.value,
                language=contract.language,
                status=PROCESSING_STATUS,
            )
            for seg in prepared:
                self.segments.create_segment(
                    transcript_asset_id=asset.id,
                    video_id=contract.video_id,
                    start_seconds=seg.start_time,
                    end_seconds=seg.end_time,
                    segment_text=seg.normalized_text,
                    original_text=seg.original_text,
                    language=contract.language,
                )
            self.assets.update_asset_status(asset.id, COMPLETED_STATUS)

        return IngestionResult(
            asset_id=asset.id,
            video_id=contract.video_id,
            content_right_id=contract.content_right_id,
            source_type=contract.source_type.value,
            language=contract.language,
            status=COMPLETED_STATUS,
            segment_count=report.segment_count,
            overlapping_pair_count=report.overlapping_pair_count,
            segments=tuple(prepared),
        )


__all__ = [
    "COMPLETED_STATUS",
    "FAILED_STATUS",
    "PROCESSING_STATUS",
    "IngestedSegment",
    "IngestionResult",
    "TranscriptIngestionService",
    "VideoNotFoundError",
]
