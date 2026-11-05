"""Transcript service (Goal 08): assets, status, and timestamped segments.

Data-integrity rules enforced here (on top of the Goal 06/07 DB constraints):

* A transcript asset is NEVER created without a valid, currently active rights
  record that belongs to the same video (fail closed).
* A segment is NEVER saved against an asset belonging to a different video:
  the asset is fetched and its ``video_id`` is validated before any write.
  The Goal 07 composite FK is a second, database-level guarantee.
* Multi-step operations use exactly one transaction; any exception rolls back
  and is re-raised (DB errors are never swallowed).
* No transcript ingestion or ASR is implemented — callers supply the segment
  data. No embeddings/vector search.

The service never automatically authorizes a video and never creates rights
records; it only validates existing ones via
:meth:`~app.services.content_rights_service.ContentRightsService.is_active`.
"""
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import TranscriptAsset, TranscriptSegment
from app.repositories.content_rights_repository import ContentRightsRepository
from app.repositories.transcript_asset_repository import TranscriptAssetRepository
from app.repositories.transcript_segment_repository import (
    TranscriptSegmentRepository,
)
from app.services import (
    ContentRightsNotEligibleError,
    TranscriptAssetNotFoundError,
    TranscriptAssetOwnershipError,
    transaction,
)
from app.services.content_rights_service import ContentRightsService


class TranscriptService:
    """Application operations for transcript assets/segments. Owns transactions."""

    def __init__(
        self,
        session: Session,
        assets: TranscriptAssetRepository | None = None,
        segments: TranscriptSegmentRepository | None = None,
        rights: ContentRightsRepository | None = None,
    ) -> None:
        self._session = session
        self.assets = assets or TranscriptAssetRepository(session)
        self.segments = segments or TranscriptSegmentRepository(session)
        self.rights = rights or ContentRightsRepository(session)

    def create_transcript_asset(
        self,
        *,
        video_id: uuid.UUID,
        content_right_id: uuid.UUID,
        source_type: str,
        language: str,
        status: str,
        now: datetime | None = None,
    ) -> TranscriptAsset:
        """Create an asset only when a valid, active rights record exists.

        Raises :class:`ContentRightsNotEligibleError` (fail closed) when the
        rights record is missing, belongs to another video, or is not currently
        active. The Goal 07 composite FK still enforces ownership at the DB.
        """
        rights = self.rights.get_rights_by_id(content_right_id)
        if (
            rights is None
            or rights.video_id != video_id
            or not ContentRightsService.is_active(rights, now=now)
        ):
            raise ContentRightsNotEligibleError(
                "No valid, currently active rights record for this video."
            )
        with transaction(self._session):
            return self.assets.create_transcript_asset(
                video_id=video_id,
                content_right_id=content_right_id,
                source_type=source_type,
                language=language,
                status=status,
            )

    def update_processing_status(
        self,
        asset_id: uuid.UUID,
        status: str,
    ) -> TranscriptAsset | None:
        """Update an asset's status in one transaction; ``None`` if absent."""
        with transaction(self._session):
            return self.assets.update_asset_status(asset_id, status)

    def save_segment(
        self,
        *,
        video_id: uuid.UUID,
        transcript_asset_id: uuid.UUID,
        start_seconds: float,
        end_seconds: float,
        segment_text: str,
        language: str | None = None,
    ) -> TranscriptSegment:
        """Validate asset ownership, then save one segment in one transaction."""
        self._require_asset_ownership(video_id, transcript_asset_id)
        with transaction(self._session):
            return self.segments.create_segment(
                transcript_asset_id=transcript_asset_id,
                video_id=video_id,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                segment_text=segment_text,
                language=language,
            )

    def save_segments(
        self,
        *,
        video_id: uuid.UUID,
        transcript_asset_id: uuid.UUID,
        segments: Sequence[dict[str, Any]],
    ) -> list[TranscriptSegment]:
        """Validate ownership once, then save many segments in ONE transaction.

        Each item supplies ``start_seconds``, ``end_seconds``, ``segment_text``
        and optionally ``language``. All inserts share a single commit; a
        failure in any one rolls back the whole batch.
        """
        self._require_asset_ownership(video_id, transcript_asset_id)
        with transaction(self._session):
            return [
                self.segments.create_segment(
                    transcript_asset_id=transcript_asset_id,
                    video_id=video_id,
                    start_seconds=item["start_seconds"],
                    end_seconds=item["end_seconds"],
                    segment_text=item["segment_text"],
                    language=item.get("language"),
                )
                for item in segments
            ]

    def _require_asset_ownership(
        self,
        video_id: uuid.UUID,
        asset_id: uuid.UUID,
    ) -> TranscriptAsset:
        """Fetch an asset and prove it belongs to ``video_id``."""
        asset = self.assets.get_asset_by_id(asset_id)
        if asset is None:
            raise TranscriptAssetNotFoundError(
                "The requested transcript asset does not exist."
            )
        if asset.video_id != video_id:
            raise TranscriptAssetOwnershipError(
                "The transcript asset belongs to a different video."
            )
        return asset

    def get_timestamped_segments(
        self,
        *,
        video_id: uuid.UUID | None = None,
        asset_id: uuid.UUID | None = None,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[TranscriptSegment]:
        """Read-only retrieval of timestamped, deterministically ordered segments.

        Exactly one scope (``asset_id`` or ``video_id``) is required; preferring
        ``asset_id`` keeps the query on the narrower index. No commit.
        """
        if asset_id is not None:
            return self.segments.list_segments_by_asset(
                asset_id, limit=limit, offset=offset
            )
        if video_id is not None:
            return self.segments.list_segments_by_video(
                video_id, limit=limit, offset=offset
            )
        raise ValueError("Provide video_id or asset_id to retrieve segments.")
