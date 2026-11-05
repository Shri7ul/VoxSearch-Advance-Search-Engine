"""Repository for ``voxsearch.transcript_assets`` (Goal 08).

Ownership integrity from Goal 07 is preserved: ``content_rights`` and
``video_id`` must reference a ``content_rights(id, video_id)`` pair belonging to
the SAME video (composite FK
``fk_transcript_assets_content_right_id_video_id``). The repository does not
weaken that guarantee; the owning ``TranscriptService`` additionally validates
the rights record *before* creating an asset. This repository never commits.
"""
import uuid

from sqlalchemy import select

from app.db.models import TranscriptAsset
from app.repositories.base import BaseRepository


class TranscriptAssetRepository(BaseRepository):
    """Data access for :class:`~app.db.models.TranscriptAsset` rows."""

    def create_transcript_asset(
        self,
        *,
        video_id: uuid.UUID,
        content_right_id: uuid.UUID,
        source_type: str,
        language: str,
        status: str,
    ) -> TranscriptAsset:
        """Insert an asset and flush (no commit).

        ``status`` is caller-supplied (the model documents a processing
        lifecycle but no enum is invented here). The composite FK guarantees
        ``content_right_id`` belongs to ``video_id``; a cross-video pair raises
        ``IntegrityError`` at flush and is never swallowed here.
        """
        asset = TranscriptAsset(
            video_id=video_id,
            content_right_id=content_right_id,
            source_type=source_type,
            language=language,
            status=status,
        )
        self.session.add(asset)
        self.flush()
        return asset

    def get_asset_by_id(self, asset_id: uuid.UUID) -> TranscriptAsset | None:
        """Return an asset by primary key, or ``None``."""
        return self.session.get(TranscriptAsset, asset_id)

    def list_assets_by_video(
        self,
        video_id: uuid.UUID,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[TranscriptAsset]:
        """List a video's assets, newest-first, with pagination.

        ``ix_transcript_assets_video_id`` backs the filter; ``id`` is the
        deterministic tiebreaker.
        """
        stmt = (
            select(TranscriptAsset)
            .where(TranscriptAsset.video_id == video_id)
            .order_by(TranscriptAsset.created_at.desc(), TranscriptAsset.id.desc())
        )
        stmt = self.paginate(stmt, limit=limit, offset=offset)
        return list(self.session.scalars(stmt).all())

    def update_asset_status(
        self,
        asset_id: uuid.UUID,
        status: str,
    ) -> TranscriptAsset | None:
        """Update an asset's processing status and flush (no commit).

        Returns ``None`` when the asset does not exist. Only ``status`` is
        changed; ownership columns are never touched.
        """
        asset = self.get_asset_by_id(asset_id)
        if asset is None:
            return None
        asset.status = status
        self.flush()
        return asset
