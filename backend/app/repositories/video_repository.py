"""Repository for ``voxsearch.videos`` (Goal 08, read/write, no schema changes).

Listing by channel is paginated and deterministically ordered by
``created_at DESC, id DESC``. ``youtube_video_id`` uniqueness is DB-enforced
(``uq_videos_youtube_video_id``). This repository never commits.
"""
import uuid
from datetime import datetime

from sqlalchemy import select

from app.db.models import Video
from app.repositories.base import UNSET, BaseRepository, _Unset


class VideoRepository(BaseRepository):
    """Data access for :class:`~app.db.models.Video` rows."""

    def create_video(
        self,
        *,
        channel_id: uuid.UUID,
        youtube_video_id: str,
        title: str,
        description: str | None = None,
        duration_seconds: int | None = None,
        published_at: datetime | None = None,
    ) -> Video:
        """Insert a video and flush so its UUID PK is available.

        ``channel_id`` must reference an existing channel (enforced by
        ``fk_videos_channel_id_channels``). A duplicate ``youtube_video_id``
        raises ``IntegrityError`` at flush and is never swallowed here.
        """
        video = Video(
            channel_id=channel_id,
            youtube_video_id=youtube_video_id,
            title=title,
            description=description,
            duration_seconds=duration_seconds,
            published_at=published_at,
        )
        self.session.add(video)
        self.flush()
        return video

    def get_video_by_id(self, video_id: uuid.UUID) -> Video | None:
        """Return a video by primary key, or ``None``."""
        return self.session.get(Video, video_id)

    def get_video_by_youtube_id(self, youtube_video_id: str) -> Video | None:
        """Return the video with this YouTube video ID, or ``None``."""
        stmt = select(Video).where(Video.youtube_video_id == youtube_video_id)
        return self.session.scalars(stmt).first()

    def list_videos_by_channel(
        self,
        channel_id: uuid.UUID,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[Video]:
        """List a channel's videos, newest-first, with pagination.

        ``ix_videos_channel_id`` backs the ``channel_id`` filter; the ``id``
        tiebreaker keeps ordering deterministic. A single query avoids N+1.
        """
        stmt = (
            select(Video)
            .where(Video.channel_id == channel_id)
            .order_by(Video.created_at.desc(), Video.id.desc())
        )
        stmt = self.paginate(stmt, limit=limit, offset=offset)
        return list(self.session.scalars(stmt).all())

    def update_video(
        self,
        video: Video,
        *,
        title: str | _Unset = UNSET,
        description: str | None | _Unset = UNSET,
        duration_seconds: int | None | _Unset = UNSET,
        published_at: datetime | None | _Unset = UNSET,
    ) -> Video:
        """Update only the supplied metadata fields, then flush (no commit)."""
        if not isinstance(title, _Unset):
            video.title = title
        if not isinstance(description, _Unset):
            video.description = description
        if not isinstance(duration_seconds, _Unset):
            video.duration_seconds = duration_seconds
        if not isinstance(published_at, _Unset):
            video.published_at = published_at
        self.flush()
        return video

    def delete_video(self, video: Video) -> None:
        """Delete a video row and flush (no commit)."""
        self.session.delete(video)
        self.flush()
