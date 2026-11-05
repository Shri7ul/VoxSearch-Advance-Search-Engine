"""Video service (Goal 08): create-or-retrieve, listing, metadata updates.

As with channels, ``uq_videos_youtube_video_id`` is the source of truth for
duplicate YouTube video IDs. ``create_or_get_video`` resolves duplicates
consistently and never hides a non-duplicate DB error.
"""
import uuid
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Video
from app.repositories.base import UNSET, _Unset
from app.repositories.video_repository import VideoRepository
from app.services import transaction


class VideoService:
    """Application operations for videos. Owns transaction boundaries."""

    def __init__(
        self,
        session: Session,
        videos: VideoRepository | None = None,
    ) -> None:
        self._session = session
        self.videos = videos or VideoRepository(session)

    def get_video_by_youtube_id(self, youtube_video_id: str) -> Video | None:
        """Read-only lookup; no commit."""
        return self.videos.get_video_by_youtube_id(youtube_video_id)

    def list_videos_by_channel(
        self,
        channel_id: uuid.UUID,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[Video]:
        """Read-only, paginated listing delegated to the repository."""
        return self.videos.list_videos_by_channel(
            channel_id, limit=limit, offset=offset
        )

    def create_or_get_video(
        self,
        *,
        channel_id: uuid.UUID,
        youtube_video_id: str,
        title: str,
        description: str | None = None,
        duration_seconds: int | None = None,
        published_at: datetime | None = None,
    ) -> Video:
        """Return the existing video for this YouTube ID, or create it.

        One transaction on a miss; a concurrent duplicate is rolled back and
        re-read. Discovery alone does NOT imply authorization — no rights
        record is created here.
        """
        existing = self.videos.get_video_by_youtube_id(youtube_video_id)
        if existing is not None:
            return existing
        try:
            with transaction(self._session):
                video = self.videos.create_video(
                    channel_id=channel_id,
                    youtube_video_id=youtube_video_id,
                    title=title,
                    description=description,
                    duration_seconds=duration_seconds,
                    published_at=published_at,
                )
        except IntegrityError:
            existing = self.videos.get_video_by_youtube_id(youtube_video_id)
            if existing is not None:
                return existing
            raise
        return video

    def update_video_metadata(
        self,
        video_id: uuid.UUID,
        *,
        title: str | _Unset = UNSET,
        description: str | None | _Unset = UNSET,
        duration_seconds: int | None | _Unset = UNSET,
        published_at: datetime | None | _Unset = UNSET,
    ) -> Video | None:
        """Update a video's metadata in one transaction; ``None`` if absent."""
        video = self.videos.get_video_by_id(video_id)
        if video is None:
            return None
        with transaction(self._session):
            return self.videos.update_video(
                video,
                title=title,
                description=description,
                duration_seconds=duration_seconds,
                published_at=published_at,
            )
