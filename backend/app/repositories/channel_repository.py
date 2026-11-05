"""Repository for ``voxsearch.channels`` (Goal 08, read/write, no schema changes).

All queries use SQLAlchemy 2.x ``select()`` / ``session.scalars()``. Listing is
paginated and deterministically ordered by ``created_at DESC, id DESC`` so
results never depend on physical row order. This repository never commits.
"""
import uuid

from sqlalchemy import select

from app.db.models import Channel
from app.repositories.base import UNSET, BaseRepository, _Unset


class ChannelRepository(BaseRepository):
    """Data access for :class:`~app.db.models.Channel` rows."""

    def create_channel(
        self,
        *,
        youtube_channel_id: str,
        title: str,
        description: str | None = None,
    ) -> Channel:
        """Insert a channel and flush so its UUID PK is available.

        ``youtube_channel_id`` uniqueness is enforced by the database
        (``uq_channels_youtube_channel_id``); a duplicate raises
        ``IntegrityError`` at flush. The repository does not catch it — the
        owning service decides how to handle duplicates.
        """
        channel = Channel(
            youtube_channel_id=youtube_channel_id,
            title=title,
            description=description,
        )
        self.session.add(channel)
        self.flush()
        return channel

    def get_channel_by_id(self, channel_id: uuid.UUID) -> Channel | None:
        """Return a channel by primary key, or ``None``."""
        return self.session.get(Channel, channel_id)

    def get_channel_by_youtube_id(self, youtube_channel_id: str) -> Channel | None:
        """Return the channel with this YouTube channel ID, or ``None``."""
        stmt = select(Channel).where(
            Channel.youtube_channel_id == youtube_channel_id
        )
        return self.session.scalars(stmt).first()

    def list_channels(
        self,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[Channel]:
        """List channels with stable, newest-first ordering and pagination."""
        stmt = select(Channel).order_by(
            Channel.created_at.desc(), Channel.id.desc()
        )
        stmt = self.paginate(stmt, limit=limit, offset=offset)
        return list(self.session.scalars(stmt).all())

    def update_channel(
        self,
        channel: Channel,
        *,
        title: str | _Unset = UNSET,
        description: str | None | _Unset = UNSET,
    ) -> Channel:
        """Update only the supplied metadata fields, then flush.

        ``UNSET`` distinguishes "leave unchanged" from an explicit ``None``
        (which clears the nullable ``description``). No commit is performed.
        """
        if not isinstance(title, _Unset):
            channel.title = title
        if not isinstance(description, _Unset):
            channel.description = description
        self.flush()
        return channel

    def delete_channel(self, channel: Channel) -> None:
        """Delete a channel row and flush (no commit).

        Downstream rows referencing this channel are protected by the existing
        foreign keys; nothing here drops or truncates related tables.
        """
        self.session.delete(channel)
        self.flush()
