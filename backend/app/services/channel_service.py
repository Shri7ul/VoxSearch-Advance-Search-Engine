"""Channel service (Goal 08): create-or-retrieve and metadata updates.

Duplicate YouTube channel IDs are handled consistently: the unique constraint
``uq_channels_youtube_channel_id`` is the source of truth, and
:meth:`create_or_get_channel` returns the existing row rather than inserting a
second one. A concurrent insert that wins the unique constraint is detected via
a narrow ``IntegrityError`` handler that rolls back (through ``transaction``)
and re-reads; any other error is re-raised untouched.
"""
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Channel
from app.repositories.base import UNSET, _Unset
from app.repositories.channel_repository import ChannelRepository
from app.services import transaction


class ChannelService:
    """Application operations for channels. Owns transaction boundaries."""

    def __init__(
        self,
        session: Session,
        channels: ChannelRepository | None = None,
    ) -> None:
        self._session = session
        self.channels = channels or ChannelRepository(session)

    def get_channel_by_youtube_id(
        self, youtube_channel_id: str
    ) -> Channel | None:
        """Read-only lookup; no transaction block, no commit."""
        return self.channels.get_channel_by_youtube_id(youtube_channel_id)

    def create_or_get_channel(
        self,
        *,
        youtube_channel_id: str,
        title: str,
        description: str | None = None,
    ) -> Channel:
        """Return the existing channel for this YouTube ID, or create it.

        A hit is a pure read (no commit). A miss creates + commits in one
        transaction. On a concurrent duplicate, roll back and return the row
        that won; never hide a non-duplicate DB error.
        """
        existing = self.channels.get_channel_by_youtube_id(youtube_channel_id)
        if existing is not None:
            return existing
        try:
            with transaction(self._session):
                channel = self.channels.create_channel(
                    youtube_channel_id=youtube_channel_id,
                    title=title,
                    description=description,
                )
        except IntegrityError:
            # transaction() already rolled back; re-read the winning row.
            existing = self.channels.get_channel_by_youtube_id(youtube_channel_id)
            if existing is not None:
                return existing
            raise  # not a duplicate -> never swallow the DB error
        return channel

    def update_channel_metadata(
        self,
        channel_id: uuid.UUID,
        *,
        title: str | _Unset = UNSET,
        description: str | None | _Unset = UNSET,
    ) -> Channel | None:
        """Update a channel's metadata in one transaction; ``None`` if absent."""
        channel = self.channels.get_channel_by_id(channel_id)
        if channel is None:
            return None
        with transaction(self._session):
            return self.channels.update_channel(
                channel, title=title, description=description
            )
