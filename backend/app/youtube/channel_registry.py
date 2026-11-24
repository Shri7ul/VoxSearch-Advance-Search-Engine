"""Goal 36 — channel registry built from Goal 35 discovered video metadata.

The established persistence boundary is :class:`app.services.channel_service.ChannelService`
and the existing ``channels`` table (unique ``youtube_channel_id``). This module
contains ONLY the registry orchestration:

* validate the external channel id / title / description from discovered videos;
* deduplicate channels deterministically by external YouTube channel id;
* call the accepted ``create_or_get_channel`` once per unique channel;
* return immutable channel records with the internal UUID.

No channels table migration, OAuth (Goal 37), transcript ingestion (Goal 38),
video persistence or search-pipeline change is introduced here. No network call
and no database write happens at import time.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services.channel_service import ChannelService
from app.youtube.discovery import VideoMetadata

MAX_EXTERNAL_CHANNEL_ID_CHARACTERS = 64
MAX_TITLE_CHARACTERS = 1000
MAX_DESCRIPTION_CHARACTERS = 10000


class InvalidChannelRegistryInputError(ValueError):
    """Registry input is malformed or out of bounds."""


@dataclass(frozen=True)
class RegisteredChannel:
    """One immutable channel record produced by the registry."""

    id: uuid.UUID
    youtube_channel_id: str
    title: str
    description: str | None


class ChannelRegistryService:
    """Register discovered channels using the existing channel service."""

    def __init__(self, channels: ChannelService) -> None:
        self._channels = channels

    def register_from_videos(
        self, videos: Sequence[VideoMetadata]
    ) -> tuple[RegisteredChannel, ...]:
        """Register unique channels in first-seen order from discovered videos."""
        if isinstance(videos, (str, bytes)) or not isinstance(videos, Sequence):
            raise InvalidChannelRegistryInputError("videos must be a sequence")

        unique: dict[str, tuple[str, str | None]] = {}
        for video in videos:
            if not isinstance(video, VideoMetadata):
                raise InvalidChannelRegistryInputError("invalid video metadata")
            external_id = self._validate_external_id(video.channel_id)
            title = self._validate_title(video.channel_title)
            description = self._validate_description(video.description)
            if external_id not in unique:
                unique[external_id] = (title, description)

        records: list[RegisteredChannel] = []
        for external_id, (title, description) in unique.items():
            channel = self._channels.create_or_get_channel(
                youtube_channel_id=external_id,
                title=title,
                description=description,
            )
            channel_id = getattr(channel, "id", None)
            returned_external_id = getattr(channel, "youtube_channel_id", None)
            returned_title = getattr(channel, "title", None)
            returned_description = getattr(channel, "description", None)
            if (
                not isinstance(channel_id, uuid.UUID)
                or returned_external_id != external_id
                or not isinstance(returned_title, str)
                or not returned_title.strip()
                or (
                    returned_description is not None
                    and not isinstance(returned_description, str)
                )
            ):
                raise InvalidChannelRegistryInputError(
                    "channel service returned an invalid record"
                )
            # Use the persisted row: create_or_get_channel returns the EXISTING
            # channel unchanged on a hit, so reporting request values would be
            # misleading.
            records.append(
                RegisteredChannel(
                    id=channel_id,
                    youtube_channel_id=returned_external_id,
                    title=returned_title,
                    description=returned_description,
                )
            )
        return tuple(records)

    @staticmethod
    def _validate_external_id(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidChannelRegistryInputError("channel id must be non-empty text")
        if len(value) > MAX_EXTERNAL_CHANNEL_ID_CHARACTERS:
            raise InvalidChannelRegistryInputError("channel id is too long")
        if any(char.isspace() for char in value):
            raise InvalidChannelRegistryInputError("channel id cannot contain whitespace")
        return value

    @staticmethod
    def _validate_title(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidChannelRegistryInputError("channel title must be non-empty text")
        if len(value) > MAX_TITLE_CHARACTERS:
            raise InvalidChannelRegistryInputError("channel title is too long")
        return value

    @staticmethod
    def _validate_description(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise InvalidChannelRegistryInputError("channel description must be text")
        if len(value) > MAX_DESCRIPTION_CHARACTERS:
            raise InvalidChannelRegistryInputError("channel description is too long")
        return value


__all__ = [
    "MAX_DESCRIPTION_CHARACTERS",
    "MAX_EXTERNAL_CHANNEL_ID_CHARACTERS",
    "MAX_TITLE_CHARACTERS",
    "ChannelRegistryService",
    "InvalidChannelRegistryInputError",
    "RegisteredChannel",
]
