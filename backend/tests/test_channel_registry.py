"""Tests for Goal 36 — channel registry from discovered video metadata."""
from __future__ import annotations

import uuid
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest

from app.youtube import (
    ChannelRegistryService,
    InvalidChannelRegistryInputError,
    RegisteredChannel,
    VideoMetadata,
)


def _video(
    video_id: str = "dQw4w9WgXcQ",
    channel_id: str = "UCexamplechannel0001",
    title: str = "Example Channel",
    description: str | None = "A description",
) -> VideoMetadata:
    return VideoMetadata(
        youtube_video_id=video_id,
        title="Video title",
        description=description,
        channel_id=channel_id,
        channel_title=title,
        duration_seconds=120,
        published_at="2024-01-02T03:04:05Z",
    )


class FakeChannels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = False

    def create_or_get_channel(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("database unavailable")
        return SimpleNamespace(
            id=uuid.uuid4(),
            youtube_channel_id=kwargs["youtube_channel_id"],
            title=kwargs["title"],
            description=kwargs["description"],
        )


# --- Registration -----------------------------------------------------------

def test_registers_unique_channels_in_first_seen_order() -> None:
    channels = FakeChannels()
    service = ChannelRegistryService(channels)  # type: ignore[arg-type]
    videos = [
        _video("dQw4w9WgXcQ", "UCfirst", "First"),
        _video("abcdefghijk", "UCsecond", "Second", None),
    ]
    result = service.register_from_videos(videos)
    assert [item.youtube_channel_id for item in result] == ["UCfirst", "UCsecond"]
    assert [item.title for item in result] == ["First", "Second"]
    assert result[1].description is None
    assert channels.calls == [
        {
            "youtube_channel_id": "UCfirst",
            "title": "First",
            "description": "A description",
        },
        {
            "youtube_channel_id": "UCsecond",
            "title": "Second",
            "description": None,
        },
    ]


def test_duplicate_channel_is_registered_once() -> None:
    channels = FakeChannels()
    service = ChannelRegistryService(channels)  # type: ignore[arg-type]
    result = service.register_from_videos(
        [_video("dQw4w9WgXcQ"), _video("abcdefghijk")]
    )
    assert len(result) == 1
    assert len(channels.calls) == 1


def test_empty_input_returns_empty_without_calls() -> None:
    channels = FakeChannels()
    service = ChannelRegistryService(channels)  # type: ignore[arg-type]
    assert service.register_from_videos([]) == ()
    assert channels.calls == []


def test_unicode_titles_and_ids_are_handled() -> None:
    channels = FakeChannels()
    service = ChannelRegistryService(channels)  # type: ignore[arg-type]
    result = service.register_from_videos(
        [_video(channel_id="UC_algebraically", title="চ্যানেল — স্বাগতম")]
    )
    assert result[0].title == "চ্যানেল — স্বাগতম"
    assert result[0].youtube_channel_id == "UC_algebraically"


# --- Validation and failures ------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel_id": ""},
        {"channel_id": "   "},
        {"channel_id": "UC has space"},
        {"channel_id": "x" * 65},
        {"title": ""},
        {"title": "   "},
        {"description": 123},
    ],
)
def test_invalid_channel_data_is_rejected_before_any_write(kwargs: Any) -> None:
    channels = FakeChannels()
    service = ChannelRegistryService(channels)  # type: ignore[arg-type]
    with pytest.raises(InvalidChannelRegistryInputError):
        service.register_from_videos([_video(**kwargs)])
    assert channels.calls == []


def test_non_video_input_is_rejected() -> None:
    service = ChannelRegistryService(FakeChannels())  # type: ignore[arg-type]
    with pytest.raises(InvalidChannelRegistryInputError):
        service.register_from_videos(["not-a-video"])  # type: ignore[list-item]


def test_channel_service_failure_propagates_without_fallback() -> None:
    channels = FakeChannels()
    channels.fail = True
    service = ChannelRegistryService(channels)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="database unavailable"):
        service.register_from_videos([_video()])


def test_existing_channel_returns_persisted_not_requested_metadata() -> None:
    class ExistingChannels(FakeChannels):
        def create_or_get_channel(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(
                id=uuid.uuid4(),
                youtube_channel_id=kwargs["youtube_channel_id"],
                title="Stored title",
                description="Stored description",
            )

    service = ChannelRegistryService(ExistingChannels())  # type: ignore[arg-type]
    record = service.register_from_videos(
        [_video(channel_id="UCexisting", title="Requested title")]
    )[0]
    assert record.title == "Stored title"
    assert record.description == "Stored description"


def test_mismatched_returned_external_id_is_rejected() -> None:
    class MismatchedChannels(FakeChannels):
        def create_or_get_channel(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                id=uuid.uuid4(),
                youtube_channel_id="UCdifferent",
                title="Title",
                description=None,
            )

    service = ChannelRegistryService(MismatchedChannels())  # type: ignore[arg-type]
    with pytest.raises(InvalidChannelRegistryInputError):
        service.register_from_videos([_video(channel_id="UCexpected")])


def test_invalid_returned_record_is_rejected() -> None:
    class BadChannels(FakeChannels):
        def create_or_get_channel(self, **kwargs: Any) -> Any:
            return SimpleNamespace(id="not-a-uuid")

    service = ChannelRegistryService(BadChannels())  # type: ignore[arg-type]
    with pytest.raises(InvalidChannelRegistryInputError):
        service.register_from_videos([_video()])


def test_record_is_immutable_and_has_no_oauth_or_transcript_fields() -> None:
    service = ChannelRegistryService(FakeChannels())  # type: ignore[arg-type]
    record = service.register_from_videos([_video()])[0]
    names = {field.name for field in fields(RegisteredChannel)}
    assert names == {
        "id", "youtube_channel_id", "title", "description"
    }
    with pytest.raises(Exception):
        record.title = "changed"  # type: ignore[misc]
