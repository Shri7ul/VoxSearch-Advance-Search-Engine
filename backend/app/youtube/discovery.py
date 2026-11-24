"""Goal 35 — bounded YouTube video discovery and metadata retrieval.

Uses the Goal 34 authenticated client with two official Data API v3 methods:

* ``search.list(part=id,snippet, type=video, maxResults<=50, q=...)``
* ``videos.list(part=snippet,contentDetails,status, id=...)``

No credentials, database writes, migrations, channel registry, OAuth or
transcript ingestion are involved. All responses are validated fail-closed; API
failures are never reported as an empty successful discovery.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.youtube.client import (
    MAX_VIDEO_IDS_PER_CALL,
    YouTubeApiClient,
    YouTubeAuthError,
    YouTubeResponseError,
)

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_DURATION_RE = re.compile(
    r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
)
MAX_QUERY_CHARACTERS = 200
MAX_PAGE_TOKEN_CHARACTERS = 512


class InvalidVideoDiscoveryInputError(YouTubeAuthError, ValueError):
    """Discovery input is malformed or out of bounds."""


@dataclass(frozen=True)
class VideoSearchHit:
    """One validated ``search.list`` video item (id + snippet only)."""

    video_id: str
    title: str
    channel_id: str
    published_at: str


@dataclass(frozen=True)
class VideoSearchPage:
    """One bounded page of video search hits."""

    hits: tuple[VideoSearchHit, ...]
    next_page_token: str | None = None


@dataclass(frozen=True)
class VideoMetadata:
    """Validated metadata from ``videos.list`` for one discovered video."""

    youtube_video_id: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    duration_seconds: int
    published_at: str


def validate_video_id(value: Any) -> str:
    if not isinstance(value, str) or not VIDEO_ID_RE.fullmatch(value):
        raise InvalidVideoDiscoveryInputError("invalid YouTube video id")
    return value


def parse_duration_seconds(value: Any) -> int:
    """Convert an ISO-8601 duration (e.g. PT1H2M3S) to whole seconds."""
    if not isinstance(value, str):
        raise YouTubeResponseError("invalid video duration")
    match = _DURATION_RE.fullmatch(value)
    if match is None:
        raise YouTubeResponseError("unsupported video duration format")
    days, hours, minutes, seconds = match.groups()
    if not any((days, hours, minutes, seconds)):
        raise YouTubeResponseError("empty video duration")
    total = (
        int(days or 0) * 86400
        + int(hours or 0) * 3600
        + int(minutes or 0) * 60
        + float(seconds or 0)
    )
    if total < 0 or not (0 <= total <= 604800):
        raise YouTubeResponseError("video duration is out of bounds")
    return int(total)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise YouTubeResponseError(f"missing video {field}")
    return value


def _optional_text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise YouTubeResponseError(f"invalid video {field}")
    return value


def _published_at(value: Any) -> str:
    text = _required_text(value, "publishedAt")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise YouTubeResponseError("invalid video publishedAt") from exc
    if parsed.tzinfo is None:
        raise YouTubeResponseError("video publishedAt is not timezone-aware")
    return text


def _items(payload: dict[str, Any]) -> list[Any]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise YouTubeResponseError("YouTube API response has no items list")
    return items


def search_videos(
    client: YouTubeApiClient,
    query: str,
    *,
    max_results: int = 10,
    page_token: str | None = None,
) -> VideoSearchPage:
    """Return one bounded page of video ids from ``search.list``."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidVideoDiscoveryInputError("query must be a non-empty string")
    if len(query) > MAX_QUERY_CHARACTERS:
        raise InvalidVideoDiscoveryInputError("query is too long")
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= 50
    ):
        raise InvalidVideoDiscoveryInputError("max_results must be 1..50")
    if page_token is not None and (
        not isinstance(page_token, str)
        or not page_token.strip()
        or len(page_token) > MAX_PAGE_TOKEN_CHARACTERS
    ):
        raise InvalidVideoDiscoveryInputError("invalid page token")

    params = {
        "part": "id,snippet",
        "type": "video",
        "maxResults": str(max_results),
        "q": query,
    }
    if page_token is not None:
        params["pageToken"] = page_token
    payload = client.get_json("search", params)

    hits: list[VideoSearchHit] = []
    for item in _items(payload):
        if not isinstance(item, dict):
            raise YouTubeResponseError("malformed search item")
        resource_id = item.get("id")
        snippet = item.get("snippet")
        if not isinstance(resource_id, dict) or not isinstance(snippet, dict):
            raise YouTubeResponseError("malformed search item")
        hits.append(
            VideoSearchHit(
                video_id=validate_video_id(resource_id.get("videoId")),
                title=_required_text(snippet.get("title"), "title"),
                channel_id=_required_text(snippet.get("channelId"), "channelId"),
                published_at=_published_at(snippet.get("publishedAt")),
            )
        )
    token = payload.get("nextPageToken")
    if token is not None and not isinstance(token, str):
        raise YouTubeResponseError("invalid nextPageToken")
    return VideoSearchPage(hits=tuple(hits), next_page_token=token)


def get_video_metadata(
    client: YouTubeApiClient, video_ids: list[str]
) -> tuple[VideoMetadata, ...]:
    """Return validated ``videos.list`` metadata in the requested id order."""
    if not isinstance(video_ids, list) or not 1 <= len(video_ids) <= MAX_VIDEO_IDS_PER_CALL:
        raise InvalidVideoDiscoveryInputError("video_ids must contain 1..50 ids")
    checked = [validate_video_id(item) for item in video_ids]
    if len(set(checked)) != len(checked):
        raise InvalidVideoDiscoveryInputError("duplicate video ids are not allowed")
    payload = client.get_json(
        "videos",
        {
            "part": "snippet,contentDetails,status",
            "id": ",".join(checked),
        },
    )
    by_id: dict[str, VideoMetadata] = {}
    for item in _items(payload):
        if not isinstance(item, dict):
            raise YouTubeResponseError("malformed video item")
        video_id = validate_video_id(item.get("id"))
        if video_id in by_id:
            raise YouTubeResponseError("duplicate video in API response")
        snippet = item.get("snippet")
        content = item.get("contentDetails")
        if not isinstance(snippet, dict) or not isinstance(content, dict):
            raise YouTubeResponseError("malformed video item")
        by_id[video_id] = VideoMetadata(
            youtube_video_id=video_id,
            title=_required_text(snippet.get("title"), "title"),
            description=_optional_text(snippet.get("description"), "description"),
            channel_id=_required_text(snippet.get("channelId"), "channelId"),
            channel_title=_required_text(snippet.get("channelTitle"), "channelTitle"),
            duration_seconds=parse_duration_seconds(content.get("duration")),
            published_at=_published_at(snippet.get("publishedAt")),
        )
    missing = [item for item in checked if item not in by_id]
    if missing:
        raise YouTubeResponseError("YouTube API omitted requested video metadata")
    return tuple(by_id[item] for item in checked)


class VideoDiscoveryService:
    """Discover videos and retrieve their verified metadata (read-only)."""

    def __init__(self, client: YouTubeApiClient) -> None:
        self._client = client

    def discover(self, query: str, *, max_results: int = 10) -> tuple[VideoMetadata, ...]:
        """Search then fetch metadata; empty search yields an empty tuple."""
        page = search_videos(self._client, query, max_results=max_results)
        if not page.hits:
            return ()
        return get_video_metadata(self._client, [hit.video_id for hit in page.hits])


__all__ = [
    "MAX_QUERY_CHARACTERS",
    "VideoDiscoveryService",
    "InvalidVideoDiscoveryInputError",
    "VideoMetadata",
    "VideoSearchHit",
    "VideoSearchPage",
    "get_video_metadata",
    "parse_duration_seconds",
    "search_videos",
    "validate_video_id",
]
