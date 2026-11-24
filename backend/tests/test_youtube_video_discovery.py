"""Tests for Goal 35 — bounded YouTube discovery via mocked transport."""
from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from app.youtube import (
    InvalidVideoDiscoveryInputError,
    VideoDiscoveryService,
    YouTubeApiClient,
    YouTubeCredentials,
    YouTubeQuotaError,
    YouTubeResponseError,
    get_video_metadata,
    parse_duration_seconds,
    search_videos,
    validate_video_id,
)

KEY = "AIzaSyDUMMYKEY-1234567890-abcdefghijklmno"
VIDEO_1 = "dQw4w9WgXcQ"
VIDEO_2 = "abcdefghijk"
CHANNEL = "UCabcdefghijklmnopqrstuvw"
PUBLISHED = "2024-01-02T03:04:05Z"


def _search_payload(ids=(VIDEO_1,), token=None):
    payload = {
        "kind": "youtube#searchListResponse",
        "items": [
            {
                "id": {"kind": "youtube#video", "videoId": video_id},
                "snippet": {
                    "title": f"Title {video_id}",
                    "channelId": CHANNEL,
                    "publishedAt": PUBLISHED,
                },
            }
            for video_id in ids
        ],
    }
    if token is not None:
        payload["nextPageToken"] = token
    return payload


def _videos_payload(ids=(VIDEO_1,)):
    return {
        "kind": "youtube#videoListResponse",
        "items": [
            {
                "id": video_id,
                "snippet": {
                    "title": f"Full title {video_id}",
                    "description": "A description",
                    "channelId": CHANNEL,
                    "channelTitle": "Example Channel",
                    "publishedAt": PUBLISHED,
                },
                "contentDetails": {"duration": "PT1H2M3S"},
                "status": {"embeddable": True},
            }
            for video_id in ids
        ],
    }


def _client(search=None, videos=None, status=200, body=None):
    calls: list[str] = []

    def transport(url: str, timeout: float):
        calls.append(url)
        path = urlparse(url).path
        if body is not None:
            return SimpleNamespace(status=status, body=body)
        if path.endswith("/search"):
            payload = search if search is not None else _search_payload()
        elif path.endswith("/videos"):
            payload = videos if videos is not None else _videos_payload()
        else:
            return SimpleNamespace(status=200, body=b"{}")
        return SimpleNamespace(
            status=200, body=json.dumps(payload).encode("utf-8")
        )

    client = YouTubeApiClient(
        YouTubeCredentials(api_key=SecretStr(KEY)), transport=transport
    )
    return client, calls


# --- Input validation --------------------------------------------------------

@pytest.mark.parametrize(
    "value", [None, 123, "", "   ", "x" * 201]
)
def test_invalid_query_is_rejected(value) -> None:
    client, calls = _client()
    with pytest.raises(InvalidVideoDiscoveryInputError):
        search_videos(client, value)
    assert calls == []


@pytest.mark.parametrize("max_results", [0, -1, 51, True, 1.5, "10"])
def test_invalid_max_results_is_rejected(max_results) -> None:
    client, calls = _client()
    with pytest.raises(InvalidVideoDiscoveryInputError):
        search_videos(client, "query", max_results=max_results)
    assert calls == []


@pytest.mark.parametrize("token", ["", "   ", 123, "x" * 513])
def test_invalid_page_token_is_rejected(token) -> None:
    client, calls = _client()
    with pytest.raises(InvalidVideoDiscoveryInputError):
        search_videos(client, "query", page_token=token)
    assert calls == []


@pytest.mark.parametrize("value", [None, 123, "", "short", "bad/id/value", "é" * 11])
def test_video_id_validation(value) -> None:
    with pytest.raises(InvalidVideoDiscoveryInputError):
        validate_video_id(value)


def test_video_id_batch_bounds_and_duplicates() -> None:
    client, calls = _client()
    with pytest.raises(InvalidVideoDiscoveryInputError):
        get_video_metadata(client, [VIDEO_1] * 51)
    with pytest.raises(InvalidVideoDiscoveryInputError):
        get_video_metadata(client, [])
    with pytest.raises(InvalidVideoDiscoveryInputError):
        get_video_metadata(client, [VIDEO_1, VIDEO_1])
    assert calls == []


def test_duplicate_api_result_is_rejected_not_overwritten() -> None:
    payload = _videos_payload(ids=(VIDEO_1, VIDEO_1))
    client, _ = _client(videos=payload)
    with pytest.raises(YouTubeResponseError, match="duplicate"):
        get_video_metadata(client, [VIDEO_1])


def test_channel_title_is_required_metadata() -> None:
    payload = _videos_payload(ids=(VIDEO_1,))
    del payload["items"][0]["snippet"]["channelTitle"]
    client, _ = _client(videos=payload)
    with pytest.raises(YouTubeResponseError):
        get_video_metadata(client, [VIDEO_1])


# --- ISO-8601 duration -------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [("PT1H2M3S", 3723), ("PT30S", 30), ("P1DT2H", 93600), ("PT0S", 0)],
)
def test_duration_conversion(value: str, expected: int) -> None:
    assert parse_duration_seconds(value) == expected


@pytest.mark.parametrize("value", [None, "", "1:02:03", "PT", "PXY", "PT9999999H"])
def test_invalid_duration_is_rejected(value) -> None:
    with pytest.raises(YouTubeResponseError):
        parse_duration_seconds(value)


# --- Search and metadata -----------------------------------------------------

def test_search_request_matches_official_contract() -> None:
    client, calls = _client()
    page = search_videos(client, "latency", max_results=7, page_token="NEXT")
    query = parse_qs(urlparse(calls[0]).query)
    assert query["part"] == ["id,snippet"]
    assert query["type"] == ["video"]
    assert query["maxResults"] == ["7"]
    assert query["q"] == ["latency"]
    assert query["pageToken"] == ["NEXT"]
    assert len(page.hits) == 1
    assert page.hits[0].video_id == VIDEO_1
    assert page.next_page_token is None


def test_empty_search_returns_empty_page_without_video_call() -> None:
    client, calls = _client(search={"items": []})
    page = search_videos(client, "nothing")
    assert page.hits == ()
    assert len(calls) == 1


def test_discover_returns_validated_metadata_in_order() -> None:
    client, calls = _client(
        search=_search_payload(ids=(VIDEO_1, VIDEO_2)),
        videos=_videos_payload(ids=(VIDEO_2, VIDEO_1)),
    )
    result = VideoDiscoveryService(client).discover("query", max_results=2)
    assert [item.youtube_video_id for item in result] == [VIDEO_1, VIDEO_2]
    assert all(item.duration_seconds == 3723 for item in result)
    assert all(item.channel_id == CHANNEL for item in result)
    assert all(item.channel_title == "Example Channel" for item in result)
    assert all(item.published_at == PUBLISHED for item in result)
    assert result[0].title == f"Full title {VIDEO_1}"
    assert len(calls) == 2


def test_discover_empty_search_does_not_fetch_metadata() -> None:
    client, calls = _client(search={"items": []})
    assert VideoDiscoveryService(client).discover("none") == ()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"items": [{"id": {"videoId": "bad-id"}, "snippet": {}}]},
        {"items": [{"id": {}, "snippet": {"title": "x"}}]},
        {"items": [{"id": {"videoId": VIDEO_1}}]},
        {"items": "not-a-list"},
    ],
)
def test_malformed_search_response_is_rejected(payload) -> None:
    client, _ = _client(search=payload)
    with pytest.raises((YouTubeResponseError, InvalidVideoDiscoveryInputError)):
        search_videos(client, "query")


def test_missing_metadata_for_requested_video_is_rejected() -> None:
    client, _ = _client(videos={"items": []})
    with pytest.raises(YouTubeResponseError):
        get_video_metadata(client, [VIDEO_1])


@pytest.mark.parametrize(
    "videos",
    [
        {"items": [{"id": VIDEO_1, "snippet": {}, "contentDetails": {"duration": "PT1S"}}]},
        {"items": [{"id": VIDEO_1, "snippet": {"title": "t", "channelId": CHANNEL, "channelTitle": "C", "publishedAt": PUBLISHED}, "contentDetails": {}}]},
        {"items": [{"id": VIDEO_1, "snippet": {"title": "t", "channelId": CHANNEL, "channelTitle": "C", "publishedAt": "not-a-date"}, "contentDetails": {"duration": "PT1S"}}]},
    ],
)
def test_malformed_video_metadata_is_rejected(videos) -> None:
    client, _ = _client(videos=videos)
    with pytest.raises(YouTubeResponseError):
        get_video_metadata(client, [VIDEO_1])


def test_quota_error_is_not_reported_as_empty_success() -> None:
    body = json.dumps(
        {"error": {"errors": [{"reason": "quotaExceeded"}]}}
    ).encode()
    client, _ = _client(status=403, body=body)
    with pytest.raises(YouTubeQuotaError):
        VideoDiscoveryService(client).discover("query")


def test_key_is_never_in_discovery_urls_or_errors() -> None:
    client, calls = _client()
    result = VideoDiscoveryService(client).discover("query")
    assert result
    # The key is necessarily present in the request query, never in errors.
    assert any(KEY in url for url in calls)
    # The key must not appear in any error text.
    with pytest.raises(Exception) as exc:
        get_video_metadata(client, ["bad"])
    assert KEY not in str(exc.value)
