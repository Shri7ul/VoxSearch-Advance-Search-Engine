"""Tests for Goal 38 — authorized YouTube transcript adapter (mocked transport)."""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from app.transcripts.contracts import TranscriptInput, TranscriptSource
from app.youtube import (
    YOUTUBE_FORCE_SSL_SCOPE,
    CaptionUnavailableError,
    InvalidTranscriptResponseError,
    TranscriptAdapterError,
    TranscriptTransportError,
    YouTubeTranscriptAdapter,
    parse_vtt,
)

VIDEO = "dQw4w9WgXcQ"
TOKEN = "ya29.secret-access-token"
VIDEO_UUID = uuid.uuid4()
RIGHTS_UUID = uuid.uuid4()

VTT = """WEBVTT

1
00:00:01.000 --> 00:00:03.500
Hello world

2
00:00:03.500 --> 00:00:05.000 line:0%
Second
line
"""


def _track(
    track_id: str = "sv1",
    language: str = "en",
    kind: str = "standard",
    status: str = "ready",
):
    return {
        "id": track_id,
        "snippet": {
            "language": language,
            "kind": kind,
            "status": status,
            "name": "English",
        },
    }


def _adapter(list_payload=None, vtt_body: bytes = VTT.encode(), status: int = 200, *, raise_error=None):
    captured: list[dict] = []

    def transport(url: str, headers: dict[str, str], timeout: float):
        captured.append({"url": url, "headers": dict(headers), "timeout": timeout})
        if raise_error is not None:
            raise raise_error
        body = (
            json.dumps(
                list_payload
                if list_payload is not None
                else {"items": [_track()]}
            ).encode()
            if urlparse(url).path.endswith("/captions")
            else vtt_body
        )
        return SimpleNamespace(status=status, body=body, content_type="")

    adapter = YouTubeTranscriptAdapter(
        access_token=SecretStr(TOKEN),
        granted_scopes=[YOUTUBE_FORCE_SSL_SCOPE],
        transport=transport,
    )
    return adapter, captured


# --- VTT parser -------------------------------------------------------------

def test_parse_vtt_preserves_verbatim_text_and_timestamps() -> None:
    segments = parse_vtt(VTT)
    assert len(segments) == 2
    assert segments[0].start_time == 1.0
    assert segments[0].end_time == 3.5
    assert segments[0].text == "Hello world"
    assert segments[1].text == "Second\nline"


def test_parse_vtt_supports_short_mm_ss_timestamps() -> None:
    vtt = "WEBVTT\n\n00:05.250 --> 00:07.000\nShort timestamps\n"
    segments = parse_vtt(vtt)
    assert segments[0].start_time == pytest.approx(5.25)
    assert segments[0].end_time == 7.0
    assert segments[0].text == "Short timestamps"


def test_parse_vtt_skips_empty_cues_and_headers() -> None:
    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n\n00:00:02.000 --> 00:00:03.000\nText\n"
    segments = parse_vtt(vtt)
    assert len(segments) == 1
    assert segments[0].text == "Text"


@pytest.mark.parametrize(
    "vtt",
    [
        "WEBVTT",
        "00:00:03.000 --> 00:00:01.000\nBad\n",
        "not a timestamp\ntext\n",
        "00:00:00.000 --> 00:00:01.000\n" + "x" * 8193,
    ],
)
def test_invalid_vtt_is_rejected(vtt: str) -> None:
    with pytest.raises(InvalidTranscriptResponseError):
        parse_vtt(vtt)


def test_non_string_vtt_is_rejected() -> None:
    with pytest.raises(InvalidTranscriptResponseError):
        parse_vtt(None)  # type: ignore[arg-type]


# --- Adapter configuration --------------------------------------------------

def test_scope_is_required() -> None:
    with pytest.raises(TranscriptAdapterError):
        YouTubeTranscriptAdapter(
            access_token=SecretStr(TOKEN),
            granted_scopes=["https://www.googleapis.com/auth/youtube.readonly"],
            transport=lambda *a: None,
        )


def test_secret_repr_is_masked() -> None:
    adapter, _ = _adapter()
    assert TOKEN not in repr(adapter)


@pytest.mark.parametrize("video", ["", "short", "bad/id"])
def test_invalid_video_id_is_rejected(video: str) -> None:
    adapter, calls = _adapter()
    with pytest.raises(TranscriptAdapterError):
        adapter.fetch_transcript(
            youtube_video_id=video,
            video_id=VIDEO_UUID,
            content_right_id=RIGHTS_UUID,
        )
    assert calls == []


def test_invalid_provenance_is_rejected() -> None:
    adapter, calls = _adapter()
    with pytest.raises(TranscriptAdapterError):
        adapter.fetch_transcript(
            youtube_video_id=VIDEO,
            video_id="not-a-uuid",  # type: ignore[arg-type]
            content_right_id=RIGHTS_UUID,
        )
    assert calls == []


# Caption listing and fetch --------------------------------------------------

def test_list_uses_oauth_header_and_official_endpoint() -> None:
    adapter, calls = _adapter()
    tracks = adapter.list_caption_tracks(VIDEO)
    assert tracks[0].id == "sv1"
    request = calls[0]
    assert request["url"].startswith("https://www.googleapis.com/youtube/v3/captions?")
    query = parse_qs(urlparse(request["url"]).query)
    assert query["part"] == ["snippet"]
    assert query["videoId"] == [VIDEO]
    assert request["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_fetch_returns_validated_transcript_input() -> None:
    adapter, _ = _adapter()
    result = adapter.fetch_transcript(
        youtube_video_id=VIDEO,
        video_id=VIDEO_UUID,
        content_right_id=RIGHTS_UUID,
    )
    assert isinstance(result, TranscriptInput)
    assert result.video_id == VIDEO_UUID
    assert result.content_right_id == RIGHTS_UUID
    assert result.source_type == TranscriptSource.CREATOR_PROVIDED
    assert result.language == "en"
    assert [segment.text for segment in result.segments] == ["Hello world", "Second\nline"]


def test_asr_track_maps_to_asr_source_and_language_filter() -> None:
    payload = {
        "items": [
            _track("asr1", language="bn", kind="asr"),
            _track("man1", language="en"),
        ]
    }
    adapter, _ = _adapter(list_payload=payload)
    bn = adapter.fetch_transcript(
        youtube_video_id=VIDEO,
        video_id=VIDEO_UUID,
        content_right_id=RIGHTS_UUID,
        language="bn",
    )
    assert bn.source_type == TranscriptSource.ASR_GENERATED
    assert bn.language == "bn"


def test_human_caption_preferred_over_asr_for_same_language() -> None:
    payload = {
        "items": [
            _track("asr1", kind="asr"),
            _track("man1", kind="standard"),
        ]
    }
    adapter, calls = _adapter(list_payload=payload)
    adapter.fetch_transcript(
        youtube_video_id=VIDEO,
        video_id=VIDEO_UUID,
        content_right_id=RIGHTS_UUID,
    )
    download_url = calls[1]["url"]
    assert urlparse(download_url).path.endswith("/captions/man1")
    assert parse_qs(urlparse(download_url).query) == {"tfmt": ["vtt"]}


@pytest.mark.parametrize(
    "payload",
    [
        {"items": []},
        {"items": [_track(status="failed")]},
        {"items": [{"id": "x"}]},
        {"items": "not-a-list"},
        {"items": [{"id": "x", "snippet": {"language": "", "kind": "standard", "status": "ready"}}]},
        {"items": [{"id": "bad/id", "snippet": {"language": "en", "kind": "standard", "status": "ready"}}]},
    ],
)
def test_unavailable_or_malformed_caption_metadata(payload) -> None:
    adapter, _ = _adapter(list_payload=payload)
    with pytest.raises((CaptionUnavailableError, InvalidTranscriptResponseError)):
        adapter.fetch_transcript(
            youtube_video_id=VIDEO,
            video_id=VIDEO_UUID,
            content_right_id=RIGHTS_UUID,
        )


def test_malformed_vtt_download_is_rejected() -> None:
    adapter, _ = _adapter(vtt_body=b"not vtt")
    with pytest.raises(InvalidTranscriptResponseError):
        adapter.fetch_transcript(
            youtube_video_id=VIDEO,
            video_id=VIDEO_UUID,
            content_right_id=RIGHTS_UUID,
        )


@pytest.mark.parametrize("status", [401, 403])
def test_authorization_failures_are_credential_free(status: int) -> None:
    adapter, _ = _adapter(status=status)
    with pytest.raises(TranscriptAdapterError) as exc:
        adapter.fetch_transcript(
            youtube_video_id=VIDEO,
            video_id=VIDEO_UUID,
            content_right_id=RIGHTS_UUID,
        )
    assert TOKEN not in str(exc.value)


def test_transport_failure_is_typed_and_secret_free() -> None:
    adapter, _ = _adapter(raise_error=TimeoutError("timeout"))
    with pytest.raises(TranscriptTransportError) as exc:
        adapter.fetch_transcript(
            youtube_video_id=VIDEO,
            video_id=VIDEO_UUID,
            content_right_id=RIGHTS_UUID,
        )
    assert TOKEN not in str(exc.value)
    assert "TimeoutError" in str(exc.value)
