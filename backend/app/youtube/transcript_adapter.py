"""Goal 38 — authorized YouTube transcript source adapter.

Official Data API v3 semantics (verified against Google's documentation):

* ``captions.list(part=snippet, videoId=...)`` lists caption-track METADATA and
  requires OAuth with ``youtube.force-ssl`` (or ``youtubepartner``); the API key
  alone is NOT sufficient.
* ``captions/download?id=<track>&tfmt=vtt`` returns the caption track as a
  binary VTT file and requires permission to edit the video.

This adapter owns ONLY acquisition + conversion into the existing validated
:class:`app.transcripts.contracts.TranscriptInput` shape. It does NOT reimplement
normalization, segmentation, timestamp alignment or chunking (Goals 09–13), does
not persist anything, and does not call the ingestion service. Callers may feed
the returned payload into that already-accepted pipeline.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable
from uuid import UUID

from pydantic import SecretStr

from app.transcripts.contracts import TranscriptInput, TranscriptSegmentInput, TranscriptSource
from app.youtube.discovery import validate_video_id
from app.youtube.oauth import YOUTUBE_FORCE_SSL_SCOPE

CAPTIONS_LIST_URL = "https://www.googleapis.com/youtube/v3/captions"
CAPTIONS_DOWNLOAD_URL = "https://www.googleapis.com/youtube/v3/captions/"
_CAPTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_CAPTION_TRACKS = 25
MAX_CUES = 10000
MAX_CUE_TEXT_CHARACTERS = 8192
MAX_VTT_BYTES = 4_000_000

_TIMESTAMP_RE = re.compile(
    r"^(\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3})\s*-->\s*"
    r"(\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3})"
)
_SHORT_TIMESTAMP_RE = re.compile(
    r"^([0-5]?\d):([0-5]\d)\.(\d{3})\s*-->\s*"
    r"([0-5]?\d):([0-5]\d)\.(\d{3})"
)


class TranscriptAdapterError(RuntimeError):
    """Credential-free base error for Goal 38 failures."""


class CaptionUnavailableError(TranscriptAdapterError):
    """No authorized caption track is available for the video."""


class InvalidTranscriptResponseError(TranscriptAdapterError):
    """YouTube returned malformed caption metadata or VTT data."""


class TranscriptTransportError(TranscriptAdapterError):
    """The caption endpoint could not be reached or returned an HTTP error."""


def validate_caption_id(value: Any) -> str:
    """Caption IDs are path segments; reject anything but safe URL characters."""
    if not isinstance(value, str) or not _CAPTION_ID_RE.fullmatch(value):
        raise InvalidTranscriptResponseError("invalid caption track id")
    return value


@dataclass(frozen=True)
class CaptionTrack:
    """Validated caption-track metadata from ``captions.list``."""

    id: str
    language: str
    kind: str
    status: str
    name: str


@runtime_checkable
class CaptionTransport(Protocol):
    """Injectable HTTPS GET boundary with OAuth headers."""

    def __call__(
        self, url: str, headers: dict[str, str], timeout: float
    ) -> Any: ...


def _default_transport(
    url: str, headers: dict[str, str], timeout: float
) -> Any:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _Response(
                getattr(response, "status", 200),
                response.read(MAX_VTT_BYTES + 1),
                str(response.headers.get("Content-Type", "")),
            )
    except urllib.error.HTTPError as exc:
        return _Response(exc.code, b"", "")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TranscriptTransportError(
            f"caption endpoint unreachable ({type(exc).__name__})"
        ) from exc


@dataclass(frozen=True)
class _Response:
    status: int
    body: bytes
    content_type: str = ""


def parse_vtt(text: str) -> tuple[TranscriptSegmentInput, ...]:
    """Parse WebVTT cues into validated, verbatim timestamped segments."""
    if not isinstance(text, str):
        raise InvalidTranscriptResponseError("VTT payload is not text")
    if len(text.encode("utf-8")) > MAX_VTT_BYTES:
        raise InvalidTranscriptResponseError("VTT payload is too large")
    lines = text.replace("\r\n", "\n").split("\n")
    segments: list[TranscriptSegmentInput] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        match = _TIMESTAMP_RE.match(stripped)
        short = None if match is not None else _SHORT_TIMESTAMP_RE.match(stripped)
        index += 1
        if match is None and short is None:
            continue
        if match is not None:
            hours_s, mins_s, secs_s, millis_s, eh, em, es, ems = match.groups()
            start = (
                int(hours_s) * 3600
                + int(mins_s) * 60
                + int(secs_s)
                + int(millis_s) / 1000.0
            )
            end = int(eh) * 3600 + int(em) * 60 + int(es) + int(ems) / 1000.0
        else:
            assert short is not None
            mins_s, secs_s, millis_s, em, es, ems = short.groups()
            start = int(mins_s) * 60 + int(secs_s) + int(millis_s) / 1000.0
            end = int(em) * 60 + int(es) + int(ems) / 1000.0
        if end <= start:
            raise InvalidTranscriptResponseError("VTT cue has invalid timestamps")

        cue_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            cue_lines.append(lines[index])
            index += 1
        cue_text = "\n".join(cue_lines).strip()
        if not cue_text:
            continue  # empty cue: skip, never fabricate text
        if len(cue_text) > MAX_CUE_TEXT_CHARACTERS:
            raise InvalidTranscriptResponseError("VTT cue text is too large")
        segments.append(
            TranscriptSegmentInput(start_time=start, end_time=end, text=cue_text)
        )
        if len(segments) > MAX_CUES:
            raise InvalidTranscriptResponseError("too many VTT cues")
    if not segments:
        raise InvalidTranscriptResponseError("VTT payload has no usable cues")
    return tuple(segments)


class YouTubeTranscriptAdapter:
    """Fetch authorized captions and return a validated `TranscriptInput`."""

    def __init__(
        self,
        *,
        access_token: SecretStr,
        granted_scopes: Sequence[str] | str,
        transport: CaptionTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(access_token, SecretStr):
            raise TranscriptAdapterError("access_token must be a SecretStr")
        if not access_token.get_secret_value().strip():
            raise TranscriptAdapterError("access_token is empty")
        scopes = (
            [granted_scopes]
            if isinstance(granted_scopes, str)
            else list(granted_scopes)
        )
        if YOUTUBE_FORCE_SSL_SCOPE not in scopes:
            raise TranscriptAdapterError(
                "caption access requires youtube.force-ssl scope"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < timeout_seconds <= 60
        ):
            raise TranscriptAdapterError("timeout_seconds is out of bounds")
        self._token = access_token
        self._transport: CaptionTransport = transport or _default_transport
        self._timeout = float(timeout_seconds)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token.get_secret_value()}",
            "Accept": "application/json",
        }

    def _get(self, url: str, *, accept: str = "application/json") -> _Response:
        headers = self._headers()
        headers["Accept"] = accept
        try:
            response = self._transport(url, headers, self._timeout)
        except TranscriptAdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak a token
            raise TranscriptTransportError(
                f"caption request failed ({type(exc).__name__})"
            ) from exc
        status = getattr(response, "status", None)
        if not isinstance(status, int):
            raise TranscriptTransportError("caption endpoint returned an invalid status")
        if status in (401, 403):
            raise TranscriptAdapterError("caption access was denied or unauthorized")
        if not 200 <= status < 300:
            raise TranscriptTransportError(
                f"caption request failed (HTTP {status})"
            )
        return _Response(
            status=status,
            body=getattr(response, "body", b""),
            content_type=str(getattr(response, "content_type", "")),
        )

    def list_caption_tracks(self, video_id: str) -> tuple[CaptionTrack, ...]:
        """List caption metadata; the API key alone cannot do this."""
        try:
            checked = validate_video_id(video_id)
        except ValueError as exc:
            raise TranscriptAdapterError("invalid YouTube video id") from exc
        query = urllib.parse.urlencode({"part": "snippet", "videoId": checked})
        response = self._get(f"{CAPTIONS_LIST_URL}?{query}")
        try:
            payload = json.loads(response.body)
        except (ValueError, TypeError) as exc:
            raise InvalidTranscriptResponseError("caption list is invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise InvalidTranscriptResponseError("caption list has no items")
        tracks: list[CaptionTrack] = []
        for item in payload["items"]:
            if not isinstance(item, dict):
                raise InvalidTranscriptResponseError("malformed caption track")
            validate_caption_id(item.get("id"))
            snippet = item.get("snippet")
            if not isinstance(snippet, dict):
                raise InvalidTranscriptResponseError("malformed caption snippet")
            language = snippet.get("language")
            kind = snippet.get("kind")
            status = snippet.get("status")
            name = snippet.get("name", "")
            if (
                not isinstance(language, str)
                or not language.strip()
                or not isinstance(kind, str)
                or not isinstance(status, str)
                or not isinstance(name, str)
            ):
                raise InvalidTranscriptResponseError("invalid caption metadata")
            tracks.append(
                CaptionTrack(
                    id=item["id"],
                    language=language,
                    kind=kind,
                    status=status,
                    name=name,
                )
            )
            if len(tracks) > MAX_CAPTION_TRACKS:
                raise InvalidTranscriptResponseError("too many caption tracks")
        return tuple(tracks)

    def fetch_transcript(
        self,
        *,
        youtube_video_id: str,
        video_id: UUID,
        content_right_id: UUID,
        language: str | None = None,
    ) -> TranscriptInput:
        """Return a validated TranscriptInput from an authorized caption track.

        ``youtube_video_id`` is the external 11-character ID used with the API;
        ``video_id`` / ``content_right_id`` are the internal UUID provenance
        required by the existing ingestion service.
        """
        try:
            checked_video = validate_video_id(youtube_video_id)
        except ValueError as exc:
            raise TranscriptAdapterError("invalid YouTube video id") from exc
        if not isinstance(video_id, UUID) or not isinstance(content_right_id, UUID):
            raise TranscriptAdapterError("transcript provenance must be UUIDs")
        if language is not None and (
            not isinstance(language, str) or not language.strip()
        ):
            raise TranscriptAdapterError("language must be non-empty when supplied")

        tracks = self.list_caption_tracks(checked_video)
        ready = [track for track in tracks if track.status == "ready"]
        if language is not None:
            ready = [track for track in ready if track.language == language]
        if not ready:
            raise CaptionUnavailableError("no authorized caption track is ready")
        # Prefer human-created captions over ASR; ties break by track id.
        selected = min(
            ready,
            key=lambda track: (track.kind == "asr", track.id),
        )

        # Google's discovery document: captions.download path is
        # `youtube/v3/captions/{id}` (path parameter) with `tfmt` in the query.
        safe_id = urllib.parse.quote(selected.id, safe="")
        query = urllib.parse.urlencode({"tfmt": "vtt"})
        response = self._get(
            f"{CAPTIONS_DOWNLOAD_URL}{safe_id}?{query}",
            accept="application/octet-stream",
        )
        try:
            vtt = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidTranscriptResponseError("VTT payload is not UTF-8") from exc
        segments = parse_vtt(vtt)
        source = (
            TranscriptSource.ASR_GENERATED
            if selected.kind == "asr"
            else TranscriptSource.CREATOR_PROVIDED
        )
        return TranscriptInput(
            video_id=video_id,
            content_right_id=content_right_id,
            source_type=source,
            language=selected.language,
            segments=list(segments),
        )


__all__ = [
    "CAPTIONS_DOWNLOAD_URL",
    "CAPTIONS_LIST_URL",
    "CaptionTrack",
    "CaptionTransport",
    "CaptionUnavailableError",
    "InvalidTranscriptResponseError",
    "TranscriptAdapterError",
    "TranscriptTransportError",
    "YouTubeTranscriptAdapter",
    "parse_vtt",
    "validate_caption_id",
]
