"""YouTube Data API setup (Goal 34 — authentication only).

No video discovery, metadata retrieval, channel registry or OAuth flow is
implemented here; those are Goals 35–37. Real keys come only from environment /
the git-ignored local `.env` and are held as ``SecretStr``.
"""
from app.youtube.discovery import (
    InvalidVideoDiscoveryInputError,
    VideoDiscoveryService,
    VideoMetadata,
    VideoSearchHit,
    VideoSearchPage,
    get_video_metadata,
    parse_duration_seconds,
    search_videos,
    validate_video_id,
)
from app.youtube.client import (
    DEFAULT_YOUTUBE_API_BASE_URL,
    MAX_VIDEO_IDS_PER_CALL,
    YouTubeAccessRestrictedError,
    YouTubeApiClient,
    YouTubeAuthError,
    YouTubeAuthFailedError,
    YouTubeAuthStatus,
    YouTubeCredentials,
    YouTubeNotConfiguredError,
    YouTubeQuotaError,
    YouTubeRequestError,
    YouTubeResponseError,
)

__all__ = [
    "InvalidVideoDiscoveryInputError",
    "VideoDiscoveryService",
    "VideoMetadata",
    "VideoSearchHit",
    "VideoSearchPage",
    "get_video_metadata",
    "parse_duration_seconds",
    "search_videos",
    "validate_video_id",
    "DEFAULT_YOUTUBE_API_BASE_URL",
    "MAX_VIDEO_IDS_PER_CALL",
    "YouTubeAccessRestrictedError",
    "YouTubeApiClient",
    "YouTubeAuthError",
    "YouTubeAuthFailedError",
    "YouTubeAuthStatus",
    "YouTubeCredentials",
    "YouTubeNotConfiguredError",
    "YouTubeQuotaError",
    "YouTubeRequestError",
    "YouTubeResponseError",
]
