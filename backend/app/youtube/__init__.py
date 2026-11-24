"""YouTube Data API setup (Goal 34 — authentication only).

No video discovery, metadata retrieval, channel registry or OAuth flow is
implemented here; those are Goals 35–37. Real keys come only from environment /
the git-ignored local `.env` and are held as ``SecretStr``.
"""
from app.youtube.client import (
    DEFAULT_YOUTUBE_API_BASE_URL,
    YouTubeAccessRestrictedError,
    YouTubeApiClient,
    YouTubeAuthError,
    YouTubeAuthFailedError,
    YouTubeAuthStatus,
    YouTubeCredentials,
    YouTubeNotConfiguredError,
    YouTubeQuotaError,
    YouTubeRequestError,
)

__all__ = [
    "DEFAULT_YOUTUBE_API_BASE_URL",
    "YouTubeAccessRestrictedError",
    "YouTubeApiClient",
    "YouTubeAuthError",
    "YouTubeAuthFailedError",
    "YouTubeAuthStatus",
    "YouTubeCredentials",
    "YouTubeNotConfiguredError",
    "YouTubeQuotaError",
    "YouTubeRequestError",
]
