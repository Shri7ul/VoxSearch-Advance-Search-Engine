"""Repository for ``voxsearch.source_audio_assets`` (Goal 12).

An authorized audio asset is the time base for word alignment. Its
``timeline_offset_ms`` maps the AUDIO timeline to the VIDEO timeline
(``video_ms = audio_ms + offset``); ``duration_ms`` is the available audio
window used (in the service layer) to reject segments that fall outside it.

All fields mirror the approved Goal 12 design-freeze §3.1 / §4 column set:
``storage_uri`` (OPAQUE handle — never a URL, never credentials / signed-URL
query strings), ``checksum_sha256`` (64 lowercase hex, binds exact bytes),
``mime_type`` + ``rights_basis`` (format + authorization provenance),
optional ``duration_ms`` / ``sample_rate_hz`` (NULL = unknown, NEVER guessed).

Ownership integrity (approved decision #8): the composite FK
``fk_source_audio_assets_right_video`` guarantees the referenced right belongs
to this audio's video, and the additive ``UNIQUE(id, content_right_id,
video_id)`` backs the exact-authorization-lineage FK from
``transcript_alignment_runs``. This repository never weakens those guarantees
and never commits.
"""
import re
import uuid

from sqlalchemy import select

from app.db.models import SourceAudioAsset
from app.repositories.base import BaseRepository

_CHECKSUM_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class SourceAudioAssetRepository(BaseRepository):
    """Data access for :class:`~app.db.models.SourceAudioAsset` rows."""

    def create_source_audio_asset(
        self,
        *,
        video_id: uuid.UUID,
        content_right_id: uuid.UUID,
        storage_uri: str,
        checksum_sha256: str,
        mime_type: str,
        rights_basis: str,
        timeline_offset_ms: int,
        duration_ms: int | None = None,
        sample_rate_hz: int | None = None,
    ) -> SourceAudioAsset:
        """Insert an audio asset and flush (no commit).

        The service layer validates these fields via :class:`SourceAudioAssetInput`
        before calling here; the DB CHECKs (nonblank, checksum format, positive
        duration/sample rate) are a second guard. ``duration_ms`` /
        ``sample_rate_hz`` are OPTIONAL (NULL = unknown). ``timeline_offset_ms``
        may be negative (audio starts before video 0). The composite FK is
        checked at flush; a cross-video right raises ``IntegrityError`` and is
        never swallowed here.
        """
        for name, value in (
            ("storage_uri", storage_uri),
            ("mime_type", mime_type),
            ("rights_basis", rights_basis),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank str")
        if not isinstance(checksum_sha256, str) or _CHECKSUM_SHA256_RE.fullmatch(
            checksum_sha256
        ) is None:
            raise ValueError(
                "checksum_sha256 must be exactly 64 lowercase hex characters"
            )
        if isinstance(timeline_offset_ms, bool) or not isinstance(
            timeline_offset_ms, int
        ):
            raise ValueError("timeline_offset_ms must be an int")
        if duration_ms is not None:
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
                raise ValueError("duration_ms must be an int or None")
            if duration_ms <= 0:
                raise ValueError("duration_ms must be > 0 when supplied")
        if sample_rate_hz is not None:
            if isinstance(sample_rate_hz, bool) or not isinstance(
                sample_rate_hz, int
            ):
                raise ValueError("sample_rate_hz must be an int or None")
            if sample_rate_hz <= 0:
                raise ValueError("sample_rate_hz must be > 0 when supplied")
        asset = SourceAudioAsset(
            video_id=video_id,
            content_right_id=content_right_id,
            storage_uri=storage_uri,
            checksum_sha256=checksum_sha256,
            mime_type=mime_type,
            rights_basis=rights_basis,
            duration_ms=duration_ms,
            sample_rate_hz=sample_rate_hz,
            timeline_offset_ms=timeline_offset_ms,
        )
        self.session.add(asset)
        self.flush()
        return asset

    def get_audio_asset_by_id(
        self, audio_id: uuid.UUID
    ) -> SourceAudioAsset | None:
        """Return an audio asset by primary key, or ``None``."""
        return self.session.get(SourceAudioAsset, audio_id)

    def list_audio_assets_by_video(
        self,
        video_id: uuid.UUID,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[SourceAudioAsset]:
        """List a video's audio assets, newest-first, with pagination."""
        stmt = (
            select(SourceAudioAsset)
            .where(SourceAudioAsset.video_id == video_id)
            .order_by(
                SourceAudioAsset.created_at.desc(), SourceAudioAsset.id.desc()
            )
        )
        stmt = self.paginate(stmt, limit=limit, offset=offset)
        return list(self.session.scalars(stmt).all())
