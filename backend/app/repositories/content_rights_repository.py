"""Repository for ``voxsearch.content_rights`` (Goal 08).

Status values are **not invented here**: ``create_content_rights`` requires the
caller to supply ``status`` verbatim (the model documents ``active`` /
``pending`` / ``expired`` / ``revoked`` as the current representation). The two
constants below are the values used by the *current-rights* query and by
revocation; they match those documented values and are kept in one place so the
service layer and tests share them.

"Current" rights consider all four relevant model fields:

* ``status`` must equal :data:`ACTIVE_STATUS`,
* ``revoked_at`` must be NULL,
* ``granted_at`` must be present and not in the future (not-yet-granted),
* ``expires_at`` must be NULL or in the future (not expired).

This is a structural, fail-closed lookup — it is **not** a legal authorization
determination (see ``app.services.content_rights_service``). This repository
never commits and never performs application data DML beyond explicit writes.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import or_, select

from app.db.models import ContentRight
from app.repositories.base import BaseRepository

ACTIVE_STATUS = "active"
"""The only ``status`` value treated as currently active (fail closed)."""

REVOKED_STATUS = "revoked"
"""The documented status set by :meth:`ContentRightsRepository.revoke_rights`."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(value: datetime) -> datetime:
    """Normalize a non-null datetime to aware UTC (naive values assumed UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def resolve_now(now: datetime | None = None) -> datetime:
    """Return ``now`` normalized to UTC, or the current UTC time.

    Accepting an explicit ``now`` keeps current-rights logic deterministic and
    offline-testable without a live database.
    """
    return _utc_now() if now is None else to_utc(now)


class ContentRightsRepository(BaseRepository):
    """Data access for :class:`~app.db.models.ContentRight` rows."""

    def create_content_rights(
        self,
        *,
        video_id: uuid.UUID,
        authorization_type: str,
        authorization_reference: str,
        status: str,
        granted_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> ContentRight:
        """Insert a rights record and flush (no commit).

        ``status`` is caller-supplied: this layer does not invent authorization
        types or status values. ``video_id`` is enforced by
        ``fk_content_rights_video_id_videos``.
        """
        rights = ContentRight(
            video_id=video_id,
            authorization_type=authorization_type,
            authorization_reference=authorization_reference,
            status=status,
            granted_at=granted_at,
            expires_at=expires_at,
        )
        self.session.add(rights)
        self.flush()
        return rights

    def get_rights_by_id(self, rights_id: uuid.UUID) -> ContentRight | None:
        """Return a rights record by primary key, or ``None``."""
        return self.session.get(ContentRight, rights_id)

    def list_rights_by_video(
        self,
        video_id: uuid.UUID,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[ContentRight]:
        """List all rights records for a video (any status), with pagination.

        Unlike :meth:`get_current_rights_by_video`, this is an unfiltered
        provenance/history listing ordered ``created_at DESC, id DESC``.
        """
        stmt = (
            select(ContentRight)
            .where(ContentRight.video_id == video_id)
            .order_by(ContentRight.created_at.desc(), ContentRight.id.desc())
        )
        stmt = self.paginate(stmt, limit=limit, offset=offset)
        return list(self.session.scalars(stmt).all())

    def get_current_rights_by_video(
        self,
        video_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> ContentRight | None:
        """Return the current active rights record for a video, or ``None``.

        Expired, revoked, not-yet-granted, and non-active records are excluded
        in SQL. If several eligible records exist, the most recently granted
        wins (``granted_at DESC``) with ``id DESC`` as a deterministic
        tiebreaker. A single query avoids N+1 reads.
        """
        reference = resolve_now(now)
        stmt = (
            select(ContentRight)
            .where(ContentRight.video_id == video_id)
            .where(ContentRight.status == ACTIVE_STATUS)
            .where(ContentRight.revoked_at.is_(None))
            .where(ContentRight.granted_at.is_not(None))
            .where(ContentRight.granted_at <= reference)
            .where(
                or_(
                    ContentRight.expires_at.is_(None),
                    ContentRight.expires_at > reference,
                )
            )
            .order_by(ContentRight.granted_at.desc(), ContentRight.id.desc())
        )
        return self.session.scalars(stmt).first()

    def revoke_rights(
        self,
        rights_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> ContentRight | None:
        """Mark a rights record revoked and flush (no commit).

        Sets both ``status`` to :data:`REVOKED_STATUS` and ``revoked_at`` to the
        resolved timestamp, so subsequent current-rights lookups exclude it.
        Returns ``None`` when the record does not exist.
        """
        rights = self.get_rights_by_id(rights_id)
        if rights is None:
            return None
        rights.status = REVOKED_STATUS
        rights.revoked_at = resolve_now(now)
        self.flush()
        return rights
