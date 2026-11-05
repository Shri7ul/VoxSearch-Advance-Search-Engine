"""Content-rights service (Goal 08): current rights, eligibility, revocation.

Fail-closed authorization eligibility
------------------------------------
:meth:`ContentRightsService.is_active` is a pure, offline-testable guard. A
rights record is eligible ONLY when every condition holds:

* ``status`` equals :data:`~app.repositories.content_rights_repository.ACTIVE_STATUS`
  exactly (any other / unknown / inactive status -> ``False``),
* ``revoked_at`` is NULL,
* ``granted_at`` is present and not in the future (not-yet-granted),
* ``expires_at`` is NULL or strictly in the future (not expired).

Anything uncertain (missing record, ``None`` values, unsupported status) fails
closed. This uses the model's existing representation; it does **not** invent
authorization types, status values, or legal rules.

Important (carried over from Goal 07): the composite foreign keys enforce
*relational ownership* consistency only. They do NOT enforce whether a rights
record is currently authorized, active, or unexpired — that is exactly what
this application-level, fail-closed logic provides. This service never
automatically authorizes a video and never creates rights records.
"""
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import ContentRight
from app.repositories.content_rights_repository import (
    ACTIVE_STATUS,
    ContentRightsRepository,
    resolve_now,
    to_utc,
)
from app.services import transaction


class ContentRightsService:
    """Application operations for content rights. Owns transaction boundaries."""

    def __init__(
        self,
        session: Session,
        rights: ContentRightsRepository | None = None,
    ) -> None:
        self._session = session
        self.rights = rights or ContentRightsRepository(session)

    @staticmethod
    def is_active(
        rights: ContentRight | None,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Return ``True`` only for a currently active, unexpired rights record.

        Pure function: no session, no database, no side effects. Naive
        datetimes are treated as UTC so comparisons never mix aware/naive
        values. Fails closed on every form of uncertainty.
        """
        if rights is None:
            return False
        if rights.status != ACTIVE_STATUS:
            return False
        if rights.revoked_at is not None:
            return False

        reference = resolve_now(now)
        granted = rights.granted_at
        if granted is None or to_utc(granted) > reference:
            return False  # not-yet-granted
        expires = rights.expires_at
        if expires is not None and to_utc(expires) <= reference:
            return False  # expired
        return True

    def get_current_rights(
        self,
        video_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> ContentRight | None:
        """Return the current active rights for a video, or ``None``.

        Read-only: the repository query already excludes ineligible rows, and
        the result is re-checked with :meth:`is_active` as defense in depth so
        an expired/revoked record is never returned as active. No commit.
        """
        rights = self.rights.get_current_rights_by_video(video_id, now=now)
        if rights is not None and not self.is_active(rights, now=now):
            return None
        return rights

    def is_authorized(
        self,
        video_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Eligibility check: ``True`` only when current active rights exist.

        This is a structural check, not a legal determination and not an
        automatic authorization: it never creates or activates a rights record.
        """
        return self.get_current_rights(video_id, now=now) is not None

    def list_rights(
        self,
        video_id: uuid.UUID,
        *,
        limit: int | None = 50,
        offset: int | None = 0,
    ) -> list[ContentRight]:
        """Read-only, paginated provenance/history listing (any status)."""
        return self.rights.list_rights_by_video(
            video_id, limit=limit, offset=offset
        )

    def revoke_rights(
        self,
        rights_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> ContentRight | None:
        """Revoke a rights record in one transaction; ``None`` if absent.

        The repository sets ``status='revoked'`` and ``revoked_at``; the commit
        is owned here. Rollback happens automatically on any exception.
        """
        with transaction(self._session):
            return self.rights.revoke_rights(rights_id, now=now)
