"""Shared, minimal building blocks for the Goal 08 repositories.

This is deliberately a *small* base, not a generic repository framework:

* it holds the injected :class:`~sqlalchemy.orm.Session`,
* it offers a ``flush()`` helper (never a commit),
* it offers one pagination helper that every ``list_*`` method shares.

Anything operation-specific lives in the concrete repository. Transaction
boundaries are owned by the service layer; nothing here commits or rolls back.
"""
from sqlalchemy import Select
from sqlalchemy.orm import Session

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


class _Unset:
    """Sentinel type distinguishing "argument omitted" from "explicitly None"."""


UNSET = _Unset()
"""Pass as a default so nullable columns can be intentionally cleared."""


class BaseRepository:
    """Common session access and pagination for all repositories.

    Parameters
    ----------
    session:
        The request-scoped SQLAlchemy ``Session`` supplied by ``get_db()``.
        A repository never creates, commits, or closes a session.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def flush(self) -> None:
        """Flush pending changes within the current transaction.

        Needed after ``session.add()`` so server defaults (``gen_random_uuid()``
        primary keys, ``now()`` timestamps) become available on the instance.
        This method NEVER commits; the owning service decides when the
        transaction ends.
        """
        self.session.flush()

    @staticmethod
    def normalize_pagination(
        limit: int | None, offset: int | None
    ) -> tuple[int, int]:
        """Validate and clamp pagination at the caller boundary.

        ``limit`` is capped at :data:`MAX_LIMIT` to avoid unbounded reads;
        ``offset`` must be non-negative. Invalid values raise ``ValueError``
        rather than being silently coerced, so bad input is never hidden.
        """
        effective_limit = DEFAULT_LIMIT if limit is None else limit
        effective_offset = 0 if offset is None else offset
        if effective_limit < 1:
            raise ValueError("limit must be >= 1")
        if effective_offset < 0:
            raise ValueError("offset must be >= 0")
        return min(effective_limit, MAX_LIMIT), effective_offset

    @classmethod
    def paginate(
        cls,
        stmt: Select,
        *,
        limit: int | None = DEFAULT_LIMIT,
        offset: int | None = 0,
    ) -> Select:
        """Apply validated ``LIMIT``/``OFFSET`` to a statement.

        Callers are responsible for adding a deterministic ``ORDER BY`` (with
        ``id`` as the final tiebreaker) before calling this helper.
        """
        effective_limit, effective_offset = cls.normalize_pagination(limit, offset)
        return stmt.limit(effective_limit).offset(effective_offset)
