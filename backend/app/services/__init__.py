"""Service layer for the VoxSearch backend (Goal 08).

Services are the **application/business boundary**. They orchestrate one or
more repositories, enforce the data-integrity and fail-closed rules that a
database constraint alone cannot express, and own every transaction boundary.

Transaction ownership (Goal 08 contract)
----------------------------------------
* **Services own transactions.** Only :func:`transaction` — used exclusively by
  services — calls ``session.commit()`` / ``session.rollback()``.
* **Repositories never commit** (they only ``flush()``).
* **Session lifecycle** stays with ``app.db.session.get_db()``; a ``Session`` is
  injected into each service. Services never create or close sessions.
* Each mutating service method uses exactly one :func:`transaction` block, so
  there are no nested or redundant commits. Read-only methods do not open a
  transaction block at all.
* ``flush()`` inside repositories makes server-generated UUID PKs/timestamps
  available within the same transaction before the commit.

:func:`transaction` re-raises every exception after rolling back. The broad
``except Exception`` is deliberate: a rollback must happen for *any* failure
(domain error or SQLAlchemy DB error) before the error propagates. Nothing is
swallowed, hidden, or logged away — SQLAlchemy exceptions always reach the
caller, satisfying the "never silently swallow DB errors" rule.

Domain exceptions below are raised by services (not repositories) to give
callers clear, credential-free failures instead of leaking constraint errors.

This package intentionally does NOT re-export the concrete service classes:
doing so would create an import cycle, because each service module imports
:func:`transaction` from here. Import services from their modules, e.g.
``from app.services.channel_service import ChannelService``.
"""
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session


class VoxSearchServiceError(Exception):
    """Base class for explicit, credential-free service-layer failures."""


class ContentRightsNotEligibleError(VoxSearchServiceError):
    """A rights record is missing, not active, or belongs to another video."""


class TranscriptAssetNotFoundError(VoxSearchServiceError):
    """The requested transcript asset does not exist."""


class TranscriptAssetOwnershipError(VoxSearchServiceError):
    """The transcript asset belongs to a different video than requested."""


@contextmanager
def transaction(session: Session) -> Iterator[None]:
    """Service-owned transaction boundary: commit on success, else rollback.

    Usage::

        with transaction(self._session):
            obj = self._repo.create_x(...)
        return obj

    On any exception inside the block (including a failed ``commit``), the
    session is rolled back and the exception is re-raised unchanged.
    """
    try:
        yield
        session.commit()
    except Exception:
        session.rollback()
        raise


__all__ = [
    "ContentRightsNotEligibleError",
    "TranscriptAssetNotFoundError",
    "TranscriptAssetOwnershipError",
    "VoxSearchServiceError",
    "transaction",
]
