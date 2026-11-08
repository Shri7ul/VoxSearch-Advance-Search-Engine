"""Repository layer for the VoxSearch backend (Goal 08).

Repositories are thin, stateless-per-request data-access objects built on
SQLAlchemy 2.x ``select()`` queries. They translate well-defined operations into
queries against the ``voxsearch`` tables declared in ``app.db.models`` (the five
Goal 06/07 tables plus the three Goal 12 word-alignment tables). They
intentionally contain **no** schema, migration, or application-authorization
logic.

Transaction ownership (Goal 08 contract)
----------------------------------------
* **Services own transaction boundaries.** Only ``app.services`` calls
  ``session.commit()`` / ``session.rollback()``.
* **Repositories NEVER commit.** A repository may call ``session.flush()`` (or
  the :meth:`BaseRepository.flush` helper) when a server-generated value (UUID
  primary key, ``now()`` timestamp) is needed inside the same transaction.
* **Session lifecycle** stays with the existing ``app.db.session.get_db()``
  dependency: a ``Session`` is injected into each repository and service.
* Repositories never swallow or rewrap ``SQLAlchemyError``; failures propagate
  to the owning service, which rolls back and re-raises.

Using the layer (no HTTP routes exist yet in Goal 08)::

    with SessionFactory(bind=get_engine()) as session:
        channels = ChannelRepository(session)
        service = ChannelService(session, channels=channels)
        channel = service.create_or_get_channel(
            youtube_channel_id="UC...", title="..."
        )

Imports in this package deliberately target concrete submodules (for example
``from app.repositories.base import BaseRepository``) rather than the package
itself, so the re-exports below cannot create an import cycle.
"""
from app.repositories.base import MAX_LIMIT, DEFAULT_LIMIT, UNSET, BaseRepository
from app.repositories.channel_repository import ChannelRepository
from app.repositories.content_rights_repository import (
    ACTIVE_STATUS,
    REVOKED_STATUS,
    ContentRightsRepository,
)
from app.repositories.source_audio_asset_repository import SourceAudioAssetRepository
from app.repositories.transcript_alignment_run_repository import (
    TranscriptAlignmentRunRepository,
)
from app.repositories.transcript_asset_repository import TranscriptAssetRepository
from app.repositories.transcript_segment_repository import TranscriptSegmentRepository
from app.repositories.transcript_word_repository import TranscriptWordRepository
from app.repositories.video_repository import VideoRepository

__all__ = [
    "ACTIVE_STATUS",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "REVOKED_STATUS",
    "UNSET",
    "BaseRepository",
    "ChannelRepository",
    "ContentRightsRepository",
    "SourceAudioAssetRepository",
    "TranscriptAlignmentRunRepository",
    "TranscriptAssetRepository",
    "TranscriptSegmentRepository",
    "TranscriptWordRepository",
    "VideoRepository",
]
