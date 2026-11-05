"""SQLAlchemy 2.x declarative models for the VoxSearch ``voxsearch`` schema.

Goal 06 established the initial relational schema / migration foundation and
Goal 07 added composite foreign keys that enforce cross-table video ownership
consistency. This module defines the relational metadata; it intentionally
contains:

* NO vector / embedding columns or indexes (planned for a later goal),
* NO ORM relationships (the tables stand alone; ``get_db`` sessions are used
  only for explicit queries in future goals),
* NO sample / production data,
* NO application CRUD or ingestion logic.

All objects live in the dedicated ``voxsearch`` schema so they are isolated
from Supabase-managed objects in ``public`` and the system schemas.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "voxsearch"

# --- Goal 07: cross-table video ownership integrity -----------------------
# A composite FK requires a UNIQUE constraint on exactly the referenced column
# pair. ``id`` is already the PK, but PostgreSQL requires the (id, video_id)
# pair to be declared explicitly, hence these unique constraints. Names are
# kept byte-for-byte in sync with the Goal 07 Alembic migration
# (migrations/versions/fe7b2b8e5323_enforce_transcript_ownership_integrity.py).
UQ_CONTENT_RIGHTS_ID_VIDEO_ID = "uq_content_rights_id_video_id"
UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID = "uq_transcript_assets_id_video_id"
FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO = "fk_transcript_assets_content_right_id_video_id"
FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO = "fk_transcript_segments_transcript_asset_id_video_id"


# Explicit schema-qualified metadata: every Table created from this Base is
# placed in ``voxsearch`` and ForeignKey targets are resolved against it.
class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA)


class Channel(Base):
    """A YouTube channel whose videos may be discovered for VoxSearch."""

    __tablename__ = "channels"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    youtube_channel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "youtube_channel_id", name="uq_channels_youtube_channel_id"
        ),
    )


class Video(Base):
    """A YouTube video belonging to a channel.

    Discovery alone does NOT imply authorization to process its transcript;
    authorization is represented separately by :class:`ContentRight`.
    """

    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    channel_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey(f"{SCHEMA}.channels.id", name="fk_videos_channel_id_channels"),
        nullable=False,
    )
    youtube_video_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("youtube_video_id", name="uq_videos_youtube_video_id"),
        # FK columns are not auto-indexed by PostgreSQL. The external id is
        # already indexed by its unique constraint above.
        Index("ix_videos_channel_id", "channel_id"),
    )


class ContentRight(Base):
    """Authorization / rights provenance for a video's transcript processing.

    ``authorization_type`` (e.g. explicit license, creator permission) together
    with ``authorization_reference`` records provenance. ``status`` (e.g.
    active / pending / expired / revoked) represents whether the video is
    currently authorized: an inactive status means it must NOT be processed.
    """

    __tablename__ = "content_rights"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey(f"{SCHEMA}.videos.id", name="fk_content_rights_video_id_videos"),
        nullable=False,
    )
    authorization_type: Mapped[str] = mapped_column(String(64), nullable=False)
    authorization_reference: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    granted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        # Backs the composite FK from transcript_assets(content_right_id,
        # video_id). Required by PostgreSQL even though `id` is already unique.
        UniqueConstraint("id", "video_id", name=UQ_CONTENT_RIGHTS_ID_VIDEO_ID),
        Index("ix_content_rights_video_id", "video_id"),
    )


class TranscriptAsset(Base):
    """A tracked transcript asset for a video, tied to its authorization.

    ``source_type`` (e.g. youtube_captions / asr / upload) and ``language``
    identify the asset; ``status`` tracks its processing lifecycle. The
    composite FK guarantees ``content_right_id`` belongs to this asset's
    ``video_id`` — an asset can never be tied to another video's rights record.
    """

    __tablename__ = "transcript_assets"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey(f"{SCHEMA}.videos.id", name="fk_transcript_assets_video_id_videos"),
        nullable=False,
    )
    content_right_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        # Backs the composite FK from transcript_segments(transcript_asset_id,
        # video_id). Required by PostgreSQL even though `id` is already unique.
        UniqueConstraint("id", "video_id", name=UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID),
        # Ownership guarantee: the referenced content-right must belong to the
        # SAME video. Replaces the old single-column FK on content_right_id.
        ForeignKeyConstraint(
            ["content_right_id", "video_id"],
            [
                f"{SCHEMA}.content_rights.id",
                f"{SCHEMA}.content_rights.video_id",
            ],
            name=FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO,
        ),
        Index("ix_transcript_assets_video_id", "video_id"),
        # content_right_id is the leading column of the composite FK above;
        # PostgreSQL does not auto-index it, so this index is retained.
        Index("ix_transcript_assets_content_right_id", "content_right_id"),
    )


class TranscriptSegment(Base):
    """A timestamped spoken-content segment of a transcript asset.

    ``start_seconds`` / ``end_seconds`` are offsets into the video. Check
    constraints enforce non-negative start, end strictly after start, and
    non-empty transcript text. The composite FK guarantees the referenced
    transcript asset belongs to this segment's ``video_id``.
    """

    __tablename__ = "transcript_segments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    transcript_asset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey(f"{SCHEMA}.videos.id", name="fk_transcript_segments_video_id_videos"),
        nullable=False,
    )
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    segment_text: Mapped[str] = mapped_column("text", Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        # Ownership guarantee: the referenced transcript asset must belong to
        # the SAME video. Replaces the old single-column FK on asset id.
        ForeignKeyConstraint(
            ["transcript_asset_id", "video_id"],
            [
                f"{SCHEMA}.transcript_assets.id",
                f"{SCHEMA}.transcript_assets.video_id",
            ],
            name=FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO,
        ),
        # One segment per start offset per asset; its leading column also
        # indexes transcript_asset_id, so no redundant standalone FK index.
        UniqueConstraint(
            "transcript_asset_id", "start_seconds",
            name="uq_transcript_segments_asset_start",
        ),
        Index("ix_transcript_segments_video_id", "video_id"),
        CheckConstraint(
            "start_seconds >= 0", name="ck_transcript_segments_start_nonnegative"
        ),
        CheckConstraint(
            "end_seconds > start_seconds",
            name="ck_transcript_segments_end_after_start",
        ),
        CheckConstraint(
            'length(btrim("text")) > 0',
            name="ck_transcript_segments_text_nonempty",
        ),
    )
