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
    BigInteger,
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

# --- Goal 12: word-alignment ownership integrity --------------------------
# A composite FK is valid on PostgreSQL ONLY when the referenced column tuple
# has a matching UNIQUE / PRIMARY KEY constraint. ``id`` is already the PK on
# every table, but PostgreSQL requires the EXACT referenced column SET to be
# declared. These additive UNIQUE constraints (on existing tables) and the ones
# on the three new tables are the FK targets. Names follow the APPROVED Goal 12
# design-freeze specification (§4) and are kept byte-for-byte in sync with the
# Goal 12 Alembic migration; the tests assert equality.
#
# The two ``*_id_right_video`` uniques back the EXACT-AUTHORIZATION-LINEAGE
# composite FKs required by the full implementation task (§6): a run can only
# combine a transcript asset and an audio asset authorized by the SAME right /
# SAME video. Those stronger FKs are IN ADDITION TO (never instead of) the
# §4 video-level FKs.
UQ_TRANSCRIPT_ASSETS_ID_RIGHT_VIDEO = "uq_transcript_assets_id_right_video"
UQ_TRANSCRIPT_SEGMENTS_ID_ASSET_VIDEO = "uq_transcript_segments_id_asset_video"

# source_audio_assets
UQ_SOURCE_AUDIO_ASSETS_ID_VIDEO_ID = "uq_source_audio_assets_id_video_id"
UQ_SOURCE_AUDIO_ASSETS_ID_RIGHT_VIDEO = "uq_source_audio_assets_id_right_video"
UQ_SOURCE_AUDIO_ASSETS_VIDEO_CHECKSUM = "uq_source_audio_assets_video_checksum"
FK_SOURCE_AUDIO_ASSETS_VIDEO = "fk_source_audio_assets_video"
FK_SOURCE_AUDIO_ASSETS_CONTENT_RIGHT_VIDEO = "fk_source_audio_assets_right_video"

# transcript_alignment_runs
UQ_ALIGNMENT_RUNS_ID_VIDEO_ID = "uq_alignment_runs_id_video_id"
UQ_ALIGNMENT_RUNS_ID_ASSET_VIDEO = "uq_transcript_alignment_runs_id_asset_video"
UQ_ALIGNMENT_RUNS_ATTEMPT = "uq_alignment_runs_attempt"
FK_ALIGNMENT_RUNS_VIDEO = "fk_alignment_runs_video"
FK_ALIGNMENT_RUNS_RIGHT_VIDEO = "fk_alignment_runs_right_video"
FK_ALIGNMENT_RUNS_ASSET_VIDEO = "fk_alignment_runs_asset_video"
FK_ALIGNMENT_RUNS_AUDIO_VIDEO = "fk_alignment_runs_audio_video"
FK_ALIGNMENT_RUNS_ASSET_RIGHT_VIDEO = "fk_alignment_runs_asset_right_video"
FK_ALIGNMENT_RUNS_AUDIO_RIGHT_VIDEO = "fk_alignment_runs_audio_right_video"
IX_ALIGNMENT_RUNS_ONE_CURRENT_SUCCESS = "uq_alignment_runs_current_success"

# transcript_words
UQ_TRANSCRIPT_WORDS_SEGMENT_WORD_INDEX = "uq_transcript_words_segment_word_index"
FK_TRANSCRIPT_WORDS_VIDEO = "fk_transcript_words_video"
FK_TRANSCRIPT_WORDS_SEGMENT_ASSET_VIDEO = "fk_transcript_words_segment_asset_video"
FK_TRANSCRIPT_WORDS_ASSET_VIDEO = "fk_transcript_words_asset_video"
FK_TRANSCRIPT_WORDS_RUN_VIDEO = "fk_transcript_words_run_video"
FK_TRANSCRIPT_WORDS_RUN_ASSET_VIDEO = "fk_transcript_words_run_asset_video"

# Closed alignment-run status set (approved design-freeze §4/§6). Kept here
# (and in the migration) as literals so the DB layer stays decoupled from
# ``app.transcripts``. The invariant asserted by the tests is: this DB tuple =
# the service-emittable ``app.transcripts.word_contract.ALIGNMENT_RUN_STATUSES``
# plus the single reserved label ``rejected_rights`` (DB set = service set +
# {``rejected_rights``}).
# ``rejected_rights`` is a RESERVED closed label: the final implementation task
# forbids persisting ANY run for an initial rights-gate rejection, so this goal
# NEVER emits it. It remains in the CHECK set for §4 schema compliance.
ALIGNMENT_RUN_STATUS_VALUES = (
    "pending", "running", "completed", "partial", "failed",
    "unsupported_language", "rejected_rights", "superseded",
)
_ALIGNMENT_STATUS_CHECK_SQL = (
    "status IN ("
    + ", ".join(f"'{s}'" for s in ALIGNMENT_RUN_STATUS_VALUES)
    + ")"
)


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
        # Goal 12: backs the exact-authorization-lineage FK from
        # transcript_alignment_runs(transcript_asset_id, content_right_id,
        # video_id). A run can only reference an asset under the SAME right.
        UniqueConstraint(
            "id", "content_right_id", "video_id",
            name=UQ_TRANSCRIPT_ASSETS_ID_RIGHT_VIDEO,
        ),
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

    Two text columns are deliberately kept separate (Goal 09 corrective):

    * ``segment_text`` maps to the physical ``"text"`` column and stores the
      NORMALIZED text. Its meaning is UNCHANGED since Goal 06: it remains the
      search target and is still guarded by
      ``ck_transcript_segments_text_nonempty``.
    * ``original_text`` stores the verbatim source text. It is NULLABLE because
      rows created before the corrective migration never captured an original;
      NULL is an honest "not captured" marker and is NEVER backfilled from the
      normalized text (that would fabricate an "original").
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
    original_text: Mapped[str | None] = mapped_column(
        "original_text", Text, nullable=True
    )
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
        # Goal 12: backs the exact-ownership FK from
        # transcript_words(segment_id, transcript_asset_id, video_id). A word
        # can only reference a segment belonging to the SAME asset and video.
        UniqueConstraint(
            "id", "transcript_asset_id", "video_id",
            name=UQ_TRANSCRIPT_SEGMENTS_ID_ASSET_VIDEO,
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


class SourceAudioAsset(Base):
    """An AUTHORIZED source audio asset (Goal 12 design-freeze §3.1 / §4).

    ``storage_uri`` is an OPAQUE, NON-SECRET object reference (a bucket/key or
    internal handle) — never a signed URL, never credentials, never audio bytes.
    ``checksum_sha256`` binds the row to exact bytes (64 lowercase hex).
    ``mime_type`` and ``rights_basis`` record format and authorization
    provenance. ``duration_ms`` / ``sample_rate_hz`` are OPTIONAL (NULL =
    unknown, NEVER guessed). ``timeline_offset_ms`` maps the audio timeline to
    the video timeline: ``video_ms = audio_ms + offset`` (may be negative).

    Ownership integrity: ``(content_right_id, video_id)`` is a composite FK to
    ``content_rights(id, video_id)``. ``UNIQUE(id, video_id)`` backs the run's
    §4 video-level FK; ``UNIQUE(id, content_right_id, video_id)`` backs the
    exact-authorization-lineage FK from ``transcript_alignment_runs``.
    """

    __tablename__ = "source_audio_assets"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey(
            f"{SCHEMA}.videos.id", name=FK_SOURCE_AUDIO_ASSETS_VIDEO,
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    content_right_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sample_rate_hz: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rights_basis: Mapped[str] = mapped_column(Text, nullable=False)
    timeline_offset_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        # §4: backs transcript_alignment_runs'(source_audio_asset_id, video_id)
        # composite FK. Required by PostgreSQL even though `id` is already PK.
        UniqueConstraint(
            "id", "video_id", name=UQ_SOURCE_AUDIO_ASSETS_ID_VIDEO_ID
        ),
        # Full implementation task §6: backs the exact-authorization-lineage FK
        # (source_audio_asset_id, content_right_id, video_id).
        UniqueConstraint(
            "id", "content_right_id", "video_id",
            name=UQ_SOURCE_AUDIO_ASSETS_ID_RIGHT_VIDEO,
        ),
        # §4: one asset per (video, exact bytes); prevents duplicate audio.
        UniqueConstraint(
            "video_id", "checksum_sha256",
            name=UQ_SOURCE_AUDIO_ASSETS_VIDEO_CHECKSUM,
        ),
        # §4: the right must belong to this audio's video.
        ForeignKeyConstraint(
            ["content_right_id", "video_id"],
            [
                f"{SCHEMA}.content_rights.id",
                f"{SCHEMA}.content_rights.video_id",
            ],
            name=FK_SOURCE_AUDIO_ASSETS_CONTENT_RIGHT_VIDEO,
        ),
        Index("ix_source_audio_assets_video_id", "video_id"),
        Index("ix_source_audio_assets_content_right_id", "content_right_id"),
        CheckConstraint(
            "duration_ms IS NULL OR duration_ms > 0",
            name="ck_source_audio_assets_duration_positive",
        ),
        CheckConstraint(
            "sample_rate_hz IS NULL OR sample_rate_hz > 0",
            name="ck_source_audio_assets_sample_rate_positive",
        ),
        CheckConstraint(
            "checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_source_audio_assets_checksum_sha256",
        ),
        CheckConstraint(
            "length(btrim(storage_uri)) > 0",
            name="ck_source_audio_assets_storage_uri_nonblank",
        ),
        CheckConstraint(
            "length(btrim(mime_type)) > 0",
            name="ck_source_audio_assets_mime_type_nonblank",
        ),
        CheckConstraint(
            "length(btrim(rights_basis)) > 0",
            name="ck_source_audio_assets_rights_basis_nonblank",
        ),
    )


class TranscriptAlignmentRun(Base):
    """One word-alignment ATTEMPT (Goal 12 design-freeze §3.2 / §4).

    Every attempt records WHO tried (``provider`` / ``provider_version`` /
    ``align_model_name``), WHEN (``started_at`` / ``finished_at``), WHAT
    (``segment_count`` / ``aligned_segment_count`` / ``skipped_segment_count`` /
    ``word_count`` / ``audio_duration_ms``), and a closed, credential-free
    ``status`` + ``failure_reason``. Count/timing columns are NULLABLE (NULL =
    not yet known / not attempted) and are never guessed.

    Ownership integrity — BOTH layers are present, never one instead of the
    other:

    * §4 video-level FKs: ``(content_right_id, video_id)`` -> content_rights,
      ``(transcript_asset_id, video_id)`` -> transcript_assets,
      ``(source_audio_asset_id, video_id)`` -> source_audio_assets, plus the
      single-column ``video_id`` -> videos cascade.
    * §6 exact-authorization-lineage FKs:
      ``(transcript_asset_id, content_right_id, video_id)`` -> transcript_assets
      and ``(source_audio_asset_id, content_right_id, video_id)`` ->
      source_audio_assets. Because both share ONE ``content_right_id``, a run
      can only combine an asset and audio authorized by the SAME right / video.

    A partial UNIQUE index permits at most ONE current successful run
    (``completed`` / ``partial``) per ``(transcript_asset_id,
    source_audio_asset_id)``; superseded / failed runs are excluded by WHERE.
    """

    __tablename__ = "transcript_alignment_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey(
            f"{SCHEMA}.videos.id", name=FK_ALIGNMENT_RUNS_VIDEO,
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    content_right_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    transcript_asset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    source_audio_asset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(
        Text, nullable=False, default="whisperx",
        server_default=text("'whisperx'"),
    )
    provider_version: Mapped[str] = mapped_column(Text, nullable=False)
    align_model_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_duration_ms: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    segment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    aligned_segment_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    skipped_segment_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
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
        # §4: backs transcript_words'(alignment_run_id, video_id) FK.
        UniqueConstraint(
            "id", "video_id", name=UQ_ALIGNMENT_RUNS_ID_VIDEO_ID
        ),
        # §6: backs transcript_words'(alignment_run_id, transcript_asset_id,
        # video_id) exact-lineage FK.
        UniqueConstraint(
            "id", "transcript_asset_id", "video_id",
            name=UQ_ALIGNMENT_RUNS_ID_ASSET_VIDEO,
        ),
        # §4: one attempt number per (asset, audio, provider, version).
        UniqueConstraint(
            "transcript_asset_id", "source_audio_asset_id", "provider",
            "provider_version", "attempt_number",
            name=UQ_ALIGNMENT_RUNS_ATTEMPT,
        ),
        # §4 video-level FKs.
        ForeignKeyConstraint(
            ["content_right_id", "video_id"],
            [
                f"{SCHEMA}.content_rights.id",
                f"{SCHEMA}.content_rights.video_id",
            ],
            name=FK_ALIGNMENT_RUNS_RIGHT_VIDEO,
        ),
        ForeignKeyConstraint(
            ["transcript_asset_id", "video_id"],
            [
                f"{SCHEMA}.transcript_assets.id",
                f"{SCHEMA}.transcript_assets.video_id",
            ],
            name=FK_ALIGNMENT_RUNS_ASSET_VIDEO,
        ),
        ForeignKeyConstraint(
            ["source_audio_asset_id", "video_id"],
            [
                f"{SCHEMA}.source_audio_assets.id",
                f"{SCHEMA}.source_audio_assets.video_id",
            ],
            name=FK_ALIGNMENT_RUNS_AUDIO_VIDEO,
        ),
        # §6 exact-authorization-lineage FKs (full implementation task §6).
        ForeignKeyConstraint(
            ["transcript_asset_id", "content_right_id", "video_id"],
            [
                f"{SCHEMA}.transcript_assets.id",
                f"{SCHEMA}.transcript_assets.content_right_id",
                f"{SCHEMA}.transcript_assets.video_id",
            ],
            name=FK_ALIGNMENT_RUNS_ASSET_RIGHT_VIDEO,
        ),
        ForeignKeyConstraint(
            ["source_audio_asset_id", "content_right_id", "video_id"],
            [
                f"{SCHEMA}.source_audio_assets.id",
                f"{SCHEMA}.source_audio_assets.content_right_id",
                f"{SCHEMA}.source_audio_assets.video_id",
            ],
            name=FK_ALIGNMENT_RUNS_AUDIO_RIGHT_VIDEO,
        ),
        # §4: at most ONE current successful run per (asset, audio) pair.
        Index(
            IX_ALIGNMENT_RUNS_ONE_CURRENT_SUCCESS,
            "transcript_asset_id", "source_audio_asset_id",
            unique=True,
            postgresql_where=text("status IN ('completed', 'partial')"),
        ),
        # §4 query indexes.
        Index(
            "ix_alignment_runs_asset_created",
            "transcript_asset_id", "created_at",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index("ix_alignment_runs_status", "status"),
        Index("ix_transcript_alignment_runs_video_id", "video_id"),
        Index("ix_transcript_alignment_runs_audio_id", "source_audio_asset_id"),
        CheckConstraint(
            _ALIGNMENT_STATUS_CHECK_SQL, name="ck_alignment_runs_status"
        ),
        CheckConstraint(
            "attempt_number >= 1", name="ck_alignment_runs_attempt_number"
        ),
        CheckConstraint(
            "(segment_count IS NULL OR segment_count >= 0) AND "
            "(aligned_segment_count IS NULL OR aligned_segment_count >= 0) AND "
            "(skipped_segment_count IS NULL OR skipped_segment_count >= 0) AND "
            "(word_count IS NULL OR word_count >= 0) AND "
            "(audio_duration_ms IS NULL OR audio_duration_ms >= 0)",
            name="ck_alignment_runs_counts_nonnegative",
        ),
        CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR "
            "finished_at >= started_at",
            name="ck_alignment_runs_finished_at",
        ),
        CheckConstraint(
            "length(btrim(provider_version)) > 0",
            name="ck_alignment_runs_provider_version_nonblank",
        ),
    )


class TranscriptWord(Base):
    """One VALIDATED, video-timeline aligned word (Goal 12 design-freeze §3.3/§4).

    Timestamps are INTEGER MILLISECONDS (BIGINT) in the VIDEO timeline (audio ms
    + ``timeline_offset_ms``); ``start_ms`` / ``end_ms`` are NEVER interpolated,
    backfilled, clamped, or derived from character offsets. ``char_start`` /
    ``char_end`` are code-point offsets into the segment's verbatim
    ``original_text``. ``word_index`` is the 0-based ordinal in that segment's
    original token stream (spec §7 step 8: ``word_index = ordinal``). ``word``
    is the VERBATIM token slice ``original_text[char_start:char_end]``, never
    the provider-cleaned / reconstructed text.

    Ownership integrity (§4 + §6):
    * ``(segment_id, transcript_asset_id, video_id)`` -> transcript_segments,
    * ``(transcript_asset_id, video_id)`` -> transcript_assets,
    * ``(alignment_run_id, video_id)`` -> transcript_alignment_runs,
    * ``(alignment_run_id, transcript_asset_id, video_id)`` -> alignment_runs,
    * ``video_id`` -> videos (cascade).

    All composite FKs share ``transcript_asset_id`` / ``video_id``, so a word's
    segment and its run must belong to the SAME asset and video.
    """

    __tablename__ = "transcript_words"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey(
            f"{SCHEMA}.videos.id", name=FK_TRANSCRIPT_WORDS_VIDEO,
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    transcript_asset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    segment_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    alignment_run_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    word_index: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    word: Mapped[str] = mapped_column(Text, nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        # §4: one word per index per segment. A successful replacement deletes
        # the superseded run's word set BEFORE inserting its own.
        UniqueConstraint(
            "segment_id", "word_index",
            name=UQ_TRANSCRIPT_WORDS_SEGMENT_WORD_INDEX,
        ),
        # §4 asset-level FK.
        ForeignKeyConstraint(
            ["transcript_asset_id", "video_id"],
            [
                f"{SCHEMA}.transcript_assets.id",
                f"{SCHEMA}.transcript_assets.video_id",
            ],
            name=FK_TRANSCRIPT_WORDS_ASSET_VIDEO,
        ),
        # §4 exact segment ownership (FK target:
        # uq_transcript_segments_id_asset_video).
        ForeignKeyConstraint(
            ["segment_id", "transcript_asset_id", "video_id"],
            [
                f"{SCHEMA}.transcript_segments.id",
                f"{SCHEMA}.transcript_segments.transcript_asset_id",
                f"{SCHEMA}.transcript_segments.video_id",
            ],
            name=FK_TRANSCRIPT_WORDS_SEGMENT_ASSET_VIDEO,
            ondelete="CASCADE",
        ),
        # §4 run-level FK (target: uq_alignment_runs_id_video_id).
        ForeignKeyConstraint(
            ["alignment_run_id", "video_id"],
            [
                f"{SCHEMA}.transcript_alignment_runs.id",
                f"{SCHEMA}.transcript_alignment_runs.video_id",
            ],
            name=FK_TRANSCRIPT_WORDS_RUN_VIDEO,
        ),
        # §6 exact run ownership (target: uq_transcript_alignment_runs_id_asset_video).
        ForeignKeyConstraint(
            ["alignment_run_id", "transcript_asset_id", "video_id"],
            [
                f"{SCHEMA}.transcript_alignment_runs.id",
                f"{SCHEMA}.transcript_alignment_runs.transcript_asset_id",
                f"{SCHEMA}.transcript_alignment_runs.video_id",
            ],
            name=FK_TRANSCRIPT_WORDS_RUN_ASSET_VIDEO,
        ),
        # §4 query indexes.
        Index("ix_transcript_words_segment_time", "segment_id", "start_ms"),
        Index("ix_transcript_words_alignment_run_id", "alignment_run_id"),
        Index("ix_transcript_words_video_id", "video_id"),
        CheckConstraint(
            "start_ms >= 0", name="ck_transcript_words_start_nonnegative"
        ),
        CheckConstraint(
            "end_ms > start_ms", name="ck_transcript_words_end_after_start"
        ),
        CheckConstraint(
            "word_index >= 0", name="ck_transcript_words_word_index_nonnegative"
        ),
        CheckConstraint(
            "char_start >= 0 AND char_end > char_start",
            name="ck_transcript_words_char_span_valid",
        ),
        CheckConstraint(
            "length(btrim(word)) > 0", name="ck_transcript_words_word_nonempty"
        ),
    )
