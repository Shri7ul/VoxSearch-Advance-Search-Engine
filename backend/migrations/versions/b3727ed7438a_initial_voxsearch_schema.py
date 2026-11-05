"""initial voxsearch schema

Revision ID: b3727ed7438a
Revises:
Create Date: 2026-11-06 03:35:54.249184

Goal 06 — connection-only schema foundation. Creates the dedicated
``voxsearch`` schema and five application tables (channels, videos,
content_rights, transcript_assets, transcript_segments).

Safety properties:
* Every statement is CREATE-only and scoped to the ``voxsearch`` schema.
* ``public`` and all existing Supabase objects are left untouched.
* ``CREATE SCHEMA IF NOT EXISTS`` and plain CREATE TABLE never overwrite or
  drop existing objects; if a conflicting table already exists the CREATE
  fails loudly and nothing is modified.
* No INSERT/UPDATE/DELETE, no vector columns, no sample data.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b3727ed7438a"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "voxsearch"


def _uuid_pk() -> sa.Column:
    return sa.Column(
        "id", sa.Uuid(), primary_key=True, nullable=False,
        server_default=sa.text("gen_random_uuid()"),
    )


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
    ]


def upgrade() -> None:
    # 1. Dedicated application schema. Idempotent and non-destructive; the
    #    existing `public` schema and all Supabase objects are preserved.
    op.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))

    # 2. Five application tables, created parent-first so FKs always reference
    #    an already-created table.
    op.create_table(
        "channels",
        _uuid_pk(),
        sa.Column("youtube_channel_id", sa.String(64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint(
            "youtube_channel_id", name="uq_channels_youtube_channel_id"
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "videos",
        _uuid_pk(),
        sa.Column(
            "channel_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.channels.id", name="fk_videos_channel_id_channels"
            ),
            nullable=False,
        ),
        sa.Column("youtube_video_id", sa.String(64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint(
            "youtube_video_id", name="uq_videos_youtube_video_id"
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "content_rights",
        _uuid_pk(),
        sa.Column(
            "video_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.videos.id", name="fk_content_rights_video_id_videos"
            ),
            nullable=False,
        ),
        sa.Column("authorization_type", sa.String(64), nullable=False),
        sa.Column("authorization_reference", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        schema=SCHEMA,
    )

    op.create_table(
        "transcript_assets",
        _uuid_pk(),
        sa.Column(
            "video_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.videos.id",
                name="fk_transcript_assets_video_id_videos",
            ),
            nullable=False,
        ),
        sa.Column(
            "content_right_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.content_rights.id",
                name="fk_transcript_assets_content_right_id_content_rights",
            ),
            nullable=False,
        ),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        *_timestamps(),
        schema=SCHEMA,
    )

    op.create_table(
        "transcript_segments",
        _uuid_pk(),
        sa.Column(
            "transcript_asset_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.transcript_assets.id",
                name="fk_transcript_segments_asset_id_transcript_assets",
            ),
            nullable=False,
        ),
        sa.Column(
            "video_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.videos.id",
                name="fk_transcript_segments_video_id_videos",
            ),
            nullable=False,
        ),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("language", sa.String(16), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint(
            "transcript_asset_id", "start_seconds",
            name="uq_transcript_segments_asset_start",
        ),
        sa.CheckConstraint(
            "start_seconds >= 0", name="ck_transcript_segments_start_nonnegative"
        ),
        sa.CheckConstraint(
            "end_seconds > start_seconds",
            name="ck_transcript_segments_end_after_start",
        ),
        sa.CheckConstraint(
            'length(btrim("text")) > 0',
            name="ck_transcript_segments_text_nonempty",
        ),
        schema=SCHEMA,
    )

    # 3. Indexes for frequently queried foreign keys / identifiers. PostgreSQL
    #    does not auto-index FK columns; the two external ids and the segment
    #    (transcript_asset_id, start_seconds) pair are already covered by their
    #    unique constraints, so no redundant indexes are created.
    op.create_index(
        "ix_videos_channel_id", "videos", ["channel_id"], schema=SCHEMA
    )
    op.create_index(
        "ix_content_rights_video_id", "content_rights", ["video_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_transcript_assets_video_id", "transcript_assets", ["video_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_transcript_assets_content_right_id", "transcript_assets",
        ["content_right_id"], schema=SCHEMA,
    )
    op.create_index(
        "ix_transcript_segments_video_id", "transcript_segments", ["video_id"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    """Remove ONLY the objects created by upgrade(). Never run against live DB.

    Tables are dropped child-first (reverse FK dependency order). PostgreSQL
    automatically drops each table's own constraints and indexes with it. The
    ``voxsearch`` schema is intentionally NOT dropped: it also hosts Alembic's
    ``alembic_version`` bookkeeping table, and ``DROP SCHEMA`` (without CASCADE)
    would fail while (with CASCADE) could destroy unrelated objects.
    """
    op.drop_table("transcript_segments", schema=SCHEMA)
    op.drop_table("transcript_assets", schema=SCHEMA)
    op.drop_table("content_rights", schema=SCHEMA)
    op.drop_table("videos", schema=SCHEMA)
    op.drop_table("channels", schema=SCHEMA)
