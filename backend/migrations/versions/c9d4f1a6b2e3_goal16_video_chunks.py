"""add Goal 16 video_chunks table with pgvector(1024) embedding column

Revision ID: c9d4f1a6b2e3
Revises: a7c2e9f4b1d8
Create Date: 2026-11-09 00:00:00.000000

Goal 16 — Vector database insert & persistence. Adds ONE additive table,
``voxsearch.video_chunks``, that stores a Goal 13 chunk together with its
Goal 15 1024-dimensional dense embedding as a pgvector ``vector(1024)``
column. This is DDL-only: no INSERT/UPDATE/DELETE, no backfill, no alteration
or drop of any pre-existing object.

pgvector extension
------------------
The migration deliberately does NOT emit ``CREATE EXTENSION vector``. pgvector
0.8.2 was verified pre-installed in Goal 06 and ``app/db/health.py`` probes it.
If the extension is absent, ``CREATE TABLE ... vector(1024)`` fails — that is
INTENTIONAL (fail-closed), not silently worked around. The ``VectorType`` used
here is a dependency-free SQLAlchemy UserDefinedType; no ``pgvector`` Python
package and no numpy is required by the core backend.

Safety properties
-----------------
* DDL-only and additive. ``upgrade()`` creates exactly one table; every
  constraint (UNIQUE / composite FK / CHECK) is declared inline on that table.
* All objects live in the dedicated ``voxsearch`` schema; ``public`` and all
  Supabase-managed objects are untouched.
* ``downgrade()`` drops ONLY the table created by ``upgrade()`` and is NEVER
  executed against the live database.
* NO ANN / HNSW / IVFFlat index (cosine similarity search is Goal 17).
* NO ``vector_dims`` CHECK (unverifiable offline; dimension is fixed by the
  UDT's ``get_col_spec`` + application-level validation in the service).

NOTE: this migration MUST NOT be applied to production Supabase without a
separate, explicit approval. Its predecessors (b5d2e8a14c7f and
a7c2e9f4b1d8) are likewise still pending. Offline SQL rendering and mocked
unit tests are the ONLY verification performed in this goal; there is NO local
PostgreSQL + pgvector instance available, so real-DB insertion/CAST behaviour
is explicitly NOT VERIFIED here.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.db.types import VectorType

# revision identifiers, used by Alembic.
revision: str = "c9d4f1a6b2e3"
down_revision: Union[str, None] = "a7c2e9f4b1d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "voxsearch"

# Constraint names — kept byte-for-byte in sync with app/db/models.py (asserted
# equal by the Goal 16 tests). The UNIQUE is the idempotent-upsert key.
UQ_VIDEO_CHUNKS_ASSET_CHUNK_MODEL = "uq_video_chunks_asset_chunk_model"
FK_VIDEO_CHUNKS_VIDEO = "fk_video_chunks_video"
FK_VIDEO_CHUNKS_ASSET_VIDEO = "fk_video_chunks_asset_video"
FK_VIDEO_CHUNKS_RIGHT_VIDEO = "fk_video_chunks_right_video"


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
    op.create_table(
        "video_chunks",
        _uuid_pk(),
        sa.Column(
            "video_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.videos.id", name=FK_VIDEO_CHUNKS_VIDEO,
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("transcript_asset_id", sa.Uuid(), nullable=False),
        sa.Column("content_right_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_ms", sa.BigInteger(), nullable=False),
        sa.Column("end_ms", sa.BigInteger(), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=False),
        sa.Column("timestamp_precision", sa.String(32), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("boundary_reason", sa.String(64), nullable=False),
        sa.Column("source_segment_refs", sa.JSON(), nullable=False),
        sa.Column("overlap_sentence_indices", sa.JSON(), nullable=False),
        sa.Column("diagnostics", sa.JSON(), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        # Dependency-free UserDefinedType -> renders "vector(1024)".
        sa.Column("embedding", VectorType(1024), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "transcript_asset_id", "chunk_index", "model_id",
            name=UQ_VIDEO_CHUNKS_ASSET_CHUNK_MODEL,
        ),
        sa.ForeignKeyConstraint(
            ["transcript_asset_id", "video_id"],
            ["voxsearch.transcript_assets.id", "voxsearch.transcript_assets.video_id"],
            name=FK_VIDEO_CHUNKS_ASSET_VIDEO,
        ),
        sa.ForeignKeyConstraint(
            ["content_right_id", "video_id"],
            ["voxsearch.content_rights.id", "voxsearch.content_rights.video_id"],
            name=FK_VIDEO_CHUNKS_RIGHT_VIDEO,
        ),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name="ck_video_chunks_chunk_index_nonnegative",
        ),
        sa.CheckConstraint(
            "start_ms >= 0 AND end_ms > start_ms AND duration_ms >= 0",
            name="ck_video_chunks_time_window",
        ),
        sa.CheckConstraint(
            "length(btrim(model_id)) > 0",
            name="ck_video_chunks_model_id_nonblank",
        ),
        sa.CheckConstraint(
            'length(btrim("text")) > 0',
            name="ck_video_chunks_text_nonblank",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    """Drop ONLY the Goal 16 table. NEVER run against the live DB."""
    op.drop_table("video_chunks", schema=SCHEMA)
