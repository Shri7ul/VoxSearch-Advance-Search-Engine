"""add Goal 12 word-alignment tables and ownership constraints

Revision ID: a7c2e9f4b1d8
Revises: b5d2e8a14c7f
Create Date: 2026-11-07 00:00:00.000000

Goal 12 — Timestamp preservation. Adds the three new tables needed for
word-level alignment, the additive UNIQUE constraints that back the composite
ownership FKs, and every CHECK / index / partial-index required by the APPROVED
Goal 12 design-freeze specification (§3, §4) plus the full implementation
task's §6 exact-authorization-lineage FKs and §7 ``timeline_offset_ms``.

Existing seconds-based transcript columns, the ``original_text`` column, and
every pre-existing row are LEFT UNTOUCHED.

Safety properties
-----------------
* DDL-only and additive. No INSERT / UPDATE / DELETE, no backfill, no table
  rename, no column alteration, no DROP of any pre-existing object in
  ``upgrade()``.
* Does NOT alter ``start_seconds`` / ``end_seconds`` (still float SECONDS) or
  the normalized ``"text"`` column. New word timestamps use INTEGER
  MILLISECONDS (BIGINT) in NEW tables only.
* The two UNIQUE constraints added to ``transcript_assets`` /
  ``transcript_segments`` are pure ADD CONSTRAINT (no data rewrite) and are the
  referenced tuples required by PostgreSQL for the new composite FKs.
* All objects live in the dedicated ``voxsearch`` schema; ``public`` and all
  Supabase-managed objects are untouched.
* ``downgrade()`` reverses ONLY what ``upgrade()`` created (child-first) and
  is NEVER executed against the live database.

Composite ownership integrity (exact authorization lineage)
-----------------------------------------------------------
PostgreSQL requires a matching UNIQUE / PK on the referenced column tuple of
every composite FK. The new tables and two additive uniques therefore declare:

* content_rights(id, video_id)                         [existing, Goal 07]
* transcript_assets(id, video_id)                      [existing, Goal 07]
* transcript_assets(id, content_right_id, video_id)   [NEW additive unique]
* transcript_segments(id, transcript_asset_id, video_id) [NEW additive unique]
* source_audio_assets(id, video_id)                    [NEW]
* source_audio_assets(id, content_right_id, video_id) [NEW, exact lineage]
* transcript_alignment_runs(id, video_id)             [NEW]
* transcript_alignment_runs(id, transcript_asset_id, video_id) [NEW, exact lineage]

A partial UNIQUE index enforces at most ONE current successful run
(status IN ('completed','partial')) per (transcript_asset_id,
source_audio_asset_id).

NOTE: this migration MUST NOT be applied to production Supabase without a
separate, explicit approval (the predecessor b5d2e8a14c7f is likewise still
pending). Offline SQL review and isolated-disposable-database tests are the only
verification performed in this goal.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a7c2e9f4b1d8"
down_revision: Union[str, None] = "b5d2e8a14c7f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "voxsearch"

# Constraint / index names — kept byte-for-byte in sync with app/db/models.py
# (asserted equal by the Goal 12 tests). Names follow the approved design-freeze
# §4; the two ``*_id_right_video`` uniques / FKs implement §6 exact lineage.
UQ_TRANSCRIPT_ASSETS_ID_RIGHT_VIDEO = "uq_transcript_assets_id_right_video"
UQ_TRANSCRIPT_SEGMENTS_ID_ASSET_VIDEO = "uq_transcript_segments_id_asset_video"

UQ_SOURCE_AUDIO_ASSETS_ID_VIDEO_ID = "uq_source_audio_assets_id_video_id"
UQ_SOURCE_AUDIO_ASSETS_ID_RIGHT_VIDEO = "uq_source_audio_assets_id_right_video"
UQ_SOURCE_AUDIO_ASSETS_VIDEO_CHECKSUM = "uq_source_audio_assets_video_checksum"
FK_SOURCE_AUDIO_ASSETS_VIDEO = "fk_source_audio_assets_video"
FK_SOURCE_AUDIO_ASSETS_CONTENT_RIGHT_VIDEO = "fk_source_audio_assets_right_video"

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

UQ_TRANSCRIPT_WORDS_SEGMENT_WORD_INDEX = "uq_transcript_words_segment_word_index"
FK_TRANSCRIPT_WORDS_VIDEO = "fk_transcript_words_video"
FK_TRANSCRIPT_WORDS_SEGMENT_ASSET_VIDEO = "fk_transcript_words_segment_asset_video"
FK_TRANSCRIPT_WORDS_ASSET_VIDEO = "fk_transcript_words_asset_video"
FK_TRANSCRIPT_WORDS_RUN_VIDEO = "fk_transcript_words_run_video"
FK_TRANSCRIPT_WORDS_RUN_ASSET_VIDEO = "fk_transcript_words_run_asset_video"

# Closed status set (in sync with app.transcripts.word_contract.ALIGNMENT_RUN_STATUSES
# and app.db.models.ALIGNMENT_RUN_STATUS_VALUES). ``rejected_rights`` is a
# RESERVED label: the final implementation task forbids persisting ANY run for
# an initial rights-gate rejection, so this goal never emits it.
_ALIGNMENT_STATUS_CHECK_SQL = (
    "status IN ('pending', 'running', 'completed', 'partial', 'failed', "
    "'unsupported_language', 'rejected_rights', 'superseded')"
)


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
    # 1. Additive UNIQUE constraints on EXISTING tables. These are the FK
    #    targets for the new tables' composite ownership FKs. Declared FIRST so
    #    the CREATE TABLE ... REFERENCES statements below resolve cleanly.
    #    NOTE: ADD CONSTRAINT UNIQUE on a populated table builds an index and
    #    takes a brief lock; explicit production approval is required (§C4).
    op.create_unique_constraint(
        UQ_TRANSCRIPT_ASSETS_ID_RIGHT_VIDEO,
        "transcript_assets", ["id", "content_right_id", "video_id"],
        schema=SCHEMA,
    )
    op.create_unique_constraint(
        UQ_TRANSCRIPT_SEGMENTS_ID_ASSET_VIDEO,
        "transcript_segments", ["id", "transcript_asset_id", "video_id"],
        schema=SCHEMA,
    )

    # 2. source_audio_assets — AUTHORIZED audio clip (design-freeze §3.1 / §4).
    #    Opaque storage_uri, SHA-256 checksum, MIME type, OPTIONAL duration /
    #    sample rate, rights provenance, and the §7 audio->video timeline offset.
    op.create_table(
        "source_audio_assets",
        _uuid_pk(),
        sa.Column(
            "video_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.videos.id", name=FK_SOURCE_AUDIO_ASSETS_VIDEO,
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("content_right_id", sa.Uuid(), nullable=False),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("checksum_sha256", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("sample_rate_hz", sa.Integer(), nullable=True),
        sa.Column("rights_basis", sa.Text(), nullable=False),
        sa.Column("timeline_offset_ms", sa.BigInteger(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "id", "video_id", name=UQ_SOURCE_AUDIO_ASSETS_ID_VIDEO_ID,
        ),
        sa.UniqueConstraint(
            "id", "content_right_id", "video_id",
            name=UQ_SOURCE_AUDIO_ASSETS_ID_RIGHT_VIDEO,
        ),
        sa.UniqueConstraint(
            "video_id", "checksum_sha256",
            name=UQ_SOURCE_AUDIO_ASSETS_VIDEO_CHECKSUM,
        ),
        sa.ForeignKeyConstraint(
            ["content_right_id", "video_id"],
            ["voxsearch.content_rights.id", "voxsearch.content_rights.video_id"],
            name=FK_SOURCE_AUDIO_ASSETS_CONTENT_RIGHT_VIDEO,
        ),
        sa.CheckConstraint(
            "duration_ms IS NULL OR duration_ms > 0",
            name="ck_source_audio_assets_duration_positive",
        ),
        sa.CheckConstraint(
            "sample_rate_hz IS NULL OR sample_rate_hz > 0",
            name="ck_source_audio_assets_sample_rate_positive",
        ),
        sa.CheckConstraint(
            "checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_source_audio_assets_checksum_sha256",
        ),
        sa.CheckConstraint(
            "length(btrim(storage_uri)) > 0",
            name="ck_source_audio_assets_storage_uri_nonblank",
        ),
        sa.CheckConstraint(
            "length(btrim(mime_type)) > 0",
            name="ck_source_audio_assets_mime_type_nonblank",
        ),
        sa.CheckConstraint(
            "length(btrim(rights_basis)) > 0",
            name="ck_source_audio_assets_rights_basis_nonblank",
        ),
        schema=SCHEMA,
    )

    # 3. transcript_alignment_runs — one alignment ATTEMPT (§3.2 / §4).
    #    Every attempt records provider/version/model, language, attempt number,
    #    status, segment/word counts, audio duration, closed failure reason and
    #    started/finished timestamps. Both §4 video-level and §6 exact-lineage
    #    composite FKs are present.
    op.create_table(
        "transcript_alignment_runs",
        _uuid_pk(),
        sa.Column(
            "video_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.videos.id", name=FK_ALIGNMENT_RUNS_VIDEO,
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("content_right_id", sa.Uuid(), nullable=False),
        sa.Column("transcript_asset_id", sa.Uuid(), nullable=False),
        sa.Column("source_audio_asset_id", sa.Uuid(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column(
            "provider", sa.Text(), nullable=False,
            server_default=sa.text("'whisperx'"),
        ),
        sa.Column("provider_version", sa.Text(), nullable=False),
        sa.Column("align_model_name", sa.Text(), nullable=True),
        sa.Column("audio_duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("segment_count", sa.Integer(), nullable=True),
        sa.Column("aligned_segment_count", sa.Integer(), nullable=True),
        sa.Column("skipped_segment_count", sa.Integer(), nullable=True),
        sa.Column("word_count", sa.Integer(), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint(
            "id", "video_id", name=UQ_ALIGNMENT_RUNS_ID_VIDEO_ID,
        ),
        sa.UniqueConstraint(
            "id", "transcript_asset_id", "video_id",
            name=UQ_ALIGNMENT_RUNS_ID_ASSET_VIDEO,
        ),
        sa.UniqueConstraint(
            "transcript_asset_id", "source_audio_asset_id", "provider",
            "provider_version", "attempt_number",
            name=UQ_ALIGNMENT_RUNS_ATTEMPT,
        ),
        # §4 video-level FKs.
        sa.ForeignKeyConstraint(
            ["content_right_id", "video_id"],
            ["voxsearch.content_rights.id", "voxsearch.content_rights.video_id"],
            name=FK_ALIGNMENT_RUNS_RIGHT_VIDEO,
        ),
        sa.ForeignKeyConstraint(
            ["transcript_asset_id", "video_id"],
            ["voxsearch.transcript_assets.id", "voxsearch.transcript_assets.video_id"],
            name=FK_ALIGNMENT_RUNS_ASSET_VIDEO,
        ),
        sa.ForeignKeyConstraint(
            ["source_audio_asset_id", "video_id"],
            [
                "voxsearch.source_audio_assets.id",
                "voxsearch.source_audio_assets.video_id",
            ],
            name=FK_ALIGNMENT_RUNS_AUDIO_VIDEO,
        ),
        # §6 exact-authorization-lineage FKs: both composite FKs share ONE
        # content_right_id, so the run can only combine assets under the SAME
        # right / SAME video.
        sa.ForeignKeyConstraint(
            ["transcript_asset_id", "content_right_id", "video_id"],
            [
                "voxsearch.transcript_assets.id",
                "voxsearch.transcript_assets.content_right_id",
                "voxsearch.transcript_assets.video_id",
            ],
            name=FK_ALIGNMENT_RUNS_ASSET_RIGHT_VIDEO,
        ),
        sa.ForeignKeyConstraint(
            ["source_audio_asset_id", "content_right_id", "video_id"],
            [
                "voxsearch.source_audio_assets.id",
                "voxsearch.source_audio_assets.content_right_id",
                "voxsearch.source_audio_assets.video_id",
            ],
            name=FK_ALIGNMENT_RUNS_AUDIO_RIGHT_VIDEO,
        ),
        sa.CheckConstraint(
            _ALIGNMENT_STATUS_CHECK_SQL, name="ck_alignment_runs_status"
        ),
        sa.CheckConstraint(
            "attempt_number >= 1", name="ck_alignment_runs_attempt_number"
        ),
        sa.CheckConstraint(
            "(segment_count IS NULL OR segment_count >= 0) AND "
            "(aligned_segment_count IS NULL OR aligned_segment_count >= 0) AND "
            "(skipped_segment_count IS NULL OR skipped_segment_count >= 0) AND "
            "(word_count IS NULL OR word_count >= 0) AND "
            "(audio_duration_ms IS NULL OR audio_duration_ms >= 0)",
            name="ck_alignment_runs_counts_nonnegative",
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR "
            "finished_at >= started_at",
            name="ck_alignment_runs_finished_at",
        ),
        sa.CheckConstraint(
            "length(btrim(provider_version)) > 0",
            name="ck_alignment_runs_provider_version_nonblank",
        ),
        schema=SCHEMA,
    )

    # 4. transcript_words — validated, VIDEO-timeline integer-ms words (§3.3/§4).
    #    ``word`` stores the VERBATIM original token slice; ``word_index`` is the
    #    0-based ordinal (§7 step 8). Child of segments, runs, assets and videos.
    op.create_table(
        "transcript_words",
        _uuid_pk(),
        sa.Column(
            "video_id", sa.Uuid(),
            sa.ForeignKey(
                "voxsearch.videos.id", name=FK_TRANSCRIPT_WORDS_VIDEO,
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("transcript_asset_id", sa.Uuid(), nullable=False),
        sa.Column("segment_id", sa.Uuid(), nullable=False),
        sa.Column("alignment_run_id", sa.Uuid(), nullable=False),
        sa.Column("word_index", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("word", sa.Text(), nullable=False),
        sa.Column("start_ms", sa.BigInteger(), nullable=False),
        sa.Column("end_ms", sa.BigInteger(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "segment_id", "word_index",
            name=UQ_TRANSCRIPT_WORDS_SEGMENT_WORD_INDEX,
        ),
        sa.ForeignKeyConstraint(
            ["transcript_asset_id", "video_id"],
            ["voxsearch.transcript_assets.id", "voxsearch.transcript_assets.video_id"],
            name=FK_TRANSCRIPT_WORDS_ASSET_VIDEO,
        ),
        sa.ForeignKeyConstraint(
            ["segment_id", "transcript_asset_id", "video_id"],
            [
                "voxsearch.transcript_segments.id",
                "voxsearch.transcript_segments.transcript_asset_id",
                "voxsearch.transcript_segments.video_id",
            ],
            name=FK_TRANSCRIPT_WORDS_SEGMENT_ASSET_VIDEO,
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["alignment_run_id", "video_id"],
            [
                "voxsearch.transcript_alignment_runs.id",
                "voxsearch.transcript_alignment_runs.video_id",
            ],
            name=FK_TRANSCRIPT_WORDS_RUN_VIDEO,
        ),
        sa.ForeignKeyConstraint(
            ["alignment_run_id", "transcript_asset_id", "video_id"],
            [
                "voxsearch.transcript_alignment_runs.id",
                "voxsearch.transcript_alignment_runs.transcript_asset_id",
                "voxsearch.transcript_alignment_runs.video_id",
            ],
            name=FK_TRANSCRIPT_WORDS_RUN_ASSET_VIDEO,
        ),
        sa.CheckConstraint(
            "start_ms >= 0", name="ck_transcript_words_start_nonnegative"
        ),
        sa.CheckConstraint(
            "end_ms > start_ms", name="ck_transcript_words_end_after_start"
        ),
        sa.CheckConstraint(
            "word_index >= 0", name="ck_transcript_words_word_index_nonnegative"
        ),
        sa.CheckConstraint(
            "char_start >= 0 AND char_end > char_start",
            name="ck_transcript_words_char_span_valid",
        ),
        sa.CheckConstraint(
            "length(btrim(word)) > 0", name="ck_transcript_words_word_nonempty"
        ),
        schema=SCHEMA,
    )

    # 5. Standalone indexes (PostgreSQL does not auto-index FK columns), the
    #    §4 query indexes, and the partial UNIQUE index that caps current
    #    successful runs at one per (asset, audio) pair.
    op.create_index(
        "ix_source_audio_assets_video_id", "source_audio_assets",
        ["video_id"], schema=SCHEMA,
    )
    op.create_index(
        "ix_source_audio_assets_content_right_id", "source_audio_assets",
        ["content_right_id"], schema=SCHEMA,
    )
    op.create_index(
        IX_ALIGNMENT_RUNS_ONE_CURRENT_SUCCESS, "transcript_alignment_runs",
        ["transcript_asset_id", "source_audio_asset_id"],
        unique=True, schema=SCHEMA,
        postgresql_where=sa.text("status IN ('completed', 'partial')"),
    )
    op.create_index(
        "ix_alignment_runs_asset_created", "transcript_alignment_runs",
        ["transcript_asset_id", "created_at"], schema=SCHEMA,
        postgresql_ops={"created_at": "DESC"},
    )
    op.create_index(
        "ix_alignment_runs_status", "transcript_alignment_runs",
        ["status"], schema=SCHEMA,
    )
    op.create_index(
        "ix_transcript_alignment_runs_video_id", "transcript_alignment_runs",
        ["video_id"], schema=SCHEMA,
    )
    op.create_index(
        "ix_transcript_alignment_runs_audio_id", "transcript_alignment_runs",
        ["source_audio_asset_id"], schema=SCHEMA,
    )
    op.create_index(
        "ix_transcript_words_segment_time", "transcript_words",
        ["segment_id", "start_ms"], schema=SCHEMA,
    )
    op.create_index(
        "ix_transcript_words_alignment_run_id", "transcript_words",
        ["alignment_run_id"], schema=SCHEMA,
    )
    op.create_index(
        "ix_transcript_words_video_id", "transcript_words",
        ["video_id"], schema=SCHEMA,
    )


def downgrade() -> None:
    """Reverse ONLY what upgrade() created. NEVER run against the live DB.

    Child tables are dropped first (their own indexes/FKs/constraints are
    dropped with them), then the two additive UNIQUE constraints on the
    pre-existing tables are removed. No existing data, column, table, or
    unrelated constraint is touched.
    """
    op.drop_table("transcript_words", schema=SCHEMA)
    op.drop_table("transcript_alignment_runs", schema=SCHEMA)
    op.drop_table("source_audio_assets", schema=SCHEMA)
    op.drop_constraint(
        UQ_TRANSCRIPT_SEGMENTS_ID_ASSET_VIDEO,
        "transcript_segments", schema=SCHEMA, type_="unique",
    )
    op.drop_constraint(
        UQ_TRANSCRIPT_ASSETS_ID_RIGHT_VIDEO,
        "transcript_assets", schema=SCHEMA, type_="unique",
    )
