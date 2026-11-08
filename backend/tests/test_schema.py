"""Unit tests for the Goal 06 schema / model / migration foundation.

These tests inspect SQLAlchemy metadata and Alembic's script directory only.
They NEVER open a network / Supabase connection. Live migration verification is
performed separately (see the Goal 06 validation report).
"""
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, DateTime, Text, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.db.models import (
    SCHEMA,
    Base,
    Channel,
    ContentRight,
    SourceAudioAsset,
    TranscriptAlignmentRun,
    TranscriptAsset,
    TranscriptSegment,
    TranscriptWord,
    Video,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = BACKEND_DIR / "migrations"
EXPECTED_TABLES = {
    "channels",
    "videos",
    "content_rights",
    "transcript_assets",
    "transcript_segments",
    # Goal 12: word-alignment tables (additive).
    "source_audio_assets",
    "transcript_alignment_runs",
    "transcript_words",
}
REVISION_ID = "b3727ed7438a"  # Goal 06 initial schema
GOAL07_REVISION = "fe7b2b8e5323"  # Goal 07 ownership integrity
GOAL09_CORRECTIVE_REVISION = "b5d2e8a14c7f"  # Goal 09 corrective: original_text
GOAL12_REVISION = "a7c2e9f4b1d8"  # Goal 12: word-alignment tables
HEAD_REVISION = GOAL12_REVISION


def tbl(name: str):
    return Base.metadata.tables[f"{SCHEMA}.{name}"]


def compiled_ddl(name: str) -> str:
    return str(CreateTable(tbl(name)).compile(dialect=postgresql.dialect()))


# --- 1. Model metadata / schema names ---------------------------------------

def test_schema_name() -> None:
    assert SCHEMA == "voxsearch"


def test_all_tables_present() -> None:
    assert set(Base.metadata.tables) == {
        f"{SCHEMA}.{n}" for n in EXPECTED_TABLES
    }


def test_every_table_in_voxsearch_schema() -> None:
    for t in Base.metadata.tables.values():
        assert t.schema == "voxsearch"


def test_declarative_classes_registered() -> None:
    for cls, name in (
        (Channel, "channels"),
        (Video, "videos"),
        (ContentRight, "content_rights"),
        (TranscriptAsset, "transcript_assets"),
        (TranscriptSegment, "transcript_segments"),
        (SourceAudioAsset, "source_audio_assets"),
        (TranscriptAlignmentRun, "transcript_alignment_runs"),
        (TranscriptWord, "transcript_words"),
    ):
        assert cls.__table__ is tbl(name)


# --- 2. UUID primary keys ---------------------------------------------------

@pytest.mark.parametrize("name", sorted(EXPECTED_TABLES))
def test_uuid_primary_key(name: str) -> None:
    t = tbl(name)
    pk = list(t.primary_key.columns)
    assert len(pk) == 1
    assert pk[0].name == "id"
    assert pk[0].type.python_type is uuid.UUID
    assert pk[0].nullable is False


@pytest.mark.parametrize("name", sorted(EXPECTED_TABLES))
def test_uuid_compiles_to_native_postgres_uuid(name: str) -> None:
    assert "id UUID" in compiled_ddl(name)


# --- 3. Foreign key definitions --------------------------------------------

# (local column, fully-qualified target column) edges. A set is used because
# Goal 07 composite FKs make ``video_id`` a member of two FK constraints.
EXPECTED_FK_EDGES = {
    "videos": {("channel_id", "voxsearch.channels.id")},
    "content_rights": {("video_id", "voxsearch.videos.id")},
    "transcript_assets": {
        ("video_id", "voxsearch.videos.id"),
        ("content_right_id", "voxsearch.content_rights.id"),
        ("video_id", "voxsearch.content_rights.video_id"),
    },
    "transcript_segments": {
        ("transcript_asset_id", "voxsearch.transcript_assets.id"),
        ("video_id", "voxsearch.transcript_assets.video_id"),
        ("video_id", "voxsearch.videos.id"),
    },
    # Goal 12: exact authorization / ownership lineage composite FKs.
    "source_audio_assets": {
        ("video_id", "voxsearch.videos.id"),
        ("content_right_id", "voxsearch.content_rights.id"),
        ("video_id", "voxsearch.content_rights.video_id"),
    },
    "transcript_alignment_runs": {
        ("video_id", "voxsearch.videos.id"),
        # §4 right-video FK (content_right_id, video_id) -> content_rights.
        ("content_right_id", "voxsearch.content_rights.id"),
        ("video_id", "voxsearch.content_rights.video_id"),
        # §4 video-level + §6 exact-authorization-lineage asset/audio FKs.
        ("transcript_asset_id", "voxsearch.transcript_assets.id"),
        ("content_right_id", "voxsearch.transcript_assets.content_right_id"),
        ("video_id", "voxsearch.transcript_assets.video_id"),
        ("source_audio_asset_id", "voxsearch.source_audio_assets.id"),
        ("content_right_id", "voxsearch.source_audio_assets.content_right_id"),
        ("video_id", "voxsearch.source_audio_assets.video_id"),
    },
    "transcript_words": {
        ("video_id", "voxsearch.videos.id"),
        # §4 asset-level FK.
        ("transcript_asset_id", "voxsearch.transcript_assets.id"),
        ("video_id", "voxsearch.transcript_assets.video_id"),
        # §4 exact segment ownership + §6 exact run ownership.
        ("segment_id", "voxsearch.transcript_segments.id"),
        ("transcript_asset_id", "voxsearch.transcript_segments.transcript_asset_id"),
        ("video_id", "voxsearch.transcript_segments.video_id"),
        ("alignment_run_id", "voxsearch.transcript_alignment_runs.id"),
        (
            "transcript_asset_id",
            "voxsearch.transcript_alignment_runs.transcript_asset_id",
        ),
        ("video_id", "voxsearch.transcript_alignment_runs.video_id"),
    },
}


@pytest.mark.parametrize("name", sorted(EXPECTED_FK_EDGES))
def test_foreign_keys_resolve_to_expected_targets(name: str) -> None:
    t = tbl(name)
    resolved = {
        (fk.parent.name, f"{fk.column.table.fullname}.{fk.column.name}")
        for fk in t.foreign_keys
    }
    assert resolved == EXPECTED_FK_EDGES[name]


def test_channels_have_no_foreign_keys() -> None:
    assert len(tbl("channels").foreign_keys) == 0


# --- 4. Unique constraints --------------------------------------------------

def test_unique_constraints() -> None:
    assert any(
        isinstance(c, UniqueConstraint) and c.name == "uq_channels_youtube_channel_id"
        for c in tbl("channels").constraints
    )
    assert any(
        isinstance(c, UniqueConstraint) and c.name == "uq_videos_youtube_video_id"
        for c in tbl("videos").constraints
    )
    # Goal 12 added a second, additive UNIQUE to transcript_segments (the FK
    # target for transcript_words). The original one is preserved unchanged.
    seg_uq = {
        c.name: set(c.columns.keys())
        for c in tbl("transcript_segments").constraints
        if isinstance(c, UniqueConstraint)
    }
    assert seg_uq == {
        "uq_transcript_segments_asset_start": {
            "transcript_asset_id", "start_seconds"
        },
        "uq_transcript_segments_id_asset_video": {
            "id", "transcript_asset_id", "video_id"
        },
    }


def test_external_identifiers_are_unique_and_not_null() -> None:
    assert tbl("channels").c.youtube_channel_id.nullable is False
    assert tbl("videos").c.youtube_video_id.nullable is False


# --- 5. Timestamp types -----------------------------------------------------

@pytest.mark.parametrize("name", sorted(EXPECTED_TABLES))
def test_created_updated_timestamps_are_timezone_aware(name: str) -> None:
    t = tbl(name)
    for colname in ("created_at", "updated_at"):
        col = t.c[colname]
        assert isinstance(col.type, DateTime)
        assert col.type.timezone is True
        assert col.nullable is False
        assert col.server_default is not None


@pytest.mark.parametrize("name", sorted(EXPECTED_TABLES))
def test_timestamps_compile_to_timestamptz(name: str) -> None:
    assert "TIMESTAMP WITH TIME ZONE" in compiled_ddl(name)


def test_nullable_timestamps() -> None:
    assert tbl("videos").c.published_at.nullable is True
    for colname in ("granted_at", "expires_at", "revoked_at"):
        assert tbl("content_rights").c[colname].nullable is True


# --- 6. Check constraints ---------------------------------------------------

def test_transcript_segment_check_constraints() -> None:
    checks = {
        c.name: str(c.sqltext)
        for c in tbl("transcript_segments").constraints
        if isinstance(c, CheckConstraint)
    }
    assert set(checks) == {
        "ck_transcript_segments_start_nonnegative",
        "ck_transcript_segments_end_after_start",
        "ck_transcript_segments_text_nonempty",
    }
    assert "start_seconds >= 0" in checks["ck_transcript_segments_start_nonnegative"]
    assert "end_seconds > start_seconds" in checks["ck_transcript_segments_end_after_start"]
    assert "btrim" in checks["ck_transcript_segments_text_nonempty"]


def test_transcript_segment_column_rules() -> None:
    t = tbl("transcript_segments")
    assert {
        "id", "transcript_asset_id", "video_id", "start_seconds",
        "end_seconds", "text", "original_text", "language", "created_at",
        "updated_at",
    } == set(t.c.keys())
    assert t.c.start_seconds.nullable is False
    assert t.c.end_seconds.nullable is False
    assert t.c.text.nullable is False
    assert t.c.language.nullable is True


def test_original_text_column_is_nullable_text() -> None:
    """The additive column is nullable TEXT with no default (legacy rows = NULL)."""
    col = tbl("transcript_segments").c.original_text
    assert isinstance(col.type, Text)
    assert col.nullable is True
    assert col.server_default is None
    assert "original_text TEXT" in compiled_ddl("transcript_segments")
    assert "original_text TEXT NOT NULL" not in compiled_ddl("transcript_segments")


def test_normalized_text_column_keeps_its_meaning() -> None:
    """The pre-existing normalized column is untouched: NOT NULL, still guarded."""
    t = tbl("transcript_segments")
    assert t.c.text.nullable is False
    assert TranscriptSegment.segment_text.property.columns[0].name == "text"
    checks = {
        c.name: str(c.sqltext)
        for c in t.constraints
        if isinstance(c, CheckConstraint)
    }
    assert "btrim" in checks["ck_transcript_segments_text_nonempty"]


def test_content_rights_authorization_fields() -> None:
    t = tbl("content_rights")
    for colname in (
        "authorization_type", "authorization_reference", "status",
        "granted_at", "expires_at", "revoked_at",
    ):
        assert colname in t.c


def test_no_vector_columns() -> None:
    # Out of scope: no embedding/vector columns anywhere.
    for t in Base.metadata.tables.values():
        for col in t.columns:
            assert "vector" not in col.name.lower()
            assert "embedding" not in col.name.lower()


# --- 7. Indexes -------------------------------------------------------------

EXPECTED_INDEXES = {
    "videos": {"ix_videos_channel_id"},
    "content_rights": {"ix_content_rights_video_id"},
    "transcript_assets": {
        "ix_transcript_assets_video_id",
        "ix_transcript_assets_content_right_id",
    },
    "transcript_segments": {"ix_transcript_segments_video_id"},
    # Goal 12: standalone indexes on the new tables. The partial UNIQUE index
    # (uq_alignment_runs_current_success) is an Index object, not a
    # UniqueConstraint, so it appears here too.
    "source_audio_assets": {
        "ix_source_audio_assets_video_id",
        "ix_source_audio_assets_content_right_id",
    },
    "transcript_alignment_runs": {
        "ix_alignment_runs_asset_created",
        "ix_alignment_runs_status",
        "ix_transcript_alignment_runs_video_id",
        "ix_transcript_alignment_runs_audio_id",
        "uq_alignment_runs_current_success",
    },
    "transcript_words": {
        "ix_transcript_words_video_id",
        "ix_transcript_words_segment_time",
        "ix_transcript_words_alignment_run_id",
    },
}


@pytest.mark.parametrize("name", sorted(EXPECTED_INDEXES))
def test_indexes_on_frequently_queried_fks(name: str) -> None:
    assert {i.name for i in tbl(name).indexes} == EXPECTED_INDEXES[name]


# --- 8. Migration revision discovery (no DB) --------------------------------

def test_migration_revision_discovery() -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    script = ScriptDirectory.from_config(cfg)

    # Exactly one head (the Goal 12 word-alignment migration).
    assert script.get_heads() == [HEAD_REVISION]

    goal06 = script.get_revision(REVISION_ID)
    assert goal06 is not None
    assert goal06.down_revision is None
    assert goal06.is_head is False

    goal07 = script.get_revision(GOAL07_REVISION)
    assert goal07 is not None
    assert goal07.down_revision == REVISION_ID
    assert goal07.is_head is False

    corrective = script.get_revision(GOAL09_CORRECTIVE_REVISION)
    assert corrective is not None
    assert corrective.down_revision == GOAL07_REVISION
    assert corrective.is_head is False

    goal12 = script.get_revision(GOAL12_REVISION)
    assert goal12 is not None
    assert goal12.down_revision == GOAL09_CORRECTIVE_REVISION
    assert goal12.is_head is True

    assert [r.revision for r in script.walk_revisions()] == [
        GOAL12_REVISION, GOAL09_CORRECTIVE_REVISION,
        GOAL07_REVISION, REVISION_ID,
    ]


def test_migration_module_is_importable_and_pure() -> None:
    import importlib.util

    path = MIGRATIONS_DIR / "versions" / f"{REVISION_ID}_initial_voxsearch_schema.py"
    spec = importlib.util.spec_from_file_location("goal06_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.revision == REVISION_ID
    assert module.down_revision is None
    assert callable(module.upgrade)
    assert callable(module.downgrade)
    # No credentials live in the migration source.
    src = path.read_text(encoding="utf-8")
    for forbidden in ("SUPABASE_", "password=", "postgresql://", "postgres://"):
        assert forbidden.lower() not in src.lower()
