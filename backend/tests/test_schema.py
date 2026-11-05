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
from sqlalchemy import CheckConstraint, DateTime, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.db.models import (
    SCHEMA,
    Base,
    Channel,
    ContentRight,
    TranscriptAsset,
    TranscriptSegment,
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
}
REVISION_ID = "b3727ed7438a"


def tbl(name: str):
    return Base.metadata.tables[f"{SCHEMA}.{name}"]


def compiled_ddl(name: str) -> str:
    return str(CreateTable(tbl(name)).compile(dialect=postgresql.dialect()))


# --- 1. Model metadata / schema names ---------------------------------------

def test_schema_name() -> None:
    assert SCHEMA == "voxsearch"


def test_all_five_tables_present() -> None:
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

EXPECTED_FKS = {
    "videos": {"channel_id": "voxsearch.channels.id"},
    "content_rights": {"video_id": "voxsearch.videos.id"},
    "transcript_assets": {
        "video_id": "voxsearch.videos.id",
        "content_right_id": "voxsearch.content_rights.id",
    },
    "transcript_segments": {
        "transcript_asset_id": "voxsearch.transcript_assets.id",
        "video_id": "voxsearch.videos.id",
    },
}


@pytest.mark.parametrize("name", sorted(EXPECTED_FKS))
def test_foreign_keys_resolve_to_expected_targets(name: str) -> None:
    t = tbl(name)
    resolved = {
        fk.parent.name: f"{fk.column.table.fullname}.{fk.column.name}"
        for fk in t.foreign_keys
    }
    assert resolved == EXPECTED_FKS[name]


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
    seg_uq = [
        c for c in tbl("transcript_segments").constraints
        if isinstance(c, UniqueConstraint)
    ]
    assert len(seg_uq) == 1
    assert seg_uq[0].name == "uq_transcript_segments_asset_start"
    assert set(seg_uq[0].columns.keys()) == {"transcript_asset_id", "start_seconds"}


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
        "end_seconds", "text", "language", "created_at", "updated_at",
    } == set(t.c.keys())
    assert t.c.start_seconds.nullable is False
    assert t.c.end_seconds.nullable is False
    assert t.c.text.nullable is False
    assert t.c.language.nullable is True


def test_content_rights_authorization_fields() -> None:
    t = tbl("content_rights")
    for colname in (
        "authorization_type", "authorization_reference", "status",
        "granted_at", "expires_at", "revoked_at",
    ):
        assert colname in t.c


def test_no_vector_columns() -> None:
    # Out of scope for Goal 06: no embedding/vector columns anywhere.
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
}


@pytest.mark.parametrize("name", sorted(EXPECTED_INDEXES))
def test_indexes_on_frequently_queried_fks(name: str) -> None:
    assert {i.name for i in tbl(name).indexes} == EXPECTED_INDEXES[name]


# --- 8. Migration revision discovery (no DB) --------------------------------

def test_migration_revision_discovery() -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    script = ScriptDirectory.from_config(cfg)

    heads = script.get_heads()
    assert heads == [REVISION_ID]

    revision = script.get_revision(REVISION_ID)
    assert revision is not None
    assert revision.down_revision is None
    assert revision.is_head is True

    all_revisions = list(script.walk_revisions())
    assert [r.revision for r in all_revisions] == [REVISION_ID]


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
