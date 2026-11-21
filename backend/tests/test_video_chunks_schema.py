"""Offline schema tests for the Goal 16 ``video_chunks`` table.

These tests inspect SQLAlchemy metadata and PostgreSQL-dialect CREATE TABLE
DDL only. They NEVER open a network / Supabase / ``.env`` connection and NEVER
apply a migration. Rendering DDL is NOT proof of runtime pgvector behaviour.
"""
import importlib.util
from pathlib import Path

from sqlalchemy import CheckConstraint, DateTime, ForeignKeyConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.db import models
from app.db.models import (
    SCHEMA,
    Base,
    FK_VIDEO_CHUNKS_ASSET_VIDEO,
    FK_VIDEO_CHUNKS_RIGHT_VIDEO,
    FK_VIDEO_CHUNKS_VIDEO,
    UQ_VIDEO_CHUNKS_ASSET_CHUNK_MODEL,
    VideoChunk,
)
from app.db.types import VectorType

BACKEND_DIR = Path(__file__).resolve().parents[1]
MIGRATION_PATH = (
    BACKEND_DIR / "migrations" / "versions"
    / "c9d4f1a6b2e3_goal16_video_chunks.py"
)

EXPECTED_TABLES = {
    "channels", "videos", "content_rights", "transcript_assets",
    "transcript_segments", "source_audio_assets",
    "transcript_alignment_runs", "transcript_words", "video_chunks",
}


def tbl(name: str):
    return Base.metadata.tables[f"{SCHEMA}.{name}"]


def compiled_ddl(name: str) -> str:
    return str(CreateTable(tbl(name)).compile(dialect=postgresql.dialect()))


def _load_migration():
    spec = importlib.util.spec_from_file_location("goal16_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- 1. Table set / registration -------------------------------------------

def test_exactly_nine_tables_in_voxsearch() -> None:
    assert set(Base.metadata.tables) == {
        f"{SCHEMA}.{n}" for n in EXPECTED_TABLES
    }
    assert all(t.schema == SCHEMA for t in Base.metadata.tables.values())


def test_video_chunk_class_is_registered() -> None:
    assert VideoChunk.__table__ is tbl("video_chunks")


def test_uuid_primary_key() -> None:
    pk = list(tbl("video_chunks").primary_key.columns)
    assert len(pk) == 1 and pk[0].name == "id"
    assert pk[0].nullable is False


# --- 2. Vector column -------------------------------------------------------

def test_embedding_column_is_vector_1024_not_null() -> None:
    col = tbl("video_chunks").c.embedding
    assert isinstance(col.type, VectorType)
    assert col.type.dim == 1024
    assert col.type.get_col_spec() == "vector(1024)"
    assert col.nullable is False


def test_embedding_ddl_renders_vector_1024_not_null() -> None:
    assert "embedding vector(1024) NOT NULL" in compiled_ddl("video_chunks")


def test_sibling_tables_have_no_vector_columns() -> None:
    """Goal 16 adds the ONLY vector column; the other 8 tables stay clean."""
    for t in Base.metadata.tables.values():
        if t.name == "video_chunks":
            continue
        for col in t.columns:
            assert "vector" not in col.name.lower()
            assert "embedding" not in col.name.lower()


# --- 3. Constraints ---------------------------------------------------------

def test_unique_upsert_key() -> None:
    uqs = {
        c.name: list(c.columns.keys())
        for c in tbl("video_chunks").constraints
        if isinstance(c, UniqueConstraint)
    }
    assert uqs == {
        UQ_VIDEO_CHUNKS_ASSET_CHUNK_MODEL: [
            "transcript_asset_id", "chunk_index", "model_id"
        ],
    }


def test_foreign_key_names() -> None:
    names = {
        c.name
        for c in tbl("video_chunks").constraints
        if isinstance(c, ForeignKeyConstraint)
    }
    assert names == {
        FK_VIDEO_CHUNKS_VIDEO,
        FK_VIDEO_CHUNKS_ASSET_VIDEO,
        FK_VIDEO_CHUNKS_RIGHT_VIDEO,
    }


def _fk_pairs(constraint_name: str) -> dict[str, str]:
    fk = next(
        c for c in tbl("video_chunks").constraints
        if isinstance(c, ForeignKeyConstraint) and c.name == constraint_name
    )
    return {
        e.parent.name: f"{e.column.table.fullname}.{e.column.name}"
        for e in fk.elements
    }


def test_composite_fk_targets_existing_declared_uniques() -> None:
    assert _fk_pairs(FK_VIDEO_CHUNKS_ASSET_VIDEO) == {
        "transcript_asset_id": "voxsearch.transcript_assets.id",
        "video_id": "voxsearch.transcript_assets.video_id",
    }
    assert _fk_pairs(FK_VIDEO_CHUNKS_RIGHT_VIDEO) == {
        "content_right_id": "voxsearch.content_rights.id",
        "video_id": "voxsearch.content_rights.video_id",
    }
    assert _fk_pairs(FK_VIDEO_CHUNKS_VIDEO) == {
        "video_id": "voxsearch.videos.id",
    }


def test_check_constraints() -> None:
    checks = {
        c.name: str(c.sqltext)
        for c in tbl("video_chunks").constraints
        if isinstance(c, CheckConstraint)
    }
    assert set(checks) == {
        "ck_video_chunks_chunk_index_nonnegative",
        "ck_video_chunks_time_window",
        "ck_video_chunks_model_id_nonblank",
        "ck_video_chunks_text_nonblank",
    }
    assert "chunk_index >= 0" in checks["ck_video_chunks_chunk_index_nonnegative"]
    assert "end_ms > start_ms" in checks["ck_video_chunks_time_window"]
    assert "btrim" in checks["ck_video_chunks_model_id_nonblank"]
    assert "btrim" in checks["ck_video_chunks_text_nonblank"]


# --- 4. NO ANN index (Goal 17 boundary) -------------------------------------

def test_no_ann_or_any_index_on_video_chunks() -> None:
    assert {i.name for i in tbl("video_chunks").indexes} == set()


def test_ddl_has_no_ann_index_or_create_extension() -> None:
    upper = compiled_ddl("video_chunks").upper()
    assert "USING HNSW" not in upper
    assert "USING IVFFLAT" not in upper
    assert "CREATE EXTENSION" not in upper


# --- 5. Timestamps ----------------------------------------------------------

def test_timestamps_are_timezone_aware_not_null() -> None:
    t = tbl("video_chunks")
    for colname in ("created_at", "updated_at"):
        col = t.c[colname]
        assert isinstance(col.type, DateTime)
        assert col.type.timezone is True
        assert col.nullable is False
        assert col.server_default is not None
    assert "TIMESTAMP WITH TIME ZONE" in compiled_ddl("video_chunks")


# --- 6. Model <-> migration constant sync ----------------------------------

def test_model_constants_match_migration_constants() -> None:
    m = _load_migration()
    assert m.UQ_VIDEO_CHUNKS_ASSET_CHUNK_MODEL == models.UQ_VIDEO_CHUNKS_ASSET_CHUNK_MODEL
    assert m.FK_VIDEO_CHUNKS_VIDEO == models.FK_VIDEO_CHUNKS_VIDEO
    assert m.FK_VIDEO_CHUNKS_ASSET_VIDEO == models.FK_VIDEO_CHUNKS_ASSET_VIDEO
    assert m.FK_VIDEO_CHUNKS_RIGHT_VIDEO == models.FK_VIDEO_CHUNKS_RIGHT_VIDEO
    assert m.revision == "c9d4f1a6b2e3"
    assert m.down_revision == "a7c2e9f4b1d8"
