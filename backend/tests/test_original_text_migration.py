"""Offline tests for the Goal 09 corrective migration (Issue 1).

Verifies that adding ``original_text`` is a SINGLE, DDL-only, additive,
backward-compatible change — and that the normalized ``"text"`` column keeps
its meaning.

These tests NEVER open a live Supabase / network connection and NEVER apply a
migration to the real database. They inspect:

* the Alembic script directory (single head, linear chain, exact revision),
* the new migration module source (DDL-only, nullable column, no credentials),
* SQLAlchemy model metadata (column type / nullability / unchanged CHECKs),
* a REAL in-memory SQLite simulation of a pre-migration row surviving the
  ``ADD COLUMN`` with no backfill (no PostgreSQL / Docker needed).

Live migration application requires separate approval and is NOT done here.
"""
import importlib.util
import re
from collections import Counter
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, Text, create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from app.db.models import SCHEMA, Base, TranscriptSegment

BACKEND_DIR = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = BACKEND_DIR / "migrations"
VERSIONS_DIR = MIGRATIONS_DIR / "versions"

GOAL06_REVISION = "b3727ed7438a"
GOAL07_REVISION = "fe7b2b8e5323"
CORRECTIVE_REVISION = "b5d2e8a14c7f"
CORRECTIVE_FILENAME = (
    f"{CORRECTIVE_REVISION}_add_original_text_to_transcript_segments.py"
)


def tbl(name: str):
    return Base.metadata.tables[f"{SCHEMA}.{name}"]


def compiled_ddl(name: str) -> str:
    from sqlalchemy.dialects import postgresql

    return str(CreateTable(tbl(name)).compile(dialect=postgresql.dialect()))


def load_module():
    path = VERSIONS_DIR / CORRECTIVE_FILENAME
    spec = importlib.util.spec_from_file_location("corrective_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bodies() -> tuple[str, str, str]:
    src = (VERSIONS_DIR / CORRECTIVE_FILENAME).read_text(encoding="utf-8")
    upgrade_src = src.split("def upgrade()")[1].split("def downgrade()")[0]
    downgrade_src = src.split("def downgrade()")[1]
    return src, upgrade_src, downgrade_src


# --- 1. Revision identity & linear history -----------------------------------


def test_revision_identity() -> None:
    m = load_module()
    assert m.revision == CORRECTIVE_REVISION
    assert m.down_revision == GOAL07_REVISION
    assert callable(m.upgrade)
    assert callable(m.downgrade)


def test_single_head_and_linear_history() -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    script = ScriptDirectory.from_config(cfg)

    assert script.get_heads() == [CORRECTIVE_REVISION]
    assert script.get_revision(CORRECTIVE_REVISION).down_revision == GOAL07_REVISION
    assert script.get_revision(GOAL07_REVISION).is_head is False
    assert [r.revision for r in script.walk_revisions()] == [
        CORRECTIVE_REVISION, GOAL07_REVISION, GOAL06_REVISION,
    ]


# --- 2. upgrade() is DDL-only: exactly one nullable ADD COLUMN ----------------


def test_upgrade_is_one_ddl_only_add_column() -> None:
    _, upgrade_src, _ = bodies()
    ops = Counter(re.findall(r"op\.(\w+)\(", upgrade_src))
    assert ops == {"add_column": 1}

    forbidden = {
        "execute", "bulk_insert", "alter_column", "drop_column",
        "create_table", "drop_table", "create_index", "drop_index",
        "create_schema", "drop_schema", "create_unique_constraint",
        "create_foreign_key", "drop_constraint",
    }
    assert not (set(ops) & forbidden)
    assert "sa.text(" not in upgrade_src  # no raw / ad-hoc SQL


def test_upgrade_column_is_nullable_text_with_no_default() -> None:
    _, upgrade_src, _ = bodies()
    assert '"transcript_segments"' in upgrade_src
    assert '"original_text"' in upgrade_src
    assert "sa.Text()" in upgrade_src
    assert "nullable=True" in upgrade_src
    assert "schema=SCHEMA" in upgrade_src
    # No DEFAULT and no data rewrite: legacy rows keep NULL honestly.
    assert "server_default" not in upgrade_src
    assert not re.search(r"\bUPDATE\b", upgrade_src, flags=re.IGNORECASE)
    assert not re.search(r"\bINSERT\b", upgrade_src, flags=re.IGNORECASE)
    assert not re.search(r"\bDELETE\b", upgrade_src, flags=re.IGNORECASE)


def test_downgrade_drops_only_the_new_column() -> None:
    _, _, downgrade_src = bodies()
    ops = Counter(re.findall(r"op\.(\w+)\(", downgrade_src))
    assert ops == {"drop_column": 1}
    assert '"transcript_segments"' in downgrade_src
    assert '"original_text"' in downgrade_src
    assert "schema=SCHEMA" in downgrade_src
    forbidden = {"drop_table", "drop_constraint", "execute", "bulk_insert"}
    assert not (set(ops) & forbidden)


def test_no_credentials_in_migration_source() -> None:
    src, _, _ = bodies()
    for forbidden in ("SUPABASE_", "password=", "postgresql://", "postgres://"):
        assert forbidden.lower() not in src.lower()


# --- 3. Model metadata: new column + normalized column unchanged -------------


def test_original_text_column_is_nullable_text() -> None:
    col = tbl("transcript_segments").c.original_text
    assert isinstance(col.type, Text)
    assert col.nullable is True
    assert col.server_default is None
    assert "original_text TEXT" in compiled_ddl("transcript_segments")
    assert "original_text TEXT NOT NULL" not in compiled_ddl("transcript_segments")


def test_normalized_text_column_keeps_meaning_and_constraints() -> None:
    t = tbl("transcript_segments")
    assert t.c.text.nullable is False
    assert TranscriptSegment.segment_text.property.columns[0].name == "text"
    checks = {
        c.name: str(c.sqltext)
        for c in t.constraints
        if isinstance(c, CheckConstraint)
    }
    # The exact original CHECK set is preserved; none added / removed.
    assert set(checks) == {
        "ck_transcript_segments_start_nonnegative",
        "ck_transcript_segments_end_after_start",
        "ck_transcript_segments_text_nonempty",
    }
    assert "btrim" in checks["ck_transcript_segments_text_nonempty"]


def test_exactly_five_tables_no_jobs_table() -> None:
    """Option A: no separate jobs table, no new table of any kind."""
    assert set(Base.metadata.tables) == {
        f"{SCHEMA}.{n}"
        for n in (
            "channels", "videos", "content_rights",
            "transcript_assets", "transcript_segments",
        )
    }
    assert "original_text" in tbl("transcript_segments").c


# --- 4. Backward compatibility: a real pre-migration row survives -----------


def test_legacy_row_survives_add_column_with_no_backfill() -> None:
    """A genuine in-memory SQLite simulation of the migration on legacy data.

    No PostgreSQL / Docker and no production connection. A row created BEFORE
    the migration (only the normalized ``"text"`` column exists) must remain
    valid afterwards: its text unchanged, ``original_text`` NULL, and the
    pre-existing non-empty CHECK still enforced. A NEW row can populate both.
    """
    engine = create_engine("sqlite://")
    try:
        with engine.begin() as conn:
            conn.execute(text(
                'CREATE TABLE transcript_segments ('
                'id INTEGER PRIMARY KEY, "text" TEXT NOT NULL, '
                'CHECK (length(trim("text")) > 0))'
            ))
            conn.execute(text(
                'INSERT INTO transcript_segments (id, "text") '
                "VALUES (1, 'normalized legacy')"
            ))
            # The migration's single DDL statement: additive, no backfill.
            conn.execute(text(
                "ALTER TABLE transcript_segments "
                "ADD COLUMN original_text TEXT"
            ))

        with engine.connect() as conn:
            row = conn.execute(text(
                'SELECT "text", original_text FROM transcript_segments '
                "WHERE id = 1"
            )).one()
            assert row[0] == "normalized legacy"  # NOT overwritten
            assert row[1] is None                  # honest "not captured"

        with engine.begin() as conn:
            # New rows can populate BOTH values separately.
            conn.execute(text(
                'INSERT INTO transcript_segments (id, "text", original_text) '
                "VALUES (2, 'normalized', '  verbatim  original  ')"
            ))
            new = conn.execute(text(
                'SELECT "text", original_text FROM transcript_segments '
                "WHERE id = 2"
            )).one()
            assert new == ("normalized", "  verbatim  original  ")

        # The original non-empty CHECK still holds for the normalized column.
        with engine.connect() as conn:
            with pytest.raises(IntegrityError):
                conn.execute(text(
                    'INSERT INTO transcript_segments (id, "text", original_text) '
                    "VALUES (3, '   ', NULL)"
                ))
    finally:
        engine.dispose()
