"""Dedicated OFFLINE tests for the Goal 16 ``video_chunks`` migration.

Covers the approved Goal 16 design (see the implementation plan):

* revision identity + single-head linear history (5 revisions, new head),
* ``upgrade()`` is DDL-only and additive: exactly one ``CREATE TABLE`` in the
  dedicated ``voxsearch`` schema, no DML, no destructive ops, no alteration of
  any pre-existing object,
* the pgvector column renders ``vector(1024)`` from the dependency-free
  ``VectorType`` UDT (no ``pgvector`` package / numpy / ``CREATE EXTENSION``),
* every named constraint (UNIQUE upsert key / composite FKs / CHECKs) is
  emitted by the REAL ``upgrade()`` function,
* NO ANN / HNSW / IVFFlat index (cosine similarity search is Goal 17),
* ``downgrade()`` drops ONLY the new table,
* model <-> migration constraint-name constant parity.

Honesty boundaries
------------------
These tests are OFFLINE: they execute the real migration functions through
Alembic's ``MigrationContext`` + ``Operations`` (``as_sql=True``, PostgreSQL
dialect) and inspect the rendered SQL string. They NEVER open a network /
Supabase / ``.env`` connection, NEVER invoke the Alembic CLI / ``env.py`` /
``get_url()``, and NEVER apply a migration to any real database. Rendering DDL
is NOT proof that a live PostgreSQL + pgvector server accepts the
``vector(1024)`` bind/CAST behaviour — that requires a real instance (NOT
AVAILABLE here; see the opt-in ``live_pgvector`` test, skipped by default).
"""
from collections import Counter
import importlib.util
from io import StringIO
from pathlib import Path
import re

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from app.db import models
from app.db.models import SCHEMA

BACKEND_DIR = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = BACKEND_DIR / "migrations"
VERSIONS_DIR = MIGRATIONS_DIR / "versions"

GOAL06_REVISION = "b3727ed7438a"
GOAL07_REVISION = "fe7b2b8e5323"
GOAL09_CORRECTIVE_REVISION = "b5d2e8a14c7f"
GOAL12_REVISION = "a7c2e9f4b1d8"
GOAL16_REVISION = "c9d4f1a6b2e3"
GOAL16_FILENAME = f"{GOAL16_REVISION}_goal16_video_chunks.py"

MODEL_MIGRATION_CONSTANTS = (
    "UQ_VIDEO_CHUNKS_ASSET_CHUNK_MODEL",
    "FK_VIDEO_CHUNKS_VIDEO",
    "FK_VIDEO_CHUNKS_ASSET_VIDEO",
    "FK_VIDEO_CHUNKS_RIGHT_VIDEO",
)


# --- helpers ---------------------------------------------------------------

def load_module():
    path = VERSIONS_DIR / GOAL16_FILENAME
    spec = importlib.util.spec_from_file_location("goal16_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bodies() -> tuple[str, str, str]:
    src = (VERSIONS_DIR / GOAL16_FILENAME).read_text(encoding="utf-8")
    upgrade_src = src.split("def upgrade()")[1].split("def downgrade()")[0]
    downgrade_src = src.split("def downgrade()")[1]
    return src, upgrade_src, downgrade_src


def _offline_sql(direction: str) -> str:
    """Execute the REAL migration function offline, capturing rendered DDL.

    Uses Alembic's ``MigrationContext`` (``as_sql=True``, PostgreSQL dialect)
    and a real ``Operations`` instance bound to the module's ``op`` global. No
    DB connection, no ``env.py`` / ``get_url()`` / ``.env`` access, no
    credentials.
    """
    module = load_module()
    buf = StringIO()
    ctx = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": buf},
    )
    module.op = Operations(ctx)
    if direction == "upgrade":
        module.upgrade()
    else:
        module.downgrade()
    return buf.getvalue()


@pytest.fixture(scope="module")
def upgrade_sql() -> str:
    return _offline_sql("upgrade")


@pytest.fixture(scope="module")
def downgrade_sql() -> str:
    return _offline_sql("downgrade")


# --- 1. Revision identity & linear history ---------------------------------

def test_revision_identity() -> None:
    m = load_module()
    assert m.revision == GOAL16_REVISION
    assert m.down_revision == GOAL12_REVISION
    assert callable(m.upgrade)
    assert callable(m.downgrade)


def test_single_head_and_five_revision_linear_history() -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    script = ScriptDirectory.from_config(cfg)

    assert script.get_heads() == [GOAL16_REVISION]
    assert script.get_revision(GOAL16_REVISION).down_revision == GOAL12_REVISION
    assert script.get_revision(GOAL16_REVISION).is_head is True
    assert script.get_revision(GOAL12_REVISION).is_head is False
    assert [r.revision for r in script.walk_revisions()] == [
        GOAL16_REVISION, GOAL12_REVISION, GOAL09_CORRECTIVE_REVISION,
        GOAL07_REVISION, GOAL06_REVISION,
    ]


def test_migration_module_has_no_credentials() -> None:
    src, _, _ = bodies()
    for forbidden in (
        "SUPABASE_", "password=", "postgresql://", "postgres://", "get_url",
    ):
        assert forbidden.lower() not in src.lower()


# --- 2. upgrade() is DDL-only and additive ----------------------------------

def test_upgrade_op_structure() -> None:
    _, upgrade_src, _ = bodies()
    ops = Counter(re.findall(r"op\.(\w+)\(", upgrade_src))
    assert ops == {"create_table": 1}


def test_upgrade_has_no_destructive_or_dml_ops() -> None:
    _, upgrade_src, _ = bodies()
    forbidden = {
        "execute", "bulk_insert", "drop_table", "drop_index", "drop_column",
        "alter_column", "add_column", "drop_constraint", "create_foreign_key",
        "drop_schema", "create_schema", "rename_table", "create_index",
        "create_unique_constraint",
    }
    used = set(re.findall(r"op\.(\w+)\(", upgrade_src))
    assert not (used & forbidden)
    # DML verbs only count as statements; "ON DELETE CASCADE" is a legal FK
    # clause, not a DELETE statement, so match the statement form explicitly.
    for dml in (
        r"\bINSERT\s+INTO\b", r"\bUPDATE\s+\w+\s+SET\b",
        r"\bDELETE\s+FROM\b", r"\bTRUNCATE\b",
    ):
        assert not re.search(dml, upgrade_src, flags=re.IGNORECASE)


def test_upgrade_does_not_touch_existing_objects() -> None:
    _, upgrade_src, _ = bodies()
    # Additive-only: the only object created is the ONE new table; no column of
    # any pre-existing table is referenced for alteration.
    for op_name in ("alter_column", "add_column", "drop_column", "rename_table"):
        assert op_name not in upgrade_src
    assert 'create_table(\n        "video_chunks"' in upgrade_src


# --- 3. downgrade() reverses only the new table ----------------------------

def test_downgrade_op_structure() -> None:
    _, _, downgrade_src = bodies()
    ops = Counter(re.findall(r"op\.(\w+)\(", downgrade_src))
    assert ops == {"drop_table": 1}


def test_downgrade_drops_only_video_chunks() -> None:
    _, _, downgrade_src = bodies()
    dropped = re.findall(r'op\.drop_table\(\s*"([^"]+)"', downgrade_src)
    assert dropped == ["video_chunks"]
    assert 'schema=SCHEMA' in downgrade_src


# --- 4. Offline SQL from the REAL upgrade()/downgrade() --------------------

def test_upgrade_sql_creates_exactly_one_schema_qualified_table(
    upgrade_sql: str,
) -> None:
    assert upgrade_sql.count("CREATE TABLE") == 1
    assert f"CREATE TABLE {SCHEMA}.video_chunks (" in upgrade_sql
    # Everything stays in the dedicated schema; public/managed objects untouched.
    for line in upgrade_sql.splitlines():
        stripped = line.strip()
        if stripped.startswith(
            ("CREATE TABLE", "ALTER TABLE", "CREATE INDEX", "CREATE UNIQUE INDEX")
        ):
            assert f"{SCHEMA}." in line


def test_upgrade_sql_has_vector_1024_column(upgrade_sql: str) -> None:
    # The dependency-free UserDefinedType renders the pgvector column type.
    assert "embedding vector(1024) NOT NULL" in upgrade_sql


def test_upgrade_sql_has_all_named_constraints(upgrade_sql: str) -> None:
    for frag in (
        "CONSTRAINT uq_video_chunks_asset_chunk_model UNIQUE "
        "(transcript_asset_id, chunk_index, model_id)",
        "CONSTRAINT fk_video_chunks_asset_video "
        "FOREIGN KEY(transcript_asset_id, video_id) "
        "REFERENCES voxsearch.transcript_assets (id, video_id)",
        "CONSTRAINT fk_video_chunks_right_video "
        "FOREIGN KEY(content_right_id, video_id) "
        "REFERENCES voxsearch.content_rights (id, video_id)",
        "CONSTRAINT fk_video_chunks_video FOREIGN KEY(video_id) "
        "REFERENCES voxsearch.videos (id) ON DELETE CASCADE",
        "CONSTRAINT ck_video_chunks_chunk_index_nonnegative "
        "CHECK (chunk_index >= 0)",
        "CONSTRAINT ck_video_chunks_time_window",
        "CONSTRAINT ck_video_chunks_model_id_nonblank",
        "CONSTRAINT ck_video_chunks_text_nonblank",
    ):
        assert frag in upgrade_sql


def test_upgrade_sql_has_no_create_extension(upgrade_sql: str) -> None:
    # pgvector 0.8.2 is assumed pre-installed (Goal 06); if absent the CREATE
    # TABLE fails fail-closed. The migration MUST NOT try to create it.
    assert "CREATE EXTENSION" not in upgrade_sql.upper()


def test_upgrade_sql_has_no_ann_index(upgrade_sql: str) -> None:
    upper = upgrade_sql.upper()
    assert "USING HNSW" not in upper
    assert "USING IVFFLAT" not in upper
    assert "CREATE INDEX" not in upper
    assert "CREATE UNIQUE INDEX" not in upper


def test_upgrade_sql_is_ddl_only_no_dml_no_credentials(
    upgrade_sql: str,
) -> None:
    upper = upgrade_sql.upper()
    # Statement-form DML only; "ON DELETE CASCADE" is a legal FK clause.
    for verb in ("INSERT INTO", "UPDATE SET", "DELETE FROM", "TRUNCATE TABLE"):
        assert verb not in upper
    assert "ALTER COLUMN" not in upper
    assert "DROP COLUMN" not in upper
    assert "CREATE EXTENSION" not in upper
    for forbidden in ("SUPABASE_", "POSTGRESQL://", "POSTGRES://", "PASSWORD="):
        assert forbidden not in upper


def test_downgrade_sql_reverses_only_new_table(downgrade_sql: str) -> None:
    # Exactly one DROP TABLE, targeting the new Goal 16 table only. The count
    # plus the exact schema-qualified match already proves no pre-existing
    # table is dropped.
    assert downgrade_sql.count("DROP TABLE") == 1
    assert f"DROP TABLE {SCHEMA}.video_chunks" in downgrade_sql
    upper = downgrade_sql.upper()
    for verb in ("CREATE TABLE", "INSERT INTO", "DELETE FROM", "DROP COLUMN"):
        assert verb not in upper


# --- 5. Model <-> migration constant parity --------------------------------

def test_model_constants_match_migration_constants() -> None:
    m = load_module()
    for name in MODEL_MIGRATION_CONSTANTS:
        assert hasattr(models, name), f"models is missing {name}"
        assert hasattr(m, name), f"migration is missing {name}"
        assert getattr(models, name) == getattr(m, name), f"mismatch on {name}"
