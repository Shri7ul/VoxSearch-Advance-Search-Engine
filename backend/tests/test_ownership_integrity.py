"""Offline tests for Goal 07: cross-table video ownership integrity.

These tests NEVER open a live Supabase / network connection. They inspect:

* SQLAlchemy 2.x model metadata,
* PostgreSQL-compiled CREATE TABLE DDL,
* the Alembic script directory and the Goal 07 migration source,
* a pure-Python simulation of composite-FK key matching.

Live constraint verification after migration is a separate step (see the Goal 07
validation report).
"""
from collections import Counter
import importlib.util
from pathlib import Path
import re

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.db import models
from app.db.models import (
    SCHEMA,
    Base,
    FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO,
    FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO,
    UQ_CONTENT_RIGHTS_ID_VIDEO_ID,
    UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = BACKEND_DIR / "migrations"
VERSIONS_DIR = MIGRATIONS_DIR / "versions"

GOAL06_REVISION = "b3727ed7438a"
GOAL07_REVISION = "fe7b2b8e5323"
GOAL07_FILENAME = f"{GOAL07_REVISION}_enforce_transcript_ownership_integrity.py"

# Exact names verified live against pg_constraint before the migration was
# written. These are the two single-column FKs that were replaced.
OLD_FK_ASSET = "fk_transcript_assets_content_right_id_content_rights"
OLD_FK_SEGMENT = "fk_transcript_segments_asset_id_transcript_assets"

# Single-column FKs that must survive the migration unchanged.
PRESERVED_VIDEO_FKS = {
    "videos": ("fk_videos_channel_id_channels", "channel_id", "voxsearch.channels.id"),
    "content_rights": (
        "fk_content_rights_video_id_videos", "video_id", "voxsearch.videos.id",
    ),
    "transcript_assets": (
        "fk_transcript_assets_video_id_videos", "video_id", "voxsearch.videos.id",
    ),
    "transcript_segments": (
        "fk_transcript_segments_video_id_videos", "video_id", "voxsearch.videos.id",
    ),
}

NEW_NAMES = {
    UQ_CONTENT_RIGHTS_ID_VIDEO_ID,
    UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID,
    FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO,
    FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO,
}

# Module-level constant identifiers used by the migration's upgrade/downgrade.
NEW_IDENTIFIERS = (
    "UQ_CONTENT_RIGHTS_ID_VIDEO_ID",
    "UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID",
    "FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO",
    "FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO",
)
OLD_IDENTIFIERS = (
    "FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_OLD",
    "FK_TRANSCRIPT_SEGMENTS_ASSET_OLD",
)


# --- helpers ---------------------------------------------------------------

def tbl(name: str):
    return Base.metadata.tables[f"{SCHEMA}.{name}"]


def compiled_ddl(name: str) -> str:
    return str(CreateTable(tbl(name)).compile(dialect=postgresql.dialect()))


def fk_by_name(name: str, constraint_name: str) -> ForeignKeyConstraint:
    matches = [
        c for c in tbl(name).constraints
        if isinstance(c, ForeignKeyConstraint) and c.name == constraint_name
    ]
    assert len(matches) == 1, f"expected exactly one FK named {constraint_name!r}"
    return matches[0]


def unique_by_name(name: str, constraint_name: str) -> UniqueConstraint:
    matches = [
        c for c in tbl(name).constraints
        if isinstance(c, UniqueConstraint) and c.name == constraint_name
    ]
    assert len(matches) == 1, f"expected exactly one UNIQUE named {constraint_name!r}"
    return matches[0]


def load_goal07_module():
    path = VERSIONS_DIR / GOAL07_FILENAME
    spec = importlib.util.spec_from_file_location("goal07_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def migration_bodies() -> tuple[str, str, str]:
    src = (VERSIONS_DIR / GOAL07_FILENAME).read_text(encoding="utf-8")
    upgrade_src = src.split("def upgrade()")[1].split("def downgrade()")[0]
    downgrade_src = src.split("def downgrade()")[1]
    return src, upgrade_src, downgrade_src


# --- 1. Composite unique constraints ---------------------------------------

def test_composite_unique_content_rights() -> None:
    c = unique_by_name("content_rights", UQ_CONTENT_RIGHTS_ID_VIDEO_ID)
    assert [col.name for col in c.columns] == ["id", "video_id"]


def test_composite_unique_transcript_assets() -> None:
    c = unique_by_name("transcript_assets", UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID)
    assert [col.name for col in c.columns] == ["id", "video_id"]


def test_composite_unique_names_are_stable_and_descriptive() -> None:
    assert UQ_CONTENT_RIGHTS_ID_VIDEO_ID == "uq_content_rights_id_video_id"
    assert UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID == "uq_transcript_assets_id_video_id"


def test_composite_unique_ddl_is_emitted() -> None:
    assert (
        f"CONSTRAINT {UQ_CONTENT_RIGHTS_ID_VIDEO_ID} UNIQUE (id, video_id)"
        in compiled_ddl("content_rights")
    )
    assert (
        f"CONSTRAINT {UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID} UNIQUE (id, video_id)"
        in compiled_ddl("transcript_assets")
    )


def test_no_other_new_unique_constraints() -> None:
    # Only the two required backing uniques were added; the Goal 06 uniques
    # remain and no extras were introduced on unrelated tables.
    assert {
        c.name for c in tbl("content_rights").constraints
        if isinstance(c, UniqueConstraint)
    } == {UQ_CONTENT_RIGHTS_ID_VIDEO_ID}
    assert {
        c.name for c in tbl("transcript_assets").constraints
        if isinstance(c, UniqueConstraint)
    } == {UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID}


# --- 2. Composite foreign key targets --------------------------------------

def test_asset_composite_fk_targets_content_rights() -> None:
    fk = fk_by_name("transcript_assets", FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO)
    assert [col.name for col in fk.columns] == ["content_right_id", "video_id"]
    assert [
        f"{e.column.table.fullname}.{e.column.name}" for e in fk.elements
    ] == ["voxsearch.content_rights.id", "voxsearch.content_rights.video_id"]


def test_segment_composite_fk_targets_transcript_assets() -> None:
    fk = fk_by_name("transcript_segments", FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO)
    assert [col.name for col in fk.columns] == ["transcript_asset_id", "video_id"]
    assert [
        f"{e.column.table.fullname}.{e.column.name}" for e in fk.elements
    ] == ["voxsearch.transcript_assets.id", "voxsearch.transcript_assets.video_id"]


def test_composite_fk_names_are_stable_and_descriptive() -> None:
    assert (
        FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO
        == "fk_transcript_assets_content_right_id_video_id"
    )
    assert (
        FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO
        == "fk_transcript_segments_transcript_asset_id_video_id"
    )


def test_composite_fk_ddl_matches_required_sql() -> None:
    assert (
        f"CONSTRAINT {FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO} "
        "FOREIGN KEY(content_right_id, video_id) "
        "REFERENCES voxsearch.content_rights (id, video_id)"
        in compiled_ddl("transcript_assets")
    )
    assert (
        f"CONSTRAINT {FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO} "
        "FOREIGN KEY(transcript_asset_id, video_id) "
        "REFERENCES voxsearch.transcript_assets (id, video_id)"
        in compiled_ddl("transcript_segments")
    )


# --- 3. Existing video foreign keys preserved ------------------------------

@pytest.mark.parametrize("name", sorted(PRESERVED_VIDEO_FKS))
def test_existing_foreign_keys_preserved(name: str) -> None:
    cname, local, remote = PRESERVED_VIDEO_FKS[name]
    fk = fk_by_name(name, cname)
    assert [col.name for col in fk.columns] == [local]
    assert [
        f"{e.column.table.fullname}.{e.column.name}" for e in fk.elements
    ] == [remote]


def test_old_single_column_fk_names_are_gone() -> None:
    all_fk_names = {
        c.name
        for t in Base.metadata.tables.values()
        for c in t.constraints
        if isinstance(c, ForeignKeyConstraint)
    }
    assert OLD_FK_ASSET not in all_fk_names
    assert OLD_FK_SEGMENT not in all_fk_names


def test_all_fk_constraint_names() -> None:
    expected = {v[0] for v in PRESERVED_VIDEO_FKS.values()} | {
        FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO,
        FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO,
    }
    actual = {
        c.name
        for t in Base.metadata.tables.values()
        for c in t.constraints
        if isinstance(c, ForeignKeyConstraint)
    }
    assert actual == expected


# --- 4. Constraint names / metadata sanity ---------------------------------

def test_four_new_constraint_names_are_exact() -> None:
    assert NEW_NAMES == {
        "uq_content_rights_id_video_id",
        "uq_transcript_assets_id_video_id",
        "fk_transcript_assets_content_right_id_video_id",
        "fk_transcript_segments_transcript_asset_id_video_id",
    }


def test_five_tables_only_and_schema_unchanged() -> None:
    assert set(Base.metadata.tables) == {
        f"{SCHEMA}.{n}" for n in (
            "channels", "videos", "content_rights",
            "transcript_assets", "transcript_segments",
        )
    }
    assert all(t.schema == SCHEMA for t in Base.metadata.tables.values())


def test_no_new_indexes_added() -> None:
    # The two new backing uniques are constraints, not Index objects; the
    # original five indexes must remain exactly as before.
    assert {i.name for i in tbl("videos").indexes} == {"ix_videos_channel_id"}
    assert {i.name for i in tbl("content_rights").indexes} == {
        "ix_content_rights_video_id"
    }
    assert {i.name for i in tbl("transcript_assets").indexes} == {
        "ix_transcript_assets_video_id",
        "ix_transcript_assets_content_right_id",
    }
    assert {i.name for i in tbl("transcript_segments").indexes} == {
        "ix_transcript_segments_video_id"
    }


# --- 5. Model / migration consistency --------------------------------------

def test_model_constants_match_migration_constants() -> None:
    m = load_goal07_module()
    assert m.UQ_CONTENT_RIGHTS_ID_VIDEO_ID == models.UQ_CONTENT_RIGHTS_ID_VIDEO_ID
    assert m.UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID == models.UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID
    assert (
        m.FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO
        == models.FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO
    )
    assert (
        m.FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO
        == models.FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO
    )


def test_migration_uses_exact_verified_old_fk_names() -> None:
    m = load_goal07_module()
    assert m.FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_OLD == OLD_FK_ASSET
    assert m.FK_TRANSCRIPT_SEGMENTS_ASSET_OLD == OLD_FK_SEGMENT


def test_migration_defines_all_new_constraint_name_literals() -> None:
    src, _, _ = migration_bodies()
    # Literal strings live in the module-level constant definitions, not in
    # the function bodies, so check the whole file.
    for name in NEW_NAMES:
        assert name in src
    assert OLD_FK_ASSET in src
    assert OLD_FK_SEGMENT in src


def test_migration_upgrade_operations() -> None:
    _, upgrade_src, _ = migration_bodies()
    ops = Counter(re.findall(r"op\.(\w+)\(", upgrade_src))
    assert ops == {
        "create_unique_constraint": 2,
        "drop_constraint": 2,
        "create_foreign_key": 2,
    }
    # Old FKs are dropped via their exact (verified) constant identifiers.
    for ident in OLD_IDENTIFIERS:
        assert ident in upgrade_src
    for ident in NEW_IDENTIFIERS:
        assert ident in upgrade_src


def test_migration_downgrade_only_reverses_new_constraints() -> None:
    _, _, downgrade_src = migration_bodies()
    ops = Counter(re.findall(r"op\.(\w+)\(", downgrade_src))
    assert ops == {
        "drop_constraint": 4,      # 2 composite FKs + 2 backing uniques
        "create_foreign_key": 2,  # 2 original single-column FKs
    }
    # It restores exactly the two old FKs and removes exactly the four new
    # constraints introduced by upgrade().
    for ident in OLD_IDENTIFIERS:
        assert ident in downgrade_src
    for ident in NEW_IDENTIFIERS:
        assert ident in downgrade_src


def test_migration_has_no_destructive_or_dml_operations() -> None:
    src, upgrade_src, downgrade_src = migration_bodies()
    forbidden_ops = {
        "drop_table", "create_table", "drop_index", "create_index",
        "add_column", "drop_column", "alter_column", "execute",
        "bulk_insert", "drop_schema", "create_schema",
    }
    for body in (upgrade_src, downgrade_src):
        used = set(re.findall(r"op\.(\w+)\(", body))
        assert not (used & forbidden_ops)
        assert "sa.text(" not in body  # no raw / ad-hoc SQL
    # No credentials anywhere in the migration file.
    for forbidden in ("SUPABASE_", "password=", "postgresql://", "postgres://"):
        assert forbidden.lower() not in src.lower()


def test_migration_revision_discovery() -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    script = ScriptDirectory.from_config(cfg)

    assert script.get_heads() == [GOAL07_REVISION]  # exactly one head

    rev = script.get_revision(GOAL07_REVISION)
    assert rev is not None
    assert rev.down_revision == GOAL06_REVISION
    assert rev.is_head is True

    assert [r.revision for r in script.walk_revisions()] == [
        GOAL07_REVISION, GOAL06_REVISION,
    ]


def test_goal07_migration_module_is_importable_and_pure() -> None:
    m = load_goal07_module()
    assert m.revision == GOAL07_REVISION
    assert m.down_revision == GOAL06_REVISION
    assert callable(m.upgrade)
    assert callable(m.downgrade)


# --- 6. Existing schema constraints remain present -------------------------

@pytest.mark.parametrize("name", sorted({
    "channels", "videos", "content_rights",
    "transcript_assets", "transcript_segments",
}))
def test_primary_keys_still_on_id(name: str) -> None:
    pk = list(tbl(name).primary_key.columns)
    assert len(pk) == 1 and pk[0].name == "id"


def test_original_unique_constraints_still_present() -> None:
    assert unique_by_name("channels", "uq_channels_youtube_channel_id")
    assert unique_by_name("videos", "uq_videos_youtube_video_id")
    seg = unique_by_name("transcript_segments", "uq_transcript_segments_asset_start")
    assert set(seg.columns.keys()) == {"transcript_asset_id", "start_seconds"}


def test_original_check_constraints_still_present() -> None:
    checks = {
        c.name for c in tbl("transcript_segments").constraints
        if isinstance(c, CheckConstraint)
    }
    assert checks == {
        "ck_transcript_segments_start_nonnegative",
        "ck_transcript_segments_end_after_start",
        "ck_transcript_segments_text_nonempty",
    }


# --- 7. Cross-video association is impossible (structural proof) -----------

def _fk_pairs(name: str, constraint_name: str) -> dict[str, str]:
    """Map each local column to its remote target for a composite FK."""
    fk = fk_by_name(name, constraint_name)
    return {
        e.parent.name: f"{e.column.table.fullname}.{e.column.name}"
        for e in fk.elements
    }


def test_asset_fk_pins_video_id_to_content_right_video_id() -> None:
    # Because video_id is part of the referenced pair, a child row can only
    # match a content_rights row whose video_id equals the child's video_id.
    assert _fk_pairs(
        "transcript_assets", FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO
    ) == {
        "content_right_id": "voxsearch.content_rights.id",
        "video_id": "voxsearch.content_rights.video_id",
    }


def test_segment_fk_pins_video_id_to_asset_video_id() -> None:
    assert _fk_pairs(
        "transcript_segments", FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO
    ) == {
        "transcript_asset_id": "voxsearch.transcript_assets.id",
        "video_id": "voxsearch.transcript_assets.video_id",
    }


def _composite_key_matches(child: dict, local_cols: list[str],
                           parent_keys: set[tuple]) -> bool:
    """Pure-Python model of a composite FK key lookup (no DB needed)."""
    return tuple(child[c] for c in local_cols) in parent_keys


def test_mismatched_video_cannot_reference_another_videos_content_right() -> None:
    fk = fk_by_name("transcript_assets", FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO)
    local = [c.name for c in fk.columns]
    CR_A, VID_A, VID_B = "cr-A", "video-A", "video-B"
    parent_keys = {(CR_A, VID_A)}  # content_rights(id, video_id)

    same_video = {"content_right_id": CR_A, "video_id": VID_A}
    other_video = {"content_right_id": CR_A, "video_id": VID_B}

    assert _composite_key_matches(same_video, local, parent_keys)
    assert not _composite_key_matches(other_video, local, parent_keys)


def test_mismatched_video_cannot_reference_another_videos_asset() -> None:
    fk = fk_by_name("transcript_segments", FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO)
    local = [c.name for c in fk.columns]
    ASSET_A, VID_A, VID_B = "asset-A", "video-A", "video-B"
    parent_keys = {(ASSET_A, VID_A)}  # transcript_assets(id, video_id)

    same_video = {"transcript_asset_id": ASSET_A, "video_id": VID_A}
    other_video = {"transcript_asset_id": ASSET_A, "video_id": VID_B}

    assert _composite_key_matches(same_video, local, parent_keys)
    assert not _composite_key_matches(other_video, local, parent_keys)
