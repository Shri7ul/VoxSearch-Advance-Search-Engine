"""Offline tests for ``VideoChunkRepository`` (Goal 16).

A ``MagicMock`` session captures the emitted SQL statement; it is compiled
against the PostgreSQL dialect WITHOUT a connection. No database, no pgvector
package, no ``.env``. These tests verify statement SHAPE and transaction
ownership, NOT live insert behaviour.
"""
import uuid

from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import Insert, Select

from app.repositories import VideoChunkRepository
from app.repositories.video_chunks import VideoChunkRepository as DirectImport


def _row(idx: int = 0) -> dict:
    return {
        "id": uuid.uuid4(),
        "video_id": uuid.uuid4(),
        "transcript_asset_id": uuid.uuid4(),
        "content_right_id": uuid.uuid4(),
        "chunk_index": idx,
        "text": f"chunk {idx}",
        "start_ms": idx * 1000,
        "end_ms": idx * 1000 + 1000,
        "duration_ms": 1000,
        "timestamp_precision": "segment_level",
        "word_count": 2,
        "boundary_reason": "target_reached",
        "source_segment_refs": [],
        "overlap_sentence_indices": [],
        "diagnostics": [],
        "model_id": "BAAI/bge-m3",
        "embedding": [1.0] + [0.0] * 1023,
    }


def _compiled(session):
    assert session.execute.call_count == 1
    stmt = session.execute.call_args[0][0]
    return stmt, str(stmt.compile(dialect=postgresql.dialect()))


# --- 1. Re-export -----------------------------------------------------------

def test_repository_is_re_exported() -> None:
    assert VideoChunkRepository is DirectImport


# --- 2. upsert_batch is a single idempotent upsert --------------------------

def test_upsert_batch_emits_insert_on_conflict_do_update() -> None:
    from unittest.mock import MagicMock

    session = MagicMock()
    VideoChunkRepository(session).upsert_batch([_row()])
    stmt, sql = _compiled(session)

    assert isinstance(stmt, Insert)
    assert "INSERT INTO voxsearch.video_chunks" in sql
    assert "ON CONFLICT (transcript_asset_id, chunk_index, model_id)" in sql
    assert "DO UPDATE SET" in sql
    assert "excluded.embedding" in sql
    assert "now()" in sql  # explicit updated_at refresh
    # CAST guard for the pgvector bind.
    assert "CAST(" in sql and "AS vector(1024))" in sql


def test_upsert_batch_refreshes_all_mutable_columns() -> None:
    from unittest.mock import MagicMock

    session = MagicMock()
    VideoChunkRepository(session).upsert_batch([_row()])
    _, sql = _compiled(session)
    for col in ("embedding", "text", "start_ms", "end_ms", "duration_ms",
                "word_count", "video_id", "content_right_id"):
        assert f"{col} = excluded.{col}" in sql


def test_upsert_batch_does_not_commit_or_flush() -> None:
    from unittest.mock import MagicMock

    session = MagicMock()
    VideoChunkRepository(session).upsert_batch([_row()])
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    session.flush.assert_not_called()


def test_upsert_batch_uses_one_statement_for_multiple_rows() -> None:
    from unittest.mock import MagicMock

    session = MagicMock()
    VideoChunkRepository(session).upsert_batch([_row(0), _row(1), _row(2)])
    _, sql = _compiled(session)  # exactly ONE execute call
    assert sql.count("CAST(") == 3  # one CAST guard per row
    assert "ON CONFLICT" in sql


def test_upsert_batch_empty_emits_nothing() -> None:
    from unittest.mock import MagicMock

    session = MagicMock()
    VideoChunkRepository(session).upsert_batch([])
    session.execute.assert_not_called()
    session.flush.assert_not_called()
    session.commit.assert_not_called()


# --- 3. list_by_asset (read-only) ------------------------------------------

def test_list_by_asset_builds_ordered_select() -> None:
    from unittest.mock import MagicMock

    session = MagicMock()
    session.scalars.return_value.all.return_value = []
    asset = uuid.uuid4()
    result = VideoChunkRepository(session).list_by_asset(asset)
    assert result == []
    stmt = session.scalars.call_args[0][0]
    assert isinstance(stmt, Select)
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "voxsearch.video_chunks" in sql
    assert "ORDER BY" in sql
    assert "chunk_index" in sql
    assert "video_chunks.id" in sql
    session.commit.assert_not_called()
    session.flush.assert_not_called()


def test_list_by_asset_optionally_filters_model_id() -> None:
    from unittest.mock import MagicMock

    session = MagicMock()
    session.scalars.return_value.all.return_value = []
    VideoChunkRepository(session).list_by_asset(
        uuid.uuid4(), model_id="BAAI/bge-m3"
    )
    stmt = session.scalars.call_args[0][0]
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "model_id" in sql
