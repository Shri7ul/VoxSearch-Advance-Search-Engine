"""Offline tests for Goal 22 PostgreSQL full-text keyword retrieval."""
from __future__ import annotations

import uuid
from dataclasses import fields
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql

from app.repositories.lexical_search import (
    FTS_CONFIGURATION,
    LexicalSearchRepository,
    LexicalSearchResult,
)
from app.repositories.video_chunks import DEFAULT_TOP_K, MAX_TOP_K


def _compiled(session: MagicMock):
    assert session.execute.call_count == 1
    stmt = session.execute.call_args.args[0]
    compiled = stmt.compile(dialect=postgresql.dialect())
    return stmt, str(compiled), compiled


def _row() -> tuple:
    return (
        uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4(),
        0, "latency matters", 0, 100, 100, "BAAI/bge-m3", 0.123,
    )


# --- SQL safety -------------------------------------------------------------

def test_search_keywords_emits_fts_functions_and_parameterized_input() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    LexicalSearchRepository(session).search_keywords("latency")
    stmt, sql, compiled = _compiled(session)

    assert isinstance(stmt, Select)
    assert "to_tsvector" in sql
    assert "websearch_to_tsquery" in sql
    assert "@@" in sql
    assert "ts_rank" in sql
    assert FTS_CONFIGURATION in compiled.params.values()
    assert "latency" in compiled.params.values()
    # PostgreSQL's two-argument FTS functions require a regconfig, not text.
    assert "CAST(" in sql
    assert "REGCONFIG" in sql
    # User text must not be interpolated into the SQL string itself.
    assert "latency" not in sql


def test_search_keywords_does_not_select_embedding() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    LexicalSearchRepository(session).search_keywords("latency")
    _, sql, _ = _compiled(session)
    assert "embedding" not in sql


def test_search_keywords_orders_deterministically_and_limits() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    LexicalSearchRepository(session).search_keywords("latency", top_k=5)
    _, sql, compiled = _compiled(session)
    assert "ORDER BY rank DESC" in sql
    assert "voxsearch.video_chunks.chunk_index ASC" in sql
    assert "voxsearch.video_chunks.id ASC" in sql
    assert "LIMIT" in sql
    assert 5 in compiled.params.values()


def test_search_keywords_default_limit_is_top_ten() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    LexicalSearchRepository(session).search_keywords("latency")
    _, _, compiled = _compiled(session)
    assert DEFAULT_TOP_K in compiled.params.values()


def test_search_keywords_applies_filters_before_order_and_limit() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    LexicalSearchRepository(session).search_keywords(
        "latency",
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        model_id="BAAI/bge-m3",
    )
    _, sql, _ = _compiled(session)
    assert sql.index("WHERE") < sql.index("ORDER BY") < sql.index("LIMIT")
    assert sql.count("AND") >= 2


@pytest.mark.parametrize("bad_k", [0, -1, MAX_TOP_K + 1, True, 1.5])
def test_search_keywords_bounds_top_k(bad_k) -> None:
    session = MagicMock()
    with pytest.raises(ValueError):
        LexicalSearchRepository(session).search_keywords("latency", top_k=bad_k)
    session.execute.assert_not_called()


# --- Result mapping ---------------------------------------------------------

def test_search_keywords_maps_rows_without_embedding_field() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = [_row()]
    results = LexicalSearchRepository(session).search_keywords("latency")
    assert len(results) == 1
    assert results[0].text == "latency matters"
    assert results[0].rank == pytest.approx(0.123)
    assert "embedding" not in {field.name for field in fields(LexicalSearchResult)}


def test_search_keywords_empty_result_is_empty_list() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    assert LexicalSearchRepository(session).search_keywords("missing") == []
