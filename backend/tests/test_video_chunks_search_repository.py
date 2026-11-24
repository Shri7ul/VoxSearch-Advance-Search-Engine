"""Offline tests for ``VideoChunkRepository.search_similar`` (Goal 17).

Verifies the SQL compilation, `<=>` cosine distance operator, deterministic
ranking (distance ASC, chunk_index ASC, id ASC), omitted embedding column, safe
parameterization, filter clauses, and result mapping.
"""
from dataclasses import dataclass, fields
from typing import Any
import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql

from app.repositories.video_chunks import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    ChunkSearchResult,
    VideoChunkRepository,
    validate_top_k,
)

DUMMY_QUERY_VECTOR = (1.0,) + (0.0,) * 1023


def _compiled(session):
    assert session.execute.call_count == 1
    stmt = session.execute.call_args[0][0]
    compiled = stmt.compile(dialect=postgresql.dialect())
    return stmt, str(compiled), compiled


# --- 1. SQL Compilation & Operator -----------------------------------------

def test_search_similar_emits_cosine_operator_and_cast() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    repo = VideoChunkRepository(session)

    repo.search_similar(DUMMY_QUERY_VECTOR, top_k=10)
    stmt, sql, compiled = _compiled(session)

    assert isinstance(stmt, Select)
    # Cosine distance operator with pgvector cast guard
    assert "embedding <=> CAST(" in sql
    assert "AS vector(1024)) AS distance" in sql
    # Ordering: distance ascending with deterministic tiebreakers
    assert "ORDER BY distance ASC, voxsearch.video_chunks.chunk_index ASC, voxsearch.video_chunks.id ASC" in sql
    # Limit
    assert "LIMIT" in sql


def test_search_similar_joins_videos_for_youtube_id() -> None:
    """Goal 32 needs the existing videos.youtube_video_id column."""
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR)
    _, sql, _ = _compiled(session)
    assert "JOIN voxsearch.videos" in sql
    assert "youtube_video_id" in sql


def test_search_similar_does_not_select_embedding_column() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    repo = VideoChunkRepository(session)

    repo.search_similar(DUMMY_QUERY_VECTOR, top_k=10)
    stmt, sql, _ = _compiled(session)

    # Selected columns should include metadata, text, and distance, but NOT the 1024-dim embedding
    selected_names = [getattr(c, "name", None) or getattr(c, "key", None) for c in stmt.selected_columns]
    assert "embedding" not in selected_names
    assert "distance" in selected_names
    assert "text" in selected_names
    assert "chunk_index" in selected_names
    assert "start_ms" in selected_names


def test_search_similar_is_safely_parameterized() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    repo = VideoChunkRepository(session)

    repo.search_similar(DUMMY_QUERY_VECTOR, top_k=7)
    _, _, compiled = _compiled(session)

    # Param is bound and processor exists
    embedding_param = [k for k in compiled.params.keys() if "embedding" in k][0]
    assert embedding_param in compiled._bind_processors
    proc = compiled._bind_processors[embedding_param]
    formatted = proc(compiled.params[embedding_param])
    assert formatted.startswith("[1.0,0.0,")
    # Top-k limit is parameterized
    limit_params = [v for k, v in compiled.params.items() if v == 7]
    assert len(limit_params) == 1


# --- 2. Filter Behavior ----------------------------------------------------

def test_search_similar_no_filters_has_no_where_clause() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR)
    _, sql, _ = _compiled(session)
    assert "WHERE" not in sql


def test_search_similar_video_id_filter() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    vid = uuid.uuid4()
    VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR, video_id=vid)
    _, sql, compiled = _compiled(session)
    assert "WHERE voxsearch.video_chunks.video_id =" in sql
    param_val = [v for k, v in compiled.params.items() if v == vid]
    assert len(param_val) == 1


def test_search_similar_transcript_asset_id_filter() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    aid = uuid.uuid4()
    VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR, transcript_asset_id=aid)
    _, sql, compiled = _compiled(session)
    assert "WHERE voxsearch.video_chunks.transcript_asset_id =" in sql
    param_val = [v for k, v in compiled.params.items() if v == aid]
    assert len(param_val) == 1


def test_search_similar_model_id_filter() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR, model_id="BAAI/bge-m3")
    _, sql, compiled = _compiled(session)
    assert "WHERE voxsearch.video_chunks.model_id =" in sql
    param_val = [v for k, v in compiled.params.items() if v == "BAAI/bge-m3"]
    assert len(param_val) == 1


def test_search_similar_combined_filters() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    vid = uuid.uuid4()
    aid = uuid.uuid4()
    VideoChunkRepository(session).search_similar(
        DUMMY_QUERY_VECTOR,
        video_id=vid,
        transcript_asset_id=aid,
        model_id="BAAI/bge-m3",
    )
    _, sql, compiled = _compiled(session)
    assert "WHERE" in sql
    assert "voxsearch.video_chunks.video_id =" in sql
    assert "voxsearch.video_chunks.transcript_asset_id =" in sql
    assert "voxsearch.video_chunks.model_id =" in sql
    assert " AND " in sql


# --- 3. Result Mapping & Contracts -----------------------------------------

@dataclass
class _FakeRow:
    id: uuid.UUID
    video_id: uuid.UUID
    transcript_asset_id: uuid.UUID
    content_right_id: uuid.UUID
    chunk_index: int
    text: str
    start_ms: int
    end_ms: int
    duration_ms: int
    timestamp_precision: str
    word_count: int
    boundary_reason: str
    source_segment_refs: list[str]
    overlap_sentence_indices: list[int]
    diagnostics: list[str]
    model_id: str
    distance: float
    youtube_video_id: str = ""


def test_search_similar_maps_rows_into_chunk_search_results() -> None:
    session = MagicMock()
    cid = uuid.uuid4()
    vid = uuid.uuid4()
    aid = uuid.uuid4()
    rid = uuid.uuid4()
    row = _FakeRow(
        id=cid,
        video_id=vid,
        transcript_asset_id=aid,
        content_right_id=rid,
        chunk_index=3,
        text="matched transcript sentence",
        start_ms=3000,
        end_ms=7500,
        duration_ms=4500,
        timestamp_precision="segment_level",
        word_count=5,
        boundary_reason="target_reached",
        source_segment_refs=["seg-1", "seg-2"],
        overlap_sentence_indices=[2],
        diagnostics=["ok"],
        model_id="BAAI/bge-m3",
        distance=0.15,
        youtube_video_id="dQw4w9WgXcQ",
    )
    session.execute.return_value.all.return_value = [row]

    results = VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR, top_k=5)
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, ChunkSearchResult)
    assert res.id == cid
    assert res.video_id == vid
    assert res.transcript_asset_id == aid
    assert res.content_right_id == rid
    assert res.chunk_index == 3
    assert res.text == "matched transcript sentence"
    assert res.start_ms == 3000
    assert res.end_ms == 7500
    assert res.duration_ms == 4500
    assert res.start_seconds == 3.0
    assert res.end_seconds == 7.5
    assert res.duration_seconds == 4.5
    assert res.distance == pytest.approx(0.15)
    assert res.score == pytest.approx(0.85)  # 1.0 - 0.15
    assert res.source_segment_refs == ("seg-1", "seg-2")
    assert res.overlap_sentence_indices == (2,)
    assert res.diagnostics == ("ok",)
    assert res.youtube_video_id == "dQw4w9WgXcQ"


def test_search_similar_empty_result_returns_empty_list() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    results = VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR)
    assert results == []


def test_search_similar_is_read_only_no_commit_or_flush() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR)
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    session.flush.assert_not_called()


# --- 4. Parameterization robustness (durable public-API assertions) ----------

def test_search_similar_sql_uses_placeholder_not_literal() -> None:
    """The query vector must be a bound parameter, not an inline literal.

    Verifies via the compiled SQL string that the embedding value is emitted
    as a %(name)s placeholder inside CAST(... AS vector(1024)) rather than
    as a raw numeric literal.  This is the durable complement to the
    _bind_processors check and does not rely on SQLAlchemy private API.
    """
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    repo = VideoChunkRepository(session)
    repo.search_similar(DUMMY_QUERY_VECTOR, top_k=10)
    _, sql, compiled = _compiled(session)

    # Must have a %(name)s placeholder wrapped in CAST, not an inline literal.
    import re
    cast_placeholder = re.search(
        r"CAST\(%\(\w+\)s AS vector\(1024\)\) AS distance", sql
    )
    assert cast_placeholder is not None, (
        f"Expected CAST(%(...)s AS vector(1024)) placeholder in SQL; got:\n{sql}"
    )
    # Both the vector binding and the LIMIT must be parameterised (no literals).
    param_keys = list(compiled.params.keys())
    embedding_params = [k for k in param_keys if "embedding" in k]
    assert len(embedding_params) == 1, (
        f"Expected exactly one embedding param, got: {embedding_params}"
    )
    limit_params = [k for k in param_keys if compiled.params[k] == 10]
    assert len(limit_params) == 1, "LIMIT value must be a bound parameter"


# --- 5. Multi-row ordering & from_row contract -------------------------------

def test_search_similar_preserves_row_order_from_execute() -> None:
    """Row order returned by session.execute().all() is preserved verbatim.

    The repository applies ORDER BY in SQL; this test confirms that the
    Python-side from_row mapping does not accidentally reorder the results.
    """
    session = MagicMock()
    cid_a, cid_b = uuid.uuid4(), uuid.uuid4()
    vid = uuid.uuid4()
    aid = uuid.uuid4()
    rid = uuid.uuid4()

    def _make_row(cid: uuid.UUID, chunk_idx: int, text: str, dist: float) -> "_FakeRow":
        return _FakeRow(
            id=cid,
            video_id=vid,
            transcript_asset_id=aid,
            content_right_id=rid,
            chunk_index=chunk_idx,
            text=text,
            start_ms=0,
            end_ms=1000,
            duration_ms=1000,
            timestamp_precision="segment_level",
            word_count=1,
            boundary_reason="target_reached",
            source_segment_refs=[],
            overlap_sentence_indices=[],
            diagnostics=[],
            model_id="BAAI/bge-m3",
            distance=dist,
        )

    # Simulate two rows already ordered by the DB (closer first)
    row_closer = _make_row(cid_a, 0, "closer chunk", 0.1)
    row_farther = _make_row(cid_b, 1, "farther chunk", 0.9)
    session.execute.return_value.all.return_value = [row_closer, row_farther]

    results = VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR, top_k=10)
    assert len(results) == 2
    assert results[0].text == "closer chunk"
    assert results[0].distance == pytest.approx(0.1)
    assert results[1].text == "farther chunk"
    assert results[1].distance == pytest.approx(0.9)
    # Ordering is preserved: first result has smaller distance
    assert results[0].distance < results[1].distance


def test_search_similar_score_is_one_minus_distance() -> None:
    """score property == 1.0 - distance for all distance values in [0, 2]."""
    session = MagicMock()
    vid, aid, rid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    for distance_val, expected_score in [
        (0.0, 1.0),    # identical vectors
        (0.5, 0.5),    # partial similarity
        (1.0, 0.0),    # orthogonal vectors
        (1.5, -0.5),   # partially opposite
        (2.0, -1.0),   # opposite vectors (anti-parallel unit vectors)
    ]:
        row = _FakeRow(
            id=uuid.uuid4(),
            video_id=vid,
            transcript_asset_id=aid,
            content_right_id=rid,
            chunk_index=0,
            text="t",
            start_ms=0,
            end_ms=100,
            duration_ms=100,
            timestamp_precision="word_level",
            word_count=1,
            boundary_reason="target_reached",
            source_segment_refs=[],
            overlap_sentence_indices=[],
            diagnostics=[],
            model_id="BAAI/bge-m3",
            distance=distance_val,
        )
        session.execute.return_value.all.return_value = [row]
        results = VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR)
        assert len(results) == 1
        result = results[0]
        assert result.distance == pytest.approx(distance_val), (
            f"distance mismatch for input {distance_val}"
        )
        assert result.score == pytest.approx(expected_score), (
            f"score={result.score} expected {expected_score} for distance={distance_val}"
        )


def test_search_similar_from_row_handles_empty_json_arrays() -> None:
    """from_row maps empty JSON lists to empty tuples without error."""
    session = MagicMock()
    vid, aid, rid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = _FakeRow(
        id=uuid.uuid4(),
        video_id=vid,
        transcript_asset_id=aid,
        content_right_id=rid,
        chunk_index=0,
        text="empty json arrays",
        start_ms=0,
        end_ms=500,
        duration_ms=500,
        timestamp_precision="segment_level",
        word_count=1,
        boundary_reason="target_reached",
        source_segment_refs=[],      # empty list -> empty tuple
        overlap_sentence_indices=[],  # empty list -> empty tuple
        diagnostics=[],               # empty list -> empty tuple
        model_id="BAAI/bge-m3",
        distance=0.3,
    )
    session.execute.return_value.all.return_value = [row]
    results = VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR)
    assert len(results) == 1
    res = results[0]
    assert res.source_segment_refs == ()
    assert res.overlap_sentence_indices == ()
    assert res.diagnostics == ()


def test_search_similar_distance_coerced_to_float() -> None:
    """distance is always a Python float, even when the DB returns a Decimal or int."""
    from decimal import Decimal
    session = MagicMock()
    vid, aid, rid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = _FakeRow(
        id=uuid.uuid4(),
        video_id=vid,
        transcript_asset_id=aid,
        content_right_id=rid,
        chunk_index=0,
        text="decimal distance",
        start_ms=0,
        end_ms=200,
        duration_ms=200,
        timestamp_precision="word_level",
        word_count=1,
        boundary_reason="target_reached",
        source_segment_refs=[],
        overlap_sentence_indices=[],
        diagnostics=[],
        model_id="BAAI/bge-m3",
        distance=Decimal("0.123456"),  # psycopg2 may return Decimal for float8
    )
    session.execute.return_value.all.return_value = [row]
    results = VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR)
    assert len(results) == 1
    assert isinstance(results[0].distance, float)
    assert results[0].distance == pytest.approx(0.123456)


# --- 6. Repository-level top-k bounds (data-access boundary) -----------------

def test_search_similar_default_top_k_uses_bounded_default() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    VideoChunkRepository(session).search_similar(DUMMY_QUERY_VECTOR)
    _, sql, compiled = _compiled(session)
    assert "LIMIT" in sql
    assert any(value == DEFAULT_TOP_K for value in compiled.params.values())


@pytest.mark.parametrize("bad_k", [0, -1, -10, MAX_TOP_K + 1, 500])
def test_search_similar_rejects_out_of_bounds_top_k(bad_k: int) -> None:
    session = MagicMock()
    with pytest.raises(ValueError):
        VideoChunkRepository(session).search_similar(
            DUMMY_QUERY_VECTOR, top_k=bad_k
        )
    session.execute.assert_not_called()


@pytest.mark.parametrize("bad_k", [True, False, 1.5, "10", None])
def test_search_similar_rejects_non_integer_top_k(bad_k) -> None:
    session = MagicMock()
    with pytest.raises(ValueError):
        VideoChunkRepository(session).search_similar(
            DUMMY_QUERY_VECTOR, top_k=bad_k
        )
    session.execute.assert_not_called()


@pytest.mark.parametrize("good_k", [1, DEFAULT_TOP_K, MAX_TOP_K])
def test_validate_top_k_accepts_inclusive_bounds(good_k: int) -> None:
    assert validate_top_k(good_k) == good_k


# --- 7. Filtering precedes ranking and limit --------------------------------

def test_search_similar_filters_are_applied_before_order_and_limit() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    VideoChunkRepository(session).search_similar(
        DUMMY_QUERY_VECTOR,
        top_k=5,
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        model_id="BAAI/bge-m3",
    )
    _, sql, _ = _compiled(session)
    assert sql.index("WHERE") < sql.index("ORDER BY") < sql.index("LIMIT")


# --- 8. Result model and DB-boundary vector validation ----------------------

def test_chunk_search_result_has_no_embedding_field() -> None:
    names = {field.name for field in fields(ChunkSearchResult)}
    assert "embedding" not in names
    assert {"id", "text", "distance", "model_id"} <= names


@pytest.mark.parametrize(
    "bad_vec",
    [
        (1.0,) * 1023,                       # wrong dimension
        (float("nan"),) + (0.0,) * 1023,     # non-finite
        (float("inf"),) + (0.0,) * 1023,     # non-finite
    ],
)
def test_search_similar_bind_processor_rejects_invalid_vector(bad_vec) -> None:
    """The VECTOR(1024) bind guard rejects bad vectors at the DB boundary."""
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    VideoChunkRepository(session).search_similar(bad_vec)
    _, _, compiled = _compiled(session)
    embedding_param = [k for k in compiled.params if "embedding" in k][0]
    processor = compiled._bind_processors[embedding_param]
    with pytest.raises(ValueError):
        processor(compiled.params[embedding_param])
