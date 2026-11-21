"""Offline tests for ``VideoChunkPersistenceService`` (Goal 16).

The 10 mandated cases are covered with a ``MagicMock`` session + repository.
NO database, NO SQLite (the ``vector`` UDT and PG-only ``ON CONFLICT`` make it
unsuitable), NO ``.env``. Transaction commit/rollback boundaries are asserted
explicitly. Live PostgreSQL + pgvector insertion is NOT VERIFIED here; it is
covered by the opt-in, skipped-by-default ``live_pgvector`` test.
"""
import math
import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from app.embeddings.contracts import DENSE_DIMENSION, EmbeddingBatch, EmbeddingVector
from app.repositories.video_chunks import VideoChunkRepository
from app.services import VoxSearchServiceError
from app.services.video_chunks import (
    EmbeddingChunkAlignmentError,
    EmbeddingPersistenceValidationError,
    InvalidChunkParentError,
    VectorPersistenceError,
    VideoChunkPersistenceService,
)
from app.transcripts.chunking import Chunk
from app.transcripts.sentence_segmentation import TimestampPrecision

# A genuine 1024-dim finite UNIT vector.
UNIT = (1.0,) + (0.0,) * (DENSE_DIMENSION - 1)
MODEL = "BAAI/bge-m3"


def _parents():
    return (uuid.uuid4(), uuid.uuid4(), uuid.uuid4())


def _chunk(
    index: int = 0,
    parents=None,
    text: str | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    duration_ms: int | None = None,
) -> Chunk:
    if parents is None:
        parents = _parents()
    if text is None:
        text = f"verbatim chunk text {index}"
    s_ms = index * 1000 if start_ms is None else start_ms
    e_ms = index * 1000 + 1000 if end_ms is None else end_ms
    d_ms = 1000 if duration_ms is None else duration_ms
    return Chunk(
        index=index,
        text=text,
        sentences=(),
        start_ms=s_ms,
        end_ms=e_ms,
        duration_ms=d_ms,
        timestamp_precision=TimestampPrecision.SEGMENT_LEVEL,
        word_count=2,
        evidence_start=0,
        source_segment_refs=("seg-1",),
        boundary_reason="target_reached",
        diagnostics=(),
        overlap_sentence_indices=(),
        video_id=parents[0],
        transcript_asset_id=parents[1],
        content_right_id=parents[2],
    )


def _unit_vec(first: float = 1.0) -> tuple[float, ...]:
    """A finite vector with L2 norm exactly ``first`` (rest are zero)."""
    return (first,) + (0.0,) * (DENSE_DIMENSION - 1)


def _batch(vectors=None, *, model_id: str = MODEL,
           dimension: int = DENSE_DIMENSION) -> EmbeddingBatch:
    if vectors is None:
        vectors = (EmbeddingVector(UNIT),)
    return EmbeddingBatch(
        vectors=tuple(vectors), model_id=model_id, dimension=dimension
    )


def _service():
    session = MagicMock()
    repo = MagicMock()
    return VideoChunkPersistenceService(session, repo=repo), session, repo


def _rows(repo):
    assert repo.upsert_batch.call_count == 1
    return repo.upsert_batch.call_args[0][0]


# --- 0. Error taxonomy ------------------------------------------------------

@pytest.mark.parametrize("exc", [
    VectorPersistenceError, InvalidChunkParentError,
    EmbeddingChunkAlignmentError, EmbeddingPersistenceValidationError,
])
def test_typed_errors_subclass_voxsearch_service_error(exc) -> None:
    assert issubclass(exc, VoxSearchServiceError)


# --- 1. Valid 1024-dimensional insertion ------------------------------------

def test_valid_1024_dim_insertion() -> None:
    svc, session, repo = _service()
    parents = _parents()
    n = svc.persist([_chunk(parents=parents)], _batch())
    assert n == 1
    session.commit.assert_called_once()
    session.rollback.assert_not_called()
    rows = _rows(repo)
    assert len(rows) == 1
    row = rows[0]
    assert len(row["embedding"]) == DENSE_DIMENSION
    assert math.isclose(math.hypot(*row["embedding"]), 1.0, abs_tol=1e-9)
    assert isinstance(row["id"], uuid.UUID)
    assert row["model_id"] == MODEL
    assert row["transcript_asset_id"] == parents[1]
    assert row["video_id"] == parents[0]
    assert row["content_right_id"] == parents[2]


# --- 2. Multiple embeddings in a batch --------------------------------------

def test_batch_of_multiple_embeddings() -> None:
    svc, session, repo = _service()
    parents = _parents()
    chunks = [_chunk(i, parents=parents) for i in range(3)]
    vectors = tuple(EmbeddingVector(_unit_vec(1.0)) for _ in range(3))
    n = svc.persist(chunks, _batch(vectors))
    assert n == 3
    rows = _rows(repo)  # exactly ONE upsert_batch call -> one statement
    assert len(rows) == 3
    assert [r["chunk_index"] for r in rows] == [0, 1, 2]
    # Positional chunk<->embedding/text alignment.
    for i, r in enumerate(rows):
        assert r["text"] == f"verbatim chunk text {i}"
        assert r["start_ms"] == i * 1000
    session.commit.assert_called_once()


def test_batch_is_atomic_one_repository_call() -> None:
    svc, _, repo = _service()
    parents = _parents()
    chunks = [_chunk(i, parents=parents) for i in range(4)]
    svc.persist(chunks, _batch(tuple(EmbeddingVector(UNIT) for _ in range(4))))
    assert repo.upsert_batch.call_count == 1
    assert len(_rows(repo)) == 4


# --- 3. Invalid vector dimension rejection ----------------------------------

def test_invalid_batch_dimension_is_rejected() -> None:
    svc, session, repo = _service()
    with pytest.raises(EmbeddingChunkAlignmentError):
        svc.persist([_chunk()], _batch(dimension=999))
    session.rollback.assert_called_once()
    session.commit.assert_not_called()
    repo.upsert_batch.assert_not_called()


def test_short_vector_is_rejected() -> None:
    svc, session, repo = _service()
    with pytest.raises(EmbeddingPersistenceValidationError):
        svc.persist([_chunk()], _batch((_unit_vec(1.0)[:11],)))
    session.rollback.assert_called_once()
    session.commit.assert_not_called()
    repo.upsert_batch.assert_not_called()


# --- 4. NaN / Infinity rejection --------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_inf_are_rejected(bad) -> None:
    svc, session, repo = _service()
    vec = (1.0, bad) + (0.0,) * (DENSE_DIMENSION - 2)
    with pytest.raises(EmbeddingPersistenceValidationError):
        svc.persist([_chunk()], _batch((vec,)))
    session.rollback.assert_called_once()
    session.commit.assert_not_called()
    repo.upsert_batch.assert_not_called()  # nothing persisted


# --- 5. Normalization validation (no silent repair) ------------------------

@pytest.mark.parametrize("first", [0.5, 2.0, 0.0])
def test_non_normalized_vector_is_rejected_not_repaired(first) -> None:
    svc, session, repo = _service()
    with pytest.raises(EmbeddingPersistenceValidationError) as ei:
        svc.persist([_chunk()], _batch((_unit_vec(first),)))
    assert "L2 norm" in str(ei.value)
    repo.upsert_batch.assert_not_called()  # never silently rescaled/persisted
    session.rollback.assert_called_once()
    session.commit.assert_not_called()


def test_within_tolerance_vector_is_stored_verbatim_not_rescaled() -> None:
    """A near-unit vector inside the documented tolerance is accepted AS-IS."""
    svc, _, repo = _service()
    slightly_off = 0.9995  # |norm - 1.0| = 5e-4 < 1e-3
    svc.persist([_chunk()], _batch((_unit_vec(slightly_off),)))
    stored = _rows(repo)[0]["embedding"]
    assert stored[0] == slightly_off  # NOT rescaled to 1.0
    assert math.hypot(*stored) == pytest.approx(slightly_off)


# --- 6. Chunk -> embedding relationship integrity --------------------------

def test_parent_and_text_integrity_is_preserved() -> None:
    svc, _, repo = _service()
    parents = _parents()
    chunk = _chunk(7, parents=parents, text="  exact verbatim text  ")
    svc.persist([chunk], _batch())
    row = _rows(repo)[0]
    assert row["video_id"] == parents[0]
    assert row["transcript_asset_id"] == parents[1]
    assert row["content_right_id"] == parents[2]
    assert row["chunk_index"] == 7
    assert row["text"] == "  exact verbatim text  "  # verbatim, unstripped
    assert row["timestamp_precision"] == "segment_level"
    assert row["source_segment_refs"] == ["seg-1"]
    assert row["boundary_reason"] == "target_reached"


# --- 7. Transaction rollback on failure -------------------------------------

def test_rollback_on_repository_failure() -> None:
    svc, session, repo = _service()
    repo.upsert_batch.side_effect = SQLAlchemyError("db down")
    with pytest.raises(SQLAlchemyError):
        svc.persist([_chunk()], _batch())
    session.rollback.assert_called_once()
    session.commit.assert_not_called()


def test_rollback_on_domain_validation_failure() -> None:
    svc, session, repo = _service()
    with pytest.raises(EmbeddingPersistenceValidationError):
        svc.persist([_chunk()], _batch((_unit_vec(0.5),)))
    session.rollback.assert_called_once()
    session.commit.assert_not_called()


# --- 8. Duplicate insertion policy (idempotent upsert) --------------------

def test_repeated_insertion_uses_idempotent_upsert() -> None:
    """Two persists of the SAME key both emit ON CONFLICT DO UPDATE."""
    session = MagicMock()
    svc = VideoChunkPersistenceService(
        session, repo=VideoChunkRepository(session)
    )
    parents = _parents()
    chunk = _chunk(0, parents=parents)
    batch = _batch()
    assert svc.persist([chunk], batch) == 1
    assert svc.persist([chunk], batch) == 1  # duplicate -> refresh, no 2nd row
    assert session.execute.call_count == 2
    for call in session.execute.call_args_list:
        sql = str(call[0][0].compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT (transcript_asset_id, chunk_index, model_id)" in sql
        assert "DO UPDATE SET" in sql


# --- 9. Alignment / parent fail-closed --------------------------------------

def test_count_mismatch_is_rejected() -> None:
    svc, session, repo = _service()
    parents = _parents()
    with pytest.raises(EmbeddingChunkAlignmentError):
        svc.persist([_chunk(0, parents), _chunk(1, parents)], _batch())
    session.rollback.assert_called_once()
    repo.upsert_batch.assert_not_called()


def test_non_embedding_batch_is_rejected() -> None:
    svc, _, _ = _service()
    with pytest.raises(EmbeddingChunkAlignmentError):
        svc.persist([_chunk()], MagicMock())  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_parents", [
    ("not-a-uuid", uuid.uuid4(), uuid.uuid4()),
    (uuid.uuid4(), None, uuid.uuid4()),
    (uuid.uuid4(), uuid.uuid4(), "also-not-a-uuid"),
])
def test_non_uuid_parent_is_rejected(bad_parents) -> None:
    svc, session, repo = _service()
    with pytest.raises(InvalidChunkParentError):
        svc.persist([_chunk(parents=bad_parents)], _batch())
    session.rollback.assert_called_once()
    repo.upsert_batch.assert_not_called()


def test_mixed_lineage_in_one_batch_is_rejected() -> None:
    svc, session, repo = _service()
    chunks = [_chunk(0, _parents()), _chunk(1, _parents())]
    vectors = tuple(EmbeddingVector(UNIT) for _ in range(2))
    with pytest.raises(InvalidChunkParentError):
        svc.persist(chunks, _batch(vectors))
    session.rollback.assert_called_once()
    repo.upsert_batch.assert_not_called()


def test_explicit_model_mismatch_is_rejected() -> None:
    svc, session, repo = _service()
    with pytest.raises(EmbeddingChunkAlignmentError):
        svc.persist([_chunk()], _batch(), model_id="different-model")
    session.rollback.assert_called_once()
    repo.upsert_batch.assert_not_called()


def test_explicit_matching_model_id_is_accepted() -> None:
    svc, _, repo = _service()
    assert svc.persist([_chunk()], _batch(), model_id=MODEL) == 1
    assert _rows(repo)[0]["model_id"] == MODEL


def test_duplicate_chunk_index_in_same_batch_is_rejected() -> None:
    svc, session, repo = _service()
    parents = _parents()
    chunks = [_chunk(0, parents), _chunk(0, parents)]
    vectors = tuple(EmbeddingVector(UNIT) for _ in range(2))
    with pytest.raises(EmbeddingChunkAlignmentError, match="duplicate chunk_index"):
        svc.persist(chunks, _batch(vectors))
    session.rollback.assert_called_once()
    repo.upsert_batch.assert_not_called()


@pytest.mark.parametrize("bad_index", [-1, -10, True, False, 1.5, "0"])
def test_invalid_chunk_index_is_rejected(bad_index) -> None:
    svc, session, repo = _service()
    parents = _parents()
    chunk = _chunk(0, parents)
    object.__setattr__(chunk, "index", bad_index)
    with pytest.raises(EmbeddingChunkAlignmentError, match="chunk_index"):
        svc.persist([chunk], _batch())
    session.rollback.assert_called_once()
    repo.upsert_batch.assert_not_called()


@pytest.mark.parametrize("start,end,dur", [
    (-1, 1000, 1000),      # start_ms < 0
    (1000, 1000, 0),       # end_ms == start_ms
    (2000, 1000, 1000),     # end_ms < start_ms
    (0, 1000, -1),         # duration_ms < 0
    (True, 1000, 1000),    # bool instead of int
    (0, "1000", 1000),     # str instead of int
])
def test_invalid_time_window_is_rejected(start, end, dur) -> None:
    svc, session, repo = _service()
    parents = _parents()
    chunk = _chunk(0, parents, start_ms=start, end_ms=end, duration_ms=dur)
    with pytest.raises(EmbeddingChunkAlignmentError, match="invalid time window"):
        svc.persist([chunk], _batch())
    session.rollback.assert_called_once()
    repo.upsert_batch.assert_not_called()


# --- 10. Empty batch / default repository ----------------------------------

def test_empty_batch_returns_zero() -> None:
    svc, session, repo = _service()
    assert svc.persist([], _batch(())) == 0
    session.commit.assert_called_once()
    session.rollback.assert_not_called()
    repo.upsert_batch.assert_called_once_with([])


def test_default_repository_is_used_when_none_injected() -> None:
    session = MagicMock()
    svc = VideoChunkPersistenceService(session)  # no repo -> builds real one
    assert svc.persist([_chunk()], _batch()) == 1
    assert session.execute.call_count == 1
    session.commit.assert_called_once()
