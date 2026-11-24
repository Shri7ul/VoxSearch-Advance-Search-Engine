"""Tests for Goal 25 — duplicate removal over Goal 24 fused candidates."""
from __future__ import annotations

import uuid
from dataclasses import fields

import pytest

from app.repositories.lexical_search import LexicalSearchResult
from app.repositories.video_chunks import ChunkSearchResult
from app.services.deduplication import DeduplicationService
from app.services.rrf import FusedCandidate


def _dense(chunk_id: uuid.UUID, index: int = 0) -> ChunkSearchResult:
    return ChunkSearchResult(
        id=chunk_id,
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=index,
        text="dense chunk",
        start_ms=0,
        end_ms=1,
        duration_ms=1,
        timestamp_precision="word",
        word_count=2,
        boundary_reason="sentence",
        source_segment_refs=(),
        overlap_sentence_indices=(),
        diagnostics=(),
        model_id="BAAI/bge-m3",
        distance=0.2,
    )


def _lexical(chunk_id: uuid.UUID, index: int = 0) -> LexicalSearchResult:
    return LexicalSearchResult(
        id=chunk_id,
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=index,
        text="lexical chunk",
        start_ms=0,
        end_ms=1,
        duration_ms=1,
        model_id="BAAI/bge-m3",
        rank=1.0,
    )


def _candidate(
    source: str, chunk_id: uuid.UUID, score: float, rank: int = 1
) -> FusedCandidate:
    chunk = _dense(chunk_id) if source == "dense" else _lexical(chunk_id)
    return FusedCandidate(
        source=source, source_rank=rank, rrf_score=score, chunk=chunk
    )


# --- Duplicate removal ------------------------------------------------------

def test_cross_source_duplicate_is_removed_and_best_kept() -> None:
    shared = uuid.uuid4()
    candidates = [
        _candidate("dense", shared, 0.020),
        _candidate("lexical", shared, 0.033),
    ]
    clean = DeduplicationService.deduplicate(candidates)
    assert len(clean) == 1
    assert clean[0].source == "lexical"
    assert clean[0].rrf_score == pytest.approx(0.033)
    assert clean[0].chunk_id == shared


def test_duplicate_within_same_source_is_removed() -> None:
    chunk_id = uuid.uuid4()
    candidates = [
        _candidate("dense", chunk_id, 0.016, rank=2),
        _candidate("dense", chunk_id, 0.017, rank=1),
    ]
    clean = DeduplicationService.deduplicate(candidates)
    assert len(clean) == 1
    assert clean[0].source_rank == 1
    assert clean[0].rrf_score == pytest.approx(0.017)


def test_unique_candidates_are_unchanged_and_order_preserved() -> None:
    ids = [uuid.uuid4() for _ in range(3)]
    candidates = [_candidate("dense", cid, 0.03 - i, rank=i + 1) for i, cid in enumerate(ids)]
    clean = DeduplicationService.deduplicate(candidates)
    assert clean == candidates
    assert [c.chunk_id for c in clean] == ids


def test_kept_representatives_keep_input_relative_order() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    candidates = [
        _candidate("dense", a, 0.05),
        _candidate("dense", b, 0.04),
        _candidate("lexical", a, 0.06),  # better copy of `a`, appears last
    ]
    clean = DeduplicationService.deduplicate(candidates)
    assert [c.chunk_id for c in clean] == [a, b]
    assert clean[0].source == "lexical"


def test_tie_keeps_dense_before_lexical() -> None:
    chunk_id = uuid.uuid4()
    candidates = [
        _candidate("lexical", chunk_id, 0.02, rank=1),
        _candidate("dense", chunk_id, 0.02, rank=1),
    ]
    clean = DeduplicationService.deduplicate(candidates)
    assert len(clean) == 1
    assert clean[0].source == "dense"


def test_empty_input_returns_empty_list() -> None:
    assert DeduplicationService.deduplicate([]) == []


# --- Integrity and scope ----------------------------------------------------

def test_input_list_is_not_mutated() -> None:
    chunk_id = uuid.uuid4()
    candidates = [
        _candidate("dense", chunk_id, 0.02),
        _candidate("lexical", chunk_id, 0.03),
    ]
    original = list(candidates)
    DeduplicationService.deduplicate(candidates)
    assert candidates == original


def test_result_objects_are_reused_not_rebuilt() -> None:
    chunk_id = uuid.uuid4()
    best = _candidate("lexical", chunk_id, 0.03)
    clean = DeduplicationService.deduplicate(
        [_candidate("dense", chunk_id, 0.02), best]
    )
    assert clean[0] is best


def test_no_embedding_exposure_and_no_reranking_fields() -> None:
    clean = DeduplicationService.deduplicate([_candidate("dense", uuid.uuid4(), 0.02)])
    assert "embedding" not in {f.name for f in fields(clean[0].chunk)}
    assert clean[0].rrf_score == pytest.approx(0.02)
    assert not hasattr(clean[0], "rerank_score")
