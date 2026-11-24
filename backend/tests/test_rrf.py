"""Tests for Goal 24 — reciprocal rank fusion combined ranking."""
from __future__ import annotations

import uuid
from dataclasses import fields

import pytest

from app.repositories.lexical_search import LexicalSearchResult
from app.repositories.video_chunks import ChunkSearchResult
from app.services.hybrid_retrieval import HybridRetrievalResult
from app.services.rrf import (
    DEFAULT_RRF_K,
    RRFFusionError,
    RRFFusionService,
)


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


def _hybrid(dense_ids, lexical_ids) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        dense=tuple(_dense(cid, i) for i, cid in enumerate(dense_ids)),
        lexical=tuple(_lexical(cid, i) for i, cid in enumerate(lexical_ids)),
    )


# --- RRF math ---------------------------------------------------------------

def test_single_source_scores_use_one_over_k_plus_rank() -> None:
    ids = [uuid.uuid4(), uuid.uuid4()]
    fused = RRFFusionService().fuse(_hybrid(ids, []))
    assert [c.rrf_score for c in fused] == pytest.approx(
        [1.0 / (DEFAULT_RRF_K + 1), 1.0 / (DEFAULT_RRF_K + 2)]
    )
    assert all(c.source == "dense" for c in fused)


def test_document_in_both_lists_gets_summed_score() -> None:
    shared = uuid.uuid4()
    fused = RRFFusionService().fuse(_hybrid([shared], [shared]))
    expected = 1.0 / (DEFAULT_RRF_K + 1) + 1.0 / (DEFAULT_RRF_K + 1)
    assert len(fused) == 2
    assert all(c.rrf_score == pytest.approx(expected) for c in fused)
    assert all(c.chunk_id == shared for c in fused)


def test_custom_k_changes_scores() -> None:
    cid = uuid.uuid4()
    fused = RRFFusionService().fuse(_hybrid([cid], []), k=0)
    assert fused[0].rrf_score == pytest.approx(1.0)


@pytest.mark.parametrize("bad_k", [-1, True, False, 1.5, "60", None])
def test_invalid_k_is_rejected(bad_k) -> None:
    with pytest.raises(RRFFusionError):
        RRFFusionService().fuse(_hybrid([uuid.uuid4()], []), k=bad_k)


# --- Ordering and duplicates ------------------------------------------------

def test_combined_order_is_descending_rrf_score() -> None:
    a, b, c = (uuid.uuid4() for _ in range(3))
    fused = RRFFusionService().fuse(_hybrid([a, b], [c]))
    scores = [item.rrf_score for item in fused]
    assert scores == sorted(scores, reverse=True)


def test_ties_break_by_source_then_rank_then_id() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    # Both documents appear at ranks summing to the same RRF score.
    fused = RRFFusionService().fuse(_hybrid([a, b], [b, a]))
    assert len(fused) == 4
    assert all(
        fused[i].rrf_score == pytest.approx(fused[i + 1].rrf_score)
        for i in range(len(fused) - 1)
    )
    assert [c.source for c in fused] == ["dense", "dense", "lexical", "lexical"]


def test_duplicates_are_preserved_for_goal_25() -> None:
    """Goal 24 fuses ranking only; it must not remove duplicate chunks."""
    shared = uuid.uuid4()
    hybrid = _hybrid([shared], [shared])
    fused = RRFFusionService().fuse(hybrid)
    assert len(fused) == 2
    assert {c.source for c in fused} == {"dense", "lexical"}
    # The original two result sets are untouched.
    assert len(hybrid.dense) == 1 and len(hybrid.lexical) == 1


def test_empty_hybrid_returns_empty_list() -> None:
    assert RRFFusionService().fuse(_hybrid([], [])) == []


# --- Integrity --------------------------------------------------------------

def test_fused_candidates_expose_no_embedding() -> None:
    cid = uuid.uuid4()
    fused = RRFFusionService().fuse(_hybrid([cid], [cid]))
    for candidate in fused:
        assert "embedding" not in {f.name for f in fields(candidate.chunk)}
        assert candidate.source_rank == 1


def test_candidates_are_immutable_and_source_rank_is_one_based() -> None:
    fused = RRFFusionService().fuse(_hybrid([uuid.uuid4()], [uuid.uuid4()]))
    with pytest.raises(Exception):
        fused[0].rrf_score = 0.0
    assert sorted(c.source_rank for c in fused) == [1, 1]
