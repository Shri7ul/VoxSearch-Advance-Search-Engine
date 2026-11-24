"""Tests for Goal 27 — candidate reranking / reordered results."""
from __future__ import annotations

import uuid
from dataclasses import fields
from typing import Any, Sequence

import pytest

from app.reranking.contracts import RerankerConfig
from app.reranking.scoring import (
    MAX_CANDIDATES,
    MAX_QUERY_CHARACTERS,
    CandidateRerankingService,
    InvalidRerankInputError,
)
from app.repositories.video_chunks import ChunkSearchResult
from app.services.rrf import FusedCandidate


class FakeScorer:
    def __init__(self, scores: Sequence[float] | None = None) -> None:
        self.scores = scores if scores is not None else []
        self.calls: list[list[list[str]]] = []

    def score(self, pairs, config: RerankerConfig) -> Sequence[float]:
        self.calls.append([list(pair) for pair in pairs])
        return self.scores


def _candidate(index: int) -> FusedCandidate:
    from app.services.rrf import DEFAULT_RRF_K

    chunk_id = uuid.uuid4()
    chunk = ChunkSearchResult(
        id=chunk_id,
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=index,
        text=f"chunk text {index}",
        start_ms=0,
        end_ms=1,
        duration_ms=1,
        timestamp_precision="word",
        word_count=3,
        boundary_reason="sentence",
        source_segment_refs=(),
        overlap_sentence_indices=(),
        diagnostics=(),
        model_id="BAAI/bge-m3",
        distance=0.2,
    )
    return FusedCandidate(
        source="dense",
        source_rank=index + 1,
        rrf_score=1.0 / (DEFAULT_RRF_K + index + 1),
        chunk=chunk,
    )


def _service(scores: Sequence[float] | None = None):
    scorer = FakeScorer(scores)
    return CandidateRerankingService(scorer), scorer


# --- Reordering -------------------------------------------------------------

def test_candidates_are_reordered_by_bge_score() -> None:
    candidates = [_candidate(i) for i in range(3)]
    service, scorer = _service([0.1, 0.9, 0.5])
    result = service.rerank("latency", candidates)
    assert [item.candidate for item in result] == [
        candidates[1], candidates[2], candidates[0]
    ]
    assert [item.score for item in result] == pytest.approx([0.9, 0.5, 0.1])
    assert [item.rank for item in result] == [1, 2, 3]


def test_original_candidate_objects_and_metadata_are_preserved() -> None:
    candidate = _candidate(0)
    service, _ = _service([0.7])
    result = service.rerank("query", [candidate])
    assert result[0].candidate is candidate
    assert result[0].candidate.rrf_score == candidate.rrf_score
    assert result[0].candidate.source == candidate.source
    assert result[0].chunk_id == candidate.chunk_id


def test_equal_scores_break_ties_by_original_order() -> None:
    candidates = [_candidate(i) for i in range(3)]
    service, _ = _service([0.5, 0.5, 0.5])
    result = service.rerank("query", candidates)
    assert [item.candidate for item in result] == candidates


def test_input_candidates_are_not_mutated() -> None:
    candidates = [_candidate(i) for i in range(2)]
    original = list(candidates)
    service, _ = _service([0.2, 0.8])
    service.rerank("query", candidates)
    assert candidates == original


def test_pairs_are_query_plus_verbatim_chunk_text() -> None:
    candidates = [_candidate(0), _candidate(1)]
    service, scorer = _service([0.4, 0.6])
    service.rerank("  my query  ", candidates)
    assert scorer.calls == [
        [["  my query  ", "chunk text 0"], ["  my query  ", "chunk text 1"]]
    ]


# --- Empty and validation ---------------------------------------------------

def test_empty_candidates_returns_empty_without_scoring() -> None:
    service, scorer = _service([])
    assert service.rerank("query", []) == []
    assert scorer.calls == []


@pytest.mark.parametrize("bad_query", ["", "   ", None, 123, True, b"x", ["q"], "x" * (MAX_QUERY_CHARACTERS + 1)])
def test_invalid_query_is_rejected(bad_query: Any) -> None:
    service, scorer = _service([0.5])
    with pytest.raises(InvalidRerankInputError):
        service.rerank(bad_query, [_candidate(0)])
    assert scorer.calls == []


def test_too_many_candidates_is_rejected() -> None:
    service, scorer = _service([0.5])
    with pytest.raises(InvalidRerankInputError):
        service.rerank("query", [_candidate(i) for i in range(MAX_CANDIDATES + 1)])
    assert scorer.calls == []


@pytest.mark.parametrize(
    "scores",
    [
        [],                       # count mismatch
        [0.5, 0.5],               # count mismatch
        [float("nan")],           # non-finite
        [float("inf")],           # non-finite
        ["0.5"],                  # non-numeric
        [True],                   # bool is not a score
    ],
)
def test_invalid_score_responses_are_rejected(scores: Any) -> None:
    service, _ = _service(scores)
    with pytest.raises(InvalidRerankInputError):
        service.rerank("query", [_candidate(0)])


# --- Failure and scope ------------------------------------------------------

def test_scorer_failure_propagates_without_fallback() -> None:
    service, scorer = _service([0.5])

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("reranker unavailable")

    scorer.score = fail  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="reranker unavailable"):
        service.rerank("query", [_candidate(0)])


def test_texts_override_changes_only_scoring_text() -> None:
    candidate = _candidate(0)
    service, scorer = _service([0.5])
    result = service.rerank(
        "query", [candidate], texts={candidate.chunk_id: "context-enriched text"}
    )
    assert scorer.calls == [[["query", "context-enriched text"]]]
    assert result[0].candidate is candidate
    assert result[0].candidate.chunk.text == "chunk text 0"


def test_missing_texts_key_falls_back_to_verbatim_chunk_text() -> None:
    candidate = _candidate(0)
    service, scorer = _service([0.5])
    service.rerank("query", [candidate], texts={uuid.uuid4(): "other"})
    assert scorer.calls == [[["query", "chunk text 0"]]]


@pytest.mark.parametrize(
    "texts",
    [
        ["not-a-mapping"],
        {True: "text"},
        {uuid.uuid4(): 123},
        {uuid.uuid4(): "   "},
    ],
)
def test_invalid_texts_mapping_is_rejected(texts: Any) -> None:
    service, scorer = _service([0.5])
    with pytest.raises(InvalidRerankInputError):
        service.rerank("query", [_candidate(0)], texts=texts)
    assert scorer.calls == []


def test_no_embedding_field_is_exposed() -> None:
    service, _ = _service([0.5])
    result = service.rerank("query", [_candidate(0)])
    assert "embedding" not in {field.name for field in fields(result[0].candidate.chunk)}
    assert "embedding" not in {field.name for field in fields(result[0])}
