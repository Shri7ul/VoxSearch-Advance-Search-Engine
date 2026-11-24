"""Tests for Goal 28 — bounded surrounding-context expansion."""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

import pytest

from app.reranking.context import (
    ContextAwareRerankingService,
    ContextChunk,
    ContextConfig,
    ContextExpansionService,
    ContextualCandidate,
)
from app.reranking.scoring import InvalidRerankInputError
from app.repositories.video_chunks import ChunkSearchResult
from app.services.rrf import FusedCandidate


def _candidate(index: int = 5, text: str = "target text") -> FusedCandidate:
    chunk = ChunkSearchResult(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=index,
        text=text,
        start_ms=500,
        end_ms=900,
        duration_ms=400,
        timestamp_precision="word",
        word_count=2,
        boundary_reason="sentence",
        source_segment_refs=("seg",),
        overlap_sentence_indices=(),
        diagnostics=(),
        model_id="BAAI/bge-m3",
        distance=0.2,
    )
    return FusedCandidate(
        source="dense", source_rank=1, rrf_score=0.016, chunk=chunk
    )


class FakeProvider:
    def __init__(self, chunks: Sequence[ContextChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else []
        self.calls: list[tuple[uuid.UUID, int]] = []

    def neighbors(
        self, candidate: FusedCandidate, *, radius: int
    ) -> Sequence[ContextChunk]:
        self.calls.append((candidate.chunk_id, radius))
        return self.chunks


def _chunk(
    index: int, text: str = "neighbor", target: FusedCandidate | None = None
) -> ContextChunk:
    target = target if target is not None else _candidate()
    chunk = target.chunk
    return ContextChunk(
        chunk_index=index,
        text=text,
        start_ms=0,
        end_ms=1,
        video_id=chunk.video_id,
        transcript_asset_id=chunk.transcript_asset_id,
        model_id=chunk.model_id,
    )


# --- Configuration ----------------------------------------------------------

@pytest.mark.parametrize("radius", [0, -1, 4, True, 1.5, "1"])
def test_invalid_radius_is_rejected(radius: Any) -> None:
    with pytest.raises(InvalidRerankInputError):
        ContextConfig(radius=radius)


@pytest.mark.parametrize("limit", [0, -1, 2001, True, "100"])
def test_invalid_context_limit_is_rejected(limit: Any) -> None:
    with pytest.raises(InvalidRerankInputError):
        ContextConfig(max_context_characters=limit)


# --- Deterministic context --------------------------------------------------

def test_context_is_sorted_bounded_and_verbatim() -> None:
    candidate = _candidate()
    provider = FakeProvider(
        [_chunk(6, "after", candidate), _chunk(4, "before", candidate)]
    )
    service = ContextExpansionService(provider)
    result = service.expand([candidate])
    assert len(result) == 1
    assert [c.chunk_index for c in result[0].context] == [4, 6]
    assert result[0].matching_text == "target text\nbefore\nafter"
    assert result[0].candidate is candidate
    assert result[0].chunk_id == candidate.chunk_id


def test_missing_and_empty_context_use_verbatim_target() -> None:
    candidate = _candidate()
    service = ContextExpansionService(FakeProvider([]))
    result = service.expand([candidate])
    assert result[0].context == ()
    assert result[0].matching_text == candidate.chunk.text


def test_context_characters_are_bounded() -> None:
    candidate = _candidate()
    provider = FakeProvider(
        [_chunk(4, "A" * 100, candidate), _chunk(6, "B" * 100, candidate)]
    )
    service = ContextExpansionService(
        provider, config=ContextConfig(max_context_characters=120)
    )
    result = service.expand([candidate])
    assert "B" not in result[0].matching_text
    assert len(result[0].matching_text) < len(candidate.chunk.text) + 200


def test_combined_text_never_exceeds_worker_text_limit() -> None:
    long_target = "T" * 8000
    candidate = _candidate(text=long_target)
    provider = FakeProvider(
        [_chunk(4, "N" * 500, candidate), _chunk(6, "M" * 500, candidate)]
    )
    result = ContextExpansionService(provider).expand([candidate])
    assert len(result[0].matching_text) <= 8192
    assert "T" * 8000 in result[0].matching_text


def test_no_context_budget_for_full_length_target() -> None:
    candidate = _candidate(text="T" * 8192)
    provider = FakeProvider([_chunk(4, "neighbor", candidate)])
    result = ContextExpansionService(provider).expand([candidate])
    assert result[0].matching_text == "T" * 8192
    assert result[0].context == ()


def test_radius_is_forwarded_to_provider() -> None:
    candidate = _candidate()
    provider = FakeProvider([])
    ContextExpansionService(
        provider, config=ContextConfig(radius=2)
    ).expand([candidate])
    assert provider.calls == [(candidate.chunk_id, 2)]


@pytest.mark.parametrize(
    "kind", ["target", "duplicate", "blank", "other_asset", "other_model"]
)
def test_invalid_context_is_rejected(kind: str) -> None:
    candidate = _candidate()
    if kind == "target":
        chunks = [_chunk(5, "target again", candidate)]
    elif kind == "duplicate":
        chunks = [_chunk(4, "a", candidate), _chunk(4, "b", candidate)]
    elif kind == "blank":
        chunks = [_chunk(4, "   ", candidate)]
    elif kind == "other_asset":
        chunks = [
            ContextChunk(
                4, "foreign", 0, 1,
                candidate.chunk.video_id, uuid.uuid4(), candidate.chunk.model_id,
            )
        ]
    else:
        chunks = [
            ContextChunk(
                4, "foreign", 0, 1,
                candidate.chunk.video_id,
                candidate.chunk.transcript_asset_id,
                "other-model",
            )
        ]
    service = ContextExpansionService(FakeProvider(chunks))
    with pytest.raises(InvalidRerankInputError):
        service.expand([candidate])


def test_empty_candidate_list_returns_empty() -> None:
    provider = FakeProvider([])
    assert ContextExpansionService(provider).expand([]) == []
    assert provider.calls == []


# --- Context-aware reranking ------------------------------------------------

def test_context_aware_reranking_preserves_original_candidates() -> None:
    candidate = _candidate()
    expansion = ContextExpansionService(FakeProvider([_chunk(4, "before", candidate)]))
    captured: dict[str, Any] = {}

    class FakeReranker:
        def rerank(self, query, candidates, *, texts=None):
            captured["query"] = query
            captured["candidates"] = candidates
            captured["texts"] = texts
            return ["reranked"]

    result = ContextAwareRerankingService(expansion, FakeReranker()).rerank(
        "query", [candidate]
    )
    assert result == ["reranked"]
    assert captured["candidates"] == [candidate]
    assert captured["candidates"][0] is candidate
    assert captured["texts"] == {
        candidate.chunk_id: "target text\nbefore"
    }


def test_no_context_does_not_add_text_override() -> None:
    candidate = _candidate()
    expansion = ContextExpansionService(FakeProvider([]))
    captured: dict[str, Any] = {}

    class FakeReranker:
        def rerank(self, query, candidates, *, texts=None):
            captured["texts"] = texts
            return []

    ContextAwareRerankingService(expansion, FakeReranker()).rerank(
        "query", [candidate]
    )
    assert captured["texts"] == {}


def test_contextual_candidate_is_immutable() -> None:
    contextual = ContextualCandidate(
        candidate=_candidate(), context=(), matching_text="target text"
    )
    with pytest.raises(Exception):
        contextual.matching_text = "other"  # type: ignore[misc]
