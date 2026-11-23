"""Tests for Goal 20 — top-10 vector retrieval orchestration.

The test doubles stand in for the accepted Goal 17 search service and Goal 19
query embedder; no database, subprocess, model or embedding generation occurs.
"""
from __future__ import annotations

import uuid
from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.embeddings.contracts import EmbeddingVector
from app.repositories.video_chunks import DEFAULT_TOP_K, ChunkSearchResult
from app.services.retrieval import (
    InvalidRetrievalRequestError,
    VectorRetrievalService,
)
from app.services.video_chunks import VideoChunkSearchService

QUERY_VECTOR = [1.0 / 32.0] * 1024
EMBEDDED = EmbeddingVector(tuple([2.0 / 32.0] * 1024))


def _result(chunk_index: int) -> ChunkSearchResult:
    return ChunkSearchResult(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=chunk_index,
        text=f"chunk {chunk_index}",
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
        distance=float(chunk_index) / 100.0,
    )


def _service(
    results: list[ChunkSearchResult] | None = None,
) -> tuple[VectorRetrievalService, MagicMock, MagicMock]:
    search = MagicMock()
    search.search.return_value = results if results is not None else []
    embedder = MagicMock()
    embedder.embed.return_value = EMBEDDED
    return VectorRetrievalService(search, embedder), search, embedder


# --- 1. Top-10 policy -------------------------------------------------------

def test_default_top_k_is_ten_and_forwarded() -> None:
    service, search, embedder = _service()
    service.retrieve(query_vector=QUERY_VECTOR)
    assert service.default_top_k == DEFAULT_TOP_K == 10
    assert search.search.call_args.kwargs["top_k"] == 10
    embedder.embed.assert_not_called()


def test_explicit_top_k_is_forwarded_within_accepted_bounds() -> None:
    service, search, _ = _service()
    service.retrieve(query_vector=QUERY_VECTOR, top_k=7)
    assert search.search.call_args.kwargs["top_k"] == 7


# --- 2. Query embedding composition ----------------------------------------

def test_natural_language_query_is_embedded_then_searched() -> None:
    service, search, embedder = _service()
    service.retrieve(query="what was said about latency?")
    embedder.embed.assert_called_once_with("what was said about latency?")
    assert search.search.call_args.args[0] == EMBEDDED


def test_precomputed_vector_bypasses_embedder() -> None:
    service, search, embedder = _service()
    service.retrieve(query_vector=QUERY_VECTOR)
    embedder.embed.assert_not_called()
    assert list(search.search.call_args.args[0]) == QUERY_VECTOR


def test_filters_are_forwarded_unchanged() -> None:
    service, search, _ = _service()
    video_id, asset_id = uuid.uuid4(), uuid.uuid4()
    service.retrieve(
        query_vector=QUERY_VECTOR,
        video_id=video_id,
        transcript_asset_id=asset_id,
        model_id="BAAI/bge-m3",
    )
    kwargs = search.search.call_args.kwargs
    assert kwargs["video_id"] == video_id
    assert kwargs["transcript_asset_id"] == asset_id
    assert kwargs["model_id"] == "BAAI/bge-m3"


# --- 3. Exactly one query source -------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "text", "query_vector": QUERY_VECTOR},
        {},
    ],
)
def test_exactly_one_query_source_is_required(kwargs: dict[str, Any]) -> None:
    service, search, embedder = _service()
    with pytest.raises(InvalidRetrievalRequestError):
        service.retrieve(**kwargs)
    search.search.assert_not_called()
    embedder.embed.assert_not_called()


# --- 4. Results and errors --------------------------------------------------

def test_ranked_order_and_result_shape_are_preserved() -> None:
    expected = [_result(0), _result(1), _result(2)]
    service, _, _ = _service(expected)
    actual = service.retrieve(query_vector=QUERY_VECTOR)
    assert actual == expected
    assert [item.chunk_index for item in actual] == [0, 1, 2]
    assert "embedding" not in {field.name for field in fields(ChunkSearchResult)}


def test_empty_result_is_returned_as_empty_list() -> None:
    service, _, _ = _service([])
    assert service.retrieve(query_vector=QUERY_VECTOR) == []


def test_search_failure_is_propagated_unchanged() -> None:
    service, search, _ = _service()
    search.search.side_effect = RuntimeError("db unavailable")
    with pytest.raises(RuntimeError, match="db unavailable"):
        service.retrieve(query_vector=QUERY_VECTOR)


def test_embedder_failure_is_propagated_unchanged() -> None:
    service, search, embedder = _service()
    embedder.embed.side_effect = RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="worker failed"):
        service.retrieve(query="text")
    search.search.assert_not_called()


def test_default_embedder_is_built_from_goal_19(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no injection, retrieval builds and uses a Goal 19 embedder."""
    from app.services import retrieval as retrieval_module

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = EMBEDDED
    monkeypatch.setattr(
        retrieval_module, "QueryEmbeddingService", lambda: fake_embedder
    )
    search = MagicMock()
    VectorRetrievalService(search).retrieve(query="text")
    fake_embedder.embed.assert_called_once_with("text")
    assert search.search.call_args.args[0] == EMBEDDED
