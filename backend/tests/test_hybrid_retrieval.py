"""Tests for Goal 23 — dense + lexical retrieval as two distinct result sets."""
from __future__ import annotations

import inspect
import uuid
from unittest.mock import MagicMock

import pytest

from app.repositories.lexical_search import LexicalSearchResult
from app.repositories.video_chunks import DEFAULT_TOP_K, ChunkSearchResult
from app.services.hybrid_retrieval import HybridRetrievalResult, HybridRetrievalService

SHARED_ID = uuid.uuid4()


def _dense_row(index: int) -> ChunkSearchResult:
    return ChunkSearchResult(
        id=SHARED_ID if index == 0 else uuid.uuid4(),
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=index,
        text=f"dense {index}",
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
        distance=float(index) / 10.0,
    )


def _lexical_row(index: int) -> LexicalSearchResult:
    return LexicalSearchResult(
        id=SHARED_ID if index == 0 else uuid.uuid4(),
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=index,
        text=f"lexical {index}",
        start_ms=0,
        end_ms=1,
        duration_ms=1,
        model_id="BAAI/bge-m3",
        rank=1.0 - index,
    )


def _service(dense=None, lexical=None):
    dense_service = MagicMock()
    dense_service.retrieve.return_value = dense if dense is not None else []
    lexical_service = MagicMock()
    lexical_service.search.return_value = lexical if lexical is not None else []
    return (
        HybridRetrievalService(dense_service, lexical_service),
        dense_service,
        lexical_service,
    )


def test_both_paths_receive_same_query_filters_and_top_k() -> None:
    service, dense, lexical = _service()
    video_id, asset_id = uuid.uuid4(), uuid.uuid4()
    service.retrieve(
        "keyword phrase",
        top_k=7,
        video_id=video_id,
        transcript_asset_id=asset_id,
        model_id="BAAI/bge-m3",
    )
    for mock in (dense.retrieve, lexical.search):
        kwargs = mock.call_args.kwargs
        assert kwargs["top_k"] == 7
        assert kwargs["video_id"] == video_id
        assert kwargs["transcript_asset_id"] == asset_id
        assert kwargs["model_id"] == "BAAI/bge-m3"
    assert dense.retrieve.call_args.kwargs["query"] == "keyword phrase"
    assert lexical.search.call_args.args[0] == "keyword phrase"


def test_two_result_sets_are_returned_unchanged_and_separate() -> None:
    dense_rows = [_dense_row(0), _dense_row(1)]
    lexical_rows = [_lexical_row(0), _lexical_row(1)]
    service, _, _ = _service(dense_rows, lexical_rows)
    result = service.retrieve("terms")
    assert isinstance(result, HybridRetrievalResult)
    assert result.dense == tuple(dense_rows)
    assert result.lexical == tuple(lexical_rows)
    assert result.dense[0].text == "dense 0"
    assert result.lexical[0].text == "lexical 0"


def test_no_fusion_ranking_or_deduplication() -> None:
    """Goal 23 stops at two result sets: a shared id stays in both (Goal 25)."""
    dense_rows = [_dense_row(0)]
    lexical_rows = [_lexical_row(0)]
    service, _, _ = _service(dense_rows, lexical_rows)
    result = service.retrieve("terms")
    assert result.dense[0].id == result.lexical[0].id == SHARED_ID
    assert not hasattr(result, "rrf_score")
    assert not hasattr(result, "combined")
    assert result.dense is not result.lexical


def test_default_top_k_is_ten_for_both_paths() -> None:
    service, dense, lexical = _service()
    service.retrieve("terms")
    assert DEFAULT_TOP_K == 10
    # The hybrid default must be the SAME shared constant used by Goals 17/20/22.
    assert inspect.signature(HybridRetrievalService.retrieve).parameters[
        "top_k"
    ].default is DEFAULT_TOP_K
    assert dense.retrieve.call_args.kwargs["top_k"] == DEFAULT_TOP_K
    assert lexical.search.call_args.kwargs["top_k"] == DEFAULT_TOP_K


def test_empty_result_sets_are_preserved() -> None:
    service, _, _ = _service()
    result = service.retrieve("missing")
    assert result.dense == ()
    assert result.lexical == ()


@pytest.mark.parametrize("failing_path", ["dense", "lexical"])
def test_failures_propagate_from_either_path(failing_path: str) -> None:
    service, dense, lexical = _service()
    failing = dense.retrieve if failing_path == "dense" else lexical.search
    failing.side_effect = RuntimeError(f"{failing_path} failed")
    with pytest.raises(RuntimeError, match=f"{failing_path} failed"):
        service.retrieve("terms")


def test_result_object_is_immutable() -> None:
    service, _, _ = _service([_dense_row(0)], [_lexical_row(0)])
    result = service.retrieve("terms")
    with pytest.raises(Exception):
        result.dense = ()
