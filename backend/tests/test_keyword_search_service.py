"""Tests for Goal 22 keyword search service validation and delegation."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from app.repositories.lexical_search import LexicalSearchResult
from app.services.lexical_search import (
    MAX_QUERY_CHARACTERS,
    InvalidKeywordFilterError,
    InvalidKeywordQueryError,
    KeywordSearchService,
)
from app.services.video_chunks import InvalidTopKError


def _service(results=None):
    repo = MagicMock()
    repo.search_keywords.return_value = results if results is not None else []
    session = MagicMock()
    return KeywordSearchService(session, repo=repo), repo, session


@pytest.mark.parametrize("bad_query", ["", "   ", None, 123, True, b"bytes", ["x"], "x" * (MAX_QUERY_CHARACTERS + 1)])
def test_invalid_query_is_rejected_without_repo_call(bad_query) -> None:
    service, repo, session = _service()
    with pytest.raises(InvalidKeywordQueryError):
        service.search(bad_query)
    repo.search_keywords.assert_not_called()
    session.commit.assert_not_called()


def test_query_is_passed_verbatim() -> None:
    service, repo, _ = _service()
    service.search("  exact  terms ")
    assert repo.search_keywords.call_args.args[0] == "  exact  terms "


@pytest.mark.parametrize("bad_k", [0, -1, 101, True, 1.5])
def test_invalid_top_k_maps_to_domain_error(bad_k) -> None:
    service, repo, _ = _service()
    with pytest.raises(InvalidTopKError):
        service.search("terms", top_k=bad_k)
    repo.search_keywords.assert_not_called()


@pytest.mark.parametrize(
    "filter_name,bad_value",
    [
        ("video_id", "not-a-uuid"),
        ("transcript_asset_id", 123),
        ("model_id", ""),
        ("model_id", True),
    ],
)
def test_invalid_filters_are_rejected(filter_name: str, bad_value) -> None:
    service, repo, _ = _service()
    with pytest.raises(InvalidKeywordFilterError):
        service.search("terms", **{filter_name: bad_value})
    repo.search_keywords.assert_not_called()


def test_filters_and_top_k_are_forwarded() -> None:
    service, repo, _ = _service()
    video_id, asset_id = uuid.uuid4(), uuid.uuid4()
    service.search(
        "terms",
        top_k=7,
        video_id=video_id,
        transcript_asset_id=asset_id,
        model_id="BAAI/bge-m3",
    )
    kwargs = repo.search_keywords.call_args.kwargs
    assert kwargs["top_k"] == 7
    assert kwargs["video_id"] == video_id
    assert kwargs["transcript_asset_id"] == asset_id
    assert kwargs["model_id"] == "BAAI/bge-m3"


def test_results_are_returned_unchanged() -> None:
    result = LexicalSearchResult(
        id=uuid.uuid4(), video_id=uuid.uuid4(), transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(), chunk_index=0, text="terms", start_ms=0,
        end_ms=1, duration_ms=1, model_id="BAAI/bge-m3", rank=0.5,
    )
    service, _, session = _service([result])
    assert service.search("terms") == [result]
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    session.flush.assert_not_called()


def test_repository_failure_is_propagated() -> None:
    service, repo, session = _service()
    repo.search_keywords.side_effect = RuntimeError("db unavailable")
    with pytest.raises(RuntimeError, match="db unavailable"):
        service.search("terms")
    session.commit.assert_not_called()
