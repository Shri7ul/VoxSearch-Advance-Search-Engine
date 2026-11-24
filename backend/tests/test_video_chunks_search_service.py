"""Offline tests for ``VideoChunkSearchService`` (Goal 17).

Verifies query vector validation (dimension 1024, finite floats, L2 normalization
without silent repair), top-k bounds (1 <= top_k <= 100), filter validation,
error taxonomy, and read-only session behavior.
"""
import math
import uuid
from unittest.mock import MagicMock

import pytest

from app.embeddings.contracts import DENSE_DIMENSION, EmbeddingVector
from app.repositories.video_chunks import ChunkSearchResult, VideoChunkRepository
from app.services import VoxSearchServiceError
from app.services.video_chunks import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    InvalidQueryVectorError,
    InvalidSearchFilterError,
    InvalidTopKError,
    SearchError,
    VideoChunkSearchService,
)

UNIT_VECTOR = (1.0,) + (0.0,) * (DENSE_DIMENSION - 1)


def _service():
    session = MagicMock()
    repo = MagicMock()
    return VideoChunkSearchService(session, repo=repo), session, repo


# --- 0. Error taxonomy ------------------------------------------------------

@pytest.mark.parametrize("exc", [
    SearchError,
    InvalidQueryVectorError,
    InvalidSearchFilterError,
    InvalidTopKError,
])
def test_search_errors_subclass_voxsearch_service_error(exc) -> None:
    assert issubclass(exc, VoxSearchServiceError)


# --- 1. Valid Search Queries -----------------------------------------------

def test_search_accepts_embedding_vector_and_forwards_to_repo() -> None:
    svc, session, repo = _service()
    repo.search_similar.return_value = []
    vec = EmbeddingVector(UNIT_VECTOR)

    res = svc.search(vec, top_k=5)
    assert res == []
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    repo.search_similar.assert_called_once_with(
        UNIT_VECTOR,
        top_k=5,
        video_id=None,
        transcript_asset_id=None,
        model_id=None,
        language=None,
        max_duration_seconds=None,
    )


def test_search_accepts_raw_sequence_and_validates_it() -> None:
    svc, _, repo = _service()
    repo.search_similar.return_value = []

    svc.search(list(UNIT_VECTOR))
    repo.search_similar.assert_called_once()
    passed_vec = repo.search_similar.call_args[0][0]
    assert len(passed_vec) == DENSE_DIMENSION
    assert passed_vec[0] == 1.0


def test_search_defaults_top_k_to_default_constant() -> None:
    svc, _, repo = _service()
    repo.search_similar.return_value = []

    svc.search(UNIT_VECTOR)
    assert repo.search_similar.call_args[1]["top_k"] == DEFAULT_TOP_K


# --- 2. Query Vector Validation (Fail-closed, no silent repair) -------------

@pytest.mark.parametrize("bad_dim", [
    (1.0,) * 10,
    (1.0,) * 1023,
    (1.0,) * 1025,
])
def test_search_rejects_wrong_vector_dimension(bad_dim) -> None:
    svc, _, repo = _service()
    with pytest.raises(InvalidQueryVectorError):
        svc.search(bad_dim)
    repo.search_similar.assert_not_called()


@pytest.mark.parametrize("bad_val", [
    float("nan"),
    float("inf"),
    float("-inf"),
])
def test_search_rejects_nan_and_inf_in_query_vector(bad_val) -> None:
    svc, _, repo = _service()
    bad_vec = (bad_val,) + (0.0,) * (DENSE_DIMENSION - 1)
    with pytest.raises(InvalidQueryVectorError):
        svc.search(bad_vec)
    repo.search_similar.assert_not_called()


@pytest.mark.parametrize("bad_item", [
    True,
    False,
    "1.0",
    None,
])
def test_search_rejects_non_real_elements_in_vector(bad_item) -> None:
    svc, _, repo = _service()
    bad_vec = (bad_item,) + (0.0,) * (DENSE_DIMENSION - 1)
    with pytest.raises(InvalidQueryVectorError):
        svc.search(bad_vec)
    repo.search_similar.assert_not_called()


@pytest.mark.parametrize("bad_container", [
    "not-a-sequence",
    b"bytes-not-allowed",
    12345,
    None,
    {"key": "value"},
])
def test_search_rejects_non_sequence_query_vector(bad_container) -> None:
    svc, _, repo = _service()
    with pytest.raises(InvalidQueryVectorError):
        svc.search(bad_container)
    repo.search_similar.assert_not_called()


@pytest.mark.parametrize("bad_norm_first", [
    0.0,   # zero vector
    0.5,   # under-normalized
    2.0,   # over-normalized
    -1.5,
])
def test_search_rejects_unnormalized_vectors_no_silent_rescaling(bad_norm_first) -> None:
    svc, _, repo = _service()
    bad_vec = (bad_norm_first,) + (0.0,) * (DENSE_DIMENSION - 1)
    with pytest.raises(InvalidQueryVectorError) as ei:
        svc.search(bad_vec)
    assert "L2 norm" in str(ei.value)
    repo.search_similar.assert_not_called()


def test_search_accepts_near_unit_vector_inside_tolerance_verbatim() -> None:
    svc, _, repo = _service()
    repo.search_similar.return_value = []
    # 0.9995 is within 1e-3 tolerance of 1.0
    near_unit = (0.9995,) + (0.0,) * (DENSE_DIMENSION - 1)
    svc.search(near_unit)
    repo.search_similar.assert_called_once()
    passed = repo.search_similar.call_args[0][0]
    assert passed[0] == 0.9995  # not silently rescaled to 1.0


# --- 3. Top-K Validation ---------------------------------------------------

@pytest.mark.parametrize("valid_k", [1, 5, 10, 50, MAX_TOP_K])
def test_search_accepts_valid_top_k(valid_k: int) -> None:
    svc, _, repo = _service()
    repo.search_similar.return_value = []
    svc.search(UNIT_VECTOR, top_k=valid_k)
    assert repo.search_similar.call_args[1]["top_k"] == valid_k


@pytest.mark.parametrize("bad_k", [0, -1, -10, MAX_TOP_K + 1, 500])
def test_search_rejects_out_of_bounds_top_k(bad_k: int) -> None:
    svc, _, repo = _service()
    with pytest.raises(InvalidTopKError):
        svc.search(UNIT_VECTOR, top_k=bad_k)
    repo.search_similar.assert_not_called()


@pytest.mark.parametrize("bad_type_k", [True, False, 1.5, "10", None])
def test_search_rejects_non_integer_top_k(bad_type_k) -> None:
    svc, _, repo = _service()
    with pytest.raises(InvalidTopKError):
        svc.search(UNIT_VECTOR, top_k=bad_type_k)
    repo.search_similar.assert_not_called()


# --- 4. Filter Validation & Forwarding --------------------------------------

def test_search_valid_filters_forwarded_to_repo() -> None:
    svc, _, repo = _service()
    repo.search_similar.return_value = []
    vid = uuid.uuid4()
    aid = uuid.uuid4()
    mid = "BAAI/bge-m3"

    svc.search(
        UNIT_VECTOR,
        video_id=vid,
        transcript_asset_id=aid,
        model_id=mid,
    )
    repo.search_similar.assert_called_once_with(
        UNIT_VECTOR,
        top_k=DEFAULT_TOP_K,
        video_id=vid,
        transcript_asset_id=aid,
        model_id=mid,
        language=None,
        max_duration_seconds=None,
    )


@pytest.mark.parametrize("bad_uuid", ["not-a-uuid", 12345, True, []])
def test_search_rejects_invalid_video_id_filter(bad_uuid) -> None:
    svc, _, repo = _service()
    with pytest.raises(InvalidSearchFilterError):
        svc.search(UNIT_VECTOR, video_id=bad_uuid)
    repo.search_similar.assert_not_called()


@pytest.mark.parametrize("bad_uuid", ["not-a-uuid", 12345, True, []])
def test_search_rejects_invalid_transcript_asset_id_filter(bad_uuid) -> None:
    svc, _, repo = _service()
    with pytest.raises(InvalidSearchFilterError):
        svc.search(UNIT_VECTOR, transcript_asset_id=bad_uuid)
    repo.search_similar.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": ""},
        {"language": "   "},
        {"language": "x" * 17},
        {"max_duration_seconds": 0},
        {"max_duration_seconds": 90000},
        {"max_duration_seconds": True},
        {"max_duration_seconds": 1.5},
    ],
)
def test_goal33_filters_are_rejected(kwargs) -> None:
    service, _, repo = _service()
    with pytest.raises(InvalidSearchFilterError):
        service.search(UNIT_VECTOR, **kwargs)
    repo.search_similar.assert_not_called()


def test_goal33_filters_are_forwarded() -> None:
    service, _, repo = _service()
    service.search(UNIT_VECTOR, language="en", max_duration_seconds=300)
    kwargs = repo.search_similar.call_args.kwargs
    assert kwargs["language"] == "en"
    assert kwargs["max_duration_seconds"] == 300


@pytest.mark.parametrize("bad_mid", ["", "   ", 123, True])
def test_search_rejects_invalid_model_id_filter(bad_mid) -> None:
    svc, _, repo = _service()
    with pytest.raises(InvalidSearchFilterError):
        svc.search(UNIT_VECTOR, model_id=bad_mid)
    repo.search_similar.assert_not_called()


# --- 5. Default Repository Construction & Read-Only Contract ---------------

def test_default_repo_used_when_none_injected() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    svc = VideoChunkSearchService(session)
    res = svc.search(UNIT_VECTOR)
    assert res == []
    assert session.execute.call_count == 1
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


# --- 6. No transaction opened on read-only search ---------------------------

def test_search_does_not_open_transaction() -> None:
    """search() must not call commit or rollback — it is read-only."""
    svc, session, repo = _service()
    repo.search_similar.return_value = []
    svc.search(UNIT_VECTOR)
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    session.flush.assert_not_called()


# --- 7. EmbeddingVector forwarded as .values, not the wrapper object --------

def test_search_forwards_values_tuple_not_embedding_vector_object() -> None:
    """The repo receives a plain tuple from EmbeddingVector.values, not an EmbeddingVector.

    This ensures the repository's VectorType bind processor sees a plain sequence
    rather than a wrapped object it cannot serialize.
    """
    svc, _, repo = _service()
    repo.search_similar.return_value = []
    vec = EmbeddingVector(UNIT_VECTOR)
    svc.search(vec)
    forwarded = repo.search_similar.call_args[0][0]
    # Must be the raw tuple, not the EmbeddingVector wrapper
    assert isinstance(forwarded, tuple), (
        f"Expected tuple forwarded to repo, got {type(forwarded).__name__}"
    )
    assert forwarded == UNIT_VECTOR


# --- 8. model_id filter — None is allowed (no model restriction) ------------

def test_search_model_id_none_is_valid_no_model_filter() -> None:
    """model_id=None is explicitly valid and means 'no model filter'."""
    svc, _, repo = _service()
    repo.search_similar.return_value = []
    svc.search(UNIT_VECTOR, model_id=None)
    assert repo.search_similar.call_args[1]["model_id"] is None


# --- 9. Validation order: top_k checked before vector ----------------------

def test_validation_order_top_k_before_vector() -> None:
    """top_k validation is performed before query vector validation.

    An invalid top_k should raise InvalidTopKError even when the vector
    is also invalid, ensuring fail-fast ordering matches the documented contract.
    """
    svc, _, repo = _service()
    # Both top_k and vector are bad; top_k is validated first
    bad_vec = (0.5,) + (0.0,) * (DENSE_DIMENSION - 1)   # bad norm
    with pytest.raises(InvalidTopKError):
        svc.search(bad_vec, top_k=0)
    repo.search_similar.assert_not_called()


# --- 10. Failure propagation (errors are never swallowed) -------------------

def test_search_propagates_repository_failure() -> None:
    svc, session, repo = _service()
    repo.search_similar.side_effect = RuntimeError("db unavailable")
    with pytest.raises(RuntimeError, match="db unavailable"):
        svc.search(UNIT_VECTOR)
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    session.flush.assert_not_called()
