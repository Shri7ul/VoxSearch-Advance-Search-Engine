"""Tests for the Goal 18 ``POST /search`` HTTP endpoint.

These are transport-level tests: they exercise the real FastAPI app, the real
Goal 17 service and repository, and a mocked SQLAlchemy session. No database,
no network, and no embedding model is involved.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.search import get_query_embedding_service, get_search_service
from app.db.session import DatabaseNotConfiguredError, get_db
from app.embeddings.contracts import InvalidEmbeddingInputError
from app.embeddings.query import (
    QueryEmbeddingService,
    QueryEmbeddingUnavailableError,
)
from app.main import app
from app.repositories.video_chunks import MAX_TOP_K
from app.services.video_chunks import VideoChunkSearchService

UNIT_VECTOR = [1.0 / 32.0] * 1024  # L2 norm == 1.0
VERSIONED_PATH = "/api/v1/search"
UNVERSIONED_PATH = "/search"


def _row() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=2,
        text="a relevant transcript chunk",
        start_ms=1200,
        end_ms=4300,
        duration_ms=3100,
        timestamp_precision="word",
        word_count=6,
        boundary_reason="sentence",
        source_segment_refs=("seg-1", "seg-2"),
        overlap_sentence_indices=(1,),
        diagnostics=(),
        model_id="BAAI/bge-m3",
        distance=0.125,
    )


@pytest.fixture
def session() -> MagicMock:
    mock = MagicMock()
    mock.execute.return_value.all.return_value = []
    return mock


@pytest.fixture
def client(session: MagicMock) -> Iterator[TestClient]:
    def override_db() -> Iterator[MagicMock]:
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def _post(client: TestClient, path: str = VERSIONED_PATH, **body: Any):
    payload = {"query_vector": UNIT_VECTOR, **body}
    return client.post(path, json=payload)


# --- 1. Route surface -------------------------------------------------------

@pytest.mark.parametrize("path", [VERSIONED_PATH, UNVERSIONED_PATH])
def test_search_endpoint_is_registered(client: TestClient, path: str) -> None:
    response = _post(client, path)
    assert response.status_code == 200
    assert response.json() == {
        "query": None,
        "top_k": 10,
        "count": 0,
        "results": [],
    }


def test_get_search_is_not_allowed(client: TestClient) -> None:
    assert client.get(VERSIONED_PATH).status_code == 405


def test_unversioned_alias_is_hidden_from_openapi(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert VERSIONED_PATH in paths
    assert UNVERSIONED_PATH not in paths


# --- 2. Successful retrieval and result mapping -----------------------------

def test_search_maps_repository_rows_to_hits(client: TestClient, session: MagicMock) -> None:
    session.execute.return_value.all.return_value = [_row()]
    response = _post(client, top_k=5)
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    hit = payload["results"][0]
    assert hit["text"] == "a relevant transcript chunk"
    assert hit["chunk_index"] == 2
    assert hit["start_ms"] == 1200
    assert hit["distance"] == pytest.approx(0.125)
    assert hit["score"] == pytest.approx(0.875)
    assert hit["source_segment_refs"] == ["seg-1", "seg-2"]
    assert "embedding" not in hit


def test_search_forwards_all_parameters(client: TestClient) -> None:
    """The endpoint passes validated typed arguments to the Goal 17 service."""
    repo = MagicMock()
    repo.search_similar.return_value = []
    service = VideoChunkSearchService(MagicMock(), repo=repo)

    def override_service() -> VideoChunkSearchService:
        return service

    app.dependency_overrides[get_search_service] = override_service
    try:
        video_id = uuid.uuid4()
        asset_id = uuid.uuid4()
        response = _post(
            client,
            top_k=7,
            video_id=str(video_id),
            transcript_asset_id=str(asset_id),
            model_id="BAAI/bge-m3",
        )
        assert response.status_code == 200
        repo.search_similar.assert_called_once()
        kwargs = repo.search_similar.call_args.kwargs
        assert kwargs["top_k"] == 7
        assert kwargs["video_id"] == video_id
        assert kwargs["transcript_asset_id"] == asset_id
        assert kwargs["model_id"] == "BAAI/bge-m3"
        assert list(repo.search_similar.call_args.args[0]) == pytest.approx(UNIT_VECTOR)
    finally:
        app.dependency_overrides.pop(get_search_service, None)


# --- 3. Request contract strictness ----------------------------------------

@pytest.mark.parametrize(
    "bad_body",
    [
        {"top_k": True},
        {"top_k": 1.5},
        {"top_k": "10"},
        {"query_vector": ["0.03125"] * 1024},
        {"query_vector": [True] * 1024},
        {"unexpected": 1},
    ],
)
def test_search_rejects_malformed_body(client: TestClient, bad_body: dict[str, Any]) -> None:
    response = client.post(
        VERSIONED_PATH, json={"query_vector": UNIT_VECTOR, **bad_body}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_search_rejects_non_finite_vector(client: TestClient, token: str) -> None:
    """NaN/Inf must fail validation, never reach the repository or response."""
    body = '{"query_vector": [' + ",".join([token] + ["0.0"] * 1023) + "]}"
    response = client.post(
        VERSIONED_PATH,
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert "NaN/Inf" in response.json()["detail"]


def test_oversized_vector_422_does_not_echo_the_vector(client: TestClient) -> None:
    """A 422 must not reflect the submitted 1024-float query vector back."""
    response = client.post(VERSIONED_PATH, json={"query_vector": [0.03125] * 1025})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert all("input" not in error and "ctx" not in error for error in detail)
    assert "0.03125" not in response.text
    assert len(response.content) < 2048


def test_empty_vector_is_http_422(client: TestClient) -> None:
    response = client.post(VERSIONED_PATH, json={"query_vector": []})
    assert response.status_code == 422


def test_malformed_json_is_http_422_without_stack_trace(client: TestClient) -> None:
    response = client.post(
        VERSIONED_PATH,
        content="{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert "Traceback" not in response.text


# --- 4. Domain error translation -------------------------------------------

@pytest.mark.parametrize("bad_k", [0, -1, MAX_TOP_K + 1])
def test_invalid_top_k_is_http_422(client: TestClient, bad_k: int) -> None:
    response = _post(client, top_k=bad_k)
    assert response.status_code == 422
    assert response.json()["detail"] == (
        f"top_k must be between 1 and {MAX_TOP_K}, got {bad_k}"
    )


def test_wrong_vector_dimension_is_http_422(client: TestClient) -> None:
    response = client.post(VERSIONED_PATH, json={"query_vector": [0.03125] * 1023})
    assert response.status_code == 422
    assert "1024" in response.json()["detail"]


def test_unnormalized_vector_is_http_422(client: TestClient) -> None:
    response = client.post(VERSIONED_PATH, json={"query_vector": [0.5] + [0.0] * 1023})
    assert response.status_code == 422


def test_invalid_model_id_is_http_422(client: TestClient) -> None:
    response = _post(client, model_id="   ")
    assert response.status_code == 422
    assert response.json()["detail"] == "model_id must be a non-empty string"


def test_invalid_uuid_is_http_422(client: TestClient) -> None:
    response = _post(client, video_id="not-a-uuid")
    assert response.status_code == 422


@pytest.mark.parametrize("good_k", [1, MAX_TOP_K])
def test_top_k_inclusive_bounds_are_accepted(client: TestClient, good_k: int) -> None:
    response = _post(client, top_k=good_k)
    assert response.status_code == 200


@pytest.mark.parametrize("explicit_k,expected_limit", [(None, 10), (7, 7)])
def test_request_limit_reaches_sql_end_to_end(
    client: TestClient, session: MagicMock, explicit_k: int | None, expected_limit: int
) -> None:
    """API -> retrieval -> service -> repository must emit the expected LIMIT.

    This is the direct test evidence for Goal 20's "Top 10 results" achieve
    condition: a request without ``top_k`` really issues ``LIMIT 10``.
    """
    from sqlalchemy.dialects import postgresql

    body = {"query_vector": UNIT_VECTOR}
    if explicit_k is not None:
        body["top_k"] = explicit_k
    response = client.post(VERSIONED_PATH, json=body)
    assert response.status_code == 200

    statement = session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    assert "LIMIT" in str(compiled)
    assert expected_limit in list(compiled.params.values())


# --- 5. OpenAPI documentation -----------------------------------------------

def test_openapi_documents_bounded_request_and_embedding_free_response(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][VERSIONED_PATH]["post"]
    request_ref = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    request_name = request_ref.rsplit("/", 1)[-1]
    request_schema = schema["components"]["schemas"][request_name]
    vector_schema = request_schema["properties"]["query_vector"]["anyOf"][0]
    assert vector_schema["maxItems"] == 1024
    assert request_schema["properties"]["query"]["anyOf"][0]["type"] == "string"
    assert request_schema["properties"]["top_k"]["default"] == 10
    assert request_schema["additionalProperties"] is False
    response_ref = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    response_name = response_ref.rsplit("/", 1)[-1]
    hit_ref = schema["components"]["schemas"][response_name]["properties"]["results"]["items"]["$ref"]
    hit_name = hit_ref.rsplit("/", 1)[-1]
    assert "embedding" not in schema["components"]["schemas"][hit_name]["properties"]


# --- 6. Read-only contract and error propagation ---------------------------

def test_search_does_not_write_or_commit(client: TestClient, session: MagicMock) -> None:
    _post(client)
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    session.flush.assert_not_called()


def test_session_is_closed_after_request(client: TestClient, session: MagicMock) -> None:
    _post(client)
    session.close.assert_called_once()


def test_unexpected_service_failure_is_not_swallowed(
    client: TestClient, session: MagicMock
) -> None:
    service = MagicMock()
    service.search.side_effect = RuntimeError("db unavailable")

    def override_service() -> Any:
        return service

    app.dependency_overrides[get_search_service] = override_service
    try:
        with pytest.raises(RuntimeError, match="db unavailable"):
            _post(client)
    finally:
        app.dependency_overrides.pop(get_search_service, None)
    session.commit.assert_not_called()


def test_database_not_configured_is_http_503() -> None:
    def override_db() -> Iterator[MagicMock]:
        raise DatabaseNotConfiguredError()
        yield  # pragma: no cover - keeps this a generator dependency

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            response = _post(client)
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()


# --- 7. Goal 19: natural-language query embedding ---------------------------

@pytest.fixture
def embedder() -> Iterator[MagicMock]:
    mock = MagicMock()

    def override() -> MagicMock:
        return mock

    app.dependency_overrides[get_query_embedding_service] = override
    try:
        yield mock
    finally:
        app.dependency_overrides.pop(get_query_embedding_service, None)


def test_query_text_is_embedded_and_used_for_retrieval(
    client: TestClient, session: MagicMock, embedder: MagicMock
) -> None:
    from sqlalchemy.dialects import postgresql

    from app.embeddings.contracts import EmbeddingVector

    embedder.embed.return_value = EmbeddingVector(tuple(UNIT_VECTOR))
    response = client.post(VERSIONED_PATH, json={"query": "what was said about latency?"})
    assert response.status_code == 200
    embedder.embed.assert_called_once_with("what was said about latency?")

    statement = session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    embedding_key = next(key for key in compiled.params if "embedding" in key)
    assert list(compiled.params[embedding_key]) == pytest.approx(UNIT_VECTOR)


def test_query_result_never_contains_raw_embedding(
    client: TestClient, session: MagicMock, embedder: MagicMock
) -> None:
    from app.embeddings.contracts import EmbeddingVector

    session.execute.return_value.all.return_value = [_row()]
    embedder.embed.return_value = EmbeddingVector(tuple(UNIT_VECTOR))
    payload = client.post(VERSIONED_PATH, json={"query": "latency"}).json()
    assert payload["count"] == 1
    assert "embedding" not in payload["results"][0]


@pytest.mark.parametrize(
    "body",
    [
        {"query": "text", "query_vector": UNIT_VECTOR},
        {},
    ],
)
def test_exactly_one_query_source_is_required(
    client: TestClient, body: dict[str, Any]
) -> None:
    response = client.post(VERSIONED_PATH, json=body)
    assert response.status_code == 422
    assert "exactly one of query or query_vector" in str(response.json()["detail"])


def test_precomputed_vector_does_not_invoke_embedder(
    client: TestClient, embedder: MagicMock
) -> None:
    assert _post(client).status_code == 200
    embedder.embed.assert_not_called()


def test_invalid_query_text_is_http_422(client: TestClient, embedder: MagicMock) -> None:
    embedder.embed.side_effect = InvalidEmbeddingInputError("invalid query text")
    response = client.post(VERSIONED_PATH, json={"query": "bad"})
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid query text"


def test_unavailable_query_worker_is_http_503(
    client: TestClient, embedder: MagicMock
) -> None:
    embedder.embed.side_effect = QueryEmbeddingUnavailableError("no isolated worker")
    response = client.post(VERSIONED_PATH, json={"query": "text"})
    assert response.status_code == 503
    assert response.json()["detail"] == "no isolated worker"


def test_query_path_does_not_write_or_commit(
    client: TestClient, session: MagicMock, embedder: MagicMock
) -> None:
    from app.embeddings.contracts import EmbeddingVector

    embedder.embed.return_value = EmbeddingVector(tuple(UNIT_VECTOR))
    assert client.post(VERSIONED_PATH, json={"query": "text"}).status_code == 200
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    session.flush.assert_not_called()
