"""Tests for Goal 21 — structured JSON search response schema."""
from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.schemas import SearchHit, SearchResponse
from app.api.search import get_query_embedding_service
from app.db.session import get_db
from app.main import app
from app.repositories.video_chunks import ChunkSearchResult

UNIT_VECTOR = [1.0 / 32.0] * 1024
PATH = "/api/v1/search"


def _result() -> ChunkSearchResult:
    return ChunkSearchResult(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=0,
        text="verbatim chunk text",
        start_ms=100,
        end_ms=900,
        duration_ms=800,
        timestamp_precision="word",
        word_count=3,
        boundary_reason="sentence",
        source_segment_refs=("seg",),
        overlap_sentence_indices=(),
        diagnostics=(),
        model_id="BAAI/bge-m3",
        distance=0.25,
    )


def _hit_payload() -> dict[str, Any]:
    return SearchHit.model_validate(_result()).model_dump(mode="json")


# --- 1. Hit schema ----------------------------------------------------------

def test_hit_maps_chunk_result_without_embedding() -> None:
    payload = _hit_payload()
    assert payload["text"] == "verbatim chunk text"
    assert payload["distance"] == pytest.approx(0.25)
    assert payload["score"] == pytest.approx(0.75)
    assert "embedding" not in payload
    assert "embedding" not in SearchHit.model_fields


@pytest.mark.parametrize("bad_distance", [-0.1, 2.1, float("nan"), float("inf")])
def test_hit_rejects_invalid_distance(bad_distance: float) -> None:
    payload = _hit_payload()
    payload["distance"] = bad_distance
    with pytest.raises(ValidationError):
        SearchHit.model_validate(payload)


@pytest.mark.parametrize(
    "distance,score",
    [(0.25, 0.9), (0.0, 0.5), (2.0, 0.0), (1.0, -0.5)],
)
def test_hit_rejects_contradictory_score_and_distance(
    distance: float, score: float
) -> None:
    """The public JSON must never present conflicting similarity values."""
    payload = _hit_payload()
    payload["distance"] = distance
    payload["score"] = score
    with pytest.raises(ValidationError, match="score must equal 1 - distance"):
        SearchHit.model_validate(payload)


@pytest.mark.parametrize(
    "distance,score",
    [(0.0, 1.0), (1.0, 0.0), (2.0, -1.0), (0.25, 0.75)],
)
def test_hit_accepts_consistent_distance_score_boundaries(
    distance: float, score: float
) -> None:
    payload = _hit_payload()
    payload["distance"] = distance
    payload["score"] = score
    hit = SearchHit.model_validate(payload)
    assert hit.distance == pytest.approx(distance)
    assert hit.score == pytest.approx(score)


def test_hit_tolerates_tiny_float_serialization_drift() -> None:
    payload = _hit_payload()
    payload["distance"] = 0.25
    payload["score"] = 0.7500000001
    assert SearchHit.model_validate(payload).score == pytest.approx(0.75)


def test_hit_rejects_extra_fields() -> None:
    payload = _hit_payload()
    payload["embedding"] = [0.0] * 1024
    with pytest.raises(ValidationError):
        SearchHit.model_validate(payload)


# --- 2. Envelope schema -----------------------------------------------------

def test_response_envelope_is_structured_and_consistent() -> None:
    hit = _hit_payload()
    response = SearchResponse(
        query="what was said?", top_k=10, count=1, results=[hit]
    )
    payload = json.loads(response.model_dump_json())
    assert set(payload) == {"query", "top_k", "count", "results"}
    assert payload["query"] == "what was said?"
    assert payload["top_k"] == 10
    assert payload["count"] == 1
    assert payload["results"][0] == hit


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 1, "results": []},            # count != len(results)
        {"count": 11, "results": [_hit_payload()] * 11},  # count > top_k
        {"top_k": 0, "count": 0, "results": []},
    ],
)
def test_response_rejects_inconsistent_envelope(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SearchResponse(**kwargs)


def test_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SearchResponse(count=0, results=[], unexpected=True)


# --- 3. OpenAPI and endpoint integration -----------------------------------

@pytest.fixture
def client() -> TestClient:
    session = MagicMock()
    session.execute.return_value.all.return_value = []

    def override_db():
        yield session  # mocked session: no real database is touched

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_openapi_documents_structured_response(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    operation = spec["paths"][PATH]["post"]
    ref = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    name = ref.rsplit("/", 1)[-1]
    schema = spec["components"]["schemas"][name]
    assert set(schema["required"]) == {"top_k", "count", "results"}
    assert set(schema["properties"]) == {"query", "top_k", "count", "results"}
    assert "description" in schema["properties"]["count"]
    assert "description" in schema["properties"]["results"]


def test_query_source_is_echoed_in_structured_response(client: TestClient) -> None:
    from app.embeddings.contracts import EmbeddingVector

    class FakeEmbedder:
        def embed(self, query: str) -> EmbeddingVector:
            return EmbeddingVector(tuple(UNIT_VECTOR))

    app.dependency_overrides[get_query_embedding_service] = lambda: FakeEmbedder()
    try:
        payload = client.post(PATH, json={"query": "latency"}).json()
    finally:
        app.dependency_overrides.pop(get_query_embedding_service, None)
    assert payload["query"] == "latency"
    assert payload["top_k"] == 10
    assert payload["count"] == 0
    assert payload["results"] == []
