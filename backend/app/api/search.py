"""Goal 18 — the ``POST /search`` transcript search endpoint.

This module is the **HTTP transport layer only**. It turns a validated JSON
request into a call to Goal 17's read-only :class:`VideoChunkSearchService`
and maps the service's domain errors onto HTTP status codes.

Scope boundary (see ``progress.md``):

* Goal 18 (this module): the search endpoint, request contract, dependency
  wiring and error translation.
* Goal 19 (NOT implemented here): natural-language query text -> embedding.
  Consequently this endpoint currently accepts a precomputed, validated
  ``query_vector``. No model is loaded and no embedding is generated here.
* Goal 20 / Goal 21: richer retrieval policies and the final public response
  schema. The minimal response envelope below exists only so the endpoint can
  return a useful, bounded payload.

Nothing in this module writes to the database: there is no commit, flush,
rollback, or DDL. The injected session is owned and closed by ``get_db``.
"""
from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.embeddings.contracts import DENSE_DIMENSION
from app.repositories.video_chunks import DEFAULT_TOP_K
from app.services.video_chunks import SearchError, VideoChunkSearchService

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    """Strict JSON body for ``POST /search``.

    ``Strict*`` scalars prevent silent coercion (e.g. ``"10"`` -> ``10``,
    ``True`` -> ``1``). The vector is length-bounded at the HTTP boundary to
    avoid unbounded request bodies; dimension, finiteness and normalization
    are still validated by the Goal 17 service via the Goal 15 contract.
    """

    model_config = ConfigDict(extra="forbid")

    query_vector: Annotated[
        list[StrictFloat], Field(min_length=1, max_length=DENSE_DIMENSION)
    ]
    top_k: StrictInt = DEFAULT_TOP_K
    video_id: UUID | None = None
    transcript_asset_id: UUID | None = None
    model_id: StrictStr | None = None


class SearchHit(BaseModel):
    """One ranked chunk returned by the endpoint (no raw embedding)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    video_id: UUID
    transcript_asset_id: UUID
    content_right_id: UUID
    chunk_index: int
    text: str
    start_ms: int
    end_ms: int
    duration_ms: int
    timestamp_precision: str
    word_count: int
    boundary_reason: str
    source_segment_refs: list[str]
    overlap_sentence_indices: list[int]
    diagnostics: list[str]
    model_id: str
    distance: float
    score: float


class SearchResponse(BaseModel):
    """Minimal response envelope for the Goal 18 endpoint."""

    count: int
    results: list[SearchHit]


def get_search_service(
    session: Annotated[Session, Depends(get_db)],
) -> VideoChunkSearchService:
    """FastAPI dependency: build the Goal 17 service over a request session."""
    return VideoChunkSearchService(session)


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search transcript chunks by cosine similarity",
)
def search(
    payload: SearchRequest,
    service: Annotated[VideoChunkSearchService, Depends(get_search_service)],
) -> SearchResponse:
    """Return the top-k chunks nearest ``payload.query_vector``.

    Domain validation failures are translated to HTTP 422 with a credential-free
    ``detail`` message. Unexpected failures propagate so FastAPI returns 500;
    they are never converted into a successful empty result.
    """
    try:
        results = service.search(
            payload.query_vector,
            top_k=payload.top_k,
            video_id=payload.video_id,
            transcript_asset_id=payload.transcript_asset_id,
            model_id=payload.model_id,
        )
    except SearchError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    return SearchResponse(
        count=len(results),
        results=[SearchHit.model_validate(result) for result in results],
    )


__all__ = [
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    "get_search_service",
    "router",
]
