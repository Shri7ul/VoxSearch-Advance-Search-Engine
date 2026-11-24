"""Goal 21 — structured JSON response schema for ``POST /search``.

This module is the single source of truth for what the search API returns. It is
independent of the HTTP transport (Goal 18), query embedding (Goal 19) and
retrieval orchestration (Goal 20): those layers produce ranked
:class:`~app.repositories.video_chunks.ChunkSearchResult` objects, and this
schema defines their public JSON representation.

Contract guarantees:

* a stable, documented envelope (``query``, ``top_k``, ``count``, ``results``);
* ``count`` always equals ``len(results)`` and never exceeds ``top_k``;
* cosine ``distance`` lies in ``[0, 2]`` and ``score = 1 - distance`` in
  ``[-1, 1]`` (pgvector ``<=>`` semantics for unit vectors);
* raw 1024-dimensional embeddings are never part of the schema;
* unknown fields, non-finite numbers and out-of-range values are rejected.
"""
from __future__ import annotations

import math
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchHit(BaseModel):
    """One ranked transcript chunk (never includes the raw embedding)."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID = Field(description="Stable video-chunk identifier")
    video_id: UUID = Field(description="Source video identifier")
    transcript_asset_id: UUID = Field(description="Source transcript asset identifier")
    content_right_id: UUID = Field(description="Content-rights record identifier")
    chunk_index: int = Field(ge=0, description="Zero-based chunk position in the asset")
    text: str = Field(description="Verbatim transcript text of the chunk")
    start_ms: int = Field(ge=0, description="Chunk start time in milliseconds")
    end_ms: int = Field(ge=0, description="Chunk end time in milliseconds")
    duration_ms: int = Field(ge=0, description="Chunk duration in milliseconds")
    timestamp_precision: str = Field(description="Precision of the source timestamps")
    word_count: int = Field(ge=0, description="Number of words in the chunk")
    boundary_reason: str = Field(description="Reason this chunk boundary was chosen")
    source_segment_refs: list[str] = Field(description="Source transcript segment references")
    overlap_sentence_indices: list[int] = Field(
        description="Overlapping sentence indices retained for context"
    )
    diagnostics: list[str] = Field(description="Non-sensitive chunking diagnostics")
    model_id: str = Field(description="Embedding model that produced the chunk vector")
    youtube_video_id: str | None = Field(
        default=None, description="Verified YouTube video id for timestamp links"
    )
    distance: float = Field(
        ge=0.0, le=2.0, description="pgvector cosine distance (<=>); 0 means identical"
    )
    score: float = Field(
        ge=-1.0, le=1.0, description="Cosine similarity, defined as 1 - distance"
    )

    @model_validator(mode="after")
    def _score_matches_distance(self) -> "SearchHit":
        """Enforce the Goal 17 ``score = 1 - distance`` contract.

        A small absolute tolerance accommodates float serialization drift; a
        contradictory pair (e.g. distance 0.25 with score 0.9) is rejected so
        the public JSON can never present two conflicting similarity values.
        """
        if not math.isclose(self.score, 1.0 - self.distance, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("score must equal 1 - distance")
        return self


class SearchResponse(BaseModel):
    """Structured JSON envelope returned by ``POST /search``."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        default=None, description="Natural-language query, when one was supplied"
    )
    top_k: int = Field(
        ge=1, le=100, description="Requested maximum number of results"
    )
    count: int = Field(ge=0, description="Number of results actually returned")
    results: list[SearchHit] = Field(
        description="Ranked chunks; ordered by distance, chunk_index, id"
    )

    @model_validator(mode="after")
    def _envelope_is_consistent(self) -> "SearchResponse":
        if self.count != len(self.results):
            raise ValueError("count must equal the number of results")
        if self.count > self.top_k:
            raise ValueError("count must not exceed top_k")
        return self


__all__ = ["SearchHit", "SearchResponse"]
