"""Goal 20 — vector retrieval returning the top 10 ranked chunks.

This is the application-level retrieval step of the semantic search pipeline.
It deliberately composes two already-accepted building blocks without changing
either one:

* :class:`~app.embeddings.query.QueryEmbeddingService` (Goal 19) turns one
  natural-language query into a validated embedding;
* :class:`~app.services.video_chunks.VideoChunkSearchService` (Goal 17) performs
  the read-only pgvector ``<=>`` top-k retrieval.

Goal 20 owns the **top-10 retrieval policy**: :data:`DEFAULT_TOP_K` is 10, so a
caller that supplies no explicit bound receives exactly the ten nearest chunks.
An explicit ``top_k`` is still forwarded and validated by Goal 17's accepted
``[1, 100]`` bound — this goal does not narrow or replace that contract.

Out of scope: the final public response schema (Goal 21), reranking, hybrid /
lexical retrieval, and any write or migration. No model is loaded here: the
query embedder keeps inference in the isolated Goal 15 subprocess.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from app.embeddings.contracts import EmbeddingVector
from app.embeddings.query import QueryEmbeddingService
from app.repositories.video_chunks import (
    DEFAULT_TOP_K,
    ChunkSearchResult,
)
from app.services.video_chunks import SearchError, VideoChunkSearchService


class InvalidRetrievalRequestError(SearchError):
    """Exactly one of ``query`` / ``query_vector`` must be provided."""


class VectorRetrievalService:
    """Compose query embedding and vector search into bounded retrieval."""

    def __init__(
        self,
        search_service: VideoChunkSearchService,
        embedder: QueryEmbeddingService | None = None,
    ) -> None:
        self._search = search_service
        self._embedder = embedder if embedder is not None else QueryEmbeddingService()

    @property
    def default_top_k(self) -> int:
        """The Goal 20 top-10 retrieval default (shared with Goal 17)."""
        return DEFAULT_TOP_K

    def retrieve(
        self,
        *,
        query: str | None = None,
        query_vector: EmbeddingVector | Sequence[float] | None = None,
        top_k: int = DEFAULT_TOP_K,
        video_id: uuid.UUID | None = None,
        transcript_asset_id: uuid.UUID | None = None,
        model_id: str | None = None,
        language: str | None = None,
        max_duration_seconds: int | None = None,
    ) -> list[ChunkSearchResult]:
        """Return at most ``top_k`` ranked chunks (10 by default).

        A natural-language ``query`` is embedded first (Goal 19); a precomputed
        ``query_vector`` is used directly. Exactly one source is required.
        Ranking, filtering, bounds and result mapping remain Goal 17's, so
        deterministic order and the no-embedding contract are preserved.
        """
        if (query is None) == (query_vector is None):
            raise InvalidRetrievalRequestError(
                "provide exactly one of query or query_vector"
            )

        if query is not None:
            vector: EmbeddingVector | Sequence[float] = self._embedder.embed(query)
        else:
            # Guarded by the mutually-exclusive check above.
            vector = query_vector  # type: ignore[assignment]

        return self._search.search(
            vector,
            top_k=top_k,
            video_id=video_id,
            transcript_asset_id=transcript_asset_id,
            model_id=model_id,
            language=language,
            max_duration_seconds=max_duration_seconds,
        )


__all__ = ["InvalidRetrievalRequestError", "VectorRetrievalService"]
