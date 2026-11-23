"""Goal 23 — dense + lexical retrieval returning two distinct result sets.

This service runs the already-accepted retrieval paths side by side:

* Goal 20 :class:`~app.services.retrieval.VectorRetrievalService` — dense cosine
  retrieval from a natural-language query (embedding via Goal 19);
* Goal 22 :class:`~app.services.lexical_search.KeywordSearchService` — exact
  PostgreSQL full-text keyword retrieval from the same query text.

It returns BOTH result lists unchanged, in their own native ordering and score
semantics. It deliberately does **not** combine, fuse, re-rank, normalize or
deduplicate them: ranked fusion is Goal 24 (RRF) and duplicate removal is Goal
25. No HTTP endpoint, migration, index, model load or write is added here.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.repositories.lexical_search import LexicalSearchResult
from app.repositories.video_chunks import DEFAULT_TOP_K, ChunkSearchResult
from app.services.lexical_search import KeywordSearchService
from app.services.retrieval import VectorRetrievalService


@dataclass(frozen=True)
class HybridRetrievalResult:
    """The two independent result sets produced by dense + lexical retrieval."""

    dense: tuple[ChunkSearchResult, ...]
    lexical: tuple[LexicalSearchResult, ...]


class HybridRetrievalService:
    """Run dense and lexical retrieval over the same query, keeping them apart."""

    def __init__(
        self,
        dense_service: VectorRetrievalService,
        lexical_service: KeywordSearchService,
    ) -> None:
        self._dense = dense_service
        self._lexical = lexical_service

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        video_id: uuid.UUID | None = None,
        transcript_asset_id: uuid.UUID | None = None,
        model_id: str | None = None,
    ) -> HybridRetrievalResult:
        """Return ``(dense, lexical)`` candidates for ``query``.

        Both paths receive the same query text, bound and filters. Each path
        performs its own accepted validation; failures propagate unchanged.
        Results are returned as immutable tuples in their original order.
        """
        dense = self._dense.retrieve(
            query=query,
            top_k=top_k,
            video_id=video_id,
            transcript_asset_id=transcript_asset_id,
            model_id=model_id,
        )
        lexical = self._lexical.search(
            query,
            top_k=top_k,
            video_id=video_id,
            transcript_asset_id=transcript_asset_id,
            model_id=model_id,
        )
        return HybridRetrievalResult(
            dense=tuple(dense),
            lexical=tuple(lexical),
        )


__all__ = ["HybridRetrievalResult", "HybridRetrievalService"]
