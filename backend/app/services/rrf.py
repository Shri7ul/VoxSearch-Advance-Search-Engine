"""Goal 24 — Reciprocal Rank Fusion (RRF) combined ranking.

This service fuses the two result sets produced by Goal 23 into ONE ranked list.
RRF is rank-based and score-scale agnostic: each occurrence of a document at
1-based rank ``r`` in a source list contributes ``1 / (k + r)``; a document
found in both lists receives the sum of its contributions. The default
``DEFAULT_RRF_K = 60`` follows the standard RRF constant.

Scope boundary:

* Goal 25 (duplicate removal) is NOT implemented. If the same chunk id appears
  in both source lists, both entries remain in the fused output, each carrying
  the document's full summed RRF score. They are adjacent in ranking; removing
  them is explicitly the next goal's job.
* No re-ranking model, no score normalization, no API endpoint, no migration and
  no new dependency is introduced. Raw embeddings are never exposed: the
  carried result objects are the accepted Goal 17 / Goal 22 value objects,
  neither of which contains an embedding field.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from app.repositories.lexical_search import LexicalSearchResult
from app.repositories.video_chunks import ChunkSearchResult
from app.services.hybrid_retrieval import HybridRetrievalResult

DEFAULT_RRF_K = 60

SourceName = Literal["dense", "lexical"]


class RRFFusionError(ValueError):
    """Invalid RRF configuration (e.g. a negative or non-integer ``k``)."""


@dataclass(frozen=True)
class FusedCandidate:
    """One source-list entry placed in the combined RRF ranking."""

    source: SourceName
    source_rank: int
    rrf_score: float
    chunk: ChunkSearchResult | LexicalSearchResult

    @property
    def chunk_id(self) -> uuid.UUID:
        return self.chunk.id


def _contributions(ids, k: int) -> dict[object, float]:
    scores: dict[object, float] = {}
    for id_list in ids:
        for rank, chunk_id in enumerate(id_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class RRFFusionService:
    """Combine Goal 23's two ranked result lists with reciprocal rank fusion."""

    def fuse(
        self,
        hybrid: HybridRetrievalResult,
        *,
        k: int = DEFAULT_RRF_K,
    ) -> list[FusedCandidate]:
        """Return all source entries ordered by descending RRF score.

        ``k`` must be a non-negative integer (``bool`` is rejected); ``k = 0``
        is valid and yields ``1 / rank``. Ties are broken deterministically by
        source name (dense before lexical), then source rank, then chunk id.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise RRFFusionError(
                f"k must be a non-negative integer, got {k!r}"
            )

        dense = list(hybrid.dense)
        lexical = list(hybrid.lexical)
        scores = _contributions(
            ([item.id for item in dense], [item.id for item in lexical]), k
        )

        candidates: list[FusedCandidate] = [
            FusedCandidate(
                source="dense",
                source_rank=rank,
                rrf_score=scores[item.id],
                chunk=item,
            )
            for rank, item in enumerate(dense, start=1)
        ]
        candidates.extend(
            FusedCandidate(
                source="lexical",
                source_rank=rank,
                rrf_score=scores[item.id],
                chunk=item,
            )
            for rank, item in enumerate(lexical, start=1)
        )

        source_order: dict[SourceName, int] = {"dense": 0, "lexical": 1}
        candidates.sort(
            key=lambda c: (
                -c.rrf_score,
                source_order[c.source],
                c.source_rank,
                str(c.chunk_id),
            )
        )
        return candidates


__all__ = [
    "DEFAULT_RRF_K",
    "FusedCandidate",
    "RRFFusionError",
    "RRFFusionService",
]
