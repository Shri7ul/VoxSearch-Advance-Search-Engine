"""Goal 25 — duplicate removal producing a clean fused result list.

Goal 24 intentionally emitted one :class:`~app.services.rrf.FusedCandidate` per
source-list occurrence, so a chunk retrieved by both dense and lexical search
appeared twice (with the same summed RRF score). This service removes those
duplicates by stable chunk id, keeping ONE representative per chunk.

Policy (deterministic and fail-closed):

* the kept representative is the candidate with the highest ``rrf_score``; ties
  are broken by source (``dense`` before ``lexical``), then source rank, then
  chunk id — the same ordering used by Goal 24;
* the returned list preserves the relative order of the kept representatives as
  they appear in the input, so a list already sorted by RRF stays sorted;
* no metadata is rewritten: the original immutable ``FusedCandidate`` object is
  returned, so dense/lexical provenance and native result semantics survive.

Out of scope: reranking (Goal 26+), score changes, HTTP endpoints, migrations,
new dependencies and any write. No embedding is ever produced or exposed.
"""
from __future__ import annotations

from app.services.rrf import FusedCandidate

_SOURCE_ORDER = {"dense": 0, "lexical": 1}


def _preference(candidate: FusedCandidate) -> tuple[float, int, int, str]:
    return (
        -candidate.rrf_score,
        _SOURCE_ORDER[candidate.source],
        candidate.source_rank,
        str(candidate.chunk_id),
    )


class DeduplicationService:
    """Remove duplicate chunk ids from a Goal 24 fused candidate list."""

    @staticmethod
    def deduplicate(
        candidates: list[FusedCandidate],
    ) -> list[FusedCandidate]:
        """Return at most one candidate per chunk id, in stable input order."""
        best: dict[object, FusedCandidate] = {}
        for candidate in candidates:
            current = best.get(candidate.chunk_id)
            if current is None or _preference(candidate) < _preference(current):
                best[candidate.chunk_id] = candidate

        seen: set[object] = set()
        clean: list[FusedCandidate] = []
        for candidate in candidates:
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            clean.append(best[candidate.chunk_id])
        return clean


__all__ = ["DeduplicationService"]
