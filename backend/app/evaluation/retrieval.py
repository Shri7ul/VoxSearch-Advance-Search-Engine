"""Goal 44 — retrieval evaluation: Recall@10, MRR and nDCG.

Consumes the validated Goal 43 dataset qrels and an injected retriever. No
network, database, model or live-search call occurs here: the retriever is a
callable returning ranked chunk ids for one query.

Metrics are deterministic, bounded and fail closed on malformed output.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from app.evaluation.dataset import EvaluationDataset, QueryCategory

DEFAULT_TOP_K = 10
MAX_TOP_K = 100


class RetrievalEvaluationError(ValueError):
    """Evaluation input or retriever output is invalid."""


@dataclass(frozen=True)
class QueryMetrics:
    """Per-query metric values (all in [0, 1])."""

    query_id: str
    category: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


@dataclass(frozen=True)
class EvaluationReport:
    """Immutable aggregate + per-category evaluation result."""

    top_k: int
    query_count: int
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    per_query: tuple[QueryMetrics, ...]
    by_category: Mapping[str, Mapping[str, float]]

    def summary(self) -> dict[str, float]:
        return {
            "recall_at_k": self.mean_recall_at_k,
            "mrr": self.mean_reciprocal_rank,
            "ndcg_at_k": self.mean_ndcg_at_k,
        }


def recall_at_k(
    ranked_ids: Sequence[str], qrels: Mapping[str, int], k: int = DEFAULT_TOP_K
) -> float:
    """Fraction of relevant chunks retrieved in the first ``k`` results."""
    _validate_k(k)
    total = len(qrels)
    if total == 0:
        return 0.0
    hits = sum(1 for chunk_id in _unique(ranked_ids[:k]) if chunk_id in qrels)
    return min(hits / total, 1.0)


def reciprocal_rank(
    ranked_ids: Sequence[str], qrels: Mapping[str, int], k: int = DEFAULT_TOP_K
) -> float:
    """Reciprocal rank of the first relevant result, or 0.0."""
    _validate_k(k)
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in qrels:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str], qrels: Mapping[str, int], k: int = DEFAULT_TOP_K
) -> float:
    """Graded nDCG using gain ``2**rel - 1`` and log2(rank + 1) discount."""
    _validate_k(k)
    if not qrels:
        return 0.0

    def dcg(ids: Sequence[str]) -> float:
        return sum(
            (2 ** qrels[chunk_id] - 1) / math.log2(rank + 1)
            for rank, chunk_id in enumerate(ids, start=1)
            if chunk_id in qrels
        )

    ideal = sorted(qrels, key=lambda chunk_id: qrels[chunk_id], reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return min(dcg(_unique(ranked_ids[:k])) / ideal_dcg, 1.0)


def evaluate_retrieval(
    dataset: EvaluationDataset,
    retriever: Callable[[str, str, int], Sequence[str]],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> EvaluationReport:
    """Evaluate every labeled query and aggregate macro-averaged metrics."""
    if not isinstance(dataset, EvaluationDataset):
        raise RetrievalEvaluationError("dataset must be an EvaluationDataset")
    _validate_k(top_k)
    if not callable(retriever):
        raise RetrievalEvaluationError("retriever must be callable")
    if dataset.size() == 0:
        raise RetrievalEvaluationError("dataset has no queries")

    qrels = dataset.qrels()
    per_query: list[QueryMetrics] = []
    for query in dataset.queries:
        ranked = retriever(query.query_id, query.text, top_k)
        if isinstance(ranked, (str, bytes)) or not isinstance(ranked, Sequence):
            raise RetrievalEvaluationError("retriever must return a sequence of ids")
        if any(not isinstance(item, str) for item in ranked):
            raise RetrievalEvaluationError("retrieved ids must be strings")
        if len(ranked) > top_k:
            raise RetrievalEvaluationError("retriever returned more ids than top_k")
        per_query.append(
            QueryMetrics(
                query_id=query.query_id,
                category=query.category.value,
                recall_at_k=recall_at_k(ranked, qrels[query.query_id], top_k),
                reciprocal_rank=reciprocal_rank(ranked, qrels[query.query_id], top_k),
                ndcg_at_k=ndcg_at_k(ranked, qrels[query.query_id], top_k),
            )
        )

    grouped: dict[str, list[QueryMetrics]] = {}
    for metric in per_query:
        grouped.setdefault(metric.category, []).append(metric)
    by_category = {
        category: {
            "recall_at_k": _mean(item.recall_at_k for item in items),
            "mrr": _mean(item.reciprocal_rank for item in items),
            "ndcg_at_k": _mean(item.ndcg_at_k for item in items),
        }
        for category, items in grouped.items()
    }
    return EvaluationReport(
        top_k=top_k,
        query_count=len(per_query),
        mean_recall_at_k=_mean(item.recall_at_k for item in per_query),
        mean_reciprocal_rank=_mean(item.reciprocal_rank for item in per_query),
        mean_ndcg_at_k=_mean(item.ndcg_at_k for item in per_query),
        per_query=tuple(per_query),
        by_category=by_category,
    )


def _unique(ids: Sequence[str]) -> tuple[str, ...]:
    """First-occurrence dedupe; repeated results must not inflate metrics."""
    return tuple(dict.fromkeys(ids))


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _validate_k(k: Any) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_TOP_K:
        raise RetrievalEvaluationError("k must be an integer between 1 and 100")


__all__ = [
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "EvaluationReport",
    "QueryMetrics",
    "RetrievalEvaluationError",
    "evaluate_retrieval",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
