"""Tests for Goal 44 — Recall@10, MRR and nDCG evaluation."""
from __future__ import annotations

import pytest

from app.evaluation import (
    RetrievalEvaluationError,
    build_default_dataset,
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    validate_dataset,
)

QRELS = {"good": 2, "ok": 1}


def test_recall_at_k_counts_relevant_hits() -> None:
    assert recall_at_k(["good", "bad", "ok"], QRELS, k=3) == pytest.approx(1.0)
    assert recall_at_k(["bad", "good"], QRELS, k=1) == pytest.approx(0.0)
    assert recall_at_k(["bad", "good"], QRELS, k=2) == pytest.approx(0.5)
    assert recall_at_k([], {}, k=10) == 0.0


def test_duplicate_results_do_not_inflate_recall_or_ndcg() -> None:
    assert recall_at_k(["good", "good", "bad"], QRELS, k=3) == pytest.approx(0.5)
    assert ndcg_at_k(["good", "good"], QRELS, k=2) == pytest.approx(
        ndcg_at_k(["good"], QRELS, k=2)
    )


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    assert reciprocal_rank(["bad", "good", "ok"], QRELS, k=3) == pytest.approx(0.5)
    assert reciprocal_rank(["good"], QRELS, k=1) == pytest.approx(1.0)
    assert reciprocal_rank(["bad"], QRELS, k=3) == 0.0


def test_ndcg_is_one_for_ideal_ordering_and_lower_otherwise() -> None:
    ideal = ndcg_at_k(["good", "ok"], QRELS, k=2)
    assert ideal == pytest.approx(1.0)
    assert ndcg_at_k(["ok", "good"], QRELS, k=2) < ideal
    assert ndcg_at_k(["bad"], QRELS, k=2) == 0.0
    assert ndcg_at_k([], {}, k=2) == 0.0


@pytest.mark.parametrize("k", [0, -1, 101, True, 1.5, "10"])
def test_k_validation_is_shared(k) -> None:
    for metric in (recall_at_k, reciprocal_rank, ndcg_at_k):
        with pytest.raises(RetrievalEvaluationError):
            metric(["good"], QRELS, k)


def _dataset():
    return validate_dataset(
        [
            {
                "query_id": "q1",
                "text": "exact term latency",
                "category": "technical_term",
                "language": "en",
                "relevant_chunk_ids": ["good", "ok"],
                "primary_relevant_chunk_id": "good",
            },
            {
                "query_id": "q2",
                "text": "paraphrase",
                "category": "semantic_paraphrase",
                "language": "en",
                "relevant_chunk_ids": ["good"],
                "primary_relevant_chunk_id": "good",
            },
        ],
        expected_size=2,
    )


def test_end_to_end_report_has_overall_and_category_metrics() -> None:
    def retriever(query_id: str, text: str, top_k: int):
        assert top_k == 10
        return ["bad", "good"] if query_id == "q1" else ["good"]

    report = evaluate_retrieval(_dataset(), retriever)
    assert report.query_count == 2
    assert report.top_k == 10
    assert set(report.by_category) == {"technical_term", "semantic_paraphrase"}
    assert report.summary()["mrr"] == pytest.approx(0.75)
    assert report.per_query[0].recall_at_k == pytest.approx(0.5)
    assert report.per_query[1].recall_at_k == pytest.approx(1.0)
    assert all(0 <= value <= 1 for value in report.summary().values())


def test_default_200_query_dataset_can_be_evaluated_deterministically() -> None:
    def retriever(query_id: str, text: str, top_k: int):
        return [f"chunk-technical_term-00"] if "technical" in query_id else []

    first = evaluate_retrieval(build_default_dataset(), retriever)
    second = evaluate_retrieval(build_default_dataset(), retriever)
    assert first.query_count == 200
    assert first == second


@pytest.mark.parametrize(
    "ranked",
    ["not-a-sequence", [123], ["x"] * 11, None],
)
def test_malformed_retriever_output_is_rejected(ranked) -> None:
    with pytest.raises(RetrievalEvaluationError):
        evaluate_retrieval(_dataset(), lambda *args: ranked)


def test_invalid_dataset_and_retriever_are_rejected() -> None:
    with pytest.raises(RetrievalEvaluationError):
        evaluate_retrieval(object(), lambda *args: [])  # type: ignore[arg-type]
    with pytest.raises(RetrievalEvaluationError):
        evaluate_retrieval(_dataset(), None)  # type: ignore[arg-type]
