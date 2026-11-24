"""Tests for Goal 45 — timestamp MAE evaluation."""
from __future__ import annotations

import pytest

from app.evaluation import (
    TimestampEvaluationError,
    TimestampJudgment,
    TimestampPrediction,
    evaluate_timestamps,
)


def _judgment(start=1000, end=3000):
    return TimestampJudgment("q1", "chunk-1", start, end)


def test_mae_is_exact_and_both_phases_are_reported() -> None:
    judgments = [_judgment(1000, 3000), _judgment(2000, 5000)]
    predictions = [
        TimestampPrediction(1100, 2900),   # errors 100, 100
        TimestampPrediction(1800, 5300),   # errors 200, 300
    ]
    result = evaluate_timestamps(judgments, lambda _: predictions.pop(0))
    assert result.judgment_count == 2
    assert result.start_mae_ms == pytest.approx(150.0)
    assert result.end_mae_ms == pytest.approx(200.0)
    assert result.combined_mae_ms == pytest.approx(175.0)
    assert [item.start_error_ms for item in result.per_judgment] == [100, 200]


def test_mapping_inputs_and_immutable_output() -> None:
    raw = {
        "query_id": "q1",
        "chunk_id": "c1",
        "expected_start_ms": 0,
        "expected_end_ms": 1000,
    }
    result = evaluate_timestamps([raw], lambda _: {"start_ms": 50, "end_ms": 950})
    assert result.summary()["start_mae_ms"] == pytest.approx(50.0)
    with pytest.raises(Exception):
        result.start_mae_ms = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_start_ms": -1},
        {"expected_start_ms": True},
        {"expected_end_ms": "1000"},
        {"expected_end_ms": 0},
        {"query_id": ""},
        {"chunk_id": ""},
        {"query_id": 123},
        {"chunk_id": None},
    ],
)
def test_invalid_judgments_are_rejected(kwargs) -> None:
    base = {
        "query_id": "q1",
        "chunk_id": "c1",
        "expected_start_ms": 0,
        "expected_end_ms": 1000,
    }
    base.update(kwargs)
    with pytest.raises(TimestampEvaluationError):
        TimestampJudgment(**base)


@pytest.mark.parametrize(
    "prediction",
    [
        {"start_ms": -1, "end_ms": 10},
        {"start_ms": 0, "end_ms": 0},
        {"start_ms": 0, "end_ms": "10"},
        {"start_ms": 0, "end_ms": 10, "unexpected": 1},
    ],
)
def test_invalid_predictions_are_rejected(prediction) -> None:
    with pytest.raises(TimestampEvaluationError):
        evaluate_timestamps([_judgment()], lambda _: prediction)


def test_empty_and_oversized_judgments_are_rejected() -> None:
    with pytest.raises(TimestampEvaluationError):
        evaluate_timestamps([], lambda _: TimestampPrediction(0, 1))
    with pytest.raises(TimestampEvaluationError):
        evaluate_timestamps([_judgment()] * 3, lambda _: TimestampPrediction(0, 1), max_judgments=2)


def test_malformed_inputs_fail_closed() -> None:
    with pytest.raises(TimestampEvaluationError):
        evaluate_timestamps("not-a-sequence", lambda _: None)
    with pytest.raises(TimestampEvaluationError):
        evaluate_timestamps([_judgment()], None)  # type: ignore[arg-type]
    with pytest.raises(TimestampEvaluationError):
        evaluate_timestamps([{"bad": 1}], lambda _: TimestampPrediction(0, 1))


def test_predictor_exception_propagates() -> None:
    def boom(_):
        raise RuntimeError("predictor failed")

    with pytest.raises(RuntimeError, match="predictor failed"):
        evaluate_timestamps([_judgment()], boom)
