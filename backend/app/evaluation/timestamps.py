"""Goal 45 — timestamp evaluation with Mean Absolute Error (MAE).

Consumes explicit timestamp judgments and an injected predictor. No network,
database, video service or model call occurs here; every boundary is injected
for deterministic tests.

Errors are reported in integer milliseconds and phases (start/end).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_JUDGMENTS = 10000


class TimestampEvaluationError(ValueError):
    """Timestamp judgment or prediction is malformed."""


@dataclass(frozen=True)
class TimestampJudgment:
    """One ground-truth timestamp window for a query/chunk pair."""

    query_id: str
    chunk_id: str
    expected_start_ms: int
    expected_end_ms: int

    def __post_init__(self) -> None:
        _validate_ms(self.expected_start_ms, "expected_start_ms")
        _validate_ms(self.expected_end_ms, "expected_end_ms")
        if self.expected_end_ms <= self.expected_start_ms:
            raise TimestampEvaluationError("expected_end_ms must exceed start")
        if (
            not isinstance(self.query_id, str)
            or not isinstance(self.chunk_id, str)
            or not self.query_id.strip()
            or not self.chunk_id.strip()
        ):
            raise TimestampEvaluationError("judgment ids must be non-empty strings")


@dataclass(frozen=True)
class TimestampPrediction:
    """A predicted timestamp window (milliseconds)."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        _validate_ms(self.start_ms, "start_ms")
        _validate_ms(self.end_ms, "end_ms")
        if self.end_ms <= self.start_ms:
            raise TimestampEvaluationError("predicted end must exceed start")


@dataclass(frozen=True)
class JudgmentError:
    """Absolute errors for one judgment."""

    query_id: str
    chunk_id: str
    start_error_ms: int
    end_error_ms: int


@dataclass(frozen=True)
class TimestampEvaluation:
    """Immutable aggregate timestamp-MAE result."""

    judgment_count: int
    start_mae_ms: float
    end_mae_ms: float
    combined_mae_ms: float
    per_judgment: tuple[JudgmentError, ...]

    def summary(self) -> dict[str, float]:
        return {
            "start_mae_ms": self.start_mae_ms,
            "end_mae_ms": self.end_mae_ms,
            "combined_mae_ms": self.combined_mae_ms,
        }


def evaluate_timestamps(
    judgments: Sequence[TimestampJudgment | Mapping[str, Any]],
    predictor: Callable[[TimestampJudgment], TimestampPrediction | Mapping[str, Any]],
    *,
    max_judgments: int = DEFAULT_MAX_JUDGMENTS,
) -> TimestampEvaluation:
    """Predict every judgment and report macro-averaged absolute errors."""
    if isinstance(judgments, (str, bytes)) or not isinstance(judgments, Sequence):
        raise TimestampEvaluationError("judgments must be a sequence")
    if (
        isinstance(max_judgments, bool)
        or not isinstance(max_judgments, int)
        or not 1 <= max_judgments <= DEFAULT_MAX_JUDGMENTS
    ):
        raise TimestampEvaluationError("max_judgments is out of bounds")
    if not judgments:
        raise TimestampEvaluationError("no timestamp judgments supplied")
    if len(judgments) > max_judgments:
        raise TimestampEvaluationError("too many timestamp judgments")
    if not callable(predictor):
        raise TimestampEvaluationError("predictor must be callable")

    errors: list[JudgmentError] = []
    for raw in judgments:
        try:
            judgment = (
                raw
                if isinstance(raw, TimestampJudgment)
                else TimestampJudgment(**_mapping(raw))
            )
        except (TypeError, KeyError) as exc:
            raise TimestampEvaluationError("malformed timestamp judgment") from exc
        predicted = predictor(judgment)
        try:
            prediction = (
                predicted
                if isinstance(predicted, TimestampPrediction)
                else TimestampPrediction(**_mapping(predicted))
            )
        except (TypeError, KeyError) as exc:
            raise TimestampEvaluationError("malformed timestamp prediction") from exc
        errors.append(
            JudgmentError(
                query_id=judgment.query_id,
                chunk_id=judgment.chunk_id,
                start_error_ms=abs(judgment.expected_start_ms - prediction.start_ms),
                end_error_ms=abs(judgment.expected_end_ms - prediction.end_ms),
            )
        )

    start_mae = sum(item.start_error_ms for item in errors) / len(errors)
    end_mae = sum(item.end_error_ms for item in errors) / len(errors)
    return TimestampEvaluation(
        judgment_count=len(errors),
        start_mae_ms=start_mae,
        end_mae_ms=end_mae,
        combined_mae_ms=(start_mae + end_mae) / 2,
        per_judgment=tuple(errors),
    )


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TimestampEvaluationError("timestamp record must be a mapping")
    return dict(value)


def _validate_ms(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TimestampEvaluationError(f"{field} must be a non-negative integer")


__all__ = [
    "DEFAULT_MAX_JUDGMENTS",
    "JudgmentError",
    "TimestampEvaluation",
    "TimestampEvaluationError",
    "TimestampJudgment",
    "TimestampPrediction",
    "evaluate_timestamps",
]
