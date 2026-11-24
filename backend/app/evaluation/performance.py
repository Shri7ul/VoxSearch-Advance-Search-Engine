"""Goal 46 — P95 latency benchmark.

Measures an injected callable with an injected monotonic clock and reports
deterministic latency percentiles in milliseconds. No network, database, model
or external benchmark service is used; all measurements are injectable.

LIMITATION: tests with an injected clock validate the metric math and control
flow only — they are NOT real-world performance measurements. Real timing
requires a live environment and is not claimed here. Optional
``warmup_iterations`` lets callers exclude cold-start effects from the measured
samples; it defaults to 0 and is never included in the report.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class PerformanceEvaluationError(ValueError):
    """Benchmark configuration or samples are invalid."""


MAX_ITERATIONS = 10000


@dataclass(frozen=True)
class LatencyReport:
    """Immutable latency summary for one operation."""

    operation: str
    sample_count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def summary(self) -> dict[str, float]:
        return {
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
        }


def percentile(values: Sequence[float], quantile: float) -> float:
    """Deterministic nearest-rank percentile over validated samples."""
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise PerformanceEvaluationError("quantile must be numeric")
    if not 0 < quantile <= 1:
        raise PerformanceEvaluationError("quantile must be in (0, 1]")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PerformanceEvaluationError("samples must be a sequence")
    if not values:
        raise PerformanceEvaluationError("no latency samples")
    ordered: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise PerformanceEvaluationError("latency samples must be finite non-negative numbers")
        ordered.append(float(value))
    ordered.sort()
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class LatencyBenchmark:
    """Run an injected operation with an injected clock."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic

    def run(
        self,
        operation: str,
        action: Callable[[], Any],
        *,
        iterations: int = 100,
        warmup_iterations: int = 0,
    ) -> LatencyReport:
        """Execute ``action`` and summarize measured latencies.

        Warm-up iterations are executed unmeasured before sampling so cold
        starts do not distort p50/p95.
        """
        if not isinstance(operation, str) or not operation.strip():
            raise PerformanceEvaluationError("operation must be non-empty")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or not 1 <= iterations <= MAX_ITERATIONS
        ):
            raise PerformanceEvaluationError("iterations is out of bounds")
        if (
            isinstance(warmup_iterations, bool)
            or not isinstance(warmup_iterations, int)
            or not 0 <= warmup_iterations <= 1000
        ):
            raise PerformanceEvaluationError("warmup_iterations is out of bounds")
        if not callable(action):
            raise PerformanceEvaluationError("action must be callable")
        for _ in range(warmup_iterations):
            action()
        samples: list[float] = []
        for _ in range(iterations):
            start = self._clock()
            action()
            end = self._clock()
            elapsed = end - start
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise PerformanceEvaluationError("clock returned an invalid duration")
            samples.append(elapsed * 1000.0)
        return LatencyReport(
            operation=operation,
            sample_count=len(samples),
            mean_ms=sum(samples) / len(samples),
            p50_ms=percentile(samples, 0.5),
            p95_ms=percentile(samples, 0.95),
            max_ms=max(samples),
        )


__all__ = [
    "MAX_ITERATIONS",
    "LatencyBenchmark",
    "LatencyReport",
    "PerformanceEvaluationError",
    "percentile",
]
