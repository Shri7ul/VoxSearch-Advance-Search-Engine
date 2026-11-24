"""Tests for Goal 46 — P95 latency benchmark."""
from __future__ import annotations

import pytest

from app.evaluation import (
    LatencyBenchmark,
    PerformanceEvaluationError,
    percentile,
)


def test_percentile_is_deterministic_nearest_rank() -> None:
    assert percentile([5, 1, 3, 2, 4], 0.5) == pytest.approx(3.0)
    assert percentile([1, 2, 3, 4, 5], 0.95) == pytest.approx(5.0)
    assert percentile([10], 0.95) == pytest.approx(10.0)
    assert percentile([1, 2, 3, 4], 0.25) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "samples,q",
    [
        ([], 0.95),
        ("not-a-sequence", 0.95),
        ([1, 2], 0),
        ([1, 2], 1.1),
        ([1, -2], 0.95),
        ([float("nan")], 0.95),
        ([True], 0.95),
        ([1], "0.5"),
    ],
)
def test_invalid_samples_or_quantiles_are_rejected(samples, q) -> None:
    with pytest.raises(PerformanceEvaluationError):
        percentile(samples, q)


def _clock(values):
    iterator = iter(values)
    return lambda: next(iterator)


def test_benchmark_reports_p95_from_injected_clock() -> None:
    # 10ms, 20ms, 15ms -> sorted [10, 15, 20]
    clock = _clock([0.0, 0.010, 0.010, 0.030, 0.030, 0.045])
    report = LatencyBenchmark(clock=clock).run("search", lambda: None, iterations=3)
    assert report.operation == "search"
    assert report.sample_count == 3
    assert report.mean_ms == pytest.approx(15.0)
    assert report.p50_ms == pytest.approx(15.0)
    assert report.p95_ms == pytest.approx(20.0)
    assert report.max_ms == pytest.approx(20.0)
    assert report.summary()["p95_ms"] == pytest.approx(20.0)


def test_warmup_is_executed_but_not_measured() -> None:
    calls = []

    def action():
        calls.append(1)

    clock = _clock([0.0, 0.002])
    report = LatencyBenchmark(clock=clock).run(
        "op", action, iterations=1, warmup_iterations=2
    )
    assert len(calls) == 3
    assert report.sample_count == 1
    assert report.p95_ms == pytest.approx(2.0)


@pytest.mark.parametrize("warmup", [-1, 1001, True, 1.5])
def test_warmup_bounds_are_enforced(warmup) -> None:
    with pytest.raises(PerformanceEvaluationError):
        LatencyBenchmark(clock=_clock([0.0, 0.1])).run(
            "op", lambda: None, iterations=1, warmup_iterations=warmup
        )


def test_benchmark_is_deterministic_for_same_clock_sequence() -> None:
    values = [0.0, 0.002, 0.002, 0.009]
    first = LatencyBenchmark(clock=_clock(values)).run("op", lambda: None, iterations=2)
    second = LatencyBenchmark(clock=_clock(values)).run("op", lambda: None, iterations=2)
    assert first == second


def test_report_is_immutable() -> None:
    report = LatencyBenchmark(clock=_clock([0.0, 0.001])).run("op", lambda: None, iterations=1)
    with pytest.raises(Exception):
        report.p95_ms = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("iterations", [0, -1, 10001, True, 1.5])
def test_iterations_are_bounded(iterations) -> None:
    with pytest.raises(PerformanceEvaluationError):
        LatencyBenchmark(clock=_clock([0.0])).run(
            "op", lambda: None, iterations=iterations
        )


def test_invalid_operation_and_action_are_rejected() -> None:
    benchmark = LatencyBenchmark(clock=_clock([0.0, 0.1]))
    with pytest.raises(PerformanceEvaluationError):
        benchmark.run("", lambda: None, iterations=1)
    with pytest.raises(PerformanceEvaluationError):
        benchmark.run("op", None, iterations=1)  # type: ignore[arg-type]


def test_clock_rollback_is_rejected() -> None:
    benchmark = LatencyBenchmark(clock=_clock([1.0, 0.5]))
    with pytest.raises(PerformanceEvaluationError, match="clock"):
        benchmark.run("op", lambda: None, iterations=1)
