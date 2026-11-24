"""Tests for Goal 48 — structured logs and bounded metrics."""
from __future__ import annotations

import pytest

from app.monitoring import (
    LogLevel,
    MetricsError,
    MetricsRegistry,
    MonitoringError,
    build_log_event,
    redact_fields,
)

SECRET = "super-secret-value"


def test_log_event_is_deterministic_and_redacts_secrets() -> None:
    fields = {
        "query": "latency",
        "password": SECRET,
        "nested": {"api_key": SECRET, "safe": 1},
        "tags": ["ok", "token", {"password": SECRET, "safe": 2}],
    }
    event = build_log_event(LogLevel.INFO, "search completed", fields, timestamp=100.0)
    assert event.timestamp == 100.0
    assert event.level == LogLevel.INFO
    assert SECRET not in repr(event)
    assert event.fields["password"] == "[REDACTED]"
    assert event.fields["nested"]["api_key"] == "[REDACTED]"
    assert event.fields["tags"] == [
        "ok",
        "[REDACTED]",
        {"password": "[REDACTED]", "safe": 2},
    ]
    with pytest.raises(TypeError):
        event.fields["message"] = "mutated"  # type: ignore[index]
    assert event == build_log_event(
        LogLevel.INFO, "search completed", fields, timestamp=100.0
    )


@pytest.mark.parametrize(
    "fields",
    [None, {}, {"DATABASE_URL": SECRET}, {"Authorization": SECRET}, {"secret": SECRET}],
)
def test_common_secret_keys_are_never_exposed(fields) -> None:
    event = build_log_event(LogLevel.ERROR, "failure", fields, timestamp=1.0)
    assert SECRET not in repr(event)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"level": "info"},
        {"message": ""},
        {"message": "x" * 2001},
        {"timestamp": True},
        {"timestamp": "now"},
    ],
)
def test_invalid_log_events_are_rejected(kwargs) -> None:
    base: dict = {"level": LogLevel.INFO, "message": "ok", "timestamp": 1.0}
    base.update(kwargs)
    with pytest.raises(MonitoringError):
        build_log_event(**base)


def test_redact_fields_rejects_bad_shapes() -> None:
    with pytest.raises(MonitoringError):
        redact_fields(["not", "a", "mapping"])
    with pytest.raises(MonitoringError):
        redact_fields({"": 1})
    with pytest.raises(MonitoringError):
        redact_fields({f"k{i}": 1 for i in range(33)})


# --- Metrics ----------------------------------------------------------------

def test_increment_and_observe_report_exact_snapshot() -> None:
    registry = MetricsRegistry()
    registry.increment("searches", 2)
    registry.increment("searches", 3)
    registry.observe("latency_ms", 10)
    registry.observe("latency_ms", 30)
    snapshot = registry.snapshot()
    searches = snapshot["searches"]
    assert searches.count == 2
    assert searches.total == pytest.approx(5.0)
    assert searches.last == pytest.approx(3.0)
    assert searches.maximum == pytest.approx(3.0)
    assert searches.mean() == pytest.approx(2.5)
    assert snapshot["latency_ms"].mean() == pytest.approx(20.0)
    assert list(snapshot) == ["latency_ms", "searches"]


def test_snapshot_is_immutable_copy() -> None:
    registry = MetricsRegistry()
    registry.increment("hits")
    first = registry.snapshot()
    registry.increment("hits")
    assert first["hits"].count == 1
    assert registry.get("hits") is not None
    assert registry.get("hits").count == 2  # type: ignore[union-attr]


@pytest.mark.parametrize("name", ["", "Bad", "9name", "a" * 65, "has space", None])
def test_invalid_metric_names_are_rejected(name) -> None:
    registry = MetricsRegistry()
    with pytest.raises(MetricsError):
        registry.increment(name)


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "1"])
def test_invalid_metric_values_are_rejected(value) -> None:
    registry = MetricsRegistry()
    with pytest.raises(MetricsError):
        registry.observe("latency_ms", value)


def test_registry_is_thread_safe_under_concurrent_updates() -> None:
    import threading

    registry = MetricsRegistry()

    def worker() -> None:
        for _ in range(100):
            registry.increment("hits", 1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    hits = registry.get("hits")
    assert hits is not None
    assert hits.count == 800
    assert hits.total == pytest.approx(800.0)


def test_metric_series_limit_is_enforced() -> None:
    registry = MetricsRegistry()
    for index in range(256):
        registry.increment(f"metric_{index}")
    with pytest.raises(MetricsError):
        registry.increment("one_too_many")
