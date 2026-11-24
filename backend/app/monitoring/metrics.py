"""Goal 48 — bounded in-process metrics registry (no external backend)."""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Any

MAX_METRIC_SERIES = 256
_MAX_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_SAMPLES_PER_SERIES = 10000


class MetricsError(ValueError):
    """Metric name or value is invalid."""


@dataclass(frozen=True)
class MetricSnapshot:
    """Immutable point-in-time view of one metric series."""

    name: str
    count: int
    total: float
    last: float
    maximum: float

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class MetricsRegistry:
    """Thread-safe, bounded counter/observation registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._series: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _validate_name(name: Any) -> str:
        if not isinstance(name, str) or not _MAX_NAME.fullmatch(name):
            raise MetricsError("metric name is invalid")
        return name

    @staticmethod
    def _validate_value(value: Any, field: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise MetricsError(f"{field} must be a finite non-negative number")
        return float(value)

    def increment(self, name: str, value: float = 1.0) -> None:
        key = self._validate_name(name)
        amount = self._validate_value(value, "value")
        with self._lock:
            self._ensure_series(key)
            series = self._series[key]
            series["count"] += 1
            series["total"] += amount
            series["last"] = amount
            series["maximum"] = max(series["maximum"], amount)

    def observe(self, name: str, value_ms: float) -> None:
        key = self._validate_name(name)
        amount = self._validate_value(value_ms, "value_ms")
        with self._lock:
            self._ensure_series(key)
            series = self._series[key]
            if series["count"] >= _MAX_SAMPLES_PER_SERIES:
                raise MetricsError("metric series sample limit reached")
            series["count"] += 1
            series["total"] += amount
            series["last"] = amount
            series["maximum"] = max(series["maximum"], amount)

    def _ensure_series(self, key: str) -> None:
        if key not in self._series and len(self._series) >= MAX_METRIC_SERIES:
            raise MetricsError("metric series limit reached")
        self._series.setdefault(
            key, {"count": 0, "total": 0.0, "last": 0.0, "maximum": 0.0}
        )

    def snapshot(self) -> dict[str, MetricSnapshot]:
        with self._lock:
            return {
                name: MetricSnapshot(
                    name=name,
                    count=series["count"],
                    total=series["total"],
                    last=series["last"],
                    maximum=series["maximum"],
                )
                for name, series in sorted(self._series.items())
            }

    def get(self, name: str) -> MetricSnapshot | None:
        key = self._validate_name(name)
        return self.snapshot().get(key)


__all__ = [
    "MAX_METRIC_SERIES",
    "MetricSnapshot",
    "MetricsError",
    "MetricsRegistry",
]
