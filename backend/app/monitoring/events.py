"""Goal 48 — secret-safe structured log events.

Pure functions only: no logging handler is installed, no I/O occurs, and no
ambient logger state is mutated. Callers pass an explicit timestamp so events
are deterministic in tests.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from dataclasses import dataclass
from enum import Enum
from typing import Any

_MAX_MESSAGE_CHARS = 2000
_MAX_FIELDS = 32
_SENSITIVE_KEY = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|credential|authorization|database_url|dsn)",
    re.IGNORECASE,
)
_REDACTED = "[REDACTED]"


class MonitoringError(ValueError):
    """Monitoring input is invalid."""


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class LogEvent:
    """Immutable structured log event."""

    timestamp: float
    level: LogLevel
    message: str
    fields: Mapping[str, Any]


def redact_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Copy fields, replacing values of secret-like keys with a fixed marker."""
    if not isinstance(fields, Mapping):
        raise MonitoringError("log fields must be a mapping")
    if len(fields) > _MAX_FIELDS:
        raise MonitoringError("too many log fields")
    cleaned: dict[str, Any] = {}
    for key, value in fields.items():
        if not isinstance(key, str) or not key:
            raise MonitoringError("log field keys must be non-empty strings")
        if _SENSITIVE_KEY.search(key):
            cleaned[key] = _REDACTED
        elif isinstance(value, Mapping):
            cleaned[key] = redact_fields(value)
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [
                redact_fields(item)
                if isinstance(item, Mapping)
                else (_REDACTED if isinstance(item, str) and _SENSITIVE_KEY.search(item) else item)
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def build_log_event(
    level: LogLevel,
    message: str,
    fields: Mapping[str, Any] | None = None,
    *,
    timestamp: float,
) -> LogEvent:
    """Build a validated, redacted log event."""
    if not isinstance(level, LogLevel):
        raise MonitoringError("level must be a LogLevel")
    if not isinstance(message, str) or not message.strip():
        raise MonitoringError("message must be non-empty")
    if len(message) > _MAX_MESSAGE_CHARS:
        raise MonitoringError("message is too long")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise MonitoringError("timestamp must be numeric")
    return LogEvent(
        timestamp=float(timestamp),
        level=level,
        message=message,
        fields=MappingProxyType(redact_fields(fields or {})),
    )


__all__ = ["LogEvent", "LogLevel", "MonitoringError", "build_log_event", "redact_fields"]
