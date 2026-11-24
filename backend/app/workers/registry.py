"""Goal 40 — deterministic in-process task registry."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from app.workers.jobs import JobValidationError


class TaskRegistry:
    """Map closed task names to handler callables, fail-closed on duplicates."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(
        self, name: str, handler: Callable[[dict[str, Any]], Any]
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise JobValidationError("task name must be non-empty")
        if name in self._handlers:
            raise JobValidationError("duplicate task registration")
        if not callable(handler):
            raise JobValidationError("task handler must be callable")
        self._handlers[name] = handler

    def get(self, name: str) -> Callable[[dict[str, Any]], Any]:
        handler = self._handlers.get(name)
        if handler is None:
            raise JobValidationError("unknown task")
        return handler

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def register_many(
        self, handlers: Mapping[str, Callable[[dict[str, Any]], Any]]
    ) -> None:
        for name, handler in handlers.items():
            self.register(name, handler)


__all__ = ["TaskRegistry"]
