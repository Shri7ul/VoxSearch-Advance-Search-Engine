"""Goal 40 — execute one validated background job via the task registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.workers.jobs import JobEnvelope, JobValidationError
from app.workers.registry import TaskRegistry


@dataclass(frozen=True)
class JobResult:
    """Immutable result of one background-job execution."""

    job_id: str
    task: str
    output: Any


class JobDispatcher:
    """Validate, look up and run background jobs without retry logic."""

    def __init__(self, registry: TaskRegistry | None = None) -> None:
        self._registry = registry or TaskRegistry()
        # Minimal, dependency-free health job so a default worker is usable.
        # Indexing / ingestion jobs belong to Goal 42 and are NOT registered.
        if "ping" not in self._registry.names():
            self._registry.register("ping", lambda payload: {"pong": True})

    @property
    def registry(self) -> TaskRegistry:
        return self._registry

    def execute(self, value: Any) -> JobResult:
        envelope = (
            value
            if isinstance(value, JobEnvelope)
            else JobEnvelope.from_mapping(value)
        )
        handler = self._registry.get(envelope.task)
        output = handler(dict(envelope.payload))
        return JobResult(
            job_id=str(envelope.job_id), task=envelope.task, output=output
        )


__all__ = ["JobDispatcher", "JobResult"]
