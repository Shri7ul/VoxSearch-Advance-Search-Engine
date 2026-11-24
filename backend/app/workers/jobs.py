"""Goal 40 — validated background-job envelope (no execution logic)."""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_TASK_NAME_CHARACTERS = 128
MAX_PAYLOAD_BYTES = 65536
MAX_JOB_ENVELOPE_BYTES = 131072


class JobValidationError(ValueError):
    """Background job payload is malformed or out of bounds."""


@dataclass(frozen=True)
class JobEnvelope:
    """One validated background job submitted through the worker boundary."""

    job_id: uuid.UUID
    task: str
    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "JobEnvelope":
        if not isinstance(value, Mapping):
            raise JobValidationError("job must be a mapping")
        job_id = value.get("job_id")
        task = value.get("task")
        payload = value.get("payload", {})
        if not isinstance(job_id, uuid.UUID):
            if isinstance(job_id, str):
                try:
                    job_id = uuid.UUID(job_id)
                except ValueError as exc:
                    raise JobValidationError("job_id is not a UUID") from exc
            else:
                raise JobValidationError("job_id must be a UUID")
        if (
            not isinstance(task, str)
            or not task.strip()
            or len(task) > MAX_TASK_NAME_CHARACTERS
            or any(char.isspace() for char in task)
        ):
            raise JobValidationError("task name is invalid")
        if not isinstance(payload, dict):
            raise JobValidationError("job payload must be an object")
        try:
            import json

            body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise JobValidationError("job payload is not JSON-serializable") from exc
        if len(body.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise JobValidationError("job payload is too large")
        envelope = cls(job_id=job_id, task=task, payload=dict(payload))
        if len(body) + len(str(job_id)) + len(task) > MAX_JOB_ENVELOPE_BYTES:
            raise JobValidationError("job envelope is too large")
        return envelope


__all__ = [
    "MAX_JOB_ENVELOPE_BYTES",
    "MAX_PAYLOAD_BYTES",
    "MAX_TASK_NAME_CHARACTERS",
    "JobEnvelope",
    "JobValidationError",
]
