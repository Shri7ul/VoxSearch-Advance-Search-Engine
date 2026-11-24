"""VoxSearch background-worker boundary (Goal 40 — Celery worker).

No import-time Celery import, no broker connection, no retry policy (Goal 41)
and no indexing automation (Goal 42).
"""
from app.workers.celery_app import (
    CeleryConfig,
    CeleryWorkerError,
    WorkerUnavailableError,
    create_celery_app,
    submit_job,
)
from app.workers.dispatcher import JobDispatcher, JobResult
from app.workers.jobs import JobEnvelope, JobValidationError
from app.workers.registry import TaskRegistry
from app.workers.retry import (
    PermanentJobError,
    RetryConfigError,
    RetryError,
    RetryExhaustedError,
    RetryExecutor,
    RetryPolicy,
    RetryResult,
    RetryableJobError,
    is_retryable,
)

__all__ = [
    "CeleryConfig",
    "CeleryWorkerError",
    "WorkerUnavailableError",
    "create_celery_app",
    "submit_job",
    "JobDispatcher",
    "JobEnvelope",
    "JobResult",
    "JobValidationError",
    "TaskRegistry",
    "PermanentJobError",
    "RetryConfigError",
    "RetryError",
    "RetryExhaustedError",
    "RetryExecutor",
    "RetryPolicy",
    "RetryResult",
    "RetryableJobError",
    "is_retryable",
]
