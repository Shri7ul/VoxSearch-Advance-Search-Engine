"""Goal 41 — bounded retry mechanism for failed background jobs.

This is a domain-level retry layer used by the Goal 40 dispatcher. It does NOT
configure Celery's own autoretry, so there is a single documented retry site.
Permanent errors fail immediately; transient errors are retried up to a bounded
attempt count with deterministic exponential backoff (no jitter/randomness).

No credentials, payload contents or raw exception messages are included in
errors. No Celery import, network call or DB write happens at import time.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.workers.jobs import JobEnvelope, JobValidationError

MAX_ATTEMPTS_LIMIT = 10
MAX_DELAY_LIMIT_SECONDS = 300.0


class RetryError(RuntimeError):
    """Base class for credential-free Goal 41 failures."""


class RetryConfigError(RetryError, ValueError):
    """Retry configuration is invalid."""


class PermanentJobError(RetryError):
    """A job failure that must never be retried."""


class RetryableJobError(RetryError):
    """A transient job failure eligible for bounded retry."""


class RetryExhaustedError(RetryError):
    """All permitted attempts failed; the last failure type is recorded."""


@dataclass(frozen=True)
class RetryPolicy:
    """Validated, immutable retry bounds."""

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= MAX_ATTEMPTS_LIMIT
        ):
            raise RetryConfigError("max_attempts is out of bounds")
        if (
            isinstance(self.base_delay_seconds, bool)
            or not isinstance(self.base_delay_seconds, (int, float))
            or not 0 <= self.base_delay_seconds <= MAX_DELAY_LIMIT_SECONDS
        ):
            raise RetryConfigError("base_delay_seconds is out of bounds")
        if (
            isinstance(self.max_delay_seconds, bool)
            or not isinstance(self.max_delay_seconds, (int, float))
            or not 0 <= self.max_delay_seconds <= MAX_DELAY_LIMIT_SECONDS
        ):
            raise RetryConfigError("max_delay_seconds is out of bounds")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise RetryConfigError(
                "max_delay_seconds must be >= base_delay_seconds"
            )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RetryPolicy":
        cfg = settings or get_settings()
        return cls(
            max_attempts=cfg.retry_max_attempts,
            base_delay_seconds=cfg.retry_base_delay_seconds,
            max_delay_seconds=cfg.retry_max_delay_seconds,
        )

    def backoff_seconds(self, attempt: int) -> float:
        """Delay before retrying after ``attempt`` (1-based) failed."""
        if not isinstance(attempt, int) or attempt < 1:
            raise RetryConfigError("attempt must be a positive integer")
        if self.base_delay_seconds == 0:
            return 0.0
        # Cap the exponent so an extreme attempt number cannot overflow into a
        # huge integer before the configured delay ceiling is applied.
        exponent = min(attempt - 1, 64)
        return min(
            self.base_delay_seconds * (2 ** exponent),
            self.max_delay_seconds,
        )


@dataclass(frozen=True)
class RetryResult:
    """Immutable outcome of a successful (possibly retried) job execution."""

    job_id: str
    task: str
    attempts: int
    output: Any


def is_retryable(error: BaseException) -> bool:
    """Classify only explicitly retryable errors; fail closed otherwise."""
    if isinstance(error, (JobValidationError, PermanentJobError, RetryConfigError)):
        return False
    return isinstance(error, RetryableJobError)


class RetryExecutor:
    """Execute one validated job with bounded, deterministic retries."""

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        classify: Callable[[BaseException], bool] = is_retryable,
    ) -> None:
        self._policy = policy or RetryPolicy()
        self._sleep = sleep
        self._classify = classify

    @property
    def policy(self) -> RetryPolicy:
        return self._policy

    def execute(
        self,
        job: Any,
        handler: Callable[[dict[str, Any]], Any],
    ) -> RetryResult:
        """Run ``handler`` for a validated job envelope."""
        envelope = (
            job if isinstance(job, JobEnvelope) else JobEnvelope.from_mapping(job)
        )
        if not callable(handler):
            raise PermanentJobError("job handler is not callable")

        last_type = "UnknownError"
        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                output = handler(dict(envelope.payload))
            except Exception as exc:
                last_type = type(exc).__name__
                if not self._classify(exc) or attempt == self._policy.max_attempts:
                    # Suppress the raw cause: it may contain secret payload
                    # details. Only the exception type name is retained above.
                    raise RetryExhaustedError(
                        f"job failed after {attempt} attempt(s) ({last_type})"
                    ) from None
                self._sleep(self._policy.backoff_seconds(attempt))
            else:
                return RetryResult(
                    job_id=str(envelope.job_id),
                    task=envelope.task,
                    attempts=attempt,
                    output=output,
                )
        raise RetryExhaustedError(
            f"job failed after {self._policy.max_attempts} attempt(s) ({last_type})"
        )


__all__ = [
    "MAX_ATTEMPTS_LIMIT",
    "MAX_DELAY_LIMIT_SECONDS",
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
