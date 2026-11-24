"""Tests for Goal 41 — bounded failed-job retry (fully mocked sleeps)."""
from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.core.config import Settings
from app.workers import (
    JobDispatcher,
    JobValidationError,
    PermanentJobError,
    RetryConfigError,
    RetryExhaustedError,
    RetryExecutor,
    RetryPolicy,
    RetryableJobError,
    TaskRegistry,
    is_retryable,
)
from app.workers.retry import RetryResult

JOB_ID = uuid.uuid4()
JOB = {"job_id": str(JOB_ID), "task": "ping", "payload": {"n": 1}}
SECRET = "secret-payload-value"


def _policy(**overrides: Any) -> RetryPolicy:
    base: dict[str, Any] = {
        "max_attempts": 3,
        "base_delay_seconds": 1.0,
        "max_delay_seconds": 4.0,
    }
    base.update(overrides)
    return RetryPolicy(**base)


def _executor(policy: RetryPolicy | None = None):
    sleeps: list[float] = []
    executor = RetryExecutor(policy or _policy(), sleep=sleeps.append)
    return executor, sleeps


# --- Policy -----------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": 11},
        {"max_attempts": True},
        {"base_delay_seconds": -1},
        {"base_delay_seconds": "1"},
        {"max_delay_seconds": -1},
        {"max_delay_seconds": 500},
        {"base_delay_seconds": 10, "max_delay_seconds": 1},
    ],
)
def test_invalid_policy_is_rejected(kwargs: Any) -> None:
    with pytest.raises(RetryConfigError):
        _policy(**kwargs)


def test_policy_from_settings() -> None:
    policy = RetryPolicy.from_settings(
        Settings(
            retry_max_attempts=5,
            retry_base_delay_seconds=2.0,
            retry_max_delay_seconds=9.0,
        )
    )
    assert policy.max_attempts == 5
    assert policy.backoff_seconds(3) == 8.0


def test_backoff_is_deterministic_and_capped() -> None:
    policy = _policy()
    assert [policy.backoff_seconds(n) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 4.0]
    assert _policy(base_delay_seconds=0).backoff_seconds(3) == 0.0
    assert policy.backoff_seconds(10**6) == policy.max_delay_seconds
    with pytest.raises(RetryConfigError):
        policy.backoff_seconds(0)


# --- Classification ---------------------------------------------------------

def test_only_explicit_transient_errors_are_retried() -> None:
    assert is_retryable(RetryableJobError("transient")) is True
    assert is_retryable(JobValidationError("bad job")) is False
    assert is_retryable(PermanentJobError("permanent")) is False
    assert is_retryable(RuntimeError("bug")) is False
    assert is_retryable(RetryConfigError("config")) is False


# --- Executor ---------------------------------------------------------------

def test_success_on_first_attempt_does_not_sleep() -> None:
    executor, sleeps = _executor()
    result = executor.execute(JOB, lambda payload: {"pong": payload["n"]})
    assert isinstance(result, RetryResult)
    assert result.attempts == 1
    assert result.job_id == str(JOB_ID)
    assert result.task == "ping"
    assert result.output == {"pong": 1}
    assert sleeps == []


def test_transient_failure_is_retried_with_bounded_backoff() -> None:
    executor, sleeps = _executor()
    calls = []

    def flaky(payload):
        calls.append(payload)
        if len(calls) < 3:
            raise RetryableJobError("temporary")
        return "ok"

    result = executor.execute(JOB, flaky)
    assert result.attempts == 3
    assert result.output == "ok"
    assert sleeps == [1.0, 2.0]
    assert all(call == {"n": 1} for call in calls)


def test_exhaustion_is_bounded_and_credential_free() -> None:
    executor, sleeps = _executor(_policy(max_attempts=2, base_delay_seconds=0))

    def always_fail(payload):
        raise RetryableJobError(f"failed with {SECRET}")

    with pytest.raises(RetryExhaustedError) as exc:
        executor.execute(JOB, always_fail)
    assert "2 attempt" in str(exc.value)
    assert SECRET not in str(exc.value)
    assert exc.value.__cause__ is None  # raw, secret-bearing cause is suppressed
    assert sleeps == [0.0]


def test_permanent_error_is_never_retried() -> None:
    executor, sleeps = _executor()
    calls = []

    def permanent(payload):
        calls.append(1)
        raise PermanentJobError("permanent")

    with pytest.raises(RetryExhaustedError) as exc:
        executor.execute(JOB, permanent)
    assert "1 attempt" in str(exc.value)
    assert calls == [1]
    assert sleeps == []


def test_unknown_exception_fails_closed_without_retry() -> None:
    executor, sleeps = _executor()

    def buggy(payload):
        raise RuntimeError("unexpected bug")

    with pytest.raises(RetryExhaustedError):
        executor.execute(JOB, buggy)
    assert sleeps == []


def test_control_flow_exceptions_are_not_swallowed() -> None:
    executor, sleeps = _executor()

    def interrupted(payload):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        executor.execute(JOB, interrupted)
    assert sleeps == []


def test_invalid_job_is_rejected_before_handler_call() -> None:
    executor, sleeps = _executor()
    with pytest.raises(JobValidationError):
        executor.execute({"job_id": "bad", "task": "ping"}, lambda p: None)
    assert sleeps == []


# --- Dispatcher integration -------------------------------------------------

def test_dispatcher_with_retry_policy_retries_transient_failure() -> None:
    registry = TaskRegistry()
    calls = []

    def flaky(payload):
        calls.append(payload)
        if len(calls) < 2:
            raise RetryableJobError("transient")
        return {"pong": True}

    registry.register("ping", flaky)
    sleeps: list[float] = []
    executor = RetryExecutor(
        _policy(max_attempts=2, base_delay_seconds=0), sleep=sleeps.append
    )
    dispatcher = JobDispatcher(registry, retry_executor=executor)
    result = dispatcher.execute(JOB)
    assert result.output == {"pong": True}
    assert sleeps == [0.0]


def test_dispatcher_without_policy_keeps_goal_40_behavior() -> None:
    registry = TaskRegistry()
    registry.register("ping", lambda payload: {"pong": True})
    result = JobDispatcher(registry).execute(JOB)
    assert result.output == {"pong": True}
