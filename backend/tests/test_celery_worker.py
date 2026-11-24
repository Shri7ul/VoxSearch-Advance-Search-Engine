"""Tests for Goal 40 — Celery worker with an injected fake Celery module."""
from __future__ import annotations

import sys
import uuid
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.queue.redis import RedisConfig
from app.workers import (
    CeleryConfig,
    CeleryWorkerError,
    JobDispatcher,
    JobEnvelope,
    JobValidationError,
    TaskRegistry,
    WorkerUnavailableError,
    create_celery_app,
    submit_job,
)
from app.workers import celery_app as celery_module_under_test

PASSWORD = "redis-secret-password"
JOB_ID = uuid.uuid4()


def _config(**overrides: Any) -> CeleryConfig:
    return CeleryConfig(
        redis=RedisConfig(
            host="redis.local",
            port=6380,
            db=2,
            password=SecretStr(PASSWORD),
            queue_key="voxsearch:jobs",
        ),
        **overrides,
    )


class FakeCelery:
    instances: list["FakeCelery"] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.tasks: dict[str, Any] = {}
        FakeCelery.instances.append(self)

    def task(self, *, name: str):
        def register(fn):
            self.tasks[name] = fn
            return fn

        return register


class FakeCeleryModule:
    Celery = FakeCelery


# --- Configuration ----------------------------------------------------------

def test_celery_not_imported_at_module_import() -> None:
    # The core app must import without the optional Celery dependency.
    assert "celery" not in sys.modules


def test_broker_url_encodes_password_and_is_masked_in_repr() -> None:
    config = _config()
    assert config.broker_url() == (
        "redis://redis-secret-password@redis.local:6380/2"
    )
    assert PASSWORD not in repr(config)
    assert PASSWORD not in str(config)


def test_broker_url_without_password() -> None:
    config = CeleryConfig(
        redis=RedisConfig(host="localhost", port=6379, db=0)
    )
    assert config.broker_url() == "redis://localhost:6379/0"


def test_from_settings_uses_redis_settings() -> None:
    config = CeleryConfig.from_settings(
        Settings(
            redis_host="redis.local",
            redis_port=6380,
            redis_db=2,
            redis_queue_key="custom:queue",
        )
    )
    assert config.broker_url().endswith("/2")
    assert config.redis.queue_key == "custom:queue"


@pytest.mark.parametrize("task_name", ["", "   ", "x" * 129, None])
def test_invalid_task_name_is_rejected(task_name: Any) -> None:
    with pytest.raises(CeleryWorkerError):
        CeleryConfig(redis=_config().redis, task_name=task_name)


# --- Job envelope, registry, dispatcher ------------------------------------

def test_job_envelope_accepts_uuid_or_string_id() -> None:
    envelope = JobEnvelope.from_mapping(
        {"job_id": str(JOB_ID), "task": "ping", "payload": {"value": 1}}
    )
    assert envelope.job_id == JOB_ID
    assert envelope.payload == {"value": 1}


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"job_id": "not-a-uuid", "task": "ping"},
        {"job_id": str(JOB_ID), "task": ""},
        {"job_id": str(JOB_ID), "task": "has space"},
        {"job_id": str(JOB_ID), "task": "ping", "payload": []},
        {"job_id": str(JOB_ID), "task": "ping", "payload": {"blob": "x" * 70000}},
        {"job_id": str(JOB_ID), "task": "ping", "payload": {"bad": object()}},
    ],
)
def test_invalid_jobs_are_rejected(value: Any) -> None:
    with pytest.raises(JobValidationError):
        JobEnvelope.from_mapping(value)


def test_registry_is_fail_closed_on_duplicates_and_unknown() -> None:
    registry = TaskRegistry()
    registry.register("ping", lambda payload: {"pong": True})
    assert registry.names() == ("ping",)
    with pytest.raises(JobValidationError):
        registry.register("ping", lambda payload: None)
    with pytest.raises(JobValidationError):
        registry.get("unknown")


def test_dispatcher_executes_registered_job() -> None:
    registry = TaskRegistry()
    registry.register("ping", lambda payload: {"pong": payload.get("n", 0)})
    result = JobDispatcher(registry).execute(
        {"job_id": str(JOB_ID), "task": "ping", "payload": {"n": 3}}
    )
    assert result.job_id == str(JOB_ID)
    assert result.task == "ping"
    assert result.output == {"pong": 3}


def test_default_dispatcher_has_a_runnable_health_job() -> None:
    result = JobDispatcher().execute(
        {"job_id": str(JOB_ID), "task": "ping", "payload": {}}
    )
    assert result.output == {"pong": True}


def test_submit_job_uses_celery_protocol_bridge() -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def send_task(self, name: str, *, args: list[Any]):
            self.calls.append((name, args))
            return "async-result"

    app = FakeApp()
    result = submit_job(
        app,
        {"job_id": str(JOB_ID), "task": "ping", "payload": {"n": 1}},
        config=_config(),
    )
    assert result == "async-result"
    name, args = app.calls[0]
    assert name == "voxsearch.execute_job"
    assert args[0] == {
        "job_id": str(JOB_ID),
        "task": "ping",
        "payload": {"n": 1},
    }


def test_submit_job_rejects_invalid_jobs_and_apps() -> None:
    class FakeApp:
        def send_task(self, *args: Any, **kwargs: Any):
            raise AssertionError("must not send")

    with pytest.raises(JobValidationError):
        submit_job(FakeApp(), {"job_id": "bad", "task": "ping"})
    with pytest.raises(CeleryWorkerError):
        submit_job(object(), {"job_id": str(JOB_ID), "task": "ping"})


def test_dispatcher_rejects_unknown_task_before_execution() -> None:
    with pytest.raises(JobValidationError):
        JobDispatcher().execute({"job_id": str(JOB_ID), "task": "missing"})


# --- Celery app factory -----------------------------------------------------

def test_app_factory_registers_task_and_uses_broker() -> None:
    FakeCelery.instances.clear()
    registry = TaskRegistry()
    registry.register("ping", lambda payload: {"pong": True})
    app = create_celery_app(
        _config(),
        dispatcher=JobDispatcher(registry),
        celery_module=FakeCeleryModule(),
    )
    assert isinstance(app, FakeCelery)
    assert app.kwargs["broker"] == "redis://redis-secret-password@redis.local:6380/2"
    assert app.kwargs["task_default_queue"] == "voxsearch:jobs"
    assert "voxsearch.execute_job" in app.tasks
    output = app.tasks["voxsearch.execute_job"](
        {"job_id": str(JOB_ID), "task": "ping", "payload": {}}
    )
    assert output["output"] == {"pong": True}


def test_missing_celery_is_typed_without_secret_leak(monkeypatch) -> None:
    def fail(name: str):
        raise ImportError("No module named celery")

    monkeypatch.setattr(celery_module_under_test.importlib, "import_module", fail)
    with pytest.raises(WorkerUnavailableError) as exc:
        create_celery_app(_config())
    assert PASSWORD not in str(exc.value)


def test_injected_celery_module_must_expose_celery() -> None:
    with pytest.raises(WorkerUnavailableError):
        create_celery_app(_config(), celery_module=object())
