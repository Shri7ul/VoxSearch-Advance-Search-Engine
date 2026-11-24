"""Goal 40 — lazily constructed Celery worker app.

Celery is an OPTIONAL dependency declared in ``requirements-worker.txt`` and is
imported only inside :func:`create_celery_app`. Constructing the app does not
connect to Redis and does not start a worker. No retry policy (Goal 41) and no
indexing automation (Goal 42) are configured here.
"""
from __future__ import annotations

import importlib
import urllib.parse
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.queue.redis import RedisConfig
from app.workers.dispatcher import JobDispatcher
from app.workers.jobs import JobEnvelope

DEFAULT_TASK_NAME = "voxsearch.execute_job"


class CeleryWorkerError(RuntimeError):
    """Base class for credential-free Goal 40 failures."""


class WorkerUnavailableError(CeleryWorkerError):
    """Celery is not installed in this environment."""


@dataclass(frozen=True)
class CeleryConfig:
    """Validated worker configuration; secrets stay masked."""

    redis: RedisConfig
    task_name: str = DEFAULT_TASK_NAME

    def __post_init__(self) -> None:
        if not isinstance(self.redis, RedisConfig):
            raise CeleryWorkerError("redis configuration is invalid")
        if (
            not isinstance(self.task_name, str)
            or not self.task_name.strip()
            or len(self.task_name) > 128
        ):
            raise CeleryWorkerError("task name is invalid")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "CeleryConfig":
        return cls(redis=RedisConfig.from_settings(settings))

    def broker_url(self) -> str:
        """Build a Redis broker URL; the password is urlencoded, never logged."""
        password = self.redis.password
        credentials = ""
        if password is not None:
            credentials = (
                urllib.parse.quote(password.get_secret_value(), safe="") + "@"
            )
        return (
            f"redis://{credentials}{self.redis.host}:{self.redis.port}/{self.redis.db}"
        )


def create_celery_app(
    config: CeleryConfig | None = None,
    *,
    dispatcher: JobDispatcher | None = None,
    celery_module: Any | None = None,
) -> Any:
    """Create a Celery app with one registered background-job task.

    ``celery_module`` is an injection seam for deterministic tests. Real use
    imports ``celery`` lazily so the core backend never depends on it.
    """
    cfg = config or CeleryConfig.from_settings()
    if celery_module is None:
        try:
            celery_module = importlib.import_module("celery")
        except ImportError as exc:
            raise WorkerUnavailableError(
                "Celery is not installed; install requirements-worker.txt"
            ) from exc
    if not hasattr(celery_module, "Celery"):
        raise WorkerUnavailableError("injected celery module is invalid")
    active_dispatcher = dispatcher or JobDispatcher()
    app = celery_module.Celery(
        "voxsearch",
        broker=cfg.broker_url(),
        task_default_queue=cfg.redis.queue_key,
    )

    def execute_job(payload: dict[str, Any]) -> dict[str, Any]:
        result = active_dispatcher.execute(payload)
        return {
            "job_id": result.job_id,
            "task": result.task,
            "output": result.output,
        }

    app.task(name=cfg.task_name)(execute_job)
    return app


def submit_job(
    app: Any,
    job: Any,
    *,
    config: CeleryConfig | None = None,
) -> Any:
    """Submit a validated job through Celery's own Redis message envelope.

    Goal 39's raw JSON LPUSH/BRPOP primitives are generic; Celery expects its
    own protocol envelope. This bridge keeps both contracts intact instead of
    feeding raw JSON into the worker queue.
    """
    if app is None or not hasattr(app, "send_task"):
        raise CeleryWorkerError("Celery app does not support send_task")
    envelope = (
        job if hasattr(job, "job_id") else JobEnvelope.from_mapping(job)
    )
    if not hasattr(envelope, "job_id") or not hasattr(envelope, "task"):
        raise CeleryWorkerError("invalid job envelope")
    task_name = config.task_name if config is not None else DEFAULT_TASK_NAME
    payload = {
        "job_id": str(envelope.job_id),
        "task": envelope.task,
        "payload": dict(getattr(envelope, "payload", {})),
    }
    return app.send_task(task_name, args=[payload])


__all__ = [
    "DEFAULT_TASK_NAME",
    "CeleryConfig",
    "CeleryWorkerError",
    "WorkerUnavailableError",
    "create_celery_app",
    "submit_job",
]
