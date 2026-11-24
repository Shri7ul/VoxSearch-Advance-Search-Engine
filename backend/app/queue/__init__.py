"""VoxSearch queue package (Goal 39 — Redis setup).

Only connection setup and the minimal queue primitives are implemented here.
No Celery workers (Goal 40), retries (41) or indexing automation (42).
"""
from app.queue.redis import (
    QueueError,
    QueueProtocolError,
    QueueStatus,
    QueueUnavailableError,
    RedisClient,
    RedisConfig,
    RedisQueue,
    encode_command,
    parse_resp_reply,
)

__all__ = [
    "QueueError",
    "QueueProtocolError",
    "QueueStatus",
    "QueueUnavailableError",
    "RedisClient",
    "RedisConfig",
    "RedisQueue",
    "encode_command",
    "parse_resp_reply",
]
