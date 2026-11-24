"""Goal 39 — Redis queue setup with a dependency-free RESP client.

The queue is operational at the primitive level: validated configuration,
health probe (PING), bounded JSON `LPUSH` enqueue, blocking `BRPOP` dequeue and
`LLEN` size. No new pip dependency is introduced; a minimal RESP2 client uses
stdlib sockets by default and is fully injectable for deterministic tests.

No connection is opened at import time. Celery workers, retries and automated
indexing belong to Goals 40–42 and are not implemented here.
"""
from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import SecretStr

from app.core.config import Settings, get_settings

DEFAULT_QUEUE_KEY = "voxsearch:jobs"
MAX_PAYLOAD_BYTES = 65536
MAX_DEQUEUE_TIMEOUT_SECONDS = 60
DEFAULT_SOCKET_TIMEOUT_SECONDS = 5.0


class QueueError(RuntimeError):
    """Base class for credential-free Goal 39 failures."""


class QueueUnavailableError(QueueError):
    """Redis could not be reached."""


class QueueProtocolError(QueueError):
    """Redis returned a malformed or unexpected reply."""


@dataclass(frozen=True)
class RedisConfig:
    """Validated, immutable Redis connection settings."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: SecretStr | None = None
    queue_key: str = DEFAULT_QUEUE_KEY
    socket_timeout_seconds: float = DEFAULT_SOCKET_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise QueueError("Redis host must be non-empty")
        if any(char.isspace() for char in self.host):
            raise QueueError("Redis host cannot contain whitespace")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise QueueError("Redis port is out of bounds")
        if isinstance(self.db, bool) or not isinstance(self.db, int) or not 0 <= self.db <= 15:
            raise QueueError("Redis db is out of bounds")
        if self.password is not None and not isinstance(self.password, SecretStr):
            raise QueueError("Redis password must be a SecretStr")
        if (
            not isinstance(self.queue_key, str)
            or not self.queue_key.strip()
            or len(self.queue_key) > 128
        ):
            raise QueueError("Redis queue key is invalid")
        if (
            isinstance(self.socket_timeout_seconds, bool)
            or not isinstance(self.socket_timeout_seconds, (int, float))
            or not 0.1 <= self.socket_timeout_seconds <= 30
        ):
            raise QueueError("Redis socket timeout is out of bounds")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RedisConfig":
        cfg = settings or get_settings()
        return cls(
            host=cfg.redis_host,
            port=cfg.redis_port,
            db=cfg.redis_db,
            password=cfg.redis_password,
            queue_key=cfg.redis_queue_key,
        )


@runtime_checkable
class RedisClient(Protocol):
    """Minimal command boundary used by the queue."""

    def execute(self, *parts: str) -> Any: ...


@dataclass(frozen=True)
class QueueStatus:
    """Result of a successful Redis health probe."""

    reachable: bool = True
    queue_key: str = DEFAULT_QUEUE_KEY


def encode_command(*parts: str) -> bytes:
    """Encode one RESP2 command array; all parts are length-prefixed."""
    if not parts:
        raise QueueProtocolError("empty Redis command")
    out = f"*{len(parts)}\r\n".encode("ascii")
    for part in parts:
        if not isinstance(part, str):
            raise QueueProtocolError("Redis command parts must be strings")
        data = part.encode("utf-8")
        out += f"${len(data)}\r\n".encode("ascii") + data + b"\r\n"
    return out


def parse_resp_reply(payload: bytes) -> Any:
    """Parse one RESP2 reply without logging command content."""
    if not isinstance(payload, bytes) or not payload:
        raise QueueProtocolError("Redis returned an empty reply")
    if not payload.endswith(b"\r\n"):
        raise QueueProtocolError("Redis reply is not newline-terminated")
    head, _, rest = payload.partition(b"\r\n")
    prefix, value = head[:1], head[1:]
    if prefix == b"+":
        return value.decode("utf-8")
    if prefix == b"-":
        # Deliberately do not surface the Redis error text.
        raise QueueProtocolError("Redis returned an error reply")
    if prefix == b":":
        try:
            return int(value)
        except ValueError as exc:
            raise QueueProtocolError("Redis returned an invalid integer") from exc
    if prefix == b"$":
        try:
            length = int(value)
        except ValueError as exc:
            raise QueueProtocolError("Redis returned an invalid bulk length") from exc
        if length == -1:
            return None
        if length > MAX_PAYLOAD_BYTES:
            raise QueueProtocolError("Redis bulk reply is too large")
        if length < 0 or len(rest) < length + 2:
            raise QueueProtocolError("Redis bulk reply is truncated")
        return rest[:length].decode("utf-8")
    if prefix == b"*":
        try:
            count = int(value)
        except ValueError as exc:
            raise QueueProtocolError("Redis returned an invalid array length") from exc
        if count == -1:
            return None
        if not 0 <= count <= 1024:
            raise QueueProtocolError("Redis array length is out of bounds")
        return _parse_array(rest, count)
    raise QueueProtocolError("Redis reply has an unknown type")


def _parse_array(data: bytes, count: int) -> list[Any]:
    items: list[Any] = []
    remaining = data
    for _ in range(count):
        end = remaining.find(b"\r\n")
        if end < 0:
            raise QueueProtocolError("Redis array reply is truncated")
        head, remaining = remaining[: end + 2], remaining[end + 2 :]
        prefix = head[:1]
        if prefix in (b"+", b"-", b":"):
            items.append(parse_resp_reply(head))
        elif prefix in (b"$", b"*"):
            consumed, item = _parse_nested(head, remaining)
            items.append(item)
            remaining = consumed
        else:
            raise QueueProtocolError("Redis array item has an unknown type")
    return items


def _parse_nested(head: bytes, remaining: bytes) -> tuple[bytes, Any]:
    prefix = head[:1]
    value = head[1:-2]
    if prefix == b"$":
        length = int(value)
        if length == -1:
            return remaining, None
        if length < 0 or len(remaining) < length + 2:
            raise QueueProtocolError("Redis bulk item is truncated")
        return remaining[length + 2 :], remaining[:length].decode("utf-8")
    count = int(value)
    if count == -1:
        return remaining, None
    if not 0 <= count <= 1024:
        raise QueueProtocolError("Redis nested array length is out of bounds")
    return remaining, _parse_array(remaining, count)


class SocketRedisClient:
    """Default stdlib RESP2 client: one short-lived connection per command."""

    def __init__(self, config: RedisConfig) -> None:
        self._config = config

    def execute(self, *parts: str) -> Any:
        request = encode_command(*parts)
        try:
            sock = socket.create_connection(
                (self._config.host, self._config.port),
                timeout=self._config.socket_timeout_seconds,
            )
        except (OSError, socket.timeout) as exc:
            raise QueueUnavailableError(
                f"Redis connection failed ({type(exc).__name__})"
            ) from exc
        with sock:
            sock.settimeout(self._config.socket_timeout_seconds)
            try:
                with sock.makefile("rb") as stream:
                    reader = _RespReader(stream)
                    if self._config.password is not None:
                        sock.sendall(
                            encode_command(
                                "AUTH", self._config.password.get_secret_value()
                            )
                        )
                        parse_resp_reply(reader.read_reply())
                    if self._config.db:
                        sock.sendall(encode_command("SELECT", str(self._config.db)))
                        parse_resp_reply(reader.read_reply())
                    sock.sendall(request)
                    return parse_resp_reply(reader.read_reply())
            except QueueError:
                raise
            except (OSError, socket.timeout) as exc:
                raise QueueUnavailableError(
                    f"Redis request failed ({type(exc).__name__})"
                ) from exc


class _RespReader:
    """Read one complete RESP2 reply (including nested arrays) from a stream."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def read_reply(self) -> bytes:
        prefix = self._stream.read(1)
        if not prefix:
            raise QueueProtocolError("Redis closed the connection")
        line = self._stream.readline()
        if not line.endswith(b"\r\n"):
            raise QueueProtocolError("Redis reply is not newline-terminated")
        head = prefix + line
        if prefix not in (b"$", b"*"):
            return head
        try:
            count = int(line[:-2])
        except ValueError as exc:
            raise QueueProtocolError("Redis returned an invalid length") from exc
        if count == -1:
            return head
        if count < 0 or count > 1024:
            raise QueueProtocolError("Redis length is out of bounds")
        if prefix == b"$":
            if count > MAX_PAYLOAD_BYTES:
                raise QueueProtocolError("Redis bulk reply is too large")
            return head + self._stream.read(count + 2)
        return head + b"".join(self.read_reply() for _ in range(count))


class RedisQueue:
    """Operational Redis queue primitives over an injectable client."""

    def __init__(self, config: RedisConfig, client: RedisClient | None = None) -> None:
        self._config = config
        self._client: RedisClient = client or SocketRedisClient(config)

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, *, client: RedisClient | None = None
    ) -> "RedisQueue":
        return cls(RedisConfig.from_settings(settings), client=client)

    def _call(self, *parts: str) -> Any:
        try:
            return self._client.execute(*parts)
        except QueueError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak a DSN or password
            raise QueueUnavailableError(
                f"Redis request failed ({type(exc).__name__})"
            ) from exc

    def health(self) -> QueueStatus:
        reply = self._call("PING")
        if reply != "PONG":
            raise QueueProtocolError("Redis PING did not return PONG")
        return QueueStatus(queue_key=self._config.queue_key)

    def enqueue(self, payload: Mapping[str, Any]) -> int:
        """Push a bounded JSON payload; return the queue length after push."""
        if not isinstance(payload, Mapping):
            raise QueueError("queue payload must be a mapping")
        try:
            body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise QueueError("queue payload is not JSON-serializable") from exc
        if len(body.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise QueueError("queue payload is too large")
        reply = self._call("LPUSH", self._config.queue_key, body)
        if not isinstance(reply, int) or isinstance(reply, bool) or reply < 1:
            raise QueueProtocolError("Redis LPUSH returned an invalid length")
        return reply

    def dequeue(self, timeout_seconds: int = 1) -> dict[str, Any] | None:
        """Pop one JSON payload with BRPOP; ``None`` means the wait timed out."""
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= MAX_DEQUEUE_TIMEOUT_SECONDS
        ):
            raise QueueError("dequeue timeout is out of bounds")
        # A blocking BRPOP cannot outlive the socket timeout; reject instead
        # of silently failing after a few seconds.
        if timeout_seconds > self._config.socket_timeout_seconds:
            raise QueueError(
                "dequeue timeout cannot exceed the Redis socket timeout"
            )
        reply = self._call(
            "BRPOP", self._config.queue_key, str(timeout_seconds)
        )
        if reply is None:
            return None
        if not isinstance(reply, list) or len(reply) != 2:
            raise QueueProtocolError("Redis BRPOP returned an invalid reply")
        key, body = reply
        if key != self._config.queue_key or not isinstance(body, str):
            raise QueueProtocolError("Redis BRPOP returned an unexpected item")
        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise QueueProtocolError("queued payload is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise QueueProtocolError("queued payload is not an object")
        return payload

    def size(self) -> int:
        reply = self._call("LLEN", self._config.queue_key)
        if not isinstance(reply, int) or isinstance(reply, bool) or reply < 0:
            raise QueueProtocolError("Redis LLEN returned an invalid length")
        return reply


__all__ = [
    "DEFAULT_QUEUE_KEY",
    "MAX_PAYLOAD_BYTES",
    "QueueError",
    "QueueProtocolError",
    "QueueStatus",
    "QueueUnavailableError",
    "RedisClient",
    "RedisConfig",
    "RedisQueue",
    "SocketRedisClient",
    "encode_command",
    "parse_resp_reply",
]
