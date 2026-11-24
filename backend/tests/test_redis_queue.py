"""Tests for Goal 39 — Redis queue setup with an injected client."""
from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.queue import (
    QueueError,
    QueueProtocolError,
    QueueUnavailableError,
    RedisConfig,
    RedisQueue,
    encode_command,
    parse_resp_reply,
)

PASSWORD = "redis-secret-value"
KEY = "voxsearch:jobs"


def _config(**overrides: Any) -> RedisConfig:
    base: dict[str, Any] = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": SecretStr(PASSWORD),
        "queue_key": KEY,
    }
    base.update(overrides)
    return RedisConfig(**base)


class FakeRedis:
    def __init__(self, reply: Any = "PONG") -> None:
        self.reply = reply
        self.calls: list[tuple[str, ...]] = []

    def execute(self, *parts: str) -> Any:
        self.calls.append(parts)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# --- Configuration ----------------------------------------------------------

def test_config_masks_password() -> None:
    config = _config()
    assert PASSWORD not in repr(config)
    assert PASSWORD not in str(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"host": ""},
        {"host": "has space"},
        {"port": 0},
        {"port": 70000},
        {"port": True},
        {"db": -1},
        {"db": 16},
        {"password": "plain"},
        {"queue_key": ""},
        {"queue_key": "x" * 129},
        {"socket_timeout_seconds": 0},
        {"socket_timeout_seconds": 31},
    ],
)
def test_invalid_config_is_rejected(kwargs: Any) -> None:
    with pytest.raises(QueueError):
        _config(**kwargs)


def test_from_settings() -> None:
    config = RedisConfig.from_settings(
        Settings(
            redis_host="redis.local",
            redis_port=6380,
            redis_db=2,
            redis_password=SecretStr(PASSWORD),
            redis_queue_key="custom:queue",
        )
    )
    assert config.host == "redis.local"
    assert config.queue_key == "custom:queue"


# --- RESP encoding and parsing ---------------------------------------------

def test_encode_command_is_length_prefixed() -> None:
    assert encode_command("LPUSH", KEY, "hi") == (
        b"*3\r\n$5\r\nLPUSH\r\n$14\r\nvoxsearch:jobs\r\n$2\r\nhi\r\n"
    )


@pytest.mark.parametrize(
    "payload,expected",
    [
        (b"+PONG\r\n", "PONG"),
        (b":12\r\n", 12),
        (b"$5\r\nhello\r\n", "hello"),
        (b"$-1\r\n", None),
        (b"*-1\r\n", None),
        (b"*2\r\n$3\r\nkey\r\n$2\r\nhi\r\n", ["key", "hi"]),
    ],
)
def test_parse_resp_reply(payload: bytes, expected: Any) -> None:
    assert parse_resp_reply(payload) == expected


def test_oversized_declared_bulk_reply_is_rejected() -> None:
    with pytest.raises(QueueProtocolError, match="too large"):
        parse_resp_reply(b"$99999999\r\n")


@pytest.mark.parametrize(
    "payload",
    [b"", b"garbage", b"-ERR secret=value\r\n", b"*:bad\r\n", b"$5\r\nab"],
)
def test_malformed_or_error_replies_are_rejected(payload: bytes) -> None:
    with pytest.raises(QueueProtocolError) as exc:
        parse_resp_reply(payload)
    assert PASSWORD not in str(exc.value)
    assert "ERR" not in str(exc.value)


# --- Queue primitives -------------------------------------------------------

def test_health_probe() -> None:
    fake = FakeRedis("PONG")
    status = RedisQueue(_config(), client=fake).health()
    assert status.reachable is True
    assert status.queue_key == KEY
    assert fake.calls == [("PING",)]


def test_failed_health_probe_is_rejected() -> None:
    with pytest.raises(QueueProtocolError):
        RedisQueue(_config(), client=FakeRedis("NO")).health()


def test_unavailable_redis_maps_to_typed_error() -> None:
    fake = FakeRedis(ConnectionError("host:6379"))
    with pytest.raises(QueueUnavailableError) as exc:
        RedisQueue(_config(), client=fake).health()
    assert "6379" not in str(exc.value)
    assert PASSWORD not in str(exc.value)


def test_enqueue_sends_bounded_json_payload() -> None:
    fake = FakeRedis(2)
    length = RedisQueue(_config(), client=fake).enqueue({"job": "index", "id": 1})
    assert length == 2
    assert fake.calls[0][0] == "LPUSH"
    assert fake.calls[0][1] == KEY
    assert json.loads(fake.calls[0][2]) == {"id": 1, "job": "index"}


def test_enqueue_rejects_invalid_payloads() -> None:
    queue = RedisQueue(_config(), client=FakeRedis(1))
    with pytest.raises(QueueError):
        queue.enqueue("not-a-mapping")  # type: ignore[arg-type]
    with pytest.raises(QueueError):
        queue.enqueue({"blob": "x" * 70000})
    with pytest.raises(QueueError):
        queue.enqueue({"bad": object()})
    assert not queue._client.calls  # type: ignore[attr-defined]


def test_dequeue_returns_payload_from_brpop_array() -> None:
    fake = FakeRedis([KEY, '{"job": "index"}'])
    result = RedisQueue(_config(), client=fake).dequeue(timeout_seconds=3)
    assert result == {"job": "index"}
    assert fake.calls == [("BRPOP", KEY, "3")]


def test_dequeue_empty_timeout_returns_none() -> None:
    assert RedisQueue(_config(), client=FakeRedis(None)).dequeue() is None


@pytest.mark.parametrize("timeout", [-1, 0, 61, True, 1.5])
def test_dequeue_timeout_is_validated(timeout: Any) -> None:
    fake = FakeRedis(None)
    with pytest.raises(QueueError):
        RedisQueue(_config(), client=fake).dequeue(timeout_seconds=timeout)
    assert fake.calls == []


def test_dequeue_cannot_outlive_socket_timeout() -> None:
    fake = FakeRedis(None)
    queue = RedisQueue(_config(socket_timeout_seconds=5.0), client=fake)
    with pytest.raises(QueueError, match="socket timeout"):
        queue.dequeue(timeout_seconds=6)
    assert fake.calls == []


@pytest.mark.parametrize(
    "reply",
    [
        [KEY, "not-json"],
        [KEY, "[1,2]"],
        ["other", "{}"],
        ["only-one"],
    ],
)
def test_malformed_brpop_replies_are_rejected(reply: Any) -> None:
    with pytest.raises(QueueProtocolError):
        RedisQueue(_config(), client=FakeRedis(reply)).dequeue()


def test_size_reports_llen() -> None:
    fake = FakeRedis(4)
    assert RedisQueue(_config(), client=fake).size() == 4
    assert fake.calls == [("LLEN", KEY)]
