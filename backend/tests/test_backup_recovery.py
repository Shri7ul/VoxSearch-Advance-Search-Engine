"""Tests for Goal 49 — backup integrity and recovery round-trip."""
from __future__ import annotations

import pytest

from app.deployment import (
    BackupError,
    create_backup,
    restore_backup,
    verify_backup,
)
from app.deployment.backup import MAX_ENTRIES, MAX_ENTRY_BYTES, MAX_TOTAL_BYTES

ENTRIES = {"config.json": b'{"k":1}', "chunks.json": b"[1,2,3]"}


def _manifest(**overrides):
    return create_backup(ENTRIES, name="daily", created_at=100.0, **overrides)


def test_backup_restore_round_trip_is_recovered() -> None:
    manifest = _manifest()
    restored: dict[str, bytes] = {}
    report = restore_backup(
        manifest,
        lambda key, value: restored.update({key: value}),
        reader=lambda: restored,
    )
    assert report.recovered is True
    assert report.restored_entries == 2
    assert restored == ENTRIES


def test_recovery_is_not_claimed_without_content_verification() -> None:
    restored: dict[str, bytes] = {}
    report = restore_backup(
        _manifest(), lambda key, value: restored.update({key: value})
    )
    assert report.checksum_valid is True
    assert report.content_matches is False
    assert report.recovered is False
    assert restored == ENTRIES  # bytes were written, but not verified


def test_read_back_mismatch_is_reported() -> None:
    restored: dict[str, bytes] = {}
    report = restore_backup(
        _manifest(),
        lambda key, value: restored.update({key: value}),
        reader=lambda: {"config.json": b"corrupted", "chunks.json": b"[1,2,3]"},
    )
    assert report.content_matches is False
    assert report.recovered is False


def test_checksum_is_deterministic_and_detects_tampering() -> None:
    first = _manifest()
    second = _manifest()
    assert first.checksum == second.checksum
    assert verify_backup(first) is True
    object.__setattr__(first, "checksum", "bad")
    assert verify_backup(first) is False
    restored: dict[str, bytes] = {}
    report = restore_backup(first, lambda k, v: restored.update({k: v}))
    assert report.recovered is False
    assert restored == {}


def test_entries_are_immutable_and_secret_names_rejected() -> None:
    manifest = _manifest()
    with pytest.raises(TypeError):
        manifest.entries["config.json"] = b"mutated"  # type: ignore[index]
    for name in (".env", "DATABASE_URL", "api_key", "secret.txt", "password.json"):
        with pytest.raises(BackupError, match="invalid or secret-like"):
            create_backup({name: b"x"}, name="daily", created_at=1.0)


@pytest.mark.parametrize(
    "entries",
    [
        {},
        {f"k{i}": b"x" for i in range(MAX_ENTRIES + 1)},
        {"a": b""},
        {"a": b"x" * (MAX_ENTRY_BYTES + 1)},
    ],
)
def test_invalid_entries_are_rejected(entries) -> None:
    with pytest.raises(BackupError):
        create_backup(entries, name="daily", created_at=1.0)


def test_total_size_limit_is_enforced() -> None:
    entries = {f"k{i}": b"x" * 900_000 for i in range(5)}
    with pytest.raises(BackupError, match="total size"):
        create_backup(entries, name="daily", created_at=1.0)


def test_invalid_manifest_and_writer_are_rejected() -> None:
    with pytest.raises(BackupError):
        verify_backup(object())  # type: ignore[arg-type]
    with pytest.raises(BackupError):
        restore_backup(_manifest(), None)  # type: ignore[arg-type]


def test_writer_failure_propagates() -> None:
    def fail(key: str, value: bytes) -> None:
        raise RuntimeError("sink unavailable")

    with pytest.raises(RuntimeError, match="sink unavailable"):
        restore_backup(_manifest(), fail)
