"""Goal 49 — backup + security with a deterministic recovery test.

This is a library-level, injectable backup primitive: no database, filesystem,
network or external service is touched. Entries are in-memory byte payloads,
secrets are rejected by name, and a SHA-256 checksum guards restoration.

"Recovery tested" here means a deterministic backup -> restore -> compare
round-trip is implemented and tested. It is NOT a tested restore of a live
production database. A report never claims content equality unless an injected
reader is supplied and the restored bytes are actually compared. The injected
sink may be only partially written if ``writer`` raises; restoration is not
transactional.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

MAX_ENTRIES = 256
MAX_ENTRY_BYTES = 1_000_000
MAX_TOTAL_BYTES = 4_000_000
_SECRET_NAME = re.compile(
    r"(\.env|password|passwd|secret|token|api[_-]?key|credential|private[_-]?key|database|\bdsn\b|connection)",
    re.IGNORECASE,
)


class BackupError(ValueError):
    """Backup input or restoration failed validation."""


@dataclass(frozen=True)
class BackupManifest:
    """Immutable, checksummed backup payload."""

    name: str
    version: str
    created_at: float
    entries: Mapping[str, bytes]
    checksum: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise BackupError("backup name must be non-empty")
        if not isinstance(self.version, str) or not self.version.strip():
            raise BackupError("backup version must be non-empty")
        if isinstance(self.created_at, bool) or not isinstance(
            self.created_at, (int, float)
        ):
            raise BackupError("created_at must be numeric")


@dataclass(frozen=True)
class RecoveryReport:
    """Immutable outcome of a restore/recovery check."""

    name: str
    restored_entries: int
    checksum_valid: bool
    content_matches: bool

    @property
    def recovered(self) -> bool:
        return self.checksum_valid and self.content_matches


def _canonical(entries: Mapping[str, bytes]) -> bytes:
    return json.dumps(
        {key: value.hex() for key, value in sorted(entries.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def create_backup(
    entries: Mapping[str, bytes],
    *,
    name: str,
    version: str = "1.0",
    created_at: float,
) -> BackupManifest:
    """Validate entries and build a checksummed in-memory backup."""
    if not isinstance(entries, Mapping) or not entries:
        raise BackupError("backup entries must be a non-empty mapping")
    if len(entries) > MAX_ENTRIES:
        raise BackupError("too many backup entries")
    total = 0
    clean: dict[str, bytes] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not key or _SECRET_NAME.search(key):
            raise BackupError("backup entry name is invalid or secret-like")
        if not isinstance(value, bytes):
            raise BackupError("backup entry values must be bytes")
        if not value or len(value) > MAX_ENTRY_BYTES:
            raise BackupError("backup entry size is out of bounds")
        total += len(value)
        clean[key] = bytes(value)
    if total > MAX_TOTAL_BYTES:
        raise BackupError("backup total size is out of bounds")
    return BackupManifest(
        name=name,
        version=version,
        created_at=float(created_at),
        entries=MappingProxyType(clean),
        checksum=hashlib.sha256(_canonical(clean)).hexdigest(),
    )


def verify_backup(manifest: BackupManifest) -> bool:
    """Recompute the checksum without exposing entry contents."""
    if not isinstance(manifest, BackupManifest):
        raise BackupError("manifest must be a BackupManifest")
    return hashlib.sha256(_canonical(manifest.entries)).hexdigest() == manifest.checksum


def restore_backup(
    manifest: BackupManifest,
    writer: Callable[[str, bytes], None],
    *,
    reader: Callable[[], Mapping[str, bytes]] | None = None,
) -> RecoveryReport:
    """Verify then restore entries into an injected sink.

    ``content_matches`` is only True when ``reader`` is supplied and every
    restored byte matches the manifest. Without a reader only checksum
    integrity is established, so recovery is NOT claimed.
    """
    if not isinstance(manifest, BackupManifest):
        raise BackupError("manifest must be a BackupManifest")
    if not callable(writer):
        raise BackupError("writer must be callable")
    if reader is not None and not callable(reader):
        raise BackupError("reader must be callable")
    if not verify_backup(manifest):
        return RecoveryReport(
            name=manifest.name,
            restored_entries=0,
            checksum_valid=False,
            content_matches=False,
        )
    for key, value in manifest.entries.items():
        writer(key, bytes(value))
    content_matches = False
    if reader is not None:
        restored = reader()
        content_matches = isinstance(restored, Mapping) and all(
            restored.get(key) == value for key, value in manifest.entries.items()
        )
    return RecoveryReport(
        name=manifest.name,
        restored_entries=len(manifest.entries),
        checksum_valid=True,
        content_matches=content_matches,
    )


__all__ = [
    "MAX_ENTRIES",
    "MAX_ENTRY_BYTES",
    "MAX_TOTAL_BYTES",
    "BackupError",
    "BackupManifest",
    "RecoveryReport",
    "create_backup",
    "restore_backup",
    "verify_backup",
]
