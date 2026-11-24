"""Goal 50 — compliance review and launch gate (final roadmap goal).

This module produces an auditable readiness report from explicit facts and
refuses to declare a public MVP ready while any blocker remains. It performs
NO deployment, network call, database write or migration. "Launch" here is a
documented decision gate, not an executed production release.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

EXPECTED_GOALS = frozenset(range(1, 51))


class ComplianceError(ValueError):
    """Compliance facts are malformed."""


class LaunchBlockedError(ComplianceError):
    """Public launch is blocked by unresolved checks."""


@dataclass(frozen=True)
class ComplianceCheck:
    """One immutable compliance check result."""

    check_id: str
    description: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ComplianceReport:
    """Immutable compliance summary with explicit blockers."""

    checks: tuple[ComplianceCheck, ...]
    blockers: tuple[str, ...]
    public_mvp_ready: bool

    def summary(self) -> dict[str, Any]:
        return {
            "checks_total": len(self.checks),
            "checks_passed": sum(1 for check in self.checks if check.passed),
            "blockers": list(self.blockers),
            "public_mvp_ready": self.public_mvp_ready,
        }


_REQUIRED_FACTS = (
    "accepted_goals",
    "live_db_verified",
    "live_youtube_verified",
    "docker_runtime_verified",
    "production_migration_applied",
    "secrets_in_repository",
    "backup_recovery_tested",
    "monitoring_integration_tested",
    "evaluation_reports_present",
)


def build_compliance_report(
    facts: Mapping[str, Any], *, evidence: Mapping[str, str] | None = None
) -> ComplianceReport:
    """Evaluate explicit facts into a deterministic readiness report.

    When ``evidence`` is supplied, every asserted fact must carry a non-empty
    evidence string. This prevents a positive readiness decision from resting
    on bare boolean flags alone. Without ``evidence`` the caller explicitly
    accepts that weaker mode; the gate still validates all fact types.
    """
    if not isinstance(facts, Mapping) or any(key not in facts for key in _REQUIRED_FACTS):
        raise ComplianceError("compliance facts are incomplete")
    goals = facts["accepted_goals"]
    if not isinstance(goals, (set, frozenset)) or any(
        not isinstance(goal, int) for goal in goals
    ):
        raise ComplianceError("accepted_goals must be a set of integers")
    for key in _REQUIRED_FACTS:
        if key != "accepted_goals" and not isinstance(facts[key], bool):
            raise ComplianceError(f"{key} must be a boolean")
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            raise ComplianceError("evidence must be a mapping")
        for key in _REQUIRED_FACTS:
            value = evidence.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ComplianceError(f"missing evidence for {key}")

    checks = [
        ComplianceCheck(
            "goals",
            "All 50 roadmap goals are accepted",
            frozenset(goals) == EXPECTED_GOALS,
            f"{len(set(goals) & EXPECTED_GOALS)}/50 accepted",
        ),
        ComplianceCheck(
            "live_db",
            "Live PostgreSQL/pgvector runtime verified",
            bool(facts["live_db_verified"]),
            "opt-in live verification not performed"
            if not facts["live_db_verified"]
            else "verified",
        ),
        ComplianceCheck(
            "live_youtube",
            "Live YouTube API/OAuth verified",
            bool(facts["live_youtube_verified"]),
            "external credentials not available"
            if not facts["live_youtube_verified"]
            else "verified",
        ),
        ComplianceCheck(
            "docker_runtime",
            "Production Docker image built and run",
            bool(facts["docker_runtime_verified"]),
            "static validation only" if not facts["docker_runtime_verified"] else "verified",
        ),
        ComplianceCheck(
            "migrations",
            "No unapproved production migration applied",
            not bool(facts["production_migration_applied"]),
            "migration state must be explicitly approved",
        ),
        ComplianceCheck(
            "secrets",
            "No secrets committed to the repository",
            not bool(facts["secrets_in_repository"]),
            "secret scan required",
        ),
        ComplianceCheck(
            "backup",
            "Backup recovery round-trip tested",
            bool(facts["backup_recovery_tested"]),
            "recovery evidence required",
        ),
        ComplianceCheck(
            "monitoring",
            "Monitoring integration tested",
            bool(facts["monitoring_integration_tested"]),
            "library-level tests only" if not facts["monitoring_integration_tested"] else "verified",
        ),
        ComplianceCheck(
            "evaluation",
            "Search quality evaluation reports present",
            bool(facts["evaluation_reports_present"]),
            "evaluation evidence required",
        ),
    ]
    blockers = tuple(check.check_id for check in checks if not check.passed)
    return ComplianceReport(
        checks=tuple(checks),
        blockers=blockers,
        public_mvp_ready=not blockers,
    )


def launch_gate(report: ComplianceReport) -> dict[str, str]:
    """Return a launch decision; never performs the launch itself."""
    if not isinstance(report, ComplianceReport):
        raise ComplianceError("report must be a ComplianceReport")
    if not report.public_mvp_ready:
        raise LaunchBlockedError(
            "public launch blocked by: " + ", ".join(report.blockers)
        )
    return {
        "status": "launch_ready",
        "note": "Decision gate only; no deployment was executed.",
    }


__all__ = [
    "EXPECTED_GOALS",
    "ComplianceCheck",
    "ComplianceError",
    "ComplianceReport",
    "LaunchBlockedError",
    "build_compliance_report",
    "launch_gate",
]
