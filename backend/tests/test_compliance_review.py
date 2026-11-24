"""Tests for Goal 50 — compliance review and launch gate."""
from __future__ import annotations

import pytest

from app.deployment import (
    ComplianceError,
    LaunchBlockedError,
    build_compliance_report,
    launch_gate,
)


def _facts(**overrides):
    facts = {
        "accepted_goals": frozenset(range(1, 51)),
        "live_db_verified": True,
        "live_youtube_verified": True,
        "docker_runtime_verified": True,
        "production_migration_applied": False,
        "secrets_in_repository": False,
        "backup_recovery_tested": True,
        "monitoring_integration_tested": True,
        "evaluation_reports_present": True,
    }
    facts.update(overrides)
    return facts


def test_ready_report_passes_launch_gate_without_deploying() -> None:
    report = build_compliance_report(_facts())
    assert report.public_mvp_ready is True
    assert report.blockers == ()
    assert len(report.checks) == 9
    decision = launch_gate(report)
    assert decision["status"] == "launch_ready"
    assert "no deployment" in decision["note"].lower()


def test_current_repository_state_is_honestly_blocked() -> None:
    report = build_compliance_report(
        _facts(
            accepted_goals=frozenset(range(1, 49)),
            live_db_verified=False,
            live_youtube_verified=False,
            docker_runtime_verified=False,
            monitoring_integration_tested=False,
        )
    )
    assert report.public_mvp_ready is False
    assert {"goals", "live_db", "live_youtube", "docker_runtime", "monitoring"} <= set(
        report.blockers
    )
    with pytest.raises(LaunchBlockedError) as exc:
        launch_gate(report)
    assert "goals" in str(exc.value)
    assert "secret" not in str(exc.value).lower()


@pytest.mark.parametrize(
    "mutation",
    [
        {"accepted_goals": {1, "two"}},
        {"accepted_goals": "all"},
        {"live_db_verified": "yes"},
        {"secrets_in_repository": 0},
    ],
)
def test_invalid_facts_are_rejected(mutation) -> None:
    with pytest.raises(ComplianceError):
        build_compliance_report(_facts(**mutation))


def test_missing_facts_are_rejected() -> None:
    facts = _facts()
    del facts["backup_recovery_tested"]
    with pytest.raises(ComplianceError):
        build_compliance_report(facts)


def test_positive_readiness_requires_evidence_when_supplied() -> None:
    evidence = {key: "deterministic fixture evidence" for key in _facts()}
    report = build_compliance_report(_facts(), evidence=evidence)
    assert report.public_mvp_ready is True
    incomplete = dict(evidence)
    del incomplete["live_db_verified"]
    with pytest.raises(ComplianceError, match="live_db_verified"):
        build_compliance_report(_facts(), evidence=incomplete)
    blank = dict(evidence)
    blank["accepted_goals"] = "   "
    with pytest.raises(ComplianceError, match="accepted_goals"):
        build_compliance_report(_facts(), evidence=blank)
    with pytest.raises(ComplianceError):
        build_compliance_report(_facts(), evidence=["not", "a", "mapping"])


def test_report_is_deterministic_and_immutable() -> None:
    first = build_compliance_report(_facts())
    second = build_compliance_report(_facts())
    assert first == second
    with pytest.raises(Exception):
        first.blockers = ()  # type: ignore[misc]


def test_launch_gate_consumes_goal49_recovery_evidence() -> None:
    from app.deployment import create_backup, restore_backup

    backup = create_backup({"state.json": b"{}"}, name="prelaunch", created_at=1.0)
    restored: dict[str, bytes] = {}
    recovery = restore_backup(
        backup,
        lambda key, value: restored.update({key: value}),
        reader=lambda: restored,
    )
    report = build_compliance_report(
        _facts(backup_recovery_tested=recovery.recovered)
    )
    assert report.public_mvp_ready is True
    assert restored == {"state.json": b"{}"}


def test_launch_gate_rejects_non_reports() -> None:
    with pytest.raises(ComplianceError):
        launch_gate(object())  # type: ignore[arg-type]
