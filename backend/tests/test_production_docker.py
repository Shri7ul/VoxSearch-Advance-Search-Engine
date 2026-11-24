"""Tests for Goal 47 — production Docker artifacts (static validation)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "infrastructure" / "docker"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing artifact: {path}"
    return path.read_text(encoding="utf-8")


def test_backend_dockerfile_is_pinned_and_non_root() -> None:
    text = _read(DOCKER / "backend.Dockerfile")
    assert "FROM python:3.12.10-slim" in text
    assert "requirements-lock.txt" in text
    assert "pip install" in text and "--no-cache-dir" in text
    assert "USER appuser" in text
    assert "HEALTHCHECK" in text and "/health" in text
    assert 'CMD ["uvicorn", "app.main:app"' in text
    assert "latest" not in text.lower()


def test_frontend_dockerfile_uses_lockfile_and_multi_stage_build() -> None:
    text = _read(DOCKER / "frontend.Dockerfile")
    assert "FROM node:20.19.5-alpine" in text
    assert "npm ci" in text
    assert "package-lock.json" in text
    assert " AS builder" in text and " AS runner" in text
    assert "NEXT_TELEMETRY_DISABLED=1" in text
    assert "latest" not in text.lower()


def test_compose_defines_required_services_and_healthchecks() -> None:
    text = _read(DOCKER / "docker-compose.yml")
    for service in ("redis:", "api:", "web:"):
        assert service in text
    assert text.count("healthcheck:") >= 3  # redis, api and web
    assert "backend/.env" in text
    assert "redis-data" in text
    assert "privileged" not in text.lower()
    assert "build:" in text and "backend.Dockerfile" in text
    assert "frontend.Dockerfile" in text


def test_docker_artifacts_contain_no_hardcoded_secrets() -> None:
    pattern = re.compile(
        r"(password|passwd|secret|token|api[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9]",
        re.IGNORECASE,
    )
    for path in DOCKER.glob("*"):
        if path.is_file():
            assert not pattern.search(_read(path)), f"secret-like value in {path}"


def test_dockerignore_excludes_secrets_and_virtualenvs() -> None:
    text = _read(ROOT / ".dockerignore")
    for pattern in ("**/.env", "**/.venv*", "**/node_modules", ".git"):
        assert pattern in text


def test_frontend_lockfile_matches_package_manifest() -> None:
    manifest = json.loads(_read(ROOT / "frontend" / "package.json"))
    lock = json.loads(_read(ROOT / "frontend" / "package-lock.json"))
    assert lock["name"] == manifest["name"]
    assert lock["version"] == manifest["version"]
    assert lock["lockfileVersion"] >= 2
    for name, version in manifest["dependencies"].items():
        assert lock["packages"][""]["dependencies"][name] == version
    for name, version in manifest["devDependencies"].items():
        assert lock["packages"][""]["devDependencies"][name] == version


def test_docker_readme_states_no_build_was_executed() -> None:
    text = " ".join(_read(DOCKER / "README.md").split())
    assert "No Docker build or run was executed" in text
