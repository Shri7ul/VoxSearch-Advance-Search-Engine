"""Goal 27 worker/client protocol tests for isolated BGE reranker scoring."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from typing import Any

import pytest

from app.reranking.contracts import RerankerConfig, RerankerModelLoadError
from reranker_worker import worker
from reranker_worker.client import SubprocessRerankerProvider


def _run_worker(monkeypatch: pytest.MonkeyPatch, request: dict[str, Any], scorer: Any):
    fake_module = types.ModuleType("FlagEmbedding")
    fake_module.FlagReranker = lambda *a, **kw: types.SimpleNamespace(
        compute_score=scorer
    )
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)
    stdin = io.BytesIO(json.dumps(request).encode("utf-8"))
    stdout = io.StringIO()
    monkeypatch.setattr(worker.sys, "stdin", types.SimpleNamespace(buffer=stdin))
    monkeypatch.setattr(worker.sys, "stdout", stdout)
    assert worker.main() == 0
    return json.loads(stdout.getvalue())


# --- Worker protocol --------------------------------------------------------

def test_worker_scores_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_worker(
        monkeypatch,
        {"model_id": "reranker", "pairs": [["q1", "d1"], ["q2", "d2"]]},
        lambda pairs: [0.1, 0.9],
    )
    assert result == {"scores": [0.1, 0.9], "provider_error": None}


def test_worker_normalizes_single_float_score(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_worker(
        monkeypatch,
        {"pairs": [["query", "document"]]},
        lambda pairs: 0.42,
    )
    assert result["scores"] == [0.42]


@pytest.mark.parametrize(
    "scores",
    [
        [0.1, 0.2],                     # count mismatch
        [float("nan")],                 # non-finite
        ["bad"],                        # non-numeric
    ],
)
def test_worker_rejects_bad_scores_with_closed_reason(
    monkeypatch: pytest.MonkeyPatch, scores: Any
) -> None:
    result = _run_worker(
        monkeypatch,
        {"pairs": [["query", "document"]]},
        lambda pairs: scores,
    )
    assert result["provider_error"] == "invalid_output"
    assert "scores" not in result or result.get("scores") == []


def test_worker_rejects_too_many_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_worker(
        monkeypatch,
        {"pairs": [["q", "d"]] * 51},
        lambda pairs: [0.5] * 51,
    )
    assert result["provider_error"] == "invalid_request"


def test_worker_maps_scoring_exception_to_closed_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(pairs: Any) -> Any:
        raise RuntimeError("C:/secret/path must not leak")

    result = _run_worker(monkeypatch, {"pairs": [["q", "d"]]}, broken)
    assert result["provider_error"] == "scoring_error"
    assert "secret" not in json.dumps(result)


# --- Client protocol --------------------------------------------------------

def test_client_returns_cleaned_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    from reranker_worker import client as client_module

    captured: dict[str, Any] = {}

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"scores": [0.75, 0.25], "provider_error": None}).encode(),
        )

    monkeypatch.setattr(client_module.subprocess, "run", fake_run)
    provider = SubprocessRerankerProvider(worker_python="python", timeout_seconds=9.0)
    scores = provider.score([["q1", "d1"], ["q2", "d2"]], RerankerConfig())
    assert scores == [0.75, 0.25]
    assert captured["argv"] == ["python", "-m", "reranker_worker.worker"]
    assert captured["kwargs"]["shell"] is False
    assert "DATABASE_URL" not in captured["kwargs"]["env"]


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"scores": [], "provider_error": None}, RerankerModelLoadError),
        ({"scores": [float("nan")], "provider_error": None}, RerankerModelLoadError),
        ({"provider_error": "scoring_error"}, RerankerModelLoadError),
        ({"provider_error": "invalid_request"}, Exception),
    ],
)
def test_client_maps_malformed_scoring_responses(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], expected: type
) -> None:
    from reranker_worker import client as client_module

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload).encode())

    monkeypatch.setattr(client_module.subprocess, "run", fake_run)
    provider = SubprocessRerankerProvider(worker_python="python", timeout_seconds=9.0)
    with pytest.raises(expected):
        provider.score([["q", "d"]], RerankerConfig())
