"""Opt-in live BGE reranker model-load smoke test (Goal 26).

Skipped by default. Requires an already-provisioned isolated interpreter with
FlagEmbedding and cached BGE reranker weights. Never downloads a model and never
touches a database.
"""
from __future__ import annotations

import os

import pytest

from app.reranking.contracts import RERANKER_PYTHON_ENV, RerankerConfig, RerankerSetup


@pytest.mark.live_reranker
def test_live_bge_reranker_model_loads() -> None:
    if os.environ.get("VOXSEARCH_LIVE_RERANKER") != "1":
        pytest.skip("set VOXSEARCH_LIVE_RERANKER=1 for live reranker verification")
    interpreter = os.environ.get(RERANKER_PYTHON_ENV, "").strip()
    if not interpreter:
        pytest.skip(f"set {RERANKER_PYTHON_ENV} to the isolated interpreter")
    model_path = os.environ.get("VOXSEARCH_RERANKER_MODEL_PATH") or None
    status = RerankerSetup(
        worker_python=interpreter,
        config=RerankerConfig(model_path=model_path),
    ).load_model()
    assert status.loaded is True
    assert status.model_id
