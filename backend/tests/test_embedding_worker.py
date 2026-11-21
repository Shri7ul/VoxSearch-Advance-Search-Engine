"""Goal 15 — isolated embedding worker / client integration tests.

Three clearly separated, honestly-labelled tiers
================================================

TIER A — "isolated worker / provider integration" (DEFAULT, always runs)
------------------------------------------------------------------------
These tests launch the REAL ``embedding_worker.worker`` subprocess through the
REAL ``SubprocessEmbeddingProvider`` client. A controlled ``FlagEmbedding``
provider DOUBLE (pure stdlib) is injected onto the worker's ``PYTHONPATH``.
The double:

* records every model/encode API call to a JSONL sidecar, and
* returns ``dense_vecs`` shaped exactly like the real ``BGEM3FlagModel``.

What this GENUINELY verifies against the real subprocess boundary:

* real argv / ``shell=False`` / JSON stdin-stdout / controlled env / timeout;
* the model is loaded ONCE per call (never once per chunk) with dense-only,
  fp16-off, CPU-only flags;
* bounded, ORDERED batching over the single loaded model;
* strict 1024-dim / finite output validation and deterministic 1:1 ordering;
* typed error mapping (load / inference / invalid-output failures);
* fail-closed request gating (URL model path, invalid input, over-length text
  never silently truncated);
* NO credentials forwarded, HF forced OFFLINE, CPU thread caps applied.

This runs with NO heavy dependencies, NO model weights, NO network, and NO
changes to the core venv. It does NOT claim real-weights inference — that is
Tier B / the standalone smoke test.

TIER B — "live real-weights" (OPT-IN, skipped by default)
---------------------------------------------------------
Gated behind ``VOXSEARCH_LIVE_EMBEDDING=1`` plus an explicitly-provided
isolated embedding interpreter and cached model weights. SKIPPED in the default
suite; we never claim it ran here.

Client-parsing unit tests
-------------------------
Pure ``subprocess.run`` mocks exercise the client's response parsing / error
mapping (no process launched), covering malformed JSON, non-zero exit, timeout,
launch OSError, oversized output, and wrong vector count.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.embeddings import (
    DENSE_DIMENSION,
    MAX_MODEL_ID_CHARACTERS,
    MAX_MODEL_PATH_CHARACTERS,
    MAX_REQUEST_BYTES,
    MAX_TEXT_CHARACTERS,
    MAX_TEXTS_PER_CALL,
    EmbeddingConfig,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    InvalidEmbeddingOutputError,
    generate_embeddings,
    validate_texts,
)
from app.embeddings import contracts as _contracts
from embedding_worker import client as _client
from embedding_worker import worker as _worker
from embedding_worker.client import SubprocessEmbeddingProvider

BACKEND = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Controlled FlagEmbedding provider DOUBLE (pure stdlib, injected on PYTHONPATH)
# ---------------------------------------------------------------------------
_FAKE_FLAGEMBEDDING_BODY = r'''
import json as _json
import math as _math
import os as _os


def _record(call, **kw):
    rec = {"call": call}
    rec.update(kw)
    with open(SIDECAR, "a", encoding="utf-8") as _f:
        _f.write(_json.dumps(rec) + "\n")


class _FakeTokenizer:
    def encode(self, text, **kwargs):
        # Approximate subword tokens: words + 2 special tokens. Always short for
        # the smoke texts; used to exercise the token-length guard deliberately.
        return [0] * (len(text.split()) + 2)


class _BadTokenizer:
    """Returns an unusable shape so the length guard cannot measure tokens."""

    def encode(self, text, **kwargs):
        return "not-a-token-id-list"


class BGEM3FlagModel:
    def __init__(self, name, **kwargs):
        if SCENARIO == "load_raise":
            raise RuntimeError("simulated load failure")
        _record(
            "load",
            name=name,
            use_fp16=kwargs.get("use_fp16"),
            devices=kwargs.get("devices"),
            return_dense=kwargs.get("return_dense"),
            return_sparse=kwargs.get("return_sparse"),
            return_colbert_vecs=kwargs.get("return_colbert_vecs"),
            offline=_os.environ.get("HF_HUB_OFFLINE"),
            secret_present=("VOXSEARCH_TEST_SECRET" in _os.environ),
            omp=_os.environ.get("OMP_NUM_THREADS"),
        )
        self.name = name
        if SCENARIO == "no_tokenizer":
            self.tokenizer = None
        elif SCENARIO == "bad_tokenizer":
            self.tokenizer = _BadTokenizer()
        else:
            self.tokenizer = _FakeTokenizer()
        self._n = 0

    def encode(self, sentences, **kwargs):
        _record(
            "encode",
            n=len(sentences),
            batch_size=kwargs.get("batch_size"),
            max_length=kwargs.get("max_length"),
            return_dense=kwargs.get("return_dense"),
            return_sparse=kwargs.get("return_sparse"),
            return_colbert_vecs=kwargs.get("return_colbert_vecs"),
        )
        if SCENARIO == "raise":
            raise RuntimeError("simulated inference failure")
        rows = []
        for _ in sentences:
            self._n += 1
            # Unit-norm 1024-dim vector: values[0] is a small distinct ordering
            # marker, values[1] carries the rest of the L2 norm -> norm == 1.0.
            marker = float(self._n) / 1000.0
            rest = _math.sqrt(1.0 - marker * marker)
            row = [marker, rest] + [0.0] * (DIM - 2)
            if SCENARIO == "nan":
                row[0] = float("nan")
            if SCENARIO == "bad_norm":
                row = [2.0] + [0.0] * (DIM - 1)  # L2 norm = 2.0 -> invalid
            rows.append(row)
        if SCENARIO == "wrong_dim":
            rows = [r[:200] for r in rows]
        if SCENARIO == "wrong_count":
            rows = rows[:-1] if len(rows) > 1 else []
        return {"dense_vecs": rows}
'''


def _write_fake(fake_dir: Path, scenario: str, sidecar: Path) -> None:
    fake_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "import sys\n"
        f"SIDECAR = {str(sidecar)!r}\n"
        f"SCENARIO = {scenario!r}\n"
        f"DIM = {DENSE_DIMENSION}\n"
    )
    (fake_dir / "FlagEmbedding.py").write_text(
        header + _FAKE_FLAGEMBEDDING_BODY, encoding="utf-8"
    )


def _calls(sidecar: Path) -> list[dict]:
    if not sidecar.exists():
        return []
    return [
        json.loads(line)
        for line in sidecar.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _provider(monkeypatch: pytest.MonkeyPatch, fake_dir: Path) -> SubprocessEmbeddingProvider:
    # _controlled_env() allowlists PYTHONPATH, so the worker inherits backend/
    # (forced) + this fake dir -> `import FlagEmbedding` resolves to the double.
    monkeypatch.setenv("PYTHONPATH", str(fake_dir))
    return SubprocessEmbeddingProvider(
        worker_python=sys.executable, timeout_seconds=30.0
    )


# ===========================================================================
# Tier A — real isolated worker subprocess integration
# ===========================================================================
class TestIsolatedWorkerIntegration:
    def test_end_to_end_order_dim_finite_and_single_model_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dir = tmp_path / "fakepkg"
        sidecar = tmp_path / "calls.jsonl"
        _write_fake(fake_dir, "ok", sidecar)
        provider = _provider(monkeypatch, fake_dir)

        texts = [f"text number {i}" for i in range(5)]
        batch = generate_embeddings(texts, provider, config=EmbeddingConfig(batch_size=2))

        # Strict 1:1 ordering, exactly 1024 dims, all finite.
        assert batch.count == 5
        assert all(v.dimension == DENSE_DIMENSION for v in batch.vectors)
        assert [v.values[0] for v in batch.vectors] == [0.001, 0.002, 0.003, 0.004, 0.005]

        calls = _calls(sidecar)
        loads = [c for c in calls if c["call"] == "load"]
        encodes = [c for c in calls if c["call"] == "encode"]
        assert len(loads) == 1, "model must be loaded ONCE, not once per chunk"
        # 5 texts / batch_size 2 -> 3 bounded, ordered batches.
        assert [c["n"] for c in encodes] == [2, 2, 1]
        assert all(c["batch_size"] == 2 for c in encodes)

        # Conservative CPU / dense-only flags actually reach the model.
        load = loads[0]
        assert load["use_fp16"] is False
        assert load["devices"] == ["cpu"]
        assert load["return_dense"] is True
        assert load["return_sparse"] is False
        assert load["return_colbert_vecs"] is False
        assert load["offline"] == "1"
        assert load["omp"] == "4"

    def test_credentials_are_not_forwarded_and_offline_forced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dir = tmp_path / "fakepkg"
        sidecar = tmp_path / "calls.jsonl"
        _write_fake(fake_dir, "ok", sidecar)
        monkeypatch.setenv("VOXSEARCH_TEST_SECRET", "DO-NOT-LEAK")
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
        provider = _provider(monkeypatch, fake_dir)

        generate_embeddings(["hello"], provider)

        load = next(c for c in _calls(sidecar) if c["call"] == "load")
        assert load["secret_present"] is False
        assert load["offline"] == "1"

    @pytest.mark.parametrize(
        "scenario,expected_exc",
        [
            ("load_raise", EmbeddingModelLoadError),
            ("raise", EmbeddingInferenceError),
            ("nan", InvalidEmbeddingOutputError),
            ("wrong_dim", InvalidEmbeddingOutputError),
            ("wrong_count", InvalidEmbeddingOutputError),
            ("bad_norm", InvalidEmbeddingOutputError),
        ],
    )
    def test_worker_failures_map_to_typed_errors(
        self,
        scenario: str,
        expected_exc: type[Exception],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_dir = tmp_path / "fakepkg"
        sidecar = tmp_path / "calls.jsonl"
        _write_fake(fake_dir, scenario, sidecar)
        provider = _provider(monkeypatch, fake_dir)
        with pytest.raises(expected_exc):
            generate_embeddings(["alpha", "beta"], provider)

    # --- direct worker request gate (bypass client; zero model calls) -------
    def _run_worker_direct(
        self, payload: dict, fake_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> dict:
        monkeypatch.setenv("PYTHONPATH", str(fake_dir))
        completed = subprocess.run(
            [sys.executable, "-m", "embedding_worker.worker"],
            input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(BACKEND),
            timeout=30.0,
            env=_client._controlled_env(),
            check=False,
        )
        assert completed.returncode == 0
        return json.loads(completed.stdout.decode("utf-8"))

    @pytest.mark.parametrize(
        "payload",
        [
            {"texts": ["a"], "model_path": "https://example.com/bge-m3",
             "batch_size": 2, "max_length": 256},
            {"texts": [], "batch_size": 2, "max_length": 256},
            {"texts": ["a", ""], "batch_size": 2, "max_length": 256},
            {"texts": ["a"], "batch_size": 99, "max_length": 256},
            {"texts": ["a"], "batch_size": 2, "max_length": True},
            {"texts": ["a"], "model_id": "m" * (MAX_MODEL_ID_CHARACTERS + 1),
             "batch_size": 2, "max_length": 256},
            {"texts": ["a"], "model_path": "p" * (MAX_MODEL_PATH_CHARACTERS + 1),
             "batch_size": 2, "max_length": 256},
        ],
    )
    def test_invalid_request_fails_closed_with_zero_model_calls(
        self, payload: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dir = tmp_path / "fakepkg"
        sidecar = tmp_path / "calls.jsonl"
        _write_fake(fake_dir, "ok", sidecar)
        out = self._run_worker_direct(payload, fake_dir, monkeypatch)
        assert out["provider_error"] == "invalid_request"
        assert out["vectors"] == []
        assert _calls(sidecar) == []  # gate BEFORE any model load / inference

    def test_over_length_text_rejected_no_silent_truncation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dir = tmp_path / "fakepkg"
        sidecar = tmp_path / "calls.jsonl"
        _write_fake(fake_dir, "ok", sidecar)
        # Fake tokenizer returns words+2 tokens; a 3-word text with max_length=2
        # would be silently truncated by encode() -> must be rejected instead.
        payload = {
            "texts": ["one two three"],
            "batch_size": 2,
            "max_length": 2,
        }
        out = self._run_worker_direct(payload, fake_dir, monkeypatch)
        assert out["provider_error"] == "text_too_long"
        assert out["vectors"] == []
        calls = _calls(sidecar)
        # Model is loaded once (to obtain the tokenizer) but ZERO encode calls:
        # no truncation / inference happened.
        assert [c["call"] for c in calls] == ["load"]

    @pytest.mark.parametrize("scenario", ["no_tokenizer", "bad_tokenizer"])
    def test_length_guard_fails_closed_when_length_cannot_be_measured(
        self,
        scenario: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: the token-length guard must REJECT, not silently skip,
        when the model exposes no usable tokenizer (or it returns an unexpected
        shape). Otherwise ``encode(truncation=True)`` would silently truncate.
        """
        fake_dir = tmp_path / "fakepkg"
        sidecar = tmp_path / "calls.jsonl"
        _write_fake(fake_dir, scenario, sidecar)
        payload = {
            "texts": ["one two three four five"],
            "batch_size": 2,
            "max_length": 256,
        }
        out = self._run_worker_direct(payload, fake_dir, monkeypatch)
        assert out["provider_error"] == "text_too_long"
        assert out["vectors"] == []
        calls = _calls(sidecar)
        # Model was loaded, but the fail-closed guard stopped BEFORE encode():
        # no silent truncation / inference happened.
        assert [c["call"] for c in calls] == ["load"]


# ===========================================================================
# Client response parsing (pure subprocess.run mocks; no process launched)
# ===========================================================================
class TestClientParsing:
    def _client(self) -> SubprocessEmbeddingProvider:
        return SubprocessEmbeddingProvider(worker_python="unused-python")

    def _patch_run(self, monkeypatch, *, returncode=0, stdout=b"", exc=None):
        def fake_run(*args, **kwargs):
            if exc is not None:
                raise exc
            return subprocess.CompletedProcess(
                args[0], returncode, stdout=stdout, stderr=b""
            )

        monkeypatch.setattr(_client.subprocess, "run", fake_run)

    def test_valid_response_returns_vectors(self, monkeypatch) -> None:
        vectors = [[0.5] * DENSE_DIMENSION]
        self._patch_run(
            monkeypatch,
            stdout=json.dumps({"vectors": vectors, "provider_error": None}).encode(),
        )
        assert self._client().embed(["a"], config=EmbeddingConfig()) == vectors

    @pytest.mark.parametrize(
        "reason,expected_exc",
        [
            ("model_load_error", EmbeddingModelLoadError),
            ("inference_error", EmbeddingInferenceError),
            ("invalid_request", Exception),
            ("text_too_long", Exception),
            ("invalid_output", InvalidEmbeddingOutputError),
            ("unknown_reason", EmbeddingInferenceError),
        ],
    )
    def test_provider_error_reason_mapping(self, reason, expected_exc, monkeypatch) -> None:
        payload = {"vectors": [], "provider_error": reason}
        self._patch_run(monkeypatch, stdout=json.dumps(payload).encode())
        with pytest.raises(expected_exc):
            self._client().embed(["a"], config=EmbeddingConfig())

    def test_nonzero_exit(self, monkeypatch) -> None:
        self._patch_run(monkeypatch, returncode=3)
        with pytest.raises(EmbeddingInferenceError):
            self._client().embed(["a"], config=EmbeddingConfig())

    def test_timeout(self, monkeypatch) -> None:
        self._patch_run(monkeypatch, exc=subprocess.TimeoutExpired("cmd", 1))
        with pytest.raises(EmbeddingInferenceError):
            self._client().embed(["a"], config=EmbeddingConfig())

    def test_launch_oserror_is_model_load_error(self, monkeypatch) -> None:
        self._patch_run(monkeypatch, exc=OSError("no interpreter"))
        with pytest.raises(EmbeddingModelLoadError):
            self._client().embed(["a"], config=EmbeddingConfig())

    def test_invalid_json(self, monkeypatch) -> None:
        self._patch_run(monkeypatch, stdout=b"not-json")
        with pytest.raises(EmbeddingInferenceError):
            self._client().embed(["a"], config=EmbeddingConfig())

    def test_oversized_stdout(self, monkeypatch) -> None:
        self._patch_run(
            monkeypatch, stdout=b"x" * (_client.MAX_RESPONSE_BYTES + 1)
        )
        with pytest.raises(EmbeddingInferenceError):
            self._client().embed(["a"], config=EmbeddingConfig())

    def test_wrong_vector_count_rejected(self, monkeypatch) -> None:
        payload = {"vectors": [[0.5] * DENSE_DIMENSION], "provider_error": None}
        self._patch_run(monkeypatch, stdout=json.dumps(payload).encode())
        with pytest.raises(InvalidEmbeddingOutputError):
            self._client().embed(["a", "b"], config=EmbeddingConfig())


# ===========================================================================
# Static isolation / pin guarantees (no heavy imports, no .env)
# ===========================================================================
class TestProviderIsolation:
    def test_core_runtime_never_imports_heavy_embedding_stack(self) -> None:
        code = (
            "import sys;"
            "import app.embeddings;"
            "import app.embeddings.contracts;"
            "import app.transcripts.chunking;"
            "import embedding_worker.client;"
            "import embedding_worker.worker;"
            "assert 'torch' not in sys.modules, 'torch leaked';"
            "assert 'transformers' not in sys.modules, 'transformers leaked';"
            "assert 'FlagEmbedding' not in sys.modules, 'FlagEmbedding leaked';"
            "assert 'numpy' not in sys.modules, 'numpy leaked';"
            "print('CLEAN')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(BACKEND),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=60.0,
            check=False,
        )
        assert completed.returncode == 0
        assert completed.stdout.decode("utf-8").strip() == "CLEAN"

    def test_requirements_embedding_pins_model_stack(self) -> None:
        text = (BACKEND / "requirements-embedding.txt").read_text(encoding="utf-8")
        for pin in ("FlagEmbedding==1.4.2", "torch==2.9.1+cpu", "BAAI/bge-m3"):
            assert pin in text

    def test_core_requirements_have_no_heavy_embedding_deps(self) -> None:
        for name in ("requirements.txt", "requirements-lock.txt"):
            lowered = (BACKEND / name).read_text(encoding="utf-8").lower()
            for kw in ("torch", "transformers", "flagembed", "numpy", "cuda"):
                assert kw not in lowered, f"{kw} leaked into {name}"


# ===========================================================================
# Limit consistency (core config / client / isolated worker MUST agree)
# ===========================================================================
class TestLimitConsistency:
    """Regression: the worker keeps LOCAL copies of the bounds (it cannot
    import ``app``). If those ever drift from the authoritative constants, a
    core-valid request could be silently dropped. These assertions turn ANY
    drift into a hard test failure, so there is no hidden client/worker mismatch.
    """

    def test_worker_local_copies_match_authoritative_constants(self) -> None:
        assert _worker.MAX_REQUEST_BYTES == _contracts.MAX_REQUEST_BYTES
        assert _worker.MAX_RESPONSE_BYTES == _contracts.MAX_RESPONSE_BYTES
        assert _worker.DENSE_DIMENSION == _contracts.DENSE_DIMENSION
        assert _worker.MAX_TEXTS == _contracts.MAX_TEXTS_PER_CALL
        assert _worker.MAX_BATCH_SIZE == _contracts.MAX_BATCH_SIZE
        assert _worker.MAX_MAX_LENGTH == _contracts.MAX_MAX_LENGTH
        assert _worker.MAX_TEXT_CHARACTERS == _contracts.MAX_TEXT_CHARACTERS
        assert _worker.MAX_MODEL_ID_CHARACTERS == _contracts.MAX_MODEL_ID_CHARACTERS
        assert _worker.MAX_MODEL_PATH_CHARACTERS == _contracts.MAX_MODEL_PATH_CHARACTERS
        assert _worker.DEFAULT_MODEL_ID == _contracts.DEFAULT_MODEL_ID

    def test_client_uses_the_same_byte_caps(self) -> None:
        # client IMPORTS the authoritative constants (no hidden local copies).
        assert _client.MAX_REQUEST_BYTES == _contracts.MAX_REQUEST_BYTES
        assert _client.MAX_RESPONSE_BYTES == _contracts.MAX_RESPONSE_BYTES

    def test_config_upper_bounds_cannot_exceed_worker_hard_caps(self) -> None:
        cfg = EmbeddingConfig()  # maximal defaults
        assert cfg.max_text_characters <= _worker.MAX_TEXT_CHARACTERS
        assert cfg.max_texts <= _worker.MAX_TEXTS
        assert cfg.batch_size <= _worker.MAX_BATCH_SIZE
        assert cfg.max_length <= _worker.MAX_MAX_LENGTH

    def test_largest_core_valid_request_fits_the_byte_cap(self) -> None:
        """The WORST-CASE request the core can accept must fit both the client
        and worker byte caps (measured worst case 3,172,177 B < 4,000,000 B).
        """
        # Control chars (\x01) are the widest per-char case in JSON (6-byte
        # \u00XX escape), so this is the TRUE worst-case payload (3,172,177 B).
        cfg = EmbeddingConfig(
            model_id="\x01" * MAX_MODEL_ID_CHARACTERS,
            model_path="\x01" * MAX_MODEL_PATH_CHARACTERS,
            max_text_characters=MAX_TEXT_CHARACTERS,
            max_texts=MAX_TEXTS_PER_CALL,
        )
        texts = ["\x01" * MAX_TEXT_CHARACTERS] * MAX_TEXTS_PER_CALL
        validated = validate_texts(texts, cfg)  # proves the config is core-valid
        payload = {
            "texts": list(validated),
            "model_id": cfg.model_id,
            "model_path": cfg.model_path,
            "batch_size": cfg.batch_size,
            "max_length": cfg.max_length,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        assert len(data) <= MAX_REQUEST_BYTES
        assert len(data) <= _worker.MAX_REQUEST_BYTES


# ===========================================================================
# Tier B — opt-in LIVE real-weights (skipped by default -> OUTSTANDING)
# ===========================================================================
def _live_enabled() -> bool:
    return os.environ.get("VOXSEARCH_LIVE_EMBEDDING") == "1"


@pytest.mark.live_embedding
def test_live_bge_m3_dense_embedding(tmp_path: Path) -> None:
    """Opt-in: verify REAL BGE-M3 through the full Goal 15 code path.

    Requires (operator-supplied, NEVER provisioned by the default suite):
      * ``VOXSEARCH_LIVE_EMBEDDING=1``
      * ``VOXSEARCH_EMBEDDING_PYTHON`` — the isolated
        ``backend/.venv-embedding`` interpreter
      * ``VOXSEARCH_EMBEDDING_MODEL_PATH`` — local cached snapshot dir
      * cached BGE-M3 weights (HF cache), no network download

    SKIPPED in the default suite; we do NOT claim real-weights validation ran
    here unless this test is explicitly enabled and passes.
    """
    if not _live_enabled():
        pytest.skip(
            "opt-in live BGE-M3: set VOXSEARCH_LIVE_EMBEDDING=1 + "
            "VOXSEARCH_EMBEDDING_PYTHON + VOXSEARCH_EMBEDDING_MODEL_PATH + "
            "cached weights; NOT run in default suite (outstanding verification)"
        )

    worker_python = os.environ.get("VOXSEARCH_EMBEDDING_PYTHON", "")
    model_path = os.environ.get("VOXSEARCH_EMBEDDING_MODEL_PATH", "")
    assert worker_python and Path(worker_python).is_file()
    assert model_path and Path(model_path).is_dir()

    provider = SubprocessEmbeddingProvider(
        worker_python=worker_python, timeout_seconds=300.0
    )
    cfg = EmbeddingConfig(model_path=model_path, max_length=32, batch_size=2)
    batch = generate_embeddings(["hello world", "another short sentence"], provider, config=cfg)
    assert batch.count == 2
    assert all(v.dimension == DENSE_DIMENSION for v in batch.vectors)
    assert all(
        all(float("-inf") < x < float("inf") for x in v.values)
        for v in batch.vectors
    )
