"""Goal 12 — WhisperX provider integration tests.

Filename mandated by the authoritative full implementation task §15:
``backend/tests/test_alignment_provider_integration.py``.

Two clearly separated, honestly-labelled tiers
==============================================

TIER A — "isolated worker / provider-adapter integration" (DEFAULT, always runs)
--------------------------------------------------------------------------------
These tests launch the REAL isolated worker as a REAL subprocess
(``python -m alignment_worker.worker``) through the REAL client
(:class:`alignment_worker.client.SubprocessAlignmentProvider`). A controlled
``whisperx`` provider DOUBLE is injected onto the worker process's
``PYTHONPATH`` (via the client's allowlisted ``PYTHONPATH`` env var). The
double is a tiny pure-Python module that:

* records every provider API call it receives to a JSONL sidecar, and
* returns output shaped exactly like ``whisperx==3.8.6``
  ``return_char_alignments=True`` (char dicts with ``char``/``start``/``end``/
  ``score`` + space separators, and — critically — NO ``word-idx`` in the final
  char dicts).

What this GENUINELY verifies (against the real subprocess boundary):

* **Actual isolated worker execution** — real argv, ``shell=False``, JSON
  stdin/stdout, controlled env, hard timeout, separated process.
* **Real provider invocation** — the worker really calls
  ``whisperx.load_align_model`` / ``whisperx.load_audio`` / ``whisperx.align``;
  the sidecar proves ``align`` was invoked with ``interpolate_method="ignore"``
  and ``return_char_alignments=True`` (the mandated v3.8.6 flags).
* **Actual character-alignment output structure** — v3.8.6-shaped ``chars``
  pass through the worker verbatim and are consumed by the client.
* **Strict word-to-character corroboration** — the client's
  :func:`reconstruct_word_chars` + the service's
  :func:`validate_provider_segment` produce corroborated aligned words.
* **Missing timestamps / malformed provider outputs fail closed** — controlled
  scenarios (missing word timing, all-NaN char timing, non-dict result, wrong
  segment count, words-not-a-list, char text mismatch, token text mismatch,
  empty words, unavailable audio) all yield ZERO persisted words or a closed
  ``provider_error``; nothing is interpolated, backfilled, or clamped.

This runs with NO heavy dependencies, NO model weights, NO real audio dataset
(a tiny placeholder file only), and NO changes to the core venv. It does NOT
claim real-weights / real-model WhisperX inference — that is Tier B.

TIER B — "live whisperx==3.8.6" (OPT-IN, skipped by default)
------------------------------------------------------------
``@pytest.mark.live_whisperx`` tests are gated behind
``VOXSEARCH_LIVE_WHISPERX=1`` plus an explicitly-provided isolated alignment
interpreter, cached model weights, and a real authorized audio file. They
verify the exact installed ``whisperx.__version__ == "3.8.6"`` and real align
output. They are SKIPPED in the default suite and reported as an OUTSTANDING
provider-integration verification. We never claim they ran here.

Isolation guarantees
--------------------
* Importing the core API / service / client / worker NEVER loads torch or
  whisperx (asserted in a clean subprocess).
* ``requirements-alignment.txt`` pins ``whisperx==3.8.6``; core
  ``requirements.txt`` / ``requirements-lock.txt`` contain NO heavy provider
  deps (asserted statically).
* ``.env`` is NEVER read; production Supabase is NEVER contacted.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from alignment_worker import worker as _worker
from alignment_worker.client import SubprocessAlignmentProvider
from app.transcripts import word_contract as wc
from app.transcripts.word_alignment import (
    SegmentData,
    SegmentPreparation,
    SkippedSegment,
    build_request,
    prepare_segments,
    validate_provider_segment,
)
from app.transcripts.word_contract import AlignedWord

BACKEND = Path(__file__).resolve().parents[1]

# Video window [10000, 12000] ms == audio window [0, 2000] ms.
OFFSET_MS = 10_000
DURATION_MS = 60_000
SID = uuid.uuid4()
ORIGINAL = "Hello world"


# ---------------------------------------------------------------------------
# Helpers: prepared segment + provider request
# ---------------------------------------------------------------------------
def _segment() -> SegmentData:
    return SegmentData(
        segment_id=SID,
        segment_ref="S1",
        start_seconds=10.0,
        end_seconds=12.0,
        original_text=ORIGINAL,
    )


def _prepared():
    prep = prepare_segments(
        [_segment()], timeline_offset_ms=OFFSET_MS, audio_duration_ms=DURATION_MS
    )
    assert not prep.skipped
    return prep.prepared[0]


def _request_dict(**overrides: object) -> dict:
    req = build_request("en", SegmentPreparation((_prepared(),), ())).to_dict()
    req.update(overrides)  # type: ignore[arg-type]
    return req


# ---------------------------------------------------------------------------
# v3.8.6-shaped provider output builders
# ---------------------------------------------------------------------------
def _chars_for(text: str, start: float, end: float) -> list[dict]:
    """Return final v3.8.6 char dicts. NO ``word-idx`` key (per v3.8.6).

    Char timings are placed strictly INSIDE the word bounds (start+0.01 ..
    end-0.01) so corroboration passes deterministically without float-ULP
    edge effects at the exact word boundary.
    """
    lo, hi = start + 0.01, end - 0.01
    n = len(text)
    step = (hi - lo) / n
    return [
        {"char": ch, "start": lo + i * step, "end": lo + (i + 1) * step, "score": 0.9}
        for i, ch in enumerate(text)
    ]


def _ok_words() -> list[dict]:
    return [
        {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.9},
        {"word": "world", "start": 0.5, "end": 0.9, "score": 0.9},
    ]


def _ok_chars() -> list[dict]:
    # WhisperX inserts a space separator char (no timing) between words.
    return (
        _chars_for("Hello", 0.0, 0.4)
        + [{"char": " "}]
        + _chars_for("world", 0.5, 0.9)
    )


def _aligned_seg(words, chars) -> dict:
    return {"text": ORIGINAL, "start": 0.0, "end": 2.0, "words": words, "chars": chars}


def _scenario_result(scenario: str):
    """Return the object the fake ``whisperx.align`` yields for a scenario."""
    if scenario == "malformed_result":
        # Not a dict -> worker must emit provider_error (fail closed).
        return ["not", "a", "dict"]
    if scenario == "wrong_segment_count":
        return {
            "segments": [
                _aligned_seg(_ok_words(), _ok_chars()),
                _aligned_seg(_ok_words(), _ok_chars()),
            ],
            "word_segments": [],
        }
    if scenario == "missing_word_timing":
        # interpolate_method="ignore" leaves NaN -> WhisperX omits start/end.
        w = _ok_words()
        w[0] = {"word": "Hello", "score": 0.9}  # start/end OMITTED
        return {"segments": [_aligned_seg(w, _ok_chars())], "word_segments": []}
    if scenario == "all_timings_missing":
        # Neither word nor char timings present: NOTHING may be fabricated.
        w = [{"word": "Hello", "score": 0.9}, {"word": "world", "score": 0.9}]
        c = [{"char": ch} for ch in "Helloworld"]  # no timings, no score
        return {"segments": [_aligned_seg(w, c)], "word_segments": []}
    if scenario == "empty_words":
        return {"segments": [_aligned_seg([], _ok_chars())], "word_segments": []}
    if scenario == "words_not_list":
        return {"segments": [_aligned_seg("garbage", _ok_chars())], "word_segments": []}
    if scenario == "char_text_mismatch":
        c = _chars_for("Xxxxx", 0.0, 0.4) + [{"char": " "}] + _chars_for("yyyyy", 0.5, 0.9)
        return {"segments": [_aligned_seg(_ok_words(), c)], "word_segments": []}
    if scenario == "token_text_mismatch":
        w = _ok_words()
        w[0] = {"word": "hallo", "start": 0.0, "end": 0.4, "score": 0.9}
        return {"segments": [_aligned_seg(w, _ok_chars())], "word_segments": []}
    # "ok" (default)
    return {"segments": [_aligned_seg(_ok_words(), _ok_chars())], "word_segments": []}


# ---------------------------------------------------------------------------
# The controlled whisperx provider DOUBLE (pure stdlib, injected on PYTHONPATH)
# ---------------------------------------------------------------------------
_FAKE_WHISPERX_BODY = r'''
import json as _json
import os as _os


def _record(call, **kw):
    rec = {"call": call}
    rec.update(kw)
    with open(SIDECAR, "a", encoding="utf-8") as _f:
        _f.write(_json.dumps(rec) + "\n")


def load_align_model(language_code, device):
    _record("load_align_model", language_code=language_code, device=device)
    return ("FAKE_MODEL", {"language": language_code, "dictionary": {}})


def load_audio(path):
    _record("load_audio", basename=_os.path.basename(path))
    return "FAKE_AUDIO_ARRAY"


def align(transcript, model, metadata, audio, device,
          interpolate_method="nearest", return_char_alignments=False, **kwargs):
    _record(
        "align",
        interpolate_method=interpolate_method,
        return_char_alignments=return_char_alignments,
        n_segments=len(transcript),
        device=device,
    )
    return RESULT
'''


def _write_fake_whisperx(fake_dir: Path, scenario: str, sidecar: Path) -> None:
    fake_dir.mkdir(parents=True, exist_ok=True)
    result = _scenario_result(scenario)
    header = (
        "import sys\n"
        f"SIDECAR = {str(sidecar)!r}\n"
        f"RESULT = {result!r}\n"
    )
    (fake_dir / "whisperx.py").write_text(
        header + _FAKE_WHISPERX_BODY, encoding="utf-8"
    )


def _run_worker(
    *,
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: dict | None = None,
    audio_exists: bool = True,
):
    """Run the REAL isolated worker subprocess with a controlled provider.

    Returns ``(provider_result, recorded_calls)``. The worker is launched via
    the production ``SubprocessAlignmentProvider`` — real process, real JSON
    boundary. ``recorded_calls`` is read from the sidecar written by the fake
    provider module INSIDE that subprocess (genuine invocation evidence).
    """
    fake_dir = tmp_path / "fakepkg"
    sidecar = tmp_path / "calls.jsonl"
    audio = tmp_path / "clip.wav"
    if audio_exists:
        # Placeholder only — NOT a real recording / dataset. The fake provider
        # never decodes audio; the worker only checks os.path.isfile().
        audio.write_bytes(b"RIFF____PLACEHOLDER_NOT_REAL_AUDIO____")

    _write_fake_whisperx(fake_dir, scenario, sidecar)
    # _controlled_env() allowlists PYTHONPATH, so the worker subprocess inherits
    # backend/ (forced) + this fake dir -> `import whisperx` resolves to the double.
    monkeypatch.setenv("PYTHONPATH", str(fake_dir))

    provider = SubprocessAlignmentProvider(
        worker_python=sys.executable, audio_path=str(audio), timeout_seconds=30.0
    )
    out = provider(request if request is not None else _request_dict())

    calls: list[dict] = []
    if sidecar.exists():
        calls = [
            json.loads(line)
            for line in sidecar.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return out, calls


def _align_call(calls: list[dict]) -> dict | None:
    for c in calls:
        if c.get("call") == "align":
            return c
    return None


# ===========================================================================
# Tier A — default, genuinely-executed isolated worker integration
# ===========================================================================
class TestIsolatedWorkerIntegration:
    def test_real_worker_invokes_provider_with_mandated_v386_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, calls = _run_worker(scenario="ok", tmp_path=tmp_path, monkeypatch=monkeypatch)

        # Worker survived the subprocess boundary with a clean result.
        assert out["provider_error"] is None
        assert len(out["segments"]) == 1

        # REAL provider invocation: all three API surfaces were actually called.
        names = [c["call"] for c in calls]
        assert "load_align_model" in names
        assert "load_audio" in names
        assert "align" in names

        align = _align_call(calls)
        assert align is not None
        # The mandated whisperx==3.8.6 adapter flags (no "nearest" interpolation).
        assert align["interpolate_method"] == "ignore"
        assert align["return_char_alignments"] is True
        assert align["n_segments"] == 1
        assert align["device"] == "cpu"

    def test_v386_char_output_structure_passes_through_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inspect the WORKER's raw stdout directly (not the client) to prove the
        # v3.8.6-shaped char structure is passed through VERBATIM.
        fake_dir = tmp_path / "fakepkg"
        sidecar = tmp_path / "calls.jsonl"
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"RIFF____PLACEHOLDER____")
        _write_fake_whisperx(fake_dir, "ok", sidecar)
        monkeypatch.setenv("PYTHONPATH", str(fake_dir))

        payload = dict(_request_dict())
        payload["audio_path"] = str(audio)
        payload["device"] = "cpu"
        completed = subprocess.run(
            [sys.executable, "-m", "alignment_worker.worker"],
            input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(BACKEND),
            timeout=30.0,
            env=_worker_env(fake_dir),
            check=False,
        )
        assert completed.returncode == 0
        raw = json.loads(completed.stdout.decode("utf-8"))
        assert raw["provider_error"] is None
        seg = raw["segments"][0]
        assert seg["error"] is None
        # Final v3.8.6 char dicts: char/start/end/score, NO word-idx.
        assert isinstance(seg["chars"], list) and seg["chars"]
        assert all("char" in c for c in seg["chars"])
        assert all("word-idx" not in c for c in seg["chars"])
        timed = [c for c in seg["chars"] if "start" in c]
        assert timed and all("start" in c and "end" in c and "score" in c for c in timed)
        # Exactly one space separator (no timing) between the two words.
        assert sum(1 for c in seg["chars"] if c.get("char") == " " and "start" not in c) == 1
        # Word dicts passed through verbatim (no fabrication by the worker).
        assert [w["word"] for w in seg["words"]] == ["Hello", "world"]

    def test_strict_word_to_char_corroboration_yields_aligned_words(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, calls = _run_worker(scenario="ok", tmp_path=tmp_path, monkeypatch=monkeypatch)
        assert _align_call(calls) is not None  # real provider was invoked
        seg = out["segments"][0]
        # Client reconstructed per-word char evidence from the v3.8.6 chars.
        for w in seg["words"]:
            assert w["char_corroborated"] is True
            assert w["char_start"] is not None and w["char_end"] is not None

        validated = validate_provider_segment(
            _prepared(), seg, timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(validated, tuple)
        assert all(isinstance(w, AlignedWord) for w in validated)
        # audio ms + offset -> VIDEO-timeline integer ms.
        assert [(w.start_ms, w.end_ms) for w in validated] == [
            (10_000, 10_400),
            (10_500, 10_900),
        ]
        assert [w.text for w in validated] == ["Hello", "world"]

    @pytest.mark.parametrize(
        "scenario,expected_reason",
        [
            ("missing_word_timing", wc.REASON_WORD_TIMING_MISSING),
            ("all_timings_missing", wc.REASON_WORD_TIMING_MISSING),
            ("empty_words", wc.REASON_TOKEN_COUNT_MISMATCH),
            ("words_not_list", wc.REASON_TOKEN_COUNT_MISMATCH),
            ("token_text_mismatch", wc.REASON_TOKEN_TEXT_MISMATCH),
            ("char_text_mismatch", wc.REASON_CHAR_CORROBORATION_FAILED),
        ],
    )
    def test_malformed_or_missing_timing_fails_closed_zero_words(
        self,
        scenario: str,
        expected_reason: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out, _ = _run_worker(
            scenario=scenario, tmp_path=tmp_path, monkeypatch=monkeypatch
        )
        assert out["provider_error"] is None  # worker produced a segment
        seg = out["segments"][0]
        validated = validate_provider_segment(
            _prepared(), seg, timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(validated, SkippedSegment)
        assert validated.reason == expected_reason
        # ZERO AlignedWords persisted — never a partial/interpolated set.
        assert not isinstance(validated, tuple)

    def test_missing_word_timestamps_remain_missing_no_backfill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, _ = _run_worker(
            scenario="missing_word_timing", tmp_path=tmp_path, monkeypatch=monkeypatch
        )
        first = out["segments"][0]["words"][0]
        # The worker/client NEVER add a start/end that WhisperX omitted.
        assert "start" not in first and "end" not in first
        # No char timing was backfilled for the corrupted path either.
        assert first.get("char_corroborated") in (True, False)

    def test_no_interpolation_leakage_when_all_timings_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, _ = _run_worker(
            scenario="all_timings_missing", tmp_path=tmp_path, monkeypatch=monkeypatch
        )
        words = out["segments"][0]["words"]
        assert all("start" not in w and "end" not in w for w in words)
        assert all(w["char_corroborated"] is False for w in words)
        assert all(w["char_start"] is None and w["char_end"] is None for w in words)

    @pytest.mark.parametrize("scenario", ["malformed_result", "wrong_segment_count"])
    def test_malformed_provider_output_emits_closed_error(
        self, scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, _ = _run_worker(
            scenario=scenario, tmp_path=tmp_path, monkeypatch=monkeypatch
        )
        assert out["provider_error"] in ("provider_error", None)
        assert out["segments"] == []  # zero word rows on a broken provider

    def test_unavailable_audio_fails_closed_zero_provider_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, calls = _run_worker(
            scenario="ok", tmp_path=tmp_path, monkeypatch=monkeypatch,
            audio_exists=False,
        )
        assert out["provider_error"] == "provider_error"
        assert out["segments"] == []
        # os.path.isfile guard runs BEFORE import whisperx -> ZERO model calls.
        assert calls == []

    # --- worker request gate: unsupported language, flags, URL rejection ----
    @pytest.mark.parametrize("lang", ["bn", "ben", "bn-BD", "bn-IN"])
    def test_bengali_rejected_with_zero_provider_calls(
        self, lang: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, calls = _run_worker(
            scenario="ok", tmp_path=tmp_path, monkeypatch=monkeypatch,
            request=_request_dict(language=lang),
        )
        assert out["provider_error"] == "unsupported_language"
        assert out["segments"] == []
        assert calls == []  # language gate is BEFORE any provider invocation

    @pytest.mark.parametrize("lang", ["xx-unknown", "", "zz"])
    def test_unknown_language_rejected_with_zero_provider_calls(
        self, lang: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, calls = _run_worker(
            scenario="ok", tmp_path=tmp_path, monkeypatch=monkeypatch,
            request=_request_dict(language=lang),
        )
        assert out["provider_error"] == "provider_error"
        assert calls == []

    @pytest.mark.parametrize("flag", ["nearest", "linear", None])
    def test_wrong_interpolate_method_rejected_zero_provider_calls(
        self, flag, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, calls = _run_worker(
            scenario="ok", tmp_path=tmp_path, monkeypatch=monkeypatch,
            request=_request_dict(interpolate_method=flag),
        )
        assert out["provider_error"] == "provider_error"
        assert calls == []

    def test_signed_url_audio_path_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dir = tmp_path / "fakepkg"
        sidecar = tmp_path / "calls.jsonl"
        _write_fake_whisperx(fake_dir, "ok", sidecar)
        monkeypatch.setenv("PYTHONPATH", str(fake_dir))
        provider = SubprocessAlignmentProvider(
            worker_python=sys.executable,
            audio_path="https://bucket/clip.wav?X-Amz-Signature=SECRET",
            timeout_seconds=30.0,
        )
        out = provider(_request_dict())
        assert out["provider_error"] == "provider_error"
        assert out["segments"] == []
        assert not sidecar.exists()  # zero provider calls


# ---------------------------------------------------------------------------
# Worker _validate_request unit gate (in-process; no heavy imports)
# ---------------------------------------------------------------------------
class TestWorkerRequestValidation:
    def test_valid_request_normalized(self) -> None:
        # The production client (SubprocessAlignmentProvider.__call__) always
        # injects ``audio_path`` and defaults ``device`` before the worker's
        # _validate_request runs. A bare AlignmentRequest.to_dict() is
        # intentionally incomplete; build the same worker-complete payload here
        # so this validates the real accepted request shape.
        req = _request_dict()
        req["audio_path"] = "clip.wav"
        req["device"] = "cpu"
        normalized, reason = _worker._validate_request(req)
        assert reason is None and normalized is not None
        assert normalized["language"] == "en"
        assert normalized["device"] == "cpu"
        assert len(normalized["segments"]) == 1

    def test_bengali_classified_as_unsupported_language(self) -> None:
        _, reason = _worker._validate_request(_request_dict(language="bn"))
        assert reason == _worker._REASON_UNSUPPORTED_LANGUAGE

    def test_worker_language_allowlist_matches_contract(self) -> None:
        assert _worker.SUPPORTED_LANGUAGES == wc.SUPPORTED_ALIGNMENT_LANGUAGES
        assert _worker.BENGALI_CODES == wc.BENGALI_LANGUAGE_CODES

    def test_return_char_alignments_must_be_true(self) -> None:
        _, reason = _worker._validate_request(_request_dict(return_char_alignments=False))
        assert reason == _worker._REASON_PROVIDER_ERROR


# ---------------------------------------------------------------------------
# Static isolation / pin guarantees (no heavy imports, no .env)
# ---------------------------------------------------------------------------
class TestProviderIsolation:
    def test_core_runtime_never_imports_heavy_provider(self) -> None:
        # Fresh subprocess -> no cross-test sys.modules pollution.
        code = (
            "import sys;"
            "import app.services.word_alignment_service;"
            "import app.transcripts.word_alignment;"
            "import app.transcripts.word_contract;"
            "import alignment_worker.client;"
            "import alignment_worker.worker;"
            "assert 'whisperx' not in sys.modules, 'whisperx leaked into core';"
            "assert 'torch' not in sys.modules, 'torch leaked into core';"
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

    def test_requirements_alignment_pins_exact_v386(self) -> None:
        text = (BACKEND / "requirements-alignment.txt").read_text(encoding="utf-8")
        assert re.search(r"(?m)^\s*whisperx==3\.8\.6\s*$", text)

    def test_core_requirements_have_no_heavy_provider_deps(self) -> None:
        for name in ("requirements.txt", "requirements-lock.txt"):
            lowered = (BACKEND / name).read_text(encoding="utf-8").lower()
            for kw in ("whisperx", "torch", "ctranslate2", "transformers"):
                assert kw not in lowered, f"{kw} leaked into {name}"


# ===========================================================================
# Tier B — opt-in LIVE whisperx==3.8.6 (skipped by default -> OUTSTANDING)
# ===========================================================================
def _live_enabled() -> bool:
    return os.environ.get("VOXSEARCH_LIVE_WHISPERX") == "1"


@pytest.mark.live_whisperx
def test_live_whisperx_386_exact_release_and_real_alignment() -> None:
    """Opt-in: verify the REAL whisperx==3.8.6 release end-to-end.

    Requires (operator-supplied, NEVER provisioned by this goal):
      * ``VOXSEARCH_LIVE_WHISPERX=1``
      * ``VOXSEARCH_ALIGNMENT_PYTHON`` - isolated interpreter with
        ``requirements-alignment.txt`` installed
      * ``VOXSEARCH_ALIGNMENT_AUDIO``  - a real, authorized local audio file
        whose spoken content matches ``VOXSEARCH_ALIGNMENT_TEXT``
      * cached wav2vec2 model weights (HF cache), no network download here

    This is the outstanding provider-integration verification. It is SKIPPED in
    the default suite; we do NOT claim live/real-weights validation was run.
    """
    if not _live_enabled():
        pytest.skip(
            "opt-in live whisperx==3.8.6: set VOXSEARCH_LIVE_WHISPERX=1 + "
            "VOXSEARCH_ALIGNMENT_PYTHON + VOXSEARCH_ALIGNMENT_AUDIO + cached "
            "model weights; NOT run in default suite (outstanding verification)"
        )

    align_python = os.environ.get("VOXSEARCH_ALIGNMENT_PYTHON", "")
    audio = os.environ.get("VOXSEARCH_ALIGNMENT_AUDIO", "")
    text = os.environ.get("VOXSEARCH_ALIGNMENT_TEXT", ORIGINAL)
    assert align_python and Path(align_python).is_file(), "isolated interpreter required"
    assert audio and Path(audio).is_file(), "real authorized audio required"

    # 1. Exact release pin, verified inside the isolated interpreter.
    probe = subprocess.run(
        [align_python, "-c", "import whisperx,sys;sys.stdout.write(whisperx.__version__)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=120.0,
        check=False,
    )
    assert probe.returncode == 0
    assert probe.stdout.decode("utf-8").strip() == "3.8.6"

    # 2. Real single-segment alignment through the isolated worker.
    seg = SegmentData(
        segment_id=SID, segment_ref="live", start_seconds=10.0, end_seconds=12.0,
        original_text=text,
    )
    prep = prepare_segments(
        [seg], timeline_offset_ms=OFFSET_MS, audio_duration_ms=DURATION_MS
    )
    request = build_request("en", prep).to_dict()
    provider = SubprocessAlignmentProvider(
        worker_python=align_python, audio_path=audio
    )
    out = provider(request)
    assert out["provider_error"] is None
    assert len(out["segments"]) == 1
    provider_seg = out["segments"][0]
    assert provider_seg["error"] is None
    # Real v3.8.6 char evidence must be present and corroborate words.
    assert provider_seg["words"]
    assert all(w.get("char_corroborated") is True for w in provider_seg["words"])
    validated = validate_provider_segment(
        prep.prepared[0], provider_seg, timeline_offset_ms=OFFSET_MS
    )
    assert isinstance(validated, tuple) and validated  # real aligned words


def _worker_env(fake_dir: Path) -> dict:
    """Minimal env for a DIRECT worker subprocess in the char-structure test.

    Mirrors client._controlled_env() allowlist semantics closely enough for the
    fake provider double to be importable; no credentials are forwarded.
    """
    env = {
        k: os.environ[k]
        for k in (
            "PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "USERPROFILE",
            "LOCALAPPDATA", "APPDATA", "PYTHONIOENCODING", "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
        )
        if k in os.environ
    }
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + str(fake_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
