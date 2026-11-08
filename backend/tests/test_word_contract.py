"""Goal 12 — word_contract.py pure unit tests.

Covers the section-16 "Timestamp and contract tests" for the pure primitives:
Decimal/ROUND_HALF_UP conversion, invalid numeric inputs, negative timestamps,
boolean rejection, original-text offset correctness, Unicode/Bangla
tokenization, punctuation-only tokens, the language gate, the audio-asset input
contract, and the closed status vocabulary. All tests are offline, dependency
free and import NO torch / whisperx.
"""
from __future__ import annotations

import math
import uuid

import pytest
from pydantic import ValidationError

from app.transcripts import word_contract as wc
from app.transcripts.word_contract import (
    ALIGNMENT_RUN_STATUSES,
    BENGALI_LANGUAGE_CODES,
    CURRENT_SUCCESS_STATUSES,
    SUPPORTED_ALIGNMENT_LANGUAGES,
    SourceAudioAssetInput,
    is_alignment_language_supported,
    is_finite_number,
    language_gate_reason,
    ms_to_seconds,
    seconds_to_ms,
    tokenize_original,
    validate_provider_second,
)


# --- seconds_to_ms: Decimal + ROUND_HALF_UP ----------------------------------

class TestSecondsToMs:
    def test_integer_seconds(self) -> None:
        assert seconds_to_ms(2.0) == 2000
        assert seconds_to_ms(0.0) == 0
        assert seconds_to_ms(12) == 12000  # int input accepted

    def test_fractional_seconds_uses_decimal_str(self) -> None:
        # Decimal(str(1.2345)) * 1000 == Decimal('1234.5000'), ROUND_HALF_UP.
        assert seconds_to_ms(1.2345) == 1235
        assert seconds_to_ms(1.2344) == 1234

    def test_round_half_up_not_bankers_rounding(self) -> None:
        # 0.5 ms is the decisive boundary: Python's banker's round(0.5) == 0,
        # but ROUND_HALF_UP must give 1 ms.
        assert round(0.5) == 0
        assert seconds_to_ms(0.0005) == 1
        assert seconds_to_ms(0.0015) == 2  # 1.5 ms -> 2
        assert seconds_to_ms(0.0004) == 0

    def test_float_binary_artifact_is_deterministic(self) -> None:
        # 0.1 + 0.2 == 0.30000000000000004 in binary float; str() + Decimal
        # keeps the conversion reproducible (300 ms, not a drifting value).
        value = 0.1 + 0.2
        assert value != 0.3
        assert seconds_to_ms(value) == 300

    @pytest.mark.parametrize(
        "bad",
        [True, False, "1.0", None, float("nan"), float("inf"), float("-inf")],
    )
    def test_rejects_non_finite_or_non_numeric(self, bad: object) -> None:
        with pytest.raises(ValueError):
            seconds_to_ms(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-0.001, -1.0, -100, -1])
    def test_rejects_negative_timestamps(self, bad: float) -> None:
        with pytest.raises(ValueError):
            seconds_to_ms(bad)

    def test_large_but_finite_value(self) -> None:
        assert seconds_to_ms(1000.0) == 1_000_000


# --- ms_to_seconds: strict integer input -------------------------------------

class TestMsToSeconds:
    def test_basic(self) -> None:
        assert ms_to_seconds(1500) == 1.5
        assert ms_to_seconds(0) == 0.0
        assert ms_to_seconds(1000) == 1.0

    @pytest.mark.parametrize("bad", [True, 1.5, "100", None, -1, -100])
    def test_rejects_bool_float_str_none_negative(self, bad: object) -> None:
        with pytest.raises(ValueError):
            ms_to_seconds(bad)  # type: ignore[arg-type]

    def test_round_trip_reflects_integer_quantization(self) -> None:
        # Quantization is a representation choice: exact recovery is NOT
        # guaranteed for fractional milliseconds.
        assert ms_to_seconds(seconds_to_ms(2.0)) == 2.0
        assert ms_to_seconds(seconds_to_ms(1.2345)) == 1.235


# --- Low-level numeric validators --------------------------------------------

class TestNumericValidators:
    def test_is_finite_number(self) -> None:
        assert is_finite_number(3) is True
        assert is_finite_number(3.5) is True
        assert is_finite_number(0) is True
        assert is_finite_number(True) is False
        assert is_finite_number(float("nan")) is False
        assert is_finite_number(float("inf")) is False
        assert is_finite_number("1") is False
        assert is_finite_number(None) is False

    @pytest.mark.parametrize(
        "value,expected",
        [
            (1.2, 1.2),
            (0, 0.0),
            (3, 3.0),
        ],
    )
    def test_validate_provider_second_accepts_finite_nonnegative(
        self, value: object, expected: float
    ) -> None:
        assert validate_provider_second(value) == expected

    @pytest.mark.parametrize(
        "bad",
        [None, True, "1.0", float("nan"), float("inf"), float("-inf"), -1.0, -0.001],
    )
    def test_validate_provider_second_fails_closed(self, bad: object) -> None:
        # Missing / NaN / Inf / bool / str / negative -> None (never persisted).
        assert validate_provider_second(bad) is None


# --- tokenize_original --------------------------------------------------------

class TestTokenizeOriginal:
    def test_offsets_index_verbatim_text(self) -> None:
        tokens = tokenize_original("Hello world")
        assert [(t.text, t.char_start, t.char_end, t.ordinal) for t in tokens] == [
            ("Hello", 0, 5, 0),
            ("world", 6, 11, 1),
        ]
        for t in tokens:
            assert "Hello world"[t.char_start : t.char_end] == t.text

    def test_ordinals_are_sequential(self) -> None:
        tokens = tokenize_original("a b c d e")
        assert [t.ordinal for t in tokens] == [0, 1, 2, 3, 4]

    def test_multiple_and_unicode_whitespace(self) -> None:
        tokens = tokenize_original("  one \t two\n\nthree \u00a0 four  ")
        assert [t.text for t in tokens] == ["one", "two", "three", "four"]
        for t in tokens:
            assert t.text == "  one \t two\n\nthree \u00a0 four  "[t.char_start : t.char_end]

    def test_punctuation_preserved(self) -> None:
        tokens = tokenize_original("Hello, world! ...")
        assert [t.text for t in tokens] == ["Hello,", "world!", "..."]
        for t in tokens:
            assert "Hello, world! ..."[t.char_start : t.char_end] == t.text

    def test_punctuation_only_token(self) -> None:
        tokens = tokenize_original("...")
        assert len(tokens) == 1
        assert tokens[0].text == "..."
        assert (tokens[0].char_start, tokens[0].char_end) == (0, 3)

    def test_bangla_unicode_and_dari(self) -> None:
        text = "আমি ভালো আছি।"
        tokens = tokenize_original(text)
        assert [t.text for t in tokens] == ["আমি", "ভালো", "আছি।"]
        # Offsets are Python CODE-POINT indices into the verbatim source.
        for t in tokens:
            assert text[t.char_start : t.char_end] == t.text

    def test_bangla_standalone_punctuation_token(self) -> None:
        tokens = tokenize_original("থামুন । ধন্যবাদ")
        assert [t.text for t in tokens] == ["থামুন", "।", "ধন্যবাদ"]

    def test_empty_and_whitespace_only(self) -> None:
        assert tokenize_original("") == ()
        assert tokenize_original("   \t\n  ") == ()

    def test_non_str_raises(self) -> None:
        with pytest.raises(ValueError):
            tokenize_original(None)  # type: ignore[arg-type]

    def test_matches_goal10_max_nonwhitespace_semantics(self) -> None:
        # The tokenizer MUST tile exactly like Goal 10's _iter_tokens.
        from app.transcripts.sentence_segmentation import _iter_tokens

        for text in ["Hello world", "a \t b  c", "আমি ভালো। ধন্যবাদ", "..., ... !!"]:
            goal10 = [(s, e) for s, e in _iter_tokens(text)]
            goal12 = [(t.char_start, t.char_end) for t in tokenize_original(text)]
            assert goal12 == goal10


# --- SourceAudioAssetInput ----------------------------------------------------

def _audio_kwargs(**overrides: object) -> dict[str, object]:
    # Spec-compliant field set (design-freeze §3.1 / full task §5.A / §7).
    base: dict[str, object] = dict(
        video_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        storage_uri="bucket/authorized-clip.wav",
        checksum_sha256="a" * 64,
        mime_type="audio/wav",
        rights_basis="explicit-license",
        timeline_offset_ms=0,
        duration_ms=1000,
    )
    base.update(overrides)
    return base


class TestSourceAudioAssetInput:
    def test_valid_with_negative_offset(self) -> None:
        obj = SourceAudioAssetInput(**_audio_kwargs(timeline_offset_ms=-500))
        assert obj.timeline_offset_ms == -500  # audio may start before video 0
        assert obj.duration_ms == 1000
        assert obj.storage_uri == "bucket/authorized-clip.wav"
        assert obj.checksum_sha256 == "a" * 64
        assert obj.mime_type == "audio/wav"
        assert obj.rights_basis == "explicit-license"
        assert obj.sample_rate_hz is None  # optional, NULL = unknown

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SourceAudioAssetInput(**_audio_kwargs(unexpected="x"))

    @pytest.mark.parametrize("ref", ["", "   ", "\t\n"])
    def test_blank_storage_uri_rejected(self, ref: str) -> None:
        with pytest.raises(ValidationError):
            SourceAudioAssetInput(**_audio_kwargs(storage_uri=ref))

    @pytest.mark.parametrize("field", ["mime_type", "rights_basis"])
    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_blank_provenance_fields_rejected(self, field: str, value: str) -> None:
        with pytest.raises(ValidationError):
            SourceAudioAssetInput(**_audio_kwargs(**{field: value}))

    @pytest.mark.parametrize(
        "bad",
        [
            "not-hex",
            "A" * 64,          # uppercase rejected (must be lowercase)
            "a" * 63,          # too short
            "a" * 65,          # too long
            "g" * 64,          # out-of-range hex char
            "",
        ],
    )
    def test_checksum_must_be_64_lowercase_hex(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            SourceAudioAssetInput(**_audio_kwargs(checksum_sha256=bad))

    def test_optional_sample_rate(self) -> None:
        assert SourceAudioAssetInput(**_audio_kwargs()).sample_rate_hz is None
        obj = SourceAudioAssetInput(**_audio_kwargs(sample_rate_hz=16_000))
        assert obj.sample_rate_hz == 16_000

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "16000"])
    def test_invalid_sample_rate_rejected(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            SourceAudioAssetInput(**_audio_kwargs(sample_rate_hz=bad))

    @pytest.mark.parametrize("duration", [0, -1, -1000])
    def test_nonpositive_duration_rejected(self, duration: int) -> None:
        with pytest.raises(ValidationError):
            SourceAudioAssetInput(**_audio_kwargs(duration_ms=duration))

    def test_fractional_float_offset_rejected(self) -> None:
        # Pydantic int rejects a float with a fractional part.
        with pytest.raises(ValidationError):
            SourceAudioAssetInput(**_audio_kwargs(timeline_offset_ms=1.5))

    def test_storage_uri_is_opaque_not_a_url(self) -> None:
        # The contract carries an opaque handle; it is deliberately not parsed
        # as a URL and never interpreted by the core runtime.
        obj = SourceAudioAssetInput(**_audio_kwargs())
        assert obj.storage_uri == "bucket/authorized-clip.wav"


# --- Language gate ------------------------------------------------------------

class TestLanguageGate:
    @pytest.mark.parametrize("code", ["en", "fr", "de", "es", "it", "ja", "zh", "hi"])
    def test_supported_codes(self, code: str) -> None:
        assert is_alignment_language_supported(code) is True
        assert language_gate_reason(code) is None

    def test_case_and_whitespace_normalized(self) -> None:
        assert is_alignment_language_supported("EN") is True
        assert is_alignment_language_supported("  eN ") is True
        assert language_gate_reason(" Fr ") is None

    @pytest.mark.parametrize("code", ["bn", "ben", "bn-BD", "bn-IN", "BN-BD"])
    def test_bengali_explicitly_rejected(self, code: str) -> None:
        assert is_alignment_language_supported(code) is False
        assert language_gate_reason(code) == wc.REASON_UNSUPPORTED_LANGUAGE

    @pytest.mark.parametrize("code", ["xx", "zz", "unknown", "eng", "en-US"])
    def test_unknown_unsupported_codes(self, code: str) -> None:
        assert is_alignment_language_supported(code) is False
        assert language_gate_reason(code) == "unsupported_language"

    @pytest.mark.parametrize("code", ["", "   ", None, 123])
    def test_invalid_language(self, code: object) -> None:
        assert is_alignment_language_supported(code) is False  # type: ignore[arg-type]
        assert language_gate_reason(code) == "invalid_language"  # type: ignore[arg-type]

    def test_bengali_codes_are_all_deferred(self) -> None:
        assert BENGALI_LANGUAGE_CODES == {"bn", "ben", "bn-bd", "bn-in"}
        assert BENGALI_LANGUAGE_CODES.isdisjoint(SUPPORTED_ALIGNMENT_LANGUAGES)

    def test_allowlist_matches_isolated_worker(self) -> None:
        # word_contract and the isolated worker MUST agree (no drift).
        from alignment_worker.worker import BENGALI_CODES, SUPPORTED_LANGUAGES

        assert SUPPORTED_ALIGNMENT_LANGUAGES == SUPPORTED_LANGUAGES
        assert BENGALI_LANGUAGE_CODES == BENGALI_CODES


# --- Closed status vocabulary (3 sources must agree) -------------------------

class TestStatusVocabulary:
    def test_status_tuple_is_closed_and_consistent(self) -> None:
        from app.db.models import ALIGNMENT_RUN_STATUS_VALUES
        from app.repositories import transcript_alignment_run_repository as runs

        # Service-level EMITTABLE statuses (full task §5.B suggested set):
        # the orchestrator never persists a run for an initial rights-gate
        # rejection (ZERO runs), so ``rejected_rights`` is NOT emitted here.
        assert set(ALIGNMENT_RUN_STATUSES) == {
            "pending", "running", "completed", "partial", "failed",
            "unsupported_language", "superseded",
        }
        # The DB CHECK (design-freeze §4/§6) RESERVES ``rejected_rights`` as a
        # closed label for schema compliance, but this goal NEVER persists it.
        # It is exactly the service set plus that one reserved label.
        assert set(ALIGNMENT_RUN_STATUS_VALUES) == (
            set(ALIGNMENT_RUN_STATUSES) | {"rejected_rights"}
        )
        assert tuple(ALIGNMENT_RUN_STATUS_VALUES) == (
            "pending", "running", "completed", "partial", "failed",
            "unsupported_language", "rejected_rights", "superseded",
        )
        assert set(runs.CURRENT_SUCCESS_STATUSES) == set(CURRENT_SUCCESS_STATUSES)
        assert CURRENT_SUCCESS_STATUSES == ("completed", "partial")

    def test_ms_per_second_is_decimal_1000(self) -> None:
        from decimal import Decimal

        assert wc.MS_PER_SECOND == Decimal(1000)
