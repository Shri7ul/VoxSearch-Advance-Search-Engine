"""Goal 12 — word_alignment.py pure validation tests.

Covers the section-16 "Timestamp and contract tests" that live in the pure
domain layer: invalid word windows, out-of-parent-window timestamps,
non-monotonic word starts, missing word timestamps, token count / text mismatch,
null original text, timeline offset conversion, the pinned adapter flags, and
the fail-closed char-corroboration reconstruction. These tests prove NO
interpolated / backfilled / clamped / fabricated timestamp can become a
persisted :class:`AlignedWord`.

All offline, dependency free, NO torch / whisperx, NO database.
"""
from __future__ import annotations

import math
import uuid

import pytest

from app.transcripts import word_contract as wc
from app.transcripts.word_alignment import (
    AlignmentRequest,
    SegmentData,
    SkippedSegment,
    build_request,
    prepare_segments,
    reconstruct_word_chars,
    validate_provider_segment,
)
from app.transcripts.word_contract import AlignedWord, tokenize_original

# A 10-second clip: video window [10000, 12000] ms == audio window [0, 2000] ms.
OFFSET_MS = 10_000
DURATION_MS = 60_000
SID = uuid.uuid4()


def _segment(
    *,
    start: float = 10.0,
    end: float = 12.0,
    original_text: str | None = "Hello world",
    segment_id: uuid.UUID = SID,
    ref: str = "S1",
) -> SegmentData:
    return SegmentData(
        segment_id=segment_id,
        segment_ref=ref,
        start_seconds=start,
        end_seconds=end,
        original_text=original_text,
    )


def _prepared(**kw: object):
    prep = prepare_segments(
        [_segment(**kw)],  # type: ignore[arg-type]
        timeline_offset_ms=OFFSET_MS,
        audio_duration_ms=DURATION_MS,
    )
    assert not prep.skipped, prep.skipped
    return prep.prepared[0]


def _word(
    text: str,
    start: float,
    end: float,
    *,
    char_start: float | None = None,
    char_end: float | None = None,
    corroborated: bool = True,
) -> dict[str, object]:
    cs = start if char_start is None else char_start
    ce = end if char_end is None else char_end
    return {
        "word": text,
        "start": start,
        "end": end,
        "char_start": cs,
        "char_end": ce,
        "char_corroborated": corroborated,
    }


def _provider_segment(words: list[dict[str, object]], *, error: object = None) -> dict[str, object]:
    return {"segment_id": str(SID), "words": words, "error": error}


# --- prepare_segments: timeline offset + fail-closed skips -------------------

class TestPrepareSegments:
    def test_offset_formula(self) -> None:
        p = _prepared()
        assert p.video_start_ms == 10_000
        assert p.video_end_ms == 12_000
        assert p.audio_start_ms == 0        # 10000 - 10000
        assert p.audio_end_ms == 2_000      # 12000 - 10000
        assert [t.text for t in p.tokens] == ["Hello", "world"]

    def test_offset_zero(self) -> None:
        prep = prepare_segments(
            [_segment(start=0.0, end=2.0)],
            timeline_offset_ms=0,
            audio_duration_ms=DURATION_MS,
        )
        p = prep.prepared[0]
        assert (p.audio_start_ms, p.audio_end_ms) == (0, 2_000)

    def test_null_original_text_skipped(self) -> None:
        prep = prepare_segments(
            [_segment(original_text=None)],
            timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=DURATION_MS,
        )
        assert prep.prepared == ()
        assert prep.skipped == (SkippedSegment("S1", wc.REASON_NULL_ORIGINAL_TEXT),)

    def test_never_fabricates_from_normalized_text(self) -> None:
        # A None original must be skipped even though the caller knows a
        # normalized text; the pure layer is given NO normalized text at all.
        prep = prepare_segments(
            [_segment(original_text=None)],
            timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=DURATION_MS,
        )
        assert prep.prepared == ()

    @pytest.mark.parametrize("text", ["", "   \t\n  "])
    def test_empty_original_text_skipped(self, text: str) -> None:
        prep = prepare_segments(
            [_segment(original_text=text)],
            timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=DURATION_MS,
        )
        assert prep.prepared == ()
        assert prep.skipped[0].reason == wc.REASON_EMPTY_ORIGINAL_TEXT

    def test_inverted_window_skipped(self) -> None:
        prep = prepare_segments(
            [_segment(start=12.0, end=12.0)],
            timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=DURATION_MS,
        )
        assert prep.prepared == ()
        assert prep.skipped[0].reason == wc.REASON_INVALID_WINDOW

    def test_nonfinite_window_skipped(self) -> None:
        prep = prepare_segments(
            [_segment(start=float("nan"), end=12.0)],
            timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=DURATION_MS,
        )
        assert prep.prepared == ()
        assert prep.skipped[0].reason == wc.REASON_INVALID_WINDOW

    def test_segment_outside_audio_window_skipped(self) -> None:
        # Video [10, 12]s but audio window is only [0, 5]s of audio timeline:
        # video > offset means audio_end_ms (2000) > 5000? no -> use a segment
        # whose AUDIO window exceeds duration_ms.
        prep = prepare_segments(
            [_segment(start=10.0, end=20.0)],
            timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=5_000,  # only 5s of audio available
        )
        assert prep.prepared == ()
        assert prep.skipped[0].reason == wc.REASON_OUTSIDE_AUDIO_WINDOW

    def test_audio_start_before_zero_skipped(self) -> None:
        # A negative offset with a segment near video 0 -> negative audio start.
        prep = prepare_segments(
            [_segment(start=0.0, end=1.0)],
            timeline_offset_ms=5_000,
            audio_duration_ms=DURATION_MS,
        )
        assert prep.prepared == ()
        assert prep.skipped[0].reason == wc.REASON_OUTSIDE_AUDIO_WINDOW

    @pytest.mark.parametrize("offset", [True, 1.5, "10000"])
    def test_offset_must_be_strict_int(self, offset: object) -> None:
        with pytest.raises(ValueError):
            prepare_segments(
                [_segment()],
                timeline_offset_ms=offset,  # type: ignore[arg-type]
                audio_duration_ms=DURATION_MS,
            )


# --- build_request: pinned flags + audio timeline ----------------------------

class TestBuildRequest:
    def test_pinned_adapter_flags(self) -> None:
        p = _prepared()
        req = build_request("en", prepare_segments(
            [_segment()], timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=DURATION_MS,
        ))
        assert isinstance(req, AlignmentRequest)
        assert req.interpolate_method == "ignore"      # NOT "nearest"
        assert req.return_char_alignments is True
        assert req.language == "en"

    def test_provider_receives_audio_timeline_seconds(self) -> None:
        req = build_request("en", prepare_segments(
            [_segment()], timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=DURATION_MS,
        ))
        task = req.segments[0]
        assert task.start == 0.0    # audio seconds, NOT video 10.0
        assert task.end == 2.0
        assert task.text == "Hello world"  # VERBATIM original text

    def test_tokens_carry_original_offsets(self) -> None:
        req = build_request("en", prepare_segments(
            [_segment()], timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=DURATION_MS,
        ))
        assert req.segments[0].tokens == (
            {"text": "Hello", "char_start": 0, "char_end": 5, "ordinal": 0},
            {"text": "world", "char_start": 6, "char_end": 11, "ordinal": 1},
        )

    def test_to_dict_is_json_serializable(self) -> None:
        import json

        req = build_request("en", prepare_segments(
            [_segment()], timeline_offset_ms=OFFSET_MS,
            audio_duration_ms=DURATION_MS,
        ))
        payload = json.loads(json.dumps(req.to_dict()))
        assert payload["interpolate_method"] == "ignore"
        assert payload["return_char_alignments"] is True
        assert payload["segments"][0]["segment_id"] == str(SID)


# --- validate_provider_segment: the no-fabrication guarantee -----------------

class TestValidateProviderSegment:
    def _valid_words(self) -> list[dict[str, object]]:
        return [
            _word("Hello", 0.0, 0.4),
            _word("world", 0.5, 0.9),
        ]

    def test_full_success_persists_video_timeline_ms(self) -> None:
        p = _prepared()
        out = validate_provider_segment(
            p, _provider_segment(self._valid_words()), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, tuple)
        assert all(isinstance(w, AlignedWord) for w in out)
        # audio ms + offset -> video ms
        assert [(w.start_ms, w.end_ms) for w in out] == [(10_000, 10_400), (10_500, 10_900)]
        assert [(w.text, w.ordinal, w.char_start, w.char_end) for w in out] == [
            ("Hello", 0, 0, 5),
            ("world", 1, 6, 11),
        ]

    def test_missing_word_timing_skips_whole_segment(self) -> None:
        p = _prepared()
        words = [{"word": "Hello", "char_corroborated": True}, _word("world", 0.5, 0.9)]
        out = validate_provider_segment(
            p, _provider_segment(words), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_WORD_TIMING_MISSING

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, True, "1.0", None])
    def test_invalid_word_timing_rejected(self, bad: object) -> None:
        p = _prepared()
        words = [{"word": "Hello", "start": bad, "end": 0.4,
                  "char_start": 0.0, "char_end": 0.4, "char_corroborated": True},
                 _word("world", 0.5, 0.9)]
        out = validate_provider_segment(
            p, _provider_segment(words), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason in (
            wc.REASON_WORD_TIMING_MISSING, wc.REASON_WORD_TIMING_INVALID
        )

    def test_end_not_after_start_rejected(self) -> None:
        p = _prepared()
        words = [_word("Hello", 0.5, 0.5), _word("world", 0.6, 0.9)]
        out = validate_provider_segment(
            p, _provider_segment(words), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_WORD_TIMING_INVALID

    def test_token_count_mismatch_skips_entire_segment(self) -> None:
        p = _prepared()
        out = validate_provider_segment(
            p, _provider_segment([_word("Hello", 0.0, 0.4)]),
            timeline_offset_ms=OFFSET_MS,
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_TOKEN_COUNT_MISMATCH
        # ZERO words, never a partial/re-indexed subset.

    def test_token_text_mismatch_skips(self) -> None:
        p = _prepared()
        words = [_word("hello", 0.0, 0.4), _word("world", 0.5, 0.9)]  # case differs
        out = validate_provider_segment(
            p, _provider_segment(words), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_TOKEN_TEXT_MISMATCH

    def test_words_not_a_sequence_skips(self) -> None:
        p = _prepared()
        out = validate_provider_segment(
            p, {"segment_id": str(SID), "words": "not-a-list", "error": None},
            timeline_offset_ms=OFFSET_MS,
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_TOKEN_COUNT_MISMATCH

    def test_missing_char_corroboration_flag_rejected(self) -> None:
        p = _prepared()
        words = [_word("Hello", 0.0, 0.4, corroborated=False), _word("world", 0.5, 0.9)]
        out = validate_provider_segment(
            p, _provider_segment(words), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_CHAR_CORROBORATION_FAILED

    def test_char_timing_outside_word_rejected(self) -> None:
        p = _prepared()
        words = [_word("Hello", 0.0, 0.4, char_start=0.0, char_end=0.9),
                 _word("world", 0.5, 0.9)]
        out = validate_provider_segment(
            p, _provider_segment(words), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_CHAR_CORROBORATION_FAILED

    def test_word_outside_parent_window_rejected(self) -> None:
        p = _prepared()  # audio window [0, 2000] ms
        words = [_word("Hello", 2.5, 2.9), _word("world", 3.0, 3.4)]
        out = validate_provider_segment(
            p, _provider_segment(words), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_WORD_OUTSIDE_WINDOW

    def test_non_monotonic_word_starts_rejected(self) -> None:
        p = _prepared()
        words = [_word("Hello", 1.0, 1.4), _word("world", 0.5, 0.9)]
        out = validate_provider_segment(
            p, _provider_segment(words), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_NON_MONOTONIC_TIMING

    def test_none_provider_segment_skips(self) -> None:
        p = _prepared()
        out = validate_provider_segment(p, None, timeline_offset_ms=OFFSET_MS)
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_PROVIDER_SEGMENT_ERROR

    def test_provider_segment_error_skips(self) -> None:
        p = _prepared()
        out = validate_provider_segment(
            p, _provider_segment([], error="model failure"), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_PROVIDER_SEGMENT_ERROR

    def test_language_error_classified(self) -> None:
        p = _prepared()
        out = validate_provider_segment(
            p,
            _provider_segment([], error="No align-model for language xx"),
            timeline_offset_ms=OFFSET_MS,
        )
        assert isinstance(out, SkippedSegment)
        assert out.reason == wc.REASON_UNSUPPORTED_LANGUAGE

    def test_no_interpolation_leakage(self) -> None:
        # A word whose 'start' is missing and whose char timings are all None
        # (simulating interpolate_method="ignore" NaN output) can NEVER become
        # an AlignedWord. It must be skipped, not backfilled.
        p = _prepared()
        words = [
            {"word": "Hello", "end": 0.4, "char_start": None,
             "char_end": None, "char_corroborated": False},
            _word("world", 0.5, 0.9),
        ]
        out = validate_provider_segment(
            p, _provider_segment(words), timeline_offset_ms=OFFSET_MS
        )
        assert isinstance(out, SkippedSegment)
        assert not isinstance(out, tuple)


# --- reconstruct_word_chars: deterministic, fail-closed ---------------------

class TestReconstructWordChars:
    def test_success(self) -> None:
        words = [{"word": "Hi"}, {"word": "there"}]
        chars = [
            {"char": "H", "start": 1.0, "end": 1.1},
            {"char": "i", "start": 1.1, "end": 1.2},
            {"char": " "},  # separator, ignored
            {"char": "t", "start": 2.0, "end": 2.1},
            {"char": "h", "start": 2.1, "end": 2.2},
            {"char": "e", "start": 2.2, "end": 2.3},
            {"char": "r", "start": 2.3, "end": 2.4},
            {"char": "e", "start": 2.4, "end": 2.5},
        ]
        ev = reconstruct_word_chars(words, chars)
        assert ev == [
            {"char_start": 1.0, "char_end": 1.2, "char_corroborated": True},
            {"char_start": 2.0, "char_end": 2.5, "char_corroborated": True},
        ]

    def test_no_real_char_timings_fails_closed(self) -> None:
        # Missing/NaN char timings (interpolate_method="ignore") -> NOT
        # corroborated. No backfill / ffill / interpolation.
        words = [{"word": "Hi"}]
        chars = [{"char": "H", "start": float("nan"), "end": float("nan")},
                 {"char": "i", "start": None, "end": None}]
        ev = reconstruct_word_chars(words, chars)
        assert ev == [{"char_start": None, "char_end": None,
                       "char_corroborated": False}]

    def test_char_text_mismatch_fails_closed(self) -> None:
        words = [{"word": "Hi"}]
        chars = [{"char": "X", "start": 1.0, "end": 1.1},
                 {"char": "i", "start": 1.1, "end": 1.2}]
        ev = reconstruct_word_chars(words, chars)
        assert ev[0]["char_corroborated"] is False

    def test_char_shortage_fails_closed(self) -> None:
        words = [{"word": "Hello"}]
        chars = [{"char": "H", "start": 1.0, "end": 1.1}]
        ev = reconstruct_word_chars(words, chars)
        assert ev[0]["char_corroborated"] is False

    def test_empty_word_fails_closed(self) -> None:
        ev = reconstruct_word_chars([{"word": ""}], [])
        assert ev == [{"char_start": None, "char_end": None,
                       "char_corroborated": False}]
