"""Goal 09 — transcript input-contract and contextual-validation tests.

Pure, offline: no database, no network, no Supabase. Structural validation is
provided by Pydantic 2 (``contracts.py``); cross-segment / contextual rules by
``validation.py``.
"""
import uuid

import pytest
from pydantic import ValidationError

from app.transcripts.contracts import (
    TranscriptInput,
    TranscriptSegmentInput,
    TranscriptSource,
)
from app.transcripts.validation import (
    TranscriptValidationError,
    validate_transcript,
)


def _payload(**overrides):
    base = {
        "video_id": str(uuid.uuid4()),
        "content_right_id": str(uuid.uuid4()),
        "source_type": "creator_provided",
        "language": "bn",
        "segments": [
            {"start_time": 0, "end_time": 2.0, "text": "  প্রথম  বাক্য  "},
            {"start_time": 2.0, "end_time": 4.5, "text": "Second one."},
        ],
    }
    base.update(overrides)
    return base


def _seg(start, end, text="x"):
    return {"start_time": start, "end_time": end, "text": text}


# --- Input validation: valid --------------------------------------------------


def test_valid_transcript_parses_and_preserves_original_text() -> None:
    contract = TranscriptInput.model_validate(_payload())
    assert contract.source_type is TranscriptSource.CREATOR_PROVIDED
    assert isinstance(contract.video_id, uuid.UUID)
    assert len(contract.segments) == 2
    # Original text is preserved VERBATIM (no silent trim / repair).
    assert contract.segments[0].text == "  প্রথম  বাক্য  "
    assert isinstance(contract.segments[0].start_time, float)


@pytest.mark.parametrize("source", [s.value for s in TranscriptSource])
def test_all_supported_sources_are_accepted(source) -> None:
    assert TranscriptInput.model_validate(_payload(source_type=source)).source_type.value == source


def test_optional_speaker_and_metadata_are_accepted() -> None:
    payload = _payload(segments=[
        _seg(0, 2, "hello"),
        {**_seg(2, 4), "speaker": "speaker_1", "metadata": {"asr_model": "future-v1"}},
    ])
    contract = TranscriptInput.model_validate(payload)
    assert contract.segments[1].speaker == "speaker_1"
    assert contract.segments[1].metadata == {"asr_model": "future-v1"}


# --- Input validation: invalid (structural) -----------------------------------


def test_empty_transcript_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(segments=[]))


def test_empty_segment_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(segments=[_seg(0, 2, "   \t\n ")]))


def test_negative_start_time_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(segments=[_seg(-1, 2)]))


@pytest.mark.parametrize("bad", ["abc", None, True, "2.0", [], {}])
def test_invalid_timestamp_types_are_rejected(bad) -> None:
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(segments=[_seg(bad, 2)]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_timestamps_are_rejected(bad) -> None:
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(segments=[_seg(0, bad)]))
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(segments=[_seg(bad, 2)]))


@pytest.mark.parametrize("pair", [(2, 2), (3, 2)])
def test_end_not_after_start_is_rejected(pair) -> None:
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(segments=[_seg(*pair)]))


def test_unknown_source_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(source_type="youtube_captions"))


def test_unknown_field_is_rejected_no_silent_discard() -> None:
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(unexpected_field="x"))


def test_blank_language_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TranscriptInput.model_validate(_payload(language="   "))


# --- Contextual validation (validation.py) -----------------------------------


def test_out_of_order_segments_are_rejected() -> None:
    payload = _payload(segments=[_seg(5, 6, "a"), _seg(2, 3, "b")])
    with pytest.raises(TranscriptValidationError) as exc:
        validate_transcript(TranscriptInput.model_validate(payload))
    assert "chronologically" in str(exc.value)


def test_equal_start_times_are_rejected() -> None:
    payload = _payload(segments=[_seg(2, 3, "a"), _seg(2, 4, "b")])
    with pytest.raises(TranscriptValidationError):
        validate_transcript(TranscriptInput.model_validate(payload))


def test_segment_exceeding_known_duration_is_rejected() -> None:
    payload = _payload()  # last segment ends at 4.5
    with pytest.raises(TranscriptValidationError) as exc:
        validate_transcript(
            TranscriptInput.model_validate(payload), video_duration_seconds=3
        )
    assert "exceeds the known video duration" in str(exc.value)


def test_unknown_duration_skips_bound() -> None:
    payload = _payload(segments=[_seg(0, 100, "a")])
    report = validate_transcript(
        TranscriptInput.model_validate(payload), video_duration_seconds=None
    )
    assert report.max_end_time == 100.0


def test_valid_overlapping_segments_are_allowed_and_preserved() -> None:
    # [0,3] and [2,5] overlap legitimately (alignment / speaker-turn windows).
    payload = _payload(segments=[_seg(0, 3, "a"), _seg(2, 5, "b")])
    contract = TranscriptInput.model_validate(payload)
    report = validate_transcript(contract, video_duration_seconds=None)
    assert report.segment_count == 2
    assert report.overlapping_pair_count == 1
    # No repair: timestamps and text are untouched.
    assert [s.start_time for s in contract.segments] == [0.0, 2.0]
    assert [s.end_time for s in contract.segments] == [3.0, 5.0]


def test_validation_never_mutates_payload() -> None:
    payload = _payload(segments=[_seg(0, 3, "  keep  "), _seg(2, 5, "b")])
    contract = TranscriptInput.model_validate(payload)
    before = [s.text for s in contract.segments]
    validate_transcript(contract, video_duration_seconds=10)
    assert [s.text for s in contract.segments] == before == ["  keep  ", "b"]


def test_segment_input_is_usable_standalone() -> None:
    seg = TranscriptSegmentInput.model_validate(_seg(0, 2, "hi"))
    assert (seg.start_time, seg.end_time, seg.text) == (0.0, 2.0, "hi")
