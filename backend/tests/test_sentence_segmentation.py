"""Goal 10 — deterministic multilingual sentence-segmentation tests.

Pure, deterministic fixtures only. No database, no external APIs, no network,
no ML models, no credentials. Tests cover English, Bangla, mixed-language text,
character-offset integrity, timestamp provenance, and documented edge cases.
"""
from __future__ import annotations

import unicodedata
from dataclasses import fields

import pytest
from pydantic import ValidationError

from app.transcripts.normalization import normalize_text
from app.transcripts.sentence_segmentation import (
    EmptySourceTextError,
    Sentence,
    SentenceSegmentationInput,
    SentenceSpan,
    TimestampPrecision,
    find_sentence_spans,
    segment_sentences,
)

# --- Fixtures / helpers -------------------------------------------------------

REF = "seg-0001"
START = 12.5
END = 18.75
LANG = "bn"


def _input(source_text: str, **overrides) -> SentenceSegmentationInput:
    data = {
        "source_text": source_text,
        "source_segment_ref": REF,
        "start_seconds": START,
        "end_seconds": END,
        "language": LANG,
    }
    data.update(overrides)
    return SentenceSegmentationInput(**data)


def _sentences(source_text: str, **overrides) -> tuple[Sentence, ...]:
    return segment_sentences(_input(source_text, **overrides))


def _assert_tiling(source_text: str, spans: tuple[SentenceSpan, ...]) -> None:
    """Spans partition the source exactly: no loss, no overlap, no reorder."""
    assert spans, "non-blank input must produce at least one sentence"
    assert spans[0].char_start == 0
    assert spans[-1].char_end == len(source_text)
    for prev, cur in zip(spans, spans[1:]):
        assert prev.char_end == cur.char_start
    for span in spans:
        assert span.char_start < span.char_end
        assert source_text[span.char_start:span.char_end] == span.text
    assert "".join(s.text for s in spans) == source_text


def _assert_provenance(sentences: tuple[Sentence, ...], source_text: str) -> None:
    """Every sentence is verbatim, ordered, and carries segment-level time only."""
    assert [s.ordinal for s in sentences] == list(range(len(sentences)))
    assert sentences[0].char_start == 0
    assert sentences[-1].char_end == len(source_text)
    for prev, cur in zip(sentences, sentences[1:]):
        assert prev.char_end == cur.char_start
    for s in sentences:
        assert source_text[s.char_start:s.char_end] == s.text
        assert s.source_segment_ref == REF
        assert s.source_segment_start_seconds == START
        assert s.source_segment_end_seconds == END
        assert s.language == LANG
        assert s.timestamp_precision is TimestampPrecision.SEGMENT_LEVEL
        # Exact sentence times are NEVER fabricated from character positions.
        assert s.exact_start_seconds is None
        assert s.exact_end_seconds is None
        assert s.normalized_text == normalize_text(s.text).normalized
    assert "".join(s.text for s in sentences) == source_text


# --- 1. English ---------------------------------------------------------------


def test_english_single_sentence() -> None:
    text = "ModernBERT struggles with certain reranking workloads."
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text
    _assert_tiling(text, find_sentence_spans(text))
    _assert_provenance(result, text)


def test_english_multiple_sentences() -> None:
    text = "First one. Second one. Third one."
    result = _sentences(text)
    assert [s.text for s in result] == ["First one.", " Second one.", " Third one."]
    _assert_provenance(result, text)


def test_english_question_and_exclamation() -> None:
    text = "Is this correct? Yes! Absolutely?"
    result = _sentences(text)
    assert [s.text.strip() for s in result] == [
        "Is this correct?", "Yes!", "Absolutely?",
    ]
    _assert_provenance(result, text)


def test_english_abbreviation_dr_not_split() -> None:
    text = "Dr. Rahman discussed the model."
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text
    _assert_provenance(result, text)


def test_english_multiple_abbreviations_not_split() -> None:
    text = "Prof. Ahmed and Mr. Khan met at 3 p.m. to review the model."
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text


def test_english_dotted_acronym_not_split() -> None:
    text = "He moved to the U.S. Today is fine."
    # U.S. is a dotted acronym: its period is not a boundary. The sentence
    # therefore under-splits (documented trade-off), never loses text.
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text


def test_english_eg_and_phd_not_split() -> None:
    text = "Use e.g. Ph.D. level methods for this task."
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text


def test_english_single_letter_initial_not_split() -> None:
    text = "J. Smith wrote the paper."
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text


def test_english_decimal_number_not_split() -> None:
    text = "The value is 3.14."
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text
    _assert_provenance(result, text)


def test_english_decimal_then_next_sentence() -> None:
    text = "The value is 3.14. Then we round it."
    result = _sentences(text)
    assert [s.text.strip() for s in result] == [
        "The value is 3.14.", "Then we round it.",
    ]
    _assert_provenance(result, text)


def test_english_url_not_split_internally() -> None:
    text = "Visit https://example.com/path?q=1. Then leave."
    result = _sentences(text)
    assert [s.text.strip() for s in result] == [
        "Visit https://example.com/path?q=1.", "Then leave.",
    ]
    _assert_provenance(result, text)


def test_english_domain_and_version_not_split() -> None:
    text = "See node.js and v1.2.3 for details."
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text


def test_english_technical_terms_preserved() -> None:
    text = "ModernBERT and BGE-M3 are useful. ColBERT is different."
    result = _sentences(text)
    assert result[0].text == "ModernBERT and BGE-M3 are useful."
    assert "BGE-M3" in result[0].text and "ColBERT" in result[1].text
    _assert_provenance(result, text)


# --- 2. Bangla ----------------------------------------------------------------


def test_bangla_danda_single_sentence() -> None:
    text = "আমি বাংলায় কথা বলি ।"
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text
    assert "।" in result[0].text
    _assert_provenance(result, text)


def test_bangla_danda_multiple_sentences() -> None:
    text = "প্রথম বাক্য। দ্বিতীয় বাক্য। তৃতীয় বাক্য।"
    result = _sentences(text)
    assert [s.text.strip() for s in result] == [
        "প্রথম বাক্য।", "দ্বিতীয় বাক্য।", "তৃতীয় বাক্য।",
    ]
    _assert_provenance(result, text)


def test_bangla_question_mark() -> None:
    text = "তুমি কেমন আছ? আমি ভালো আছি।"
    result = _sentences(text)
    assert [s.text.strip() for s in result] == ["তুমি কেমন আছ?", "আমি ভালো আছি।"]
    _assert_provenance(result, text)


def test_bangla_exclamation_mark() -> None:
    text = "কী সুন্দর দিন! আবার আসবো।"
    result = _sentences(text)
    assert [s.text.strip() for s in result] == ["কী সুন্দর দিন!", "আবার আসবো।"]
    _assert_provenance(result, text)


def test_bangla_double_danda() -> None:
    text = "একটি বাক্য॥ পরের বাক্য।"
    result = _sentences(text)
    assert [s.text.strip() for s in result] == ["একটি বাক্য॥", "পরের বাক্য।"]
    _assert_provenance(result, text)


def test_bangla_unicode_preserved_code_point_for_code_point() -> None:
    # Bangla vowel signs / combining marks survive verbatim in sentence text.
    text = "আমি বাংলায় কথা বলি। তুমি কেমন আছ?"
    result = _sentences(text)
    assert result[0].text == "আমি বাংলায় কথা বলি।"
    assert [ord(c) for c in result[0].text] == [
        ord(c) for c in "আমি বাংলায় কথা বলি।"
    ]
    _assert_provenance(result, text)


def test_decomposed_unicode_is_never_rewritten_in_sentence_text() -> None:
    # A DECOMPOSED Latin sequence (e + combining acute; NFC would compose it)
    # alongside Bangla text. Sentence text must keep the ORIGINAL code points.
    text = "Cafe\u0301 মডেল নিয়ে বলি। পরে দেখবো।"
    assert unicodedata.normalize("NFC", text) != text  # control: NFC changes it
    result = _sentences(text)
    assert len(result) == 2
    assert [ord(c) for c in result[0].text] == [
        ord(c) for c in text[: result[0].char_end]
    ]
    assert "Cafe\u0301" in result[0].text  # decomposed sequence intact
    # normalized_text IS separately NFC-composed, but never replaces text and
    # never affects the character offsets.
    assert result[0].normalized_text != result[0].text
    assert result[0].normalized_text == normalize_text(result[0].text).normalized
    _assert_provenance(result, text)


def test_bangla_no_lowercasing_or_rewrite() -> None:
    text = "BGE-M3 মডেল ব্যবহার করি। এটি ভালো।"
    result = _sentences(text)
    assert "BGE-M3" in result[0].text
    assert result[0].text == "BGE-M3 মডেল ব্যবহার করি।"
    _assert_provenance(result, text)


# --- 3. Mixed language --------------------------------------------------------


def test_mixed_bangla_english_two_units() -> None:
    text = "ModernBERT is useful. তবে কিছু ক্ষেত্রে সমস্যা হয়।"
    result = _sentences(text)
    assert len(result) == 2
    assert result[0].text == "ModernBERT is useful."
    assert result[1].text == " তবে কিছু ক্ষেত্রে সমস্যা হয়।"
    assert result[1].text.strip() == "তবে কিছু ক্ষেত্রে সমস্যা হয়।"
    _assert_provenance(result, text)


def test_mixed_technical_term_inside_bangla_sentence() -> None:
    text = "ModernBERT মডেলটি দ্রুত। তবে BGE-M3 ভিন্ন।"
    result = _sentences(text)
    assert [s.text.strip() for s in result] == [
        "ModernBERT মডেলটি দ্রুত।", "তবে BGE-M3 ভিন্ন।",
    ]
    _assert_provenance(result, text)


def test_mixed_english_abbreviation_inside_bangla() -> None:
    text = "Dr. Rahman বলেন U.S. সম্পর্কে কথা।"
    result = _sentences(text)
    assert len(result) == 1  # Dr. and U.S. are not boundaries
    assert result[0].text == text
    _assert_provenance(result, text)


def test_mixed_punctuation_english_then_bangla() -> None:
    text = "Really? তবে কেন! ঠিক আছে।"
    result = _sentences(text)
    assert [s.text.strip() for s in result] == [
        "Really?", "তবে কেন!", "ঠিক আছে।",
    ]
    _assert_provenance(result, text)


def test_mixed_bangla_acronym_and_decimal() -> None:
    text = "মান 3.14 এবং U.S. মডেল। পরে দেখবো।"
    result = _sentences(text)
    assert [s.text.strip() for s in result] == [
        "মান 3.14 এবং U.S. মডেল।", "পরে দেখবো।",
    ]
    _assert_provenance(result, text)


# --- 4. Integrity -------------------------------------------------------------


def test_sentence_order_is_source_order() -> None:
    text = "One. Two. Three. Four."
    result = _sentences(text)
    assert [s.text.strip() for s in result] == ["One.", "Two.", "Three.", "Four."]
    assert [s.char_start for s in result] == sorted(s.char_start for s in result)


def test_character_offsets_reconstruct_original_text() -> None:
    text = "  Hello,  World!  তবে কেমন? ঠিক আছে।  "
    result = _sentences(text)
    for s in result:
        assert text[s.char_start:s.char_end] == s.text
    _assert_provenance(result, text)


def test_no_silent_text_loss_full_reconstruction() -> None:
    samples = [
        "A. B! C? D।",
        "  leading.  middle\n\n last!  ",
        "Dr. Smith said \"Hi.\" Then left.",
        "Wow!!! Really?? Ok...",
    ]
    for text in samples:
        result = _sentences(text)
        assert "".join(s.text for s in result) == text


def test_punctuation_preserved_in_sentence_text() -> None:
    text = "Wait , what ?! (yes) ... Done."
    result = _sentences(text)
    assert "".join(s.text for s in result) == text
    assert any("?!" in s.text for s in result)
    assert any("(yes)" in s.text for s in result)
    assert any("..." in s.text for s in result)


def test_stable_ordinals_are_deterministic() -> None:
    text = "One. Two. Three."
    first = _sentences(text)
    for _ in range(5):
        assert [s.ordinal for s in _sentences(text)] == [s.ordinal for s in first]
        assert [s.text for s in _sentences(text)] == [s.text for s in first]


def test_parent_segment_reference_preserved_on_every_sentence() -> None:
    text = "One. Two. Three."
    result = _sentences(text)
    assert {s.source_segment_ref for s in result} == {REF}


def test_inherited_timestamps_preserved_verbatim_on_every_sentence() -> None:
    text = "First. Second. Third. Fourth."
    result = _sentences(text)
    assert len(result) == 4
    assert {s.source_segment_start_seconds for s in result} == {START}
    assert {s.source_segment_end_seconds for s in result} == {END}
    # Every sentence shares the SAME parent window -> no per-sentence time was
    # derived proportionally from character offsets or sentence lengths.
    assert result[0].source_segment_start_seconds == result[-1].source_segment_start_seconds
    assert result[0].source_segment_end_seconds == result[-1].source_segment_end_seconds


def test_timestamp_precision_marked_segment_level() -> None:
    result = _sentences("One. Two.")
    assert {s.timestamp_precision for s in result} == {
        TimestampPrecision.SEGMENT_LEVEL
    }
    assert TimestampPrecision.WORD_ALIGNED not in {
        s.timestamp_precision for s in result
    }


def test_no_fabricated_sentence_level_timestamp_fields() -> None:
    result = _sentences("One. Two.")
    field_names = {f.name for f in fields(Sentence)}
    forbidden = {
        "sentence_start_seconds", "sentence_end_seconds",
        "start_seconds", "end_seconds", "start_time", "end_time",
    }
    assert field_names.isdisjoint(forbidden)
    for s in result:
        assert s.exact_start_seconds is None
        assert s.exact_end_seconds is None


def test_language_preserved_verbatim() -> None:
    result = _sentences("One. Two.", language="en")
    assert {s.language for s in result} == {"en"}


def test_normalized_text_is_separate_and_does_not_affect_offsets() -> None:
    text = "  Hello   World!  Next."
    result = _sentences(text)
    for s in result:
        assert s.text == text[s.char_start:s.char_end]  # offsets -> original
        assert s.normalized_text == normalize_text(s.text).normalized
        assert s.normalized_text != s.text  # whitespace collapsed separately
    assert "".join(s.text for s in result) == text


def test_pure_engine_is_independently_testable() -> None:
    # find_sentence_spans needs no contract, no DB, and no I/O.
    text = "Independent.  Testable!"
    spans = find_sentence_spans(text)
    assert isinstance(spans, tuple)
    assert all(isinstance(s, SentenceSpan) for s in spans)
    _assert_tiling(text, spans)


# --- 5. Edge cases ------------------------------------------------------------


def test_empty_text_raises_explicit_error() -> None:
    with pytest.raises(EmptySourceTextError):
        find_sentence_spans("")


def test_whitespace_only_text_raises_explicit_error() -> None:
    for blank in ("   ", "\t\n  \r\n", "\u00a0\u00a0"):
        with pytest.raises(EmptySourceTextError):
            find_sentence_spans(blank)


def test_empty_text_contract_rejected() -> None:
    with pytest.raises(ValidationError):
        _input("")


def test_whitespace_only_contract_rejected() -> None:
    with pytest.raises(ValidationError):
        _input("  \n\t ")


def test_no_terminal_punctuation_is_one_sentence() -> None:
    text = "Just one sentence without a stop"
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text
    _assert_provenance(result, text)


def test_repeated_punctuation_is_one_boundary() -> None:
    text = "Wow!!! Really?? Ok..."
    result = _sentences(text)
    assert [s.text.strip() for s in result] == ["Wow!!!", "Really??", "Ok..."]
    _assert_provenance(result, text)


def test_standalone_ellipsis_token_creates_no_empty_sentence() -> None:
    text = "Wait ... (yes) ... really?"
    result = _sentences(text)
    assert all(s.text.strip() for s in result)
    assert "".join(s.text for s in result) == text


def test_newlines_preserved_in_offsets() -> None:
    text = "First line.\n\nSecond line."
    result = _sentences(text)
    assert result[1].text == "\n\nSecond line."
    assert text[result[1].char_start:result[1].char_end] == result[1].text
    _assert_provenance(result, text)


def test_quotation_marks_stay_with_their_sentence() -> None:
    text = 'He said "Hi." Then left.'
    result = _sentences(text)
    assert result[0].text == 'He said "Hi."'
    assert result[1].text == " Then left."
    _assert_provenance(result, text)


def test_curly_quotes_stay_with_their_sentence() -> None:
    text = "He said “Hi.” Then left."
    result = _sentences(text)
    assert result[0].text == "He said “Hi.”"
    _assert_provenance(result, text)


def test_parentheses_preserved() -> None:
    text = "(See Fig. 1) for details. Next."
    result = _sentences(text)
    assert [s.text.strip() for s in result] == [
        "(See Fig. 1) for details.", "Next.",
    ]
    _assert_provenance(result, text)


def test_abbreviation_at_sentence_boundary_undersplits_but_keeps_text() -> None:
    text = "This uses etc. Next sentence follows."
    result = _sentences(text)
    # etc. is not a boundary -> under-split. Documented trade-off: no silent
    # text loss, no dropped sentence.
    assert len(result) == 1
    assert result[0].text == text
    _assert_provenance(result, text)


def test_leading_and_trailing_whitespace_preserved() -> None:
    text = "  Hello.  "
    result = _sentences(text)
    assert len(result) == 1
    assert result[0].text == text
    assert result[0].char_start == 0 and result[0].char_end == len(text)
    _assert_provenance(result, text)


def test_consecutive_whitespace_between_sentences_preserved() -> None:
    text = "One.    \t  Two."
    result = _sentences(text)
    assert len(result) == 2
    assert result[1].text == "    \t  Two."
    _assert_provenance(result, text)


def test_no_period_followed_by_whitespace_but_abbreviation() -> None:
    text = "Prof. Rahman spoke. Dr. Khan replied."
    result = _sentences(text)
    assert [s.text.strip() for s in result] == [
        "Prof. Rahman spoke.", "Dr. Khan replied.",
    ]
    _assert_provenance(result, text)


# --- 6. Input contract --------------------------------------------------------


def test_contract_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SentenceSegmentationInput(
            source_text="Hello.", source_segment_ref=REF,
            start_seconds=START, end_seconds=END, language="en",
            unexpected="value",
        )


def test_contract_rejects_string_timestamps() -> None:
    with pytest.raises(ValidationError):
        SentenceSegmentationInput(
            source_text="Hello.", source_segment_ref=REF,
            start_seconds="12.5", end_seconds=END, language="en",
        )


def test_contract_rejects_bool_timestamp() -> None:
    with pytest.raises(ValidationError):
        SentenceSegmentationInput(
            source_text="Hello.", source_segment_ref=REF,
            start_seconds=True, end_seconds=END, language="en",
        )


def test_contract_rejects_non_finite_timestamp() -> None:
    with pytest.raises(ValidationError):
        SentenceSegmentationInput(
            source_text="Hello.", source_segment_ref=REF,
            start_seconds=float("nan"), end_seconds=END, language="en",
        )


def test_contract_rejects_negative_start() -> None:
    with pytest.raises(ValidationError):
        _input("Hello.", start_seconds=-1.0)


def test_contract_rejects_end_not_after_start() -> None:
    with pytest.raises(ValidationError):
        _input("Hello.", start_seconds=5.0, end_seconds=5.0)
    with pytest.raises(ValidationError):
        _input("Hello.", start_seconds=5.0, end_seconds=4.0)


def test_contract_rejects_blank_labels() -> None:
    with pytest.raises(ValidationError):
        _input("Hello.", source_segment_ref="  ")
    with pytest.raises(ValidationError):
        _input("Hello.", language="  ")


def test_contract_does_not_trim_source_text() -> None:
    text = "  Hello.  "
    assert _input(text).source_text == text


def test_segment_sentences_accepts_raw_mapping() -> None:
    result = segment_sentences({
        "source_text": "One. Two.",
        "source_segment_ref": "seg-map",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "language": "en",
    })
    assert len(result) == 2
    assert {s.source_segment_ref for s in result} == {"seg-map"}
    assert {s.language for s in result} == {"en"}


def test_segment_sentences_accepts_contract_instance() -> None:
    contract = _input("One. Two.")
    result = segment_sentences(contract)
    assert len(result) == 2
    assert isinstance(result, tuple)
    assert all(isinstance(s, Sentence) for s in result)


def test_segment_sentences_rejects_bad_mapping() -> None:
    with pytest.raises(ValidationError):
        segment_sentences({"source_text": "Only text"})


def test_timestamp_precision_enum_is_closed() -> None:
    assert {p.value for p in TimestampPrecision} == {
        "segment_level", "word_aligned"
    }


# --- 7. Corpus-wide tiling / no-loss regression ------------------------------

_CORPUS = (
    "Hello world.",
    "One. Two. Three.",
    "Dr. Rahman discussed the model.",
    "The value is 3.14. Then we round it.",
    "ModernBERT is useful. তবে কিছু ক্ষেত্রে সমস্যা হয়।",
    "আমি বাংলায় কথা বলি। তুমি কেমন আছ? ভালো!",
    "Wow!!! Really?? Ok...",
    "He said \"Hi.\" Then left.",
    "First line.\n\nSecond line.",
    "  Hello.  ",
    "Just one sentence without a stop",
    "Visit https://example.com/path?q=1. Then leave.",
    "মান 3.14 এবং U.S. মডেল। পরে দেখবো।",
    "Wait ... (yes) ... really?",
)


@pytest.mark.parametrize("text", _CORPUS)
def test_corpus_spans_tile_source_exactly(text: str) -> None:
    _assert_tiling(text, find_sentence_spans(text))
    result = _sentences(text)
    _assert_provenance(result, text)
