"""Goal 12 — sentence_timestamps.py pure tests.

Covers the section-16 "Sentence timing tests": fully / partially / completely
unaligned sentences, exact sentence boundaries, word-to-sentence assignment,
mixed precision, zero word rows, and the guarantee that Goal 10 ``Sentence``
objects (including their reserved ``exact_start_seconds`` /
``exact_end_seconds``) are NEVER modified. All offline, dependency free.
"""
from __future__ import annotations

import uuid

from app.transcripts.sentence_segmentation import (
    Sentence,
    SentenceSegmentationInput,
    TimestampPrecision,
    segment_sentences,
)
from app.transcripts.sentence_timestamps import (
    AlignedSentence,
    derive_aligned_sentences,
)
from app.transcripts.word_contract import AlignedWord, Token, tokenize_original

SOURCE = "Hello world. Second sentence!"
SEG_ID = uuid.uuid4()


def _sentences() -> tuple[Sentence, ...]:
    return segment_sentences(
        SentenceSegmentationInput(
            source_text=SOURCE,
            source_segment_ref="S1",
            start_seconds=0.0,
            end_seconds=5.0,
            language="en",
        )
    )


def _tokens() -> tuple[Token, ...]:
    return tokenize_original(SOURCE)


def _word_for(token: Token, start_ms: int, end_ms: int) -> AlignedWord:
    return AlignedWord(
        segment_id=SEG_ID,
        ordinal=token.ordinal,
        text=token.text,
        char_start=token.char_start,
        char_end=token.char_end,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _all_words(tokens: tuple[Token, ...]) -> tuple[AlignedWord, ...]:
    # Deterministic, monotonically increasing video-timeline ms per ordinal.
    return tuple(
        _word_for(t, 100 + 100 * t.ordinal, 180 + 100 * t.ordinal)
        for t in tokens
    )


class TestDeriveAlignedSentences:
    def test_fully_aligned_sentence_is_word_aligned(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        words = _all_words(tokens)
        result = derive_aligned_sentences(sentences, words, tokens)
        assert len(result) == len(sentences) == 2
        assert all(isinstance(r, AlignedSentence) for r in result)
        assert all(r.precision == TimestampPrecision.WORD_ALIGNED for r in result)
        # Sentence 0: tokens ordinal 0,1 -> min start 100, max end 280.
        assert result[0].exact_start_ms == 100
        assert result[0].exact_end_ms == 280
        # Sentence 1: tokens ordinal 2,3 -> min start 300, max end 480.
        assert result[1].exact_start_ms == 300
        assert result[1].exact_end_ms == 480

    def test_word_count_matches_assigned_words(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        result = derive_aligned_sentences(sentences, _all_words(tokens), tokens)
        assert result[0].word_count == 2
        assert result[1].word_count == 2

    def test_partially_aligned_sentence_stays_segment_level(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        words = tuple(w for w in _all_words(tokens) if w.ordinal != 1)
        result = derive_aligned_sentences(sentences, words, tokens)
        # Sentence 0 missing one of its two tokens -> SEGMENT_LEVEL, no exact.
        assert result[0].precision == TimestampPrecision.SEGMENT_LEVEL
        assert result[0].exact_start_ms is None
        assert result[0].exact_end_ms is None
        assert result[0].word_count == 1
        # Sentence 1 is still fully aligned.
        assert result[1].precision == TimestampPrecision.WORD_ALIGNED

    def test_completely_unaligned_sentence_stays_segment_level(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        result = derive_aligned_sentences(sentences, (), tokens)
        assert result[0].precision == TimestampPrecision.SEGMENT_LEVEL
        assert result[0].exact_start_ms is None
        assert result[0].exact_end_ms is None
        assert result[0].word_count == 0

    def test_mixed_precision(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        # Only sentence 1's words survive -> sentence 0 SEGMENT_LEVEL,
        # sentence 1 WORD_ALIGNED.
        words = tuple(w for w in _all_words(tokens) if w.ordinal >= 2)
        result = derive_aligned_sentences(sentences, words, tokens)
        assert result[0].precision == TimestampPrecision.SEGMENT_LEVEL
        assert result[1].precision == TimestampPrecision.WORD_ALIGNED
        assert {r.precision for r in result} == {
            TimestampPrecision.SEGMENT_LEVEL, TimestampPrecision.WORD_ALIGNED
        }

    def test_exact_boundaries_use_min_start_and_max_end(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        # Deliberately non-monotonic / irregular ms to prove min/max semantics.
        mapping = {
            tokens[0].ordinal: (500, 900),
            tokens[1].ordinal: (100, 950),  # min start
            tokens[2].ordinal: (400, 7000),  # max end
            tokens[3].ordinal: (600, 650),
        }
        words = tuple(_word_for(t, *mapping[t.ordinal]) for t in tokens)
        result = derive_aligned_sentences(sentences, words, tokens)
        assert result[0].exact_start_ms == 100   # min of 500, 100
        assert result[0].exact_end_ms == 950    # max of 900, 950
        assert result[1].exact_start_ms == 400  # min of 400, 600
        assert result[1].exact_end_ms == 7000   # max of 7000, 650

    def test_word_to_sentence_assignment_by_char_start(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        # A word at char_start 6 ("world.") belongs to sentence 0 [0,12),
        # NOT sentence 1 [12,29). Only that one word is aligned.
        only = _word_for(tokens[1], 100, 200)
        result = derive_aligned_sentences(sentences, (only,), tokens)
        assert result[0].word_count == 1
        assert result[1].word_count == 0

    def test_zero_word_rows_fully_compatible_with_goal10(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        result = derive_aligned_sentences(sentences, (), tokens)
        assert len(result) == len(sentences)
        for r in result:
            assert r.precision == TimestampPrecision.SEGMENT_LEVEL
            assert r.exact_start_ms is None and r.exact_end_ms is None
            assert r.word_count == 0
            # Goal 10 sentences are always SEGMENT_LEVEL.
            assert r.sentence.timestamp_precision == TimestampPrecision.SEGMENT_LEVEL

    def test_goal10_sentence_exact_fields_unchanged(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        before = [
            (s.exact_start_seconds, s.exact_end_seconds, s.timestamp_precision)
            for s in sentences
        ]
        derive_aligned_sentences(sentences, _all_words(tokens), tokens)
        after = [
            (s.exact_start_seconds, s.exact_end_seconds, s.timestamp_precision)
            for s in sentences
        ]
        # Reserved seconds fields are NEVER populated; precision is untouched.
        assert after == before
        assert all(s.exact_start_seconds is None for s in sentences)
        assert all(s.exact_end_seconds is None for s in sentences)
        assert all(
            s.timestamp_precision == TimestampPrecision.SEGMENT_LEVEL
            for s in sentences
        )

    def test_input_sentences_not_mutated_and_identity_preserved(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        result = derive_aligned_sentences(sentences, _all_words(tokens), tokens)
        # AlignedSentence wraps the SAME Sentence instance (no copy/mutation).
        assert result[0].sentence is sentences[0]
        assert result[1].sentence is sentences[1]

    def test_empty_sentence_list(self) -> None:
        assert derive_aligned_sentences((), (), ()) == ()

    def test_full_coverage_requires_exact_ordinal_set(self) -> None:
        sentences, tokens = _sentences(), _tokens()
        # Same COUNT of words but WRONG ordinal -> not fully aligned. Build a
        # word with a foreign ordinal landing in sentence 0's span.
        words = (
            _word_for(tokens[0], 100, 180),
            AlignedWord(
                segment_id=SEG_ID, ordinal=99, text="world.",
                char_start=6, char_end=12, start_ms=200, end_ms=280,
            ),
        )
        result = derive_aligned_sentences(sentences, words, tokens)
        # sentence 0 has 2 words in span but ordinals {0, 99} != {0, 1}.
        assert result[0].precision == TimestampPrecision.SEGMENT_LEVEL
        assert result[0].exact_start_ms is None
