"""Sentence-level timestamp derivation from aligned words (Goal 12).

PURE layer. It consumes Goal 10 :class:`~app.transcripts.sentence_segmentation.Sentence`
objects UNCHANGED and a set of validated :class:`~app.transcripts.word_contract.AlignedWord`
rows, and upgrades a sentence to ``WORD_ALIGNED`` ONLY when every original token
in its span has a valid aligned word.

Design (approved decisions)
---------------------------
* Goal 10 ``Sentence`` is NOT modified: ``exact_start_seconds`` /
  ``exact_end_seconds`` remain ``None`` and are NEVER populated here. Exact
  sentence timing is expressed on the NEW :class:`AlignedSentence` in integer
  MILLISECONDS (``exact_start_ms`` / ``exact_end_ms``), matching the Goal 12
  word storage unit.
* A word belongs to a sentence when its ORIGINAL-text ``char_start`` falls in
  the sentence's half-open ``[char_start, char_end)`` span. Sentences tile
  the source text, so every word belongs to exactly one sentence.
* FULLY aligned sentence: every original token whose ``char_start`` is in the
  sentence span has a corresponding aligned word (matched by ordinal, 1:1).
  Then ``exact_start_ms = min(word.start_ms)``,
  ``exact_end_ms = max(word.end_ms)``, precision ``WORD_ALIGNED``.
* Otherwise (missing words, partial coverage, or zero words): precision stays
  ``SEGMENT_LEVEL`` and both exact values are ``None``. Bengali / unaligned
  segments therefore remain ``SEGMENT_LEVEL`` (decision #2).
* NO interpolation, NO proportional distribution, NO segment-window fallback
  presented as an exact boundary.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.transcripts.sentence_segmentation import Sentence, TimestampPrecision
from app.transcripts.word_contract import AlignedWord, Token


@dataclass(frozen=True)
class AlignedSentence:
    """A Goal 10 sentence with optional exact, word-derived timing.

    ``word_count`` is the number of aligned words assigned to this sentence
    (0 when none). ``precision`` is ``WORD_ALIGNED`` only for a fully covered
    sentence; otherwise ``SEGMENT_LEVEL`` with both exact values ``None``.
    """

    sentence: Sentence
    precision: TimestampPrecision
    exact_start_ms: int | None
    exact_end_ms: int | None
    word_count: int


def _in_span(offset: int, sentence: Sentence) -> bool:
    return sentence.char_start <= offset < sentence.char_end


def derive_aligned_sentences(
    sentences: Sequence[Sentence],
    words: Sequence[AlignedWord],
    tokens: Sequence[Token],
) -> tuple[AlignedSentence, ...]:
    """Derive :class:`AlignedSentence` values for ONE segment.

    ``sentences`` are the Goal 10 segmentation of that segment's ORIGINAL text;
    ``words`` are its validated aligned words (VIDEO-timeline ms); ``tokens``
    are the FULL original-text token stream of the segment (used to decide
    whether every token in a sentence span is covered).

    Pure and deterministic. Never mutates the input ``Sentence`` objects.
    """
    result: list[AlignedSentence] = []
    for sentence in sentences:
        sentence_words = [w for w in words if _in_span(w.char_start, sentence)]
        sentence_tokens = [t for t in tokens if _in_span(t.char_start, sentence)]

        word_ordinals = {w.ordinal for w in sentence_words}
        token_ordinals = {t.ordinal for t in sentence_tokens}

        fully_aligned = (
            bool(sentence_tokens)
            and word_ordinals == token_ordinals
            and len(sentence_words) == len(sentence_tokens)
        )

        if fully_aligned:
            exact_start = min(w.start_ms for w in sentence_words)
            exact_end = max(w.end_ms for w in sentence_words)
            result.append(
                AlignedSentence(
                    sentence=sentence,
                    precision=TimestampPrecision.WORD_ALIGNED,
                    exact_start_ms=exact_start,
                    exact_end_ms=exact_end,
                    word_count=len(sentence_words),
                )
            )
        else:
            result.append(
                AlignedSentence(
                    sentence=sentence,
                    precision=TimestampPrecision.SEGMENT_LEVEL,
                    exact_start_ms=None,
                    exact_end_ms=None,
                    word_count=len(sentence_words),
                )
            )
    return tuple(result)


__all__ = ["AlignedSentence", "derive_aligned_sentences"]
