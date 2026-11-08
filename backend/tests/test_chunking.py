"""Goal 13 — deterministic, sentence-aware chunking tests.

Pure, deterministic fixtures only: no DB, no network, no ML, no credentials.
Covers configuration validation, boundary/overlap policy, timestamp precision
(segment-level vs word-aligned vs mixed), provenance, evidence/context
separation, invalid-input rejection, and regression compatibility with the
Goal 09/10/12 contracts.
"""
from __future__ import annotations

import uuid
from dataclasses import fields

import pytest
from pydantic import ValidationError

from app.transcripts.chunking import (
    DIAG_COARSE_PRECISION,
    DIAG_MIN_DURATION_UNMET,
    DIAG_OVERLAP_COARSE,
    DIAG_OVERLAP_TRIMMED,
    DIAG_SINGLE_EXCEEDS_MAX_DURATION,
    DIAG_SINGLE_EXCEEDS_MAX_WORDS,
    DIAG_TARGET_MIN_WORDS_UNMET,
    REASON_END_OF_INPUT,
    REASON_SINGLE_SENTENCE,
    REASON_TARGET_REACHED,
    ROLE_CONTEXT,
    ROLE_EVIDENCE,
    Chunk,
    ChunkParentContext,
    ChunkSentence,
    ChunkingConfig,
    InvalidChunkInputError,
    build_production_chunks,
    chunk_sentences,
    validate_aligned_sentences,
)
from app.transcripts.sentence_segmentation import (
    Sentence,
    SentenceSegmentationInput,
    TimestampPrecision,
    segment_sentences,
)
from app.transcripts.sentence_timestamps import AlignedSentence, derive_aligned_sentences
from app.transcripts.word_contract import AlignedWord, tokenize_original

# --- Fixture builders ---------------------------------------------------------


def _sentence(
    text: str,
    *,
    ordinal: int = 0,
    char_start: int = 0,
    seg_ref: str = "seg-1",
    seg_start: float = 0.0,
    seg_end: float = 10.0,
    lang: str = "en",
) -> Sentence:
    """A verbatim Goal 10 Sentence (segment-level precision by construction)."""
    return Sentence(
        text=text,
        ordinal=ordinal,
        char_start=char_start,
        char_end=char_start + len(text),
        source_segment_ref=seg_ref,
        source_segment_start_seconds=seg_start,
        source_segment_end_seconds=seg_end,
        language=lang,
        timestamp_precision=TimestampPrecision.SEGMENT_LEVEL,
        normalized_text=text,
        exact_start_seconds=None,
        exact_end_seconds=None,
    )


def _coarse(text: str, **kw) -> AlignedSentence:
    return AlignedSentence(
        sentence=_sentence(text, **kw),
        precision=TimestampPrecision.SEGMENT_LEVEL,
        exact_start_ms=None,
        exact_end_ms=None,
        word_count=0,  # aligned-word count is irrelevant to the chunker
    )


def _exact(
    i: int,
    words: int,
    start_s: float,
    end_s: float,
    *,
    seg_ref: str | None = None,
) -> AlignedSentence:
    """A fully WORD_ALIGNED sentence with exact ms timing (monotonic seg_start)."""
    text = " ".join(["w"] * words) + "."
    return AlignedSentence(
        sentence=_sentence(
            text,
            ordinal=i,
            seg_ref=seg_ref or f"seg-{i}",
            seg_start=start_s,
            seg_end=end_s,
        ),
        precision=TimestampPrecision.WORD_ALIGNED,
        exact_start_ms=int(start_s * 1000),
        exact_end_ms=int(end_s * 1000),
        word_count=words,
    )


def _evidence_indices(chunks) -> list[int]:
    return [r.index for c in chunks for r in c.evidence_sentences]


# --- 1. Configuration ---------------------------------------------------------


def test_config_defaults_match_approved_spec():
    cfg = ChunkingConfig()
    assert cfg.target_duration_seconds == 30.0
    assert cfg.min_duration_seconds == 10.0
    assert cfg.max_duration_seconds == 60.0
    assert cfg.overlap_seconds == 5.0
    assert cfg.target_min_words == 80
    assert cfg.target_max_words == 150


def test_config_is_frozen_and_forbids_extra():
    cfg = ChunkingConfig()
    with pytest.raises(ValidationError):
        cfg.target_duration_seconds = 99.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ChunkingConfig(unknown_field=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_duration_seconds": 0},
        {"target_duration_seconds": -1},
        {"overlap_seconds": 0},
        {"overlap_seconds": float("nan")},
        {"overlap_seconds": float("inf")},
        {"target_duration_seconds": True},  # bool rejected by _finite_float
        {"target_duration_seconds": "30"},  # str rejected (no coercion)
        {"min_duration_seconds": 40, "target_duration_seconds": 30},  # min > target
        {"target_duration_seconds": 70, "max_duration_seconds": 60},  # target > max
        {"overlap_seconds": 60, "max_duration_seconds": 60},  # overlap >= max
        {"target_min_words": 0},
        {"target_min_words": 80.0},  # strict int
        {"target_min_words": True},
        {"target_min_words": 200, "target_max_words": 150},  # min > max words
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        ChunkingConfig(**overrides)


def test_config_override_is_honoured():
    cfg = ChunkingConfig(target_min_words=2, target_max_words=4)
    items = [_coarse("One two.", ordinal=i, seg_start=float(i)) for i in range(6)]
    chunks = chunk_sentences(items, config=cfg)
    # With a 2-word floor, several small chunks form instead of one big one.
    assert len(chunks) >= 2
    assert _evidence_indices(chunks) == list(range(6))


# --- 2. Empty / single / ordering / text reconstruction ----------------------


def test_empty_input_returns_empty_tuple():
    assert chunk_sentences(()) == ()
    assert chunk_sentences([]) == ()


def test_single_coarse_sentence():
    items = [_coarse("A single sentence.")]
    chunks = chunk_sentences(items)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.index == 0
    assert c.text == "A single sentence."
    assert c.sentences[0].role == ROLE_EVIDENCE
    assert c.evidence_start == 0
    assert not c.has_context
    assert c.overlap_sentence_indices == ()
    assert c.timestamp_precision == TimestampPrecision.SEGMENT_LEVEL
    assert DIAG_COARSE_PRECISION in c.diagnostics
    assert c.boundary_reason == REASON_END_OF_INPUT


def test_sentence_order_and_verbatim_text_preserved():
    texts = ["First one. ", "Second one. ", "Third one."]
    items = [
        _coarse(t, ordinal=i, char_start=sum(len(x) for x in texts[:i]), seg_start=0.0)
        for i, t in enumerate(texts)
    ]
    (chunk,) = chunk_sentences(items)
    assert [r.text for r in chunk.sentences] == texts
    assert [r.index for r in chunk.sentences] == [0, 1, 2]
    assert chunk.text == "".join(texts)  # verbatim, no loss, no rewrite
    assert all(r.role == ROLE_EVIDENCE for r in chunk.sentences)


def test_no_character_loss_across_chunks():
    items = [_exact(i, 20, i * 10, i * 10 + 10) for i in range(9)]
    chunks = chunk_sentences(items)
    reconstructed = "".join(r.text for c in chunks for r in c.evidence_sentences)
    assert reconstructed == "".join(items[k].sentence.text for k in range(9))


def test_no_fabricated_identity_or_speaker_fields():
    """Real parent IDs ARE propagated; no fabricated sentence UUID / speaker."""
    chunk_names = {f.name for f in fields(Chunk)}
    ref_names = {f.name for f in fields(ChunkSentence)}
    # Real parent provenance is now a first-class part of the Chunk contract.
    assert "video_id" in chunk_names
    assert "transcript_asset_id" in chunk_names
    assert "content_right_id" in chunk_names
    # No invented persisted identity, no speaker, no sentence UUID field.
    assert "speaker" not in ref_names
    assert "chunk_id" not in chunk_names
    assert "sentence_uuid" not in ref_names
    parent = _parent()
    (chunk,) = build_production_chunks([_coarse("A. ")], parent=parent)
    # Propagated values are the REAL ones, not fabricated.
    assert chunk.video_id == parent.video_id
    assert chunk.transcript_asset_id == parent.transcript_asset_id
    # sentence_ref is a deterministic derived property, not a stored UUID.
    assert chunk.sentences[0].sentence_ref == "seg-1:0:0:3"


# --- 3. Coarse precision: no proportional / fabricated timing ----------------


def test_coarse_sentences_never_gain_exact_timing():
    items = [
        _coarse("Alpha beta.", ordinal=i, seg_start=float(i) * 5, seg_end=float(i) * 5 + 5)
        for i in range(4)
    ]
    (chunk,) = chunk_sentences(items)
    assert chunk.timestamp_precision == TimestampPrecision.SEGMENT_LEVEL
    assert all(r.exact_start_ms is None and r.exact_end_ms is None for r in chunk.sentences)
    assert all(r.timestamp_precision == TimestampPrecision.SEGMENT_LEVEL for r in chunk.sentences)
    assert DIAG_COARSE_PRECISION in chunk.diagnostics
    # Duration targets are best-effort; no false sentence-level claim.
    assert chunk.start_ms == seconds(0.0)
    assert chunk.end_ms == seconds(20.0)  # max segment end, not a split


def seconds(value: float) -> int:
    from app.transcripts.word_contract import seconds_to_ms

    return seconds_to_ms(value)


def test_one_segment_multiple_sentences_share_window_no_proportional_split():
    # Four sentences all inherit ONE 40s parent window. The chunk extent must be
    # exactly that window — never divided proportionally among the sentences.
    items = [
        _coarse(f"Sentence {i}. ", ordinal=i, char_start=i * 12, seg_ref="seg-A", seg_start=100.0, seg_end=140.0)
        for i in range(4)
    ]
    (chunk,) = chunk_sentences(items)
    assert chunk.start_ms == seconds(100.0)
    assert chunk.end_ms == seconds(140.0)
    assert chunk.duration_ms == seconds(40.0)
    assert chunk.source_segment_refs == ("seg-A",)  # de-duplicated
    assert DIAG_COARSE_PRECISION in chunk.diagnostics


def test_multiple_segments_ordered_provenance():
    items = [
        _coarse("A. ", ordinal=0, seg_ref="s1", seg_start=0.0, seg_end=5.0),
        _coarse("B. ", ordinal=1, seg_ref="s1", seg_start=0.0, seg_end=5.0),  # same segment
        _coarse("C. ", ordinal=2, seg_ref="s2", seg_start=5.0, seg_end=9.0),
        _coarse("D. ", ordinal=3, seg_ref="s3", seg_start=9.0, seg_end=12.0),
    ]
    (chunk,) = chunk_sentences(items)
    assert chunk.source_segment_refs == ("s1", "s2", "s3")


def test_duplicate_source_refs_are_deduplicated_in_provenance():
    items = [
        _coarse("x. ", ordinal=i, seg_ref="dup", seg_start=0.0, seg_end=5.0)
        for i in range(5)
    ]
    (chunk,) = chunk_sentences(items)
    assert chunk.source_segment_refs == ("dup",)
    assert len(chunk.sentences) == 5  # sentences themselves are NOT deduplicated


def test_coarse_short_transcript_uses_wordcount_not_false_duration_enforcement():
    items = [_coarse("word. ", ordinal=i, seg_start=0.0, seg_end=2.0) for i in range(3)]
    (chunk,) = chunk_sentences(items)
    assert chunk.timestamp_precision == TimestampPrecision.SEGMENT_LEVEL
    assert DIAG_COARSE_PRECISION in chunk.diagnostics
    # min-duration cannot be claimed unmet under coarse precision (unmeasurable).
    assert DIAG_MIN_DURATION_UNMET not in chunk.diagnostics


# --- 4. Single oversized sentences (kept intact, never split) ----------------


def test_single_sentence_exceeding_max_words_kept_intact():
    text = " ".join(["word"] * 200) + "."
    items = [_coarse(text), _coarse("Short tail.")]
    chunks = chunk_sentences(items)
    assert any(c.boundary_reason == REASON_SINGLE_SENTENCE for c in chunks)
    big = [c for c in chunks if c.boundary_reason == REASON_SINGLE_SENTENCE][0]
    assert big.text == text  # not truncated
    assert DIAG_SINGLE_EXCEEDS_MAX_WORDS in big.diagnostics


def test_exact_single_sentence_exceeding_max_duration_kept_intact():
    # One 90s exact sentence + a tail: the 90s sentence must stay whole.
    items = [_exact(0, 5, 0.0, 90.0), _exact(1, 5, 90.0, 95.0)]
    chunks = chunk_sentences(items)
    first = chunks[0]
    assert first.boundary_reason == REASON_SINGLE_SENTENCE
    assert first.duration_ms == seconds(90.0)
    assert DIAG_SINGLE_EXCEEDS_MAX_DURATION in first.diagnostics


def test_oversized_first_chunk_reports_no_overlap_trimmed_diagnostic():
    # Regression (final Goal 13 review): an oversized single-sentence chunk is
    # the FIRST chunk, so there is NO overlap context to trim. Its > 60s
    # duration comes entirely from the evidence sentence (kept intact by
    # design). `overlap_trimmed_for_max_duration` must NOT be reported when no
    # context sentence was actually removed. The positive case (context IS
    # trimmed) is covered by test_overlap_never_inflates_chunk_past_max_duration.
    items = [_exact(0, 5, 0.0, 90.0), _exact(1, 5, 90.0, 95.0)]
    chunks = chunk_sentences(items)
    first = chunks[0]
    assert first.boundary_reason == REASON_SINGLE_SENTENCE
    assert first.duration_ms == seconds(90.0)
    assert first.has_context is False
    assert first.overlap_sentence_indices == ()
    assert DIAG_SINGLE_EXCEEDS_MAX_DURATION in first.diagnostics
    assert DIAG_OVERLAP_TRIMMED not in first.diagnostics


# --- 5. Exact precision: 30s target, 60s cap, overlap -----------------------


def test_exact_precision_duration_target_and_wordaligned_label():
    items = [_exact(i, 15, i * 10, i * 10 + 10) for i in range(12)]
    chunks = chunk_sentences(items)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.timestamp_precision == TimestampPrecision.WORD_ALIGNED
        # Evidence-only extent never exceeds the 60s maximum.
        ev = c.evidence_sentences
        assert ev
        assert c.boundary_reason in (REASON_TARGET_REACHED, REASON_END_OF_INPUT)
    assert _evidence_indices(chunks) == list(range(12))  # disjoint, all covered once


def test_exact_chunks_never_exceed_60s_evidence():
    items = [_exact(i, 5, i * 10, i * 10 + 10) for i in range(12)]
    chunks = chunk_sentences(items)
    for c in chunks:
        ev_start = min(r.exact_start_ms for r in c.evidence_sentences)
        ev_end = max(r.exact_end_ms for r in c.evidence_sentences)
        assert ev_end - ev_start <= seconds(60.0)


def test_overlap_reuses_complete_sentences_with_original_ids_and_timestamps():
    items = [_exact(i, 15, i * 10, i * 10 + 10) for i in range(9)]
    chunks = chunk_sentences(items)
    assert len(chunks) >= 2
    for c in chunks[1:]:
        assert c.has_context
        assert c.overlap_sentence_indices == tuple(r.index for r in c.context_sentences)
        for r in c.context_sentences:
            assert r.role == ROLE_CONTEXT
            src = items[r.index]
            # Original IDs, timestamps, and verbatim text preserved.
            assert r.text == src.sentence.text
            assert r.exact_start_ms == src.exact_start_ms
            assert r.exact_end_ms == src.exact_end_ms
            assert r.source_segment_ref == src.sentence.source_segment_ref
        # Context is the trailing suffix of the PREVIOUS chunk's evidence.
        prev_ev = [r.index for r in chunks[c.index - 1].evidence_sentences]
        assert tuple(r.index for r in c.context_sentences) == tuple(prev_ev[-len(c.context_sentences):])


def test_overlap_is_exactly_five_seconds_when_available():
    # 5s exact sentences -> a one-sentence suffix is exactly the 5s overlap.
    items = [_exact(i, 10, i * 5, i * 5 + 5) for i in range(18)]
    chunks = chunk_sentences(items)
    c1 = chunks[1]
    assert len(c1.context_sentences) == 1
    ctx = c1.context_sentences[0]
    assert ctx.exact_end_ms - ctx.exact_start_ms == seconds(5.0)


def test_overlap_chooses_closest_valid_complete_sentence_suffix():
    # 3s sentences: 1 sentence = 3s (|3-5|=2), 2 = 6s (|6-5|=1) -> 2 chosen.
    items = [_exact(i, 6, i * 3, i * 3 + 3) for i in range(30)]
    chunks = chunk_sentences(items)
    assert len(chunks) >= 2
    c1 = chunks[1]
    assert len(c1.context_sentences) == 2
    span = max(r.exact_end_ms for r in c1.context_sentences) - min(
        r.exact_start_ms for r in c1.context_sentences
    )
    assert span == seconds(6.0)


def test_overlap_never_inflates_chunk_past_max_duration():
    # 55s exact sentences: each chunk is one sentence; a 55s overlap context
    # would push the next chunk past 60s, so it must be trimmed.
    items = [_exact(i, 8, i * 55, i * 55 + 55) for i in range(4)]
    chunks = chunk_sentences(items)
    assert len(chunks) >= 2
    for c in chunks[1:]:
        assert not c.has_context  # overlap trimmed away
        assert DIAG_OVERLAP_TRIMMED in c.diagnostics


def test_context_is_not_reported_as_evidence_provenance():
    items = [_exact(i, 15, i * 10, i * 10 + 10) for i in range(9)]
    chunks = chunk_sentences(items)
    c1 = chunks[1]
    ctx_refs = {r.source_segment_ref for r in c1.context_sentences}
    assert ctx_refs.isdisjoint(set(c1.source_segment_refs)) or not ctx_refs
    # source_segment_refs come from evidence only.
    ev_refs = {r.source_segment_ref for r in c1.evidence_sentences}
    assert set(c1.source_segment_refs) == ev_refs
    assert c1.evidence_start == len(c1.context_sentences)


# --- 6. Mixed precision -------------------------------------------------------


def test_mixed_precision_chunk_is_downgraded_to_segment_level():
    items = [
        _exact(0, 10, 0.0, 5.0),
        _exact(1, 10, 5.0, 10.0),
        _coarse("coarse tail. ", ordinal=2, seg_ref="seg-c", seg_start=10.0, seg_end=18.0),
    ]
    (chunk,) = chunk_sentences(items)
    assert chunk.timestamp_precision == TimestampPrecision.SEGMENT_LEVEL
    assert DIAG_COARSE_PRECISION in chunk.diagnostics
    # Per-sentence precision is preserved verbatim.
    assert [r.timestamp_precision for r in chunk.sentences] == [
        TimestampPrecision.WORD_ALIGNED,
        TimestampPrecision.WORD_ALIGNED,
        TimestampPrecision.SEGMENT_LEVEL,
    ]


# --- 7. Unicode / multilingual / Goal 10 regression --------------------------


def test_bangla_and_unicode_text_preserved_verbatim():
    items = [
        _coarse("আমি ভালো আছি। ", ordinal=0, lang="bn", seg_start=0.0, seg_end=5.0),
        _coarse("তুমি কেমন? ", ordinal=1, lang="bn", seg_start=5.0, seg_end=9.0),
        _coarse("ঠিক আছে!", ordinal=2, lang="bn", seg_start=9.0, seg_end=12.0),
    ]
    (chunk,) = chunk_sentences(items)
    assert chunk.text == "আমি ভালো আছি। তুমি কেমন? ঠিক আছে!"
    assert all(r.language == "bn" for r in chunk.sentences)


def test_goal10_segmentation_feeds_chunker_without_resplitting():
    source = (
        "Dr. Rahman joined on 3.14.2019. "
        "Visit https://example.com/a.b today. "
        "The U.S. team shipped v1.2.3. "
        "Really?! Yes!!!"
    )
    sents = segment_sentences(
        SentenceSegmentationInput(
            source_text=source,
            source_segment_ref="seg-1",
            start_seconds=0.0,
            end_seconds=30.0,
            language="en",
        )
    )
    aligned = derive_aligned_sentences(sents, (), ())
    chunks = chunk_sentences(aligned)
    # Chunker groups the Goal 10 sentences verbatim; nothing is re-split/rewritten.
    assert "".join(r.text for c in chunks for r in c.evidence_sentences) == source
    assert all(c.timestamp_precision == TimestampPrecision.SEGMENT_LEVEL for c in chunks)


def test_goal10_goal12_wordaligned_regression_integration():
    source = "Hello world. Second sentence. Third one."
    sents = segment_sentences(
        SentenceSegmentationInput(
            source_text=source,
            source_segment_ref="seg-1",
            start_seconds=0.0,
            end_seconds=12.0,
            language="en",
        )
    )
    tokens = tokenize_original(source)
    seg_id = uuid.uuid4()
    words = tuple(
        AlignedWord(
            segment_id=seg_id,
            ordinal=t.ordinal,
            text=t.text,
            char_start=t.char_start,
            char_end=t.char_end,
            start_ms=t.ordinal * 1000,
            end_ms=t.ordinal * 1000 + 900,
        )
        for t in tokens
    )
    aligned = derive_aligned_sentences(sents, words, tokens)
    assert all(a.precision == TimestampPrecision.WORD_ALIGNED for a in aligned)
    chunks = chunk_sentences(aligned)
    assert len(chunks) >= 1
    assert all(c.timestamp_precision == TimestampPrecision.WORD_ALIGNED for c in chunks)
    # Exact ms are propagated, never fabricated from char offsets.
    assert min(r.exact_start_ms for c in chunks for r in c.evidence_sentences) == 0


def test_repeated_sentences_are_preserved_deterministically():
    items = [_coarse("Same. ", ordinal=i, seg_start=float(i)) for i in range(5)]
    chunks = chunk_sentences(items)
    assert [r.text for c in chunks for r in c.evidence_sentences] == ["Same. "] * 5


def test_determinism():
    items = [_exact(i, 12, i * 8, i * 8 + 8) for i in range(10)]
    assert chunk_sentences(items) == chunk_sentences(items)
    assert chunk_sentences(items) == chunk_sentences(list(items))


# --- 8. Invalid input: rejected, never silently repaired --------------------


def test_non_monotonic_segment_starts_rejected():
    items = [
        _coarse("A. ", seg_ref="s1", seg_start=10.0, seg_end=15.0),
        _coarse("B. ", seg_ref="s2", seg_start=5.0, seg_end=8.0),  # goes backward
    ]
    with pytest.raises(InvalidChunkInputError):
        chunk_sentences(items)


@pytest.mark.parametrize(
    "seg_start,seg_end",
    [(-1.0, 5.0), (5.0, 5.0), (6.0, 5.0), (float("nan"), 5.0), (1.0, float("inf"))],
)
def test_invalid_parent_window_rejected(seg_start, seg_end):
    items = [_coarse("A. ", seg_start=seg_start, seg_end=seg_end)]
    with pytest.raises(InvalidChunkInputError):
        chunk_sentences(items)


def test_wordaligned_missing_or_bad_exact_times_rejected():
    s = _sentence("x. ")
    bad = [
        AlignedSentence(sentence=s, precision=TimestampPrecision.WORD_ALIGNED,
                        exact_start_ms=None, exact_end_ms=1000, word_count=1),
        AlignedSentence(sentence=s, precision=TimestampPrecision.WORD_ALIGNED,
                        exact_start_ms=2000, exact_end_ms=1000, word_count=1),
        AlignedSentence(sentence=s, precision=TimestampPrecision.WORD_ALIGNED,
                        exact_start_ms=True, exact_end_ms=1000, word_count=1),
        AlignedSentence(sentence=s, precision=TimestampPrecision.WORD_ALIGNED,
                        exact_start_ms=-1, exact_end_ms=1000, word_count=1),
    ]
    for item in bad:
        with pytest.raises(InvalidChunkInputError):
            chunk_sentences([item])


def test_segment_level_with_exact_times_rejected():
    s = _sentence("x. ")
    item = AlignedSentence(
        sentence=s,
        precision=TimestampPrecision.SEGMENT_LEVEL,
        exact_start_ms=0,
        exact_end_ms=1000,
        word_count=1,
    )
    with pytest.raises(InvalidChunkInputError):
        chunk_sentences([item])


def test_wrong_element_type_rejected():
    with pytest.raises(InvalidChunkInputError):
        chunk_sentences(["not an AlignedSentence"])  # type: ignore[list-item]


def test_validate_aligned_sentences_accepts_valid_input():
    items = [_coarse("A. "), _exact(1, 5, 5.0, 9.0)]
    assert validate_aligned_sentences(items) is None


def test_min_duration_unmet_diagnostic_for_short_exact_transcript():
    items = [_exact(0, 3, 0.0, 4.0)]  # 4s < 10s min
    (chunk,) = chunk_sentences(items)
    assert chunk.timestamp_precision == TimestampPrecision.WORD_ALIGNED
    assert DIAG_MIN_DURATION_UNMET in chunk.diagnostics
    assert DIAG_TARGET_MIN_WORDS_UNMET in chunk.diagnostics


# --- 9. Provenance: real parent context + stable sentence references ---------


class _ParentStub:
    """Stand-in exposing the real Goal 09/12 parent field names (no DB)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _parent(*, with_right: bool = False) -> ChunkParentContext:
    """A real parent context built from genuine uuid.UUID values."""
    kw = {"video_id": uuid.uuid4(), "transcript_asset_id": uuid.uuid4()}
    if with_right:
        kw["content_right_id"] = uuid.uuid4()
    return ChunkParentContext(**kw)


def test_real_parent_ids_propagated_into_every_chunk():
    vid, aid, rid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    parent = ChunkParentContext(
        video_id=vid, transcript_asset_id=aid, content_right_id=rid
    )
    items = [_exact(i, 15, i * 10, i * 10 + 10) for i in range(9)]
    chunks = build_production_chunks(items, parent=parent)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.video_id == vid
        assert chunk.transcript_asset_id == aid
        assert chunk.content_right_id == rid
        assert isinstance(chunk.video_id, uuid.UUID)
        assert isinstance(chunk.transcript_asset_id, uuid.UUID)


def test_content_right_id_is_optional_and_none_when_absent():
    parent = ChunkParentContext(
        video_id=uuid.uuid4(), transcript_asset_id=uuid.uuid4()
    )
    assert parent.content_right_id is None
    (chunk,) = build_production_chunks([_coarse("Only one.")], parent=parent)
    assert chunk.video_id == parent.video_id
    assert chunk.transcript_asset_id == parent.transcript_asset_id
    assert chunk.content_right_id is None


def test_parent_context_built_from_real_goal09_and_goal12_parent_context():
    vid, aid, rid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    # Same real field names/values as TranscriptAlignmentRun and IngestionResult.
    run = _ParentStub(video_id=vid, transcript_asset_id=aid, content_right_id=rid)
    ingestion = _ParentStub(video_id=vid, asset_id=aid, content_right_id=rid)
    from_run = ChunkParentContext.from_alignment_run(run)
    from_ing = ChunkParentContext.from_ingestion_result(ingestion)
    for ctx in (from_run, from_ing):
        assert ctx.video_id == vid
        assert ctx.transcript_asset_id == aid
        assert ctx.content_right_id == rid
    (chunk,) = build_production_chunks([_coarse("x. ")], parent=from_run)
    assert chunk.video_id == vid
    assert chunk.transcript_asset_id == aid
    assert chunk.content_right_id == rid


def test_content_right_id_propagated_only_when_present_in_parent():
    vid, aid = uuid.uuid4(), uuid.uuid4()
    # A parent object without a real content_right_id -> no new rights lookup.
    run = _ParentStub(video_id=vid, transcript_asset_id=aid)
    ctx = ChunkParentContext.from_alignment_run(run)
    assert ctx.content_right_id is None
    (chunk,) = build_production_chunks([_coarse("x. ")], parent=ctx)
    assert chunk.content_right_id is None


def test_production_path_requires_valid_parent_context():
    items = [_coarse("A. ")]
    # Missing parent -> TypeError (required keyword-only argument, no default).
    with pytest.raises(TypeError):
        build_production_chunks(items)  # type: ignore[call-arg]
    # None / wrong type -> InvalidChunkInputError.
    with pytest.raises(InvalidChunkInputError):
        build_production_chunks(items, parent=None)  # type: ignore[arg-type]
    with pytest.raises(InvalidChunkInputError):
        build_production_chunks(items, parent="not-a-context")  # type: ignore[arg-type]
    # Pure primitive also rejects a non-None, wrong-type parent (fail closed).
    with pytest.raises(InvalidChunkInputError):
        chunk_sentences(items, parent=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"video_id": "not-a-uuid", "transcript_asset_id": uuid.uuid4()},
        {"video_id": uuid.uuid4(), "transcript_asset_id": 123},
        {
            "video_id": uuid.uuid4(),
            "transcript_asset_id": uuid.uuid4(),
            "content_right_id": "bad",
        },
    ],
)
def test_parent_context_rejects_non_uuid_or_fabricated_ids(kwargs):
    with pytest.raises(ValidationError):
        ChunkParentContext(**kwargs)


def test_parent_context_is_frozen_and_forbids_extra():
    ctx = _parent()
    with pytest.raises(ValidationError):
        ctx.video_id = uuid.uuid4()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ChunkParentContext(  # type: ignore[call-arg]
            video_id=uuid.uuid4(), transcript_asset_id=uuid.uuid4(), bogus=1
        )


def test_sentence_ref_is_deterministic_and_derived_from_immutable_provenance():
    items = [
        _coarse("First. ", ordinal=0, char_start=0, seg_ref="seg-A",
                seg_start=0.0, seg_end=5.0),
        _coarse("Second. ", ordinal=1, char_start=7, seg_ref="seg-A",
                seg_start=0.0, seg_end=5.0),
    ]
    (chunk,) = build_production_chunks(items, parent=_parent())
    refs = [r.sentence_ref for r in chunk.sentences]
    assert refs == ["seg-A:0:0:7", "seg-A:1:7:15"]
    # Deterministic across calls, parents, and the pure primitive.
    (again,) = build_production_chunks(items, parent=_parent())
    assert [r.sentence_ref for r in again.sentences] == refs
    (pure,) = chunk_sentences(items)
    assert [r.sentence_ref for r in pure.sentences] == refs


def test_overlap_context_and_evidence_share_identical_sentence_refs():
    items = [_exact(i, 15, i * 10, i * 10 + 10) for i in range(9)]
    chunks = build_production_chunks(items, parent=_parent())
    assert len(chunks) >= 2
    for chunk in chunks[1:]:
        for ctx in chunk.context_sentences:
            assert ctx.role == ROLE_CONTEXT
            # The same global index is EVIDENCE in exactly one chunk.
            evidence = [
                r
                for other in chunks
                for r in other.evidence_sentences
                if r.index == ctx.index
            ]
            assert len(evidence) == 1
            assert ctx.sentence_ref == evidence[0].sentence_ref


def test_multi_segment_provenance_and_ordering_preserved_with_parent():
    parent = _parent()
    items = [
        _coarse("A. ", ordinal=0, seg_ref="s1", seg_start=0.0, seg_end=5.0),
        _coarse("B. ", ordinal=1, seg_ref="s1", seg_start=0.0, seg_end=5.0),
        _coarse("C. ", ordinal=2, seg_ref="s2", seg_start=5.0, seg_end=9.0),
        _coarse("D. ", ordinal=3, seg_ref="s3", seg_start=9.0, seg_end=12.0),
    ]
    (chunk,) = build_production_chunks(items, parent=parent)
    # Ordered, de-duplicated evidence segment refs; all segments preserved.
    assert chunk.source_segment_refs == ("s1", "s2", "s3")
    assert [r.source_segment_ref for r in chunk.sentences] == [
        "s1", "s1", "s2", "s3"
    ]
    assert chunk.video_id == parent.video_id
    assert chunk.transcript_asset_id == parent.transcript_asset_id
    # Stable refs are distinct per (segment, ordinal, offsets).
    assert len({r.sentence_ref for r in chunk.sentences}) == 4


def test_pure_primitive_remains_optional_parent_backward_compatible():
    items = [_coarse("A. "), _coarse("B. ")]
    # No parent: historical behaviour; provenance fields are None.
    (chunk,) = chunk_sentences(items)
    assert chunk.video_id is None
    assert chunk.transcript_asset_id is None
    assert chunk.content_right_id is None
    assert chunk.text == "A. B. "
    assert all(r.sentence_ref for r in chunk.sentences)  # still derived
    # With parent: real IDs stamped.
    parent = _parent()
    (with_parent,) = chunk_sentences(items, parent=parent)
    assert with_parent.video_id == parent.video_id
    assert with_parent.transcript_asset_id == parent.transcript_asset_id


def test_parent_context_does_not_change_chunking_algorithm():
    items = [
        _exact(0, 15, 0.0, 10.0),
        _exact(1, 15, 10.0, 20.0),
        _coarse("coarse tail. ", ordinal=2, seg_ref="seg-c",
                seg_start=20.0, seg_end=28.0),
        _exact(3, 15, 28.0, 38.0),
    ]
    without = chunk_sentences(items)
    parent = _parent()
    with_parent = chunk_sentences(items, parent=parent)
    assert len(without) == len(with_parent)
    algo_fields = (
        "index", "text", "start_ms", "end_ms", "duration_ms",
        "timestamp_precision", "word_count", "evidence_start",
        "source_segment_refs", "boundary_reason", "diagnostics",
        "overlap_sentence_indices",
    )
    for a, b in zip(without, with_parent):
        for field in algo_fields:
            assert getattr(a, field) == getattr(b, field), field
        assert [s.sentence_ref for s in a.sentences] == [
            s.sentence_ref for s in b.sentences
        ]
        assert [s.role for s in a.sentences] == [s.role for s in b.sentences]
    # The production wrapper produces the same grouping too.
    prod = build_production_chunks(items, parent=parent)
    assert [c.text for c in prod] == [c.text for c in without]
