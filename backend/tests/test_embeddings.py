"""Goal 15 — dense BGE-M3 embedding contracts (PURE, mocked, deterministic).

Covers the 9 mandated test areas with mocks/fakes — NO heavy ML imports, NO
model weights, NO network, NO database, NO credentials:

1. A valid single-text embedding.
2. Multiple texts and output ordering.
3. Output dimension equals 1024.
4. Finite values only (NaN/Inf rejected).
5. Empty input behavior.
6. Invalid input behavior.
7. Batch processing (bounded, ordered, model loaded once).
8. Model-loading / inference failure handling (typed errors).
9. Regression compatibility with existing transcript chunking (Goal 13).

The real model is exercised separately (opt-in) in
``test_embedding_worker.py`` / the standalone smoke test — never here.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from app.embeddings import (
    DEFAULT_MODEL_ID,
    DENSE_DIMENSION,
    MAX_BATCH_SIZE,
    MAX_MAX_LENGTH,
    MAX_MODEL_ID_CHARACTERS,
    MAX_MODEL_PATH_CHARACTERS,
    MAX_TEXT_CHARACTERS,
    MAX_TEXTS_PER_CALL,
    NORMALIZATION_TOLERANCE,
    DenseEncoder,
    EmbeddingBatch,
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    EmbeddingProvider,
    EmbeddingVector,
    InProcessEmbeddingProvider,
    InvalidEmbeddingInputError,
    InvalidEmbeddingOutputError,
    generate_embeddings,
    iter_batches,
    texts_from_chunks,
    validate_output,
    validate_texts,
)
from app.transcripts.chunking import Chunk
from app.transcripts.sentence_segmentation import TimestampPrecision

# ---------------------------------------------------------------------------
# Deterministic fakes (pure stdlib — stand in for the isolated model/provider)
# ---------------------------------------------------------------------------


def _marker(seed: float) -> float:
    """Small, distinct, order-preserving value used as a vector's first coord.

    The application now REQUIRES unit L2 norm, so the first element must stay in
    (-1, 1). ``seed / 1000`` keeps the markers distinct and increasing while
    leaving plenty of room for the norm-completing second component.
    """
    return seed / 1000.0


def _vec(seed: float) -> list[float]:
    """One valid 1024-dim, finite, L2-NORMALIZED vector.

    ``values[0]`` is the small distinct ``_marker(seed)`` (proves 1:1 ordering);
    ``values[1]`` carries the rest of the norm so ``hypot(values) == 1.0`` and
    the vector passes the application's normalization validation.
    """
    marker = _marker(seed)
    rest = math.sqrt(1.0 - marker * marker)
    return [marker, rest] + [0.0] * (DENSE_DIMENSION - 2)


class FakeProvider:
    """Returns one distinct, valid vector per input text, in order."""

    def __init__(self, *, start: float = 1.0, failure: Exception | None = None):
        self._start = start
        self._failure = failure
        self.calls: list[tuple[str, ...]] = []

    def embed(
        self, texts: Sequence[str], *, config: EmbeddingConfig
    ) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        if self._failure is not None:
            raise self._failure
        return [_vec(self._start + i) for i, _ in enumerate(texts)]


class FakeEncoder:
    """DenseEncoder double: records each bounded batch, returns valid rows."""

    def __init__(self, *, failure: Exception | None = None):
        self._failure = failure
        self.batches: list[tuple[str, ...]] = []
        self._n = 0

    def encode_batch(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        self.batches.append(tuple(texts))
        if self._failure is not None:
            raise self._failure
        rows = []
        for _ in texts:
            self._n += 1
            rows.append(_vec(float(self._n)))
        return rows


def _chunk(text: str, index: int = 0) -> Chunk:
    """A minimal REAL Goal 13 Chunk (only ``text`` is read by the bridge)."""
    return Chunk(
        index=index,
        text=text,
        sentences=(),
        start_ms=0,
        end_ms=1000,
        duration_ms=1000,
        timestamp_precision=TimestampPrecision.SEGMENT_LEVEL,
        word_count=len(text.split()),
        evidence_start=0,
        source_segment_refs=(),
        boundary_reason="end_of_input",
        diagnostics=(),
        overlap_sentence_indices=(),
    )


# ===========================================================================
# 1 / 2 / 3 / 4 — valid single, ordering, 1024 dims, finite values
# ===========================================================================
class TestValidGeneration:
    def test_single_text_embedding(self) -> None:
        provider = FakeProvider()
        batch = generate_embeddings(["hello world"], provider)
        assert isinstance(batch, EmbeddingBatch)
        assert batch.count == 1
        assert batch.model_id == DEFAULT_MODEL_ID
        assert batch.dimension == DENSE_DIMENSION
        assert provider.calls == [("hello world",)]

    def test_multiple_texts_preserve_input_order(self) -> None:
        texts = ["alpha", "beta", "gamma", "delta"]
        provider = FakeProvider(start=10.0)
        batch = generate_embeddings(texts, provider)
        assert batch.count == len(texts)
        # Distinct first elements prove strict 1:1 ordering (markers are the
        # bounded ``_marker`` values, since vectors must now be unit-norm).
        assert [v.values[0] for v in batch.vectors] == [
            _marker(x) for x in (10.0, 11.0, 12.0, 13.0)
        ]

    def test_output_dimension_is_exactly_1024(self) -> None:
        batch = generate_embeddings(["a"], FakeProvider())
        assert batch.vectors[0].dimension == DENSE_DIMENSION == 1024
        assert len(batch.vectors[0].values) == 1024

    def test_all_values_finite(self) -> None:
        batch = generate_embeddings(["a", "b"], FakeProvider())
        assert all(
            math.isfinite(x) for v in batch.vectors for x in v.values
        )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nan_inf_values_rejected(self, bad: float) -> None:
        class BadProvider:
            def embed(self, texts, *, config):
                row = _vec(1.0)
                row[7] = bad
                return [row]

        with pytest.raises(InvalidEmbeddingOutputError):
            generate_embeddings(["a"], BadProvider())

    def test_output_value_object_rejects_wrong_dimension(self) -> None:
        with pytest.raises(InvalidEmbeddingOutputError):
            EmbeddingVector.from_sequence([0.0] * 10)

    def test_output_value_object_rejects_non_number(self) -> None:
        row = [0.0] * DENSE_DIMENSION
        row[0] = "x"
        with pytest.raises(InvalidEmbeddingOutputError):
            EmbeddingVector.from_sequence(row)

    def test_validate_output_rejects_wrong_count(self) -> None:
        cfg = EmbeddingConfig()
        with pytest.raises(InvalidEmbeddingOutputError):
            validate_output([_vec(1.0)], expected=2, config=cfg)


# ===========================================================================
# 5 — empty input behavior (NO provider call, NO model load)
# ===========================================================================
class TestEmptyInput:
    def test_empty_sequence_returns_empty_batch_without_provider_call(self) -> None:
        provider = FakeProvider()
        batch = generate_embeddings([], provider)
        assert batch.count == 0
        assert batch.vectors == ()
        assert batch.model_id == DEFAULT_MODEL_ID
        assert batch.dimension == DENSE_DIMENSION
        assert provider.calls == []  # zero model work for zero input

    def test_empty_tuple_same_behavior(self) -> None:
        provider = FakeProvider()
        assert generate_embeddings((), provider).count == 0
        assert provider.calls == []


# ===========================================================================
# 6 — invalid input behavior (explicit rejection, no silent repair)
# ===========================================================================
class TestInvalidInput:
    @pytest.mark.parametrize(
        "bad",
        [
            "a bare string",
            b"bytes",
            123,
            None,
            ["ok", 5],
            ["ok", ""],
            ["ok", "   "],
        ],
    )
    def test_invalid_input_rejected(self, bad) -> None:
        provider = FakeProvider()
        with pytest.raises(InvalidEmbeddingInputError):
            generate_embeddings(bad, provider)
        assert provider.calls == []  # rejected BEFORE any provider work

    def test_over_long_text_rejected_no_truncation(self) -> None:
        cfg = EmbeddingConfig(max_length=8)
        provider = FakeProvider()
        with pytest.raises(InvalidEmbeddingInputError):
            generate_embeddings(["word " * 9], provider, config=cfg)
        assert provider.calls == []

    def test_too_many_texts_rejected(self) -> None:
        cfg = EmbeddingConfig(max_texts=2)
        with pytest.raises(InvalidEmbeddingInputError):
            validate_texts(["a", "b", "c"], cfg)

    def test_character_limit_rejected(self) -> None:
        cfg = EmbeddingConfig(max_text_characters=10)
        with pytest.raises(InvalidEmbeddingInputError):
            validate_texts(["x" * 11], cfg)

    def test_validate_texts_preserves_verbatim_text(self) -> None:
        cfg = EmbeddingConfig()
        raw = "  Keep  EXACT\tspacing  "
        assert validate_texts([raw], cfg) == (raw,)

    def test_config_rejects_bad_values(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(batch_size=0)
        with pytest.raises(ValidationError):
            EmbeddingConfig(batch_size=MAX_BATCH_SIZE + 1)
        with pytest.raises(ValidationError):
            EmbeddingConfig(max_length=True)  # bool is not a strict int
        with pytest.raises(ValidationError):
            EmbeddingConfig(max_length=MAX_MAX_LENGTH + 1)
        with pytest.raises(ValidationError):
            EmbeddingConfig(model_path="https://example.com/model")
        with pytest.raises(ValidationError):
            EmbeddingConfig(unknown_field=1)

    def test_config_is_frozen(self) -> None:
        cfg = EmbeddingConfig()
        with pytest.raises(ValidationError):
            cfg.batch_size = 2  # type: ignore[misc]

    def test_config_defaults_are_conservative(self) -> None:
        cfg = EmbeddingConfig()
        assert cfg.model_id == DEFAULT_MODEL_ID
        assert cfg.model_path is None
        assert 1 <= cfg.batch_size <= MAX_BATCH_SIZE
        assert 1 <= cfg.max_length <= MAX_MAX_LENGTH
        assert cfg.max_texts <= MAX_TEXTS_PER_CALL
        assert cfg.max_text_characters <= MAX_TEXT_CHARACTERS


# ===========================================================================
# 7 — batch processing (bounded, ordered, ONE loaded model)
# ===========================================================================
class TestBatchProcessing:
    def test_iter_batches_is_bounded_ordered_and_disjoint(self) -> None:
        cfg = EmbeddingConfig(batch_size=3)
        texts = tuple(f"t{i}" for i in range(8))
        batches = list(iter_batches(texts, cfg))
        assert batches == [
            ("t0", "t1", "t2"),
            ("t3", "t4", "t5"),
            ("t6", "t7"),
        ]
        assert tuple(t for b in batches for t in b) == texts  # no loss / dup

    def test_inprocess_provider_batches_over_one_loaded_encoder(self) -> None:
        encoder = FakeEncoder()
        provider = InProcessEmbeddingProvider(encoder)
        cfg = EmbeddingConfig(batch_size=2)
        batch = generate_embeddings(
            [f"t{i}" for i in range(5)], provider, config=cfg
        )
        assert batch.count == 5
        # 5 texts / batch_size 2 -> 3 bounded batches, all on ONE encoder.
        assert len(encoder.batches) == 3
        assert all(len(b) <= 2 for b in encoder.batches)
        assert tuple(t for b in encoder.batches for t in b) == tuple(
            f"t{i}" for i in range(5)
        )
        # Global ordering preserved across batch boundaries.
        assert [v.values[0] for v in batch.vectors] == [
            _marker(float(i)) for i in range(1, 6)
        ]

    def test_inprocess_provider_wrong_batch_count_rejected(self) -> None:
        class ShortEncoder:
            def encode_batch(self, texts):
                return [_vec(1.0)]  # returns fewer rows than inputs

        provider = InProcessEmbeddingProvider(ShortEncoder())
        with pytest.raises(InvalidEmbeddingOutputError):
            generate_embeddings(["a", "b"], provider)

    def test_protocols_are_runtime_checkable(self) -> None:
        assert isinstance(FakeEncoder(), DenseEncoder)
        assert isinstance(FakeProvider(), EmbeddingProvider)


# ===========================================================================
# 8 — model-loading / inference failure handling (typed, credential-free)
# ===========================================================================
class TestFailureHandling:
    def test_unexpected_provider_exception_wrapped_as_inference_error(self) -> None:
        provider = FakeProvider(failure=RuntimeError("raw secret-ish detail"))
        with pytest.raises(EmbeddingInferenceError) as exc:
            generate_embeddings(["a"], provider)
        # The raw message / type must not be blindly propagated.
        assert "RuntimeError" in str(exc.value)
        assert "secret-ish" not in str(exc.value)

    def test_encoder_exception_wrapped_as_inference_error(self) -> None:
        encoder = FakeEncoder(failure=RuntimeError("boom"))
        provider = InProcessEmbeddingProvider(encoder)
        with pytest.raises(EmbeddingInferenceError):
            generate_embeddings(["a", "b"], provider)

    def test_embedding_error_subclasses_pass_through_unchanged(self) -> None:
        provider = FakeProvider(failure=EmbeddingModelLoadError("no weights"))
        with pytest.raises(EmbeddingModelLoadError):
            generate_embeddings(["a"], provider)

    def test_error_taxonomy(self) -> None:
        for cls in (
            InvalidEmbeddingInputError,
            InvalidEmbeddingOutputError,
            EmbeddingModelLoadError,
            EmbeddingInferenceError,
        ):
            assert issubclass(cls, EmbeddingError)


# ===========================================================================
# 9 — regression compatibility with Goal 13 transcript chunking
# ===========================================================================
class TestChunkBridge:
    def test_texts_from_chunks_reads_verbatim_text_in_order(self) -> None:
        chunks = [
            _chunk("  First  sentence. ", 0),
            _chunk("Second one.", 1),
            _chunk("Third.", 2),
        ]
        texts = texts_from_chunks(chunks)
        assert texts == ("  First  sentence. ", "Second one.", "Third.")
        # Goal 13 objects are NEVER mutated and gain NO embedding field.
        assert chunks[0].text == "  First  sentence. "
        assert not hasattr(chunks[0], "embedding")

    def test_full_path_chunk_to_embedding_is_1_to_1(self) -> None:
        chunks = [_chunk("A", 0), _chunk("B", 1)]
        provider = FakeProvider()
        batch = generate_embeddings(texts_from_chunks(chunks), provider)
        assert batch.count == len(chunks)
        assert provider.calls == [("A", "B")]

    @pytest.mark.parametrize("bad", ["a bare string", ["not a chunk"], [_chunk("x"), None]])
    def test_invalid_chunk_input_rejected(self, bad) -> None:
        with pytest.raises(InvalidEmbeddingInputError):
            texts_from_chunks(bad)

    def test_empty_chunk_text_rejected(self) -> None:
        with pytest.raises(InvalidEmbeddingInputError):
            texts_from_chunks([_chunk("   ")])


# ===========================================================================
# 10 — L2-normalization validation (application-level, documented tolerance)
# ===========================================================================
class TestL2Normalization:
    """``BGEM3FlagModel(normalize_embeddings=True)`` is contracted to return
    unit-norm dense vectors. The application VALIDATES that invariant and
    REJECTS non-normalized output; it NEVER silently rescales the values.
    """

    def test_unit_norm_vector_accepted(self) -> None:
        row = [0.0] * DENSE_DIMENSION
        row[0] = 1.0
        v = EmbeddingVector.from_sequence(row)
        assert v.dimension == DENSE_DIMENSION
        assert abs(math.hypot(*v.values) - 1.0) <= NORMALIZATION_TOLERANCE

    def test_normalized_fake_vector_passes_full_generation(self) -> None:
        batch = generate_embeddings(["a"], FakeProvider())
        assert (
            abs(math.hypot(*batch.vectors[0].values) - 1.0)
            <= NORMALIZATION_TOLERANCE
        )

    @pytest.mark.parametrize("scale", [0.0, 0.5, 2.0, 10.0])
    def test_non_normalized_vector_rejected(self, scale: float) -> None:
        row = [0.0] * DENSE_DIMENSION
        row[0] = scale
        with pytest.raises(InvalidEmbeddingOutputError):
            EmbeddingVector.from_sequence(row)

    def test_norm_within_tolerance_accepted_and_NOT_rescaled(self) -> None:
        # Just inside the tolerance: ACCEPTED, but the values are returned
        # VERBATIM — the model's output is never silently normalized.
        target = 1.0 + NORMALIZATION_TOLERANCE / 2.0
        row = [0.0] * DENSE_DIMENSION
        row[0] = target
        v = EmbeddingVector.from_sequence(row)
        assert v.values[0] == target  # NOT rescaled to 1.0

    def test_norm_outside_tolerance_rejected(self) -> None:
        row = [0.0] * DENSE_DIMENSION
        row[0] = 1.0 + NORMALIZATION_TOLERANCE * 2.0
        with pytest.raises(InvalidEmbeddingOutputError):
            EmbeddingVector.from_sequence(row)

    def test_generation_rejects_non_normalized_provider_output(self) -> None:
        class BadNormProvider:
            def embed(self, texts, *, config):
                return [
                    [2.0] + [0.0] * (DENSE_DIMENSION - 1) for _ in texts
                ]

        with pytest.raises(InvalidEmbeddingOutputError):
            generate_embeddings(["a"], BadNormProvider())


# ===========================================================================
# 11 — explicit, aligned safety limits (config rejects what worker rejects)
# ===========================================================================
class TestExplicitLimits:
    """Regression: the core config must NOT accept a value the worker's hard
    caps would reject. All limits flow from one authoritative constants set.
    """

    def test_max_text_characters_cannot_exceed_worker_cap(self) -> None:
        # MAX_TEXT_CHARACTERS is BOTH the config upper bound and the worker cap.
        assert EmbeddingConfig().max_text_characters == MAX_TEXT_CHARACTERS
        with pytest.raises(ValidationError):
            EmbeddingConfig(max_text_characters=MAX_TEXT_CHARACTERS + 1)

    def test_at_limit_config_is_accepted(self) -> None:
        cfg = EmbeddingConfig(
            max_text_characters=MAX_TEXT_CHARACTERS,
            model_id="m" * MAX_MODEL_ID_CHARACTERS,
            model_path="p" * MAX_MODEL_PATH_CHARACTERS,
        )
        assert cfg.max_text_characters == MAX_TEXT_CHARACTERS
        assert len(cfg.model_id) == MAX_MODEL_ID_CHARACTERS
        assert len(cfg.model_path) == MAX_MODEL_PATH_CHARACTERS

    def test_over_long_model_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(model_id="m" * (MAX_MODEL_ID_CHARACTERS + 1))

    def test_over_long_model_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(model_path="p" * (MAX_MODEL_PATH_CHARACTERS + 1))
