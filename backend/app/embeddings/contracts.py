"""Dense BGE-M3 embedding contracts and deterministic generation (Goal 15).

Purpose & scope
---------------
Goal 15 is **embedding generation only**, per the canonical tracker
(``progress.md``): ``15 | Embedding generation | 1024-dimensional vector``.
This is a PURE, library-level module: NO database, NO persistence, NO HTTP
route, NO migration, NO search, NO sparse / ColBERT retrieval, NO credentials,
NO network, and NO heavy ML import. It is importable by the core backend
``.venv`` (which deliberately contains no torch / transformers / FlagEmbedding
/ numpy); the actual model lives in the isolated ``.venv-embedding`` process
(see :mod:`embedding_worker`).

Architecture (mirrors Goal 12's proven isolation boundary)
---------------------------------------------------------
* This module owns the *contracts*: configuration, input/output validation,
  bounded batching, ordered results, and typed errors.
* :class:`DenseEncoder` is a **loaded** model (inference only). Model
  initialization is the caller's job, so construction is deliberately kept
  OUT of this module and OUT of :class:`InProcessEmbeddingProvider` — the model
  is loaded ONCE, never once per chunk.
* :class:`EmbeddingProvider` is a complete backend that owns the model
  lifecycle. :class:`~embedding_worker.client.SubprocessEmbeddingProvider`
  implements it by launching the isolated worker ONCE for the whole call; the
  worker loads the model a single time and encodes internally in small batches.

Consuming Goal 13 chunks without modifying Goal 12/13
-----------------------------------------------------
:func:`texts_from_chunks` reads the verbatim ``Chunk.text`` (Goal 13,
:class:`app.transcripts.chunking.Chunk`) in input order. It mutates nothing and
imports no ML code. Embeddings are returned as a separate, strictly ordered
:class:`EmbeddingBatch` 1:1 with the input — Goal 13's ``Chunk`` is NEVER given
an embedding field here. This leaves Goal 16 (vector persistence) free to
choose the storage shape without any Goal 12/13 regression.

Conservative CPU / memory policy
--------------------------------
The workstation is CPU-only with limited free RAM. Defaults are deliberately
small: ``batch_size=4``, ``max_length=256`` tokens, at most 64 texts per call,
at most 8192 characters per text. These are SAFETY bounds, not performance
claims; no throughput / latency benchmark is made. Text that would be silently
truncated by the tokenizer is REJECTED (no silent repair). Sparse / multi-vector
heads are disabled. FP16 is never enabled on CPU. Dense vectors are validated
to be L2-normalized within a documented tolerance; non-normalized model output
is REJECTED, never silently rescaled.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.transcripts.chunking import Chunk

# --- Explicit, consistent safety bounds (single source of truth) -------------
# These are the ONLY authoritative limits for Goal 15. ``client.py`` imports
# them; ``worker.py`` keeps LOCAL copies (it must not import ``app``) that are
# checked against these by ``tests/test_embedding_worker.py::TestLimitConsistency``
# so the core and the isolated process can never silently drift apart. Every
# INPUT bound below is enforced BOTH by ``EmbeddingConfig`` (config validation)
# and by the worker's request gate, so no core-accepted configuration can be
# rejected by the worker because of a limit disagreement; the two BYTE caps are
# enforced by the client (before send) and the worker (on read).
#
# Conservative, hardware-aware bounds (NOT benchmarks).
DENSE_DIMENSION = 1024
DEFAULT_MODEL_ID = "BAAI/bge-m3"

MAX_BATCH_SIZE = 8
MAX_MAX_LENGTH = 512
MAX_TEXTS_PER_CALL = 64
MAX_TEXT_CHARACTERS = 8192
MAX_MODEL_ID_CHARACTERS = 256
MAX_MODEL_PATH_CHARACTERS = 4096

# Worst-case JSON request (client -> worker), UTF-8, ``ensure_ascii=False``:
#   texts      : 64 x 8192 chars x 6 B/char = 3,145,728  (a control char expands
#                to the 6-byte JSON escape ``\u00XX`` — the widest per-char case)
#   model_id   : 256 chars x 6 B           =       1,536
#   model_path : 4096 chars x 6 B          =      24,576
#   keys/ints  :                                  ~1,337
#   -------------------------------------------------------
#   MEASURED worst case                        = 3,172,177 B
# 4,000,000 leaves ~26% headroom. The client checks this BEFORE sending; the
# worker reads ``MAX_REQUEST_BYTES + 1`` and rejects anything larger. Both use
# THIS constant, so a core-valid request can never be dropped by a size limit.
MAX_REQUEST_BYTES = 4_000_000

# MEASURED worst-case JSON response (64 vectors x 1024 finite floats with full
# 19-char float reprs, compact separators) = 1,310,883 B; 16 MB is a generous
# fail-closed cap. The worker bounds its own ``_emit`` and the client re-checks
# after the call.
MAX_RESPONSE_BYTES = 16_000_000

# L2-normalization contract. ``BGEM3FlagModel(normalize_embeddings=True)``
# returns unit-norm dense vectors; the application VALIDATES this instead of
# assuming it. A vector whose L2 norm deviates from 1.0 by more than this
# ABSOLUTE tolerance is REJECTED with ``InvalidEmbeddingOutputError`` — it is
# NEVER silently rescaled. fp32 ``F.normalize`` output is ~1.0 within rounding;
# 1e-3 is deliberate headroom for float serialization, while still rejecting
# plainly un-normalized output (e.g. norm 0.5 or 2.0). This is a correctness
# tolerance, NOT a performance/benchmark claim.
NORMALIZATION_TOLERANCE = 1e-3

# --- Explicit, credential-free error taxonomy --------------------------------


class EmbeddingError(Exception):
    """Base class for every Goal 15 failure."""


class InvalidEmbeddingInputError(EmbeddingError, ValueError):
    """Rejection of invalid / empty / oversized input (no silent repair)."""


class EmbeddingModelLoadError(EmbeddingError, RuntimeError):
    """The isolated model could not be loaded / launched."""


class EmbeddingInferenceError(EmbeddingError, RuntimeError):
    """Inference failed, timed out, or the provider could not be reached."""


class InvalidEmbeddingOutputError(EmbeddingError, ValueError):
    """Provider output has wrong shape/count, NaN, Inf, or a bad type."""


# --- Validated, immutable configuration (single source of truth) -------------


class EmbeddingConfig(BaseModel):
    """Validated, immutable embedding parameters.

    No magic numbers are scattered through the pipeline — every threshold
    flows from here. Deterministic for a given config + input.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = DEFAULT_MODEL_ID
    # Optional LOCAL path to the cached snapshot. A URL is rejected; model
    # weights are a cache-resident artifact, never fetched by this module.
    model_path: str | None = None
    batch_size: int = Field(default=4, strict=True, ge=1, le=MAX_BATCH_SIZE)
    max_length: int = Field(default=256, strict=True, ge=1, le=MAX_MAX_LENGTH)
    max_texts: int = Field(
        default=MAX_TEXTS_PER_CALL, strict=True, ge=1, le=MAX_TEXTS_PER_CALL
    )
    # NOTE: the upper bound is the SAME value as the worker's hard cap. A config
    # the core accepts can therefore never be rejected by the worker for being
    # too long (see ``tests/test_embedding_worker.py::TestLimitConsistency``).
    max_text_characters: int = Field(
        default=MAX_TEXT_CHARACTERS, strict=True, ge=1, le=MAX_TEXT_CHARACTERS
    )

    @field_validator("model_id")
    @classmethod
    def _model_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_id must be a non-empty str")
        if len(value) > MAX_MODEL_ID_CHARACTERS:
            raise ValueError(
                f"model_id exceeds {MAX_MODEL_ID_CHARACTERS} characters"
            )
        return value

    @field_validator("model_path")
    @classmethod
    def _model_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip() or "://" in value:
            raise ValueError("model_path must be a LOCAL path, not a URL")
        if len(value) > MAX_MODEL_PATH_CHARACTERS:
            raise ValueError(
                f"model_path exceeds {MAX_MODEL_PATH_CHARACTERS} characters"
            )
        return value


# --- Output value objects ----------------------------------------------------


@dataclass(frozen=True)
class EmbeddingVector:
    """One validated dense vector: exactly DENSE_DIMENSION finite, unit-norm floats."""

    values: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.values)

    @classmethod
    def from_sequence(
        cls, raw: Any, *, index: int = 0
    ) -> "EmbeddingVector":
        """Validate one raw vector: exact dims, finite floats, unit L2 norm.

        ``BGEM3FlagModel(normalize_embeddings=True)`` is contracted to return
        unit-norm dense vectors; the application VALIDATES that invariant. A
        vector whose L2 norm deviates from 1.0 by more than
        ``NORMALIZATION_TOLERANCE`` is REJECTED with
        :class:`InvalidEmbeddingOutputError` — it is NEVER silently rescaled.
        NaN/Inf and wrong dimensions are never coerced.
        """
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise InvalidEmbeddingOutputError(
                f"vector {index}: expected a sequence of numbers, "
                f"got {type(raw).__name__}"
            )
        if len(raw) != DENSE_DIMENSION:
            raise InvalidEmbeddingOutputError(
                f"vector {index}: expected {DENSE_DIMENSION} dimensions, "
                f"got {len(raw)}"
            )
        values: list[float] = []
        for pos, value in enumerate(raw):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise InvalidEmbeddingOutputError(
                    f"vector {index}[{pos}]: expected a finite real number, "
                    f"got {type(value).__name__}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise InvalidEmbeddingOutputError(
                    f"vector {index}[{pos}]: NaN/Inf is not allowed"
                )
            values.append(number)
        norm = math.hypot(*values)
        if not math.isfinite(norm) or abs(norm - 1.0) > NORMALIZATION_TOLERANCE:
            raise InvalidEmbeddingOutputError(
                f"vector {index}: L2 norm {norm:.6f} is not 1.0 within "
                f"tolerance {NORMALIZATION_TOLERANCE} (non-normalized output "
                f"is rejected, never silently rescaled)"
            )
        return cls(values=tuple(values))


@dataclass(frozen=True)
class EmbeddingBatch:
    """Ordered dense vectors, 1:1 aligned with the validated input texts."""

    vectors: tuple[EmbeddingVector, ...]
    model_id: str
    dimension: int

    @property
    def count(self) -> int:
        return len(self.vectors)


# --- Input validation (no silent repair, no normalization of verbatim text) ---


def validate_texts(texts: Any, config: EmbeddingConfig) -> tuple[str, ...]:
    """Validate the input sequence and return VERBATIM texts.

    Rejected explicitly (with 0-based index): a bare str/bytes, a non-sequence,
    more than ``config.max_texts`` texts, any non-str element, any empty /
    whitespace-only text, any text longer than ``max_text_characters``, and any
    text whose whitespace-word count exceeds ``max_length`` (a conservative
    heuristic that prevents the tokenizer from silently truncating it).

    Nothing is stripped, normalized, sorted, dropped, or repaired: Goal 13's
    verbatim chunk text is preserved exactly.
    """
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise InvalidEmbeddingInputError(
            "texts must be a sequence of str (not a bare str/bytes)"
        )
    if len(texts) > config.max_texts:
        raise InvalidEmbeddingInputError(
            f"at most {config.max_texts} texts per call, got {len(texts)}"
        )
    out: list[str] = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str):
            raise InvalidEmbeddingInputError(
                f"text {idx}: expected str, got {type(text).__name__}"
            )
        if not text.strip():
            raise InvalidEmbeddingInputError(
                f"text {idx}: must be a non-empty, non-whitespace str"
            )
        if len(text) > config.max_text_characters:
            raise InvalidEmbeddingInputError(
                f"text {idx}: {len(text)} characters exceeds "
                f"max_text_characters={config.max_text_characters}"
            )
        words = len(text.split())
        if words > config.max_length:
            raise InvalidEmbeddingInputError(
                f"text {idx}: ~{words} whitespace words exceeds the conservative "
                f"max_length={config.max_length} and would be silently truncated"
            )
        out.append(text)
    return tuple(out)


def validate_output(
    raw: Any, *, expected: int, config: EmbeddingConfig
) -> EmbeddingBatch:
    """Validate provider output: exact count, 1024 dims, finite, unit L2 norm."""
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise InvalidEmbeddingOutputError(
            f"provider output must be a sequence, got {type(raw).__name__}"
        )
    if len(raw) != expected:
        raise InvalidEmbeddingOutputError(
            f"expected {expected} vectors, got {len(raw)}"
        )
    vectors = tuple(
        EmbeddingVector.from_sequence(item, index=idx)
        for idx, item in enumerate(raw)
    )
    return EmbeddingBatch(
        vectors=vectors,
        model_id=config.model_id,
        dimension=DENSE_DIMENSION,
    )


# --- Bounded batching (never assume unlimited RAM) ---------------------------
# NOTE (deliberate duplication, kept by design): ``iter_batches`` /
# ``InProcessEmbeddingProvider`` here, and
# ``embedding_worker.worker._encode_batches`` in the isolated process, both
# implement a small bounded, ordered batch loop. They are intentionally NOT
# merged. The worker MUST NOT import :mod:`app` (it runs in the optional
# ``.venv-embedding`` interpreter WITHOUT pydantic / fastapi / sqlalchemy), so
# it cannot reuse these helpers; and this module's helpers are a PUBLIC
# contract (in ``__all__``) exercised directly by ``tests/test_embeddings.py``
# and offered as a supported in-process provider path for callers that already
# hold a loaded encoder. Removing either would break a public contract or force
# the isolated worker to import the core app (architectural churn + public-API
# effect). ``TestLimitConsistency`` pins every shared bound so the two copies
# can never drift. See ``backend/README.md`` §12.


def iter_batches(
    texts: Sequence[str], config: EmbeddingConfig
) -> Iterator[tuple[str, ...]]:
    """Yield consecutive, NON-overlapping, ORDERED small batches."""
    for start in range(0, len(texts), config.batch_size):
        yield tuple(texts[start : start + config.batch_size])


# --- Protocols: model initialization is separate from inference --------------


@runtime_checkable
class DenseEncoder(Protocol):
    """A LOADED dense model — inference ONLY.

    Initialization is deliberately NOT part of this protocol: the caller loads
    the model once and passes the resulting encoder in. This is what keeps a
    multi-gigabyte model from being reloaded for every chunk.
    """

    def encode_batch(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A complete embedding backend owning the model lifecycle."""

    def embed(
        self, texts: Sequence[str], *, config: EmbeddingConfig
    ) -> Sequence[Sequence[float]]: ...


class InProcessEmbeddingProvider:
    """Batch inference over ONE already-loaded :class:`DenseEncoder`.

    Construction (model loading) is external by design: this class NEVER loads
    or reloads a model. It only splits the validated texts into small bounded
    batches and concatenates the results in strict input order.
    """

    def __init__(self, encoder: DenseEncoder) -> None:
        self._encoder = encoder

    @property
    def encoder(self) -> DenseEncoder:
        return self._encoder

    def embed(
        self, texts: Sequence[str], *, config: EmbeddingConfig
    ) -> list[Sequence[float]]:
        out: list[Sequence[float]] = []
        for batch_index, batch in enumerate(iter_batches(texts, config)):
            try:
                encoded = self._encoder.encode_batch(batch)
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001 — explicit, credential-free
                raise EmbeddingInferenceError(
                    f"batch {batch_index}: inference failed "
                    f"({type(exc).__name__})"
                ) from exc
            if len(encoded) != len(batch):
                raise InvalidEmbeddingOutputError(
                    f"batch {batch_index}: encoder returned {len(encoded)} "
                    f"vectors for {len(batch)} texts"
                )
            out.extend(encoded)
        return out


# --- Orchestration -----------------------------------------------------------


def generate_embeddings(
    texts: Any,
    provider: EmbeddingProvider,
    *,
    config: EmbeddingConfig | None = None,
) -> EmbeddingBatch:
    """Validate input -> embed -> validate output -> return ordered batch.

    * Deterministic 1:1 input/output ordering is enforced by
      :func:`validate_output` (exact count) and by index-aligned concatenation.
    * An EMPTY input yields an explicit empty batch and performs NO provider
      call (no model load for zero work).
    * Model-loading / inference / malformed-output failures are surfaced as the
      typed :class:`EmbeddingError` subclasses; unexpected provider exceptions
      are wrapped as :class:`EmbeddingInferenceError`.
    """
    cfg = config if config is not None else EmbeddingConfig()
    validated = validate_texts(texts, cfg)
    if not validated:
        return EmbeddingBatch(
            vectors=(), model_id=cfg.model_id, dimension=DENSE_DIMENSION
        )
    try:
        raw = provider.embed(validated, config=cfg)
    except EmbeddingError:
        raise
    except Exception as exc:  # noqa: BLE001 — never leak a traceback/secret
        raise EmbeddingInferenceError(
            f"embedding provider failed ({type(exc).__name__})"
        ) from exc
    return validate_output(raw, expected=len(validated), config=cfg)


# --- Read-only bridge to Goal 13 chunks --------------------------------------


def texts_from_chunks(chunks: Any) -> tuple[str, ...]:
    """Extract VERBATIM Goal 13 ``Chunk.text`` in input order.

    Read-only: no Goal 12/13 object is mutated and no embedding is attached.
    Invalid / empty chunk text is rejected explicitly.
    """
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise InvalidEmbeddingInputError(
            "chunks must be a sequence of Goal 13 Chunk objects"
        )
    out: list[str] = []
    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, Chunk):
            raise InvalidEmbeddingInputError(
                f"chunk {idx}: expected Chunk, got {type(chunk).__name__}"
            )
        text = chunk.text
        if not isinstance(text, str) or not text.strip():
            raise InvalidEmbeddingInputError(
                f"chunk {idx}: text must be a non-empty str"
            )
        out.append(text)
    return tuple(out)


__all__ = [
    "DEFAULT_MODEL_ID",
    "DENSE_DIMENSION",
    "MAX_BATCH_SIZE",
    "MAX_MAX_LENGTH",
    "MAX_MODEL_ID_CHARACTERS",
    "MAX_MODEL_PATH_CHARACTERS",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "MAX_TEXT_CHARACTERS",
    "MAX_TEXTS_PER_CALL",
    "NORMALIZATION_TOLERANCE",
    "DenseEncoder",
    "EmbeddingBatch",
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingInferenceError",
    "EmbeddingModelLoadError",
    "EmbeddingProvider",
    "EmbeddingVector",
    "InProcessEmbeddingProvider",
    "InvalidEmbeddingInputError",
    "InvalidEmbeddingOutputError",
    "generate_embeddings",
    "iter_batches",
    "texts_from_chunks",
    "validate_output",
    "validate_texts",
]
