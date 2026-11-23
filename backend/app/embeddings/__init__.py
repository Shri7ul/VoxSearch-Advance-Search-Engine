"""VoxSearch dense embedding layer (Goal 15).

Goal 15 is **embedding generation ONLY** — the canonical tracker
(``progress.md``) requires a ``1024-dimensional vector``. This package is the
PURE, torch-free contract/orchestration layer, importable by the core backend
``.venv`` (no torch / transformers / FlagEmbedding / numpy). It provides:

* :mod:`app.embeddings.contracts` — validated immutable configuration, strict
  input/output validation (exactly 1024 dims, no NaN/Inf, L2-normalized within
  a documented tolerance — invalid output is REJECTED, never silently
  rescaled), bounded batching, a typed error taxonomy, ordering guarantees, the
  read-only Goal 13 ``Chunk.text`` bridge, and :func:`generate_embeddings`.

The real BGE-M3 model runs ONLY in the isolated ``backend/.venv-embedding``
interpreter via :mod:`embedding_worker` (a separate process with lazy heavy
imports), following the same proven boundary as Goal 12's
:mod:`alignment_worker`. Importing :mod:`app` / :mod:`app.transcripts` /
:mod:`app.embeddings` NEVER loads a model.

Explicitly OUT OF SCOPE (later goals): vector persistence / pgvector (Goal 16),
cosine similarity search (Goal 17), search API (Goal 18), sparse / ColBERT
retrieval, reranking, frontend, YouTube ingestion, workers/queues, and any
database migration. No production embedding job is run by the test suite.
"""
from app.embeddings.query import (
    QUERY_EMBEDDING_PYTHON_ENV,
    QueryEmbeddingError,
    QueryEmbeddingService,
    QueryEmbeddingUnavailableError,
)
from app.embeddings.contracts import (
    DEFAULT_MODEL_ID,
    DENSE_DIMENSION,
    MAX_BATCH_SIZE,
    MAX_MAX_LENGTH,
    MAX_MODEL_ID_CHARACTERS,
    MAX_MODEL_PATH_CHARACTERS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
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

__all__ = [
    "QUERY_EMBEDDING_PYTHON_ENV",
    "QueryEmbeddingError",
    "QueryEmbeddingService",
    "QueryEmbeddingUnavailableError",
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
