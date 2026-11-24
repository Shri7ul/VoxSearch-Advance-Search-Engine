"""Goal 42 — end-to-end indexing automation (no manual steps between stages).

Orchestrates already-accepted components in order:

1. Goal 09 transcript ingestion (validated ``TranscriptInput`` -> asset);
2. Goal 13 production chunking (real parent provenance from the ingestion result);
3. Goal 15 embedding generation (validated 1024-dim unit vectors);
4. Goal 16 chunk+embedding persistence (idempotent upsert).

Every boundary is injected, so no Redis/Celery/database connection or model load
happens at import time and tests use deterministic doubles. This goal does NOT
add a new HTTP route, a retry layer (Goal 41), a Celery task definition (Goal 40
registry owns those) or a database migration.
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.embeddings.contracts import (
    EmbeddingBatch,
    EmbeddingConfig,
    EmbeddingProvider,
    generate_embeddings,
    texts_from_chunks,
)
from app.transcripts.chunking import ChunkParentContext, build_production_chunks
from app.transcripts.contracts import TranscriptInput


class IndexingAutomationError(RuntimeError):
    """Base class for credential-free Goal 42 failures."""


class InvalidIndexingInputError(IndexingAutomationError, ValueError):
    """Automation input is malformed or out of bounds."""


@dataclass(frozen=True)
class IndexingResult:
    """Immutable outcome of one automated indexing run."""

    asset_id: str
    video_id: str
    chunk_count: int
    persisted_count: int
    model_id: str


@runtime_checkable
class IngestionBoundary(Protocol):
    def ingest(self, payload: TranscriptInput) -> Any: ...


@runtime_checkable
class PersistenceBoundary(Protocol):
    def persist(
        self, chunks: Sequence[Any], embeddings: EmbeddingBatch, *, model_id: str
    ) -> int: ...


class IndexingAutomationService:
    """Run the accepted transcript-to-index pipeline end to end."""

    def __init__(
        self,
        *,
        ingestion: IngestionBoundary,
        embedding_provider: EmbeddingProvider,
        persistence: PersistenceBoundary,
        chunker: Any = build_production_chunks,
        embedding_config: EmbeddingConfig | None = None,
        model_id: str | None = None,
    ) -> None:
        self._ingestion = ingestion
        self._provider = embedding_provider
        self._persistence = persistence
        self._chunker = chunker
        self._config = embedding_config or EmbeddingConfig()
        self._model_id = model_id or self._config.model_id
        if not isinstance(self._model_id, str) or not self._model_id.strip():
            raise InvalidIndexingInputError("model_id must be non-empty")

    def run(
        self,
        transcript: TranscriptInput | Mapping[str, Any],
        aligned_sentences: Sequence[Any],
    ) -> IndexingResult:
        """Ingest, chunk, embed and persist one transcript in order."""
        contract = (
            transcript
            if isinstance(transcript, TranscriptInput)
            else TranscriptInput.model_validate(transcript)
        )
        if isinstance(aligned_sentences, (str, bytes)) or not isinstance(
            aligned_sentences, Sequence
        ):
            raise InvalidIndexingInputError("aligned sentences must be a sequence")
        if not aligned_sentences:
            raise InvalidIndexingInputError("aligned sentences are empty")

        ingestion_result = self._ingestion.ingest(contract)
        for attribute in ("asset_id", "video_id", "content_right_id"):
            value = getattr(ingestion_result, attribute, None)
            if not isinstance(value, uuid.UUID):
                raise IndexingAutomationError(
                    f"ingestion returned invalid {attribute}"
                )
        parent = ChunkParentContext.from_ingestion_result(ingestion_result)
        chunks = self._chunker(aligned_sentences, parent=parent)
        if not isinstance(chunks, Sequence) or not chunks:
            raise IndexingAutomationError("chunking produced no chunks")
        texts = texts_from_chunks(chunks)
        embeddings = generate_embeddings(
            texts, self._provider, config=self._config
        )
        persisted = self._persistence.persist(
            chunks, embeddings, model_id=self._model_id
        )
        if (
            isinstance(persisted, bool)
            or not isinstance(persisted, int)
            or persisted != len(chunks)
        ):
            raise IndexingAutomationError(
                "persistence count does not match the embedded chunk count"
            )
        return IndexingResult(
            asset_id=str(parent.transcript_asset_id),
            video_id=str(parent.video_id),
            chunk_count=len(chunks),
            persisted_count=persisted,
            model_id=self._model_id,
        )


def build_indexing_handler(
    service: IndexingAutomationService,
    *,
    aligned_fetcher: Any = None,
):
    """Return a worker-registry handler that parses a Goal 40 job payload.

    Aligned sentences are rich domain objects and cannot be JSON-serialized,
    so real jobs carry only identifiers and an injected ``aligned_fetcher``
    resolves them (e.g. from the transcript repository).
    """

    def handler(payload: dict[str, Any]) -> dict[str, Any]:
        transcript = payload.get("transcript")
        aligned = payload.get("aligned_sentences")
        if aligned is None and aligned_fetcher is not None:
            aligned = aligned_fetcher(payload)
        if transcript is None or aligned is None:
            raise InvalidIndexingInputError(
                "indexing job requires transcript and resolvable aligned sentences"
            )
        result = service.run(transcript, aligned)
        return {
            "asset_id": result.asset_id,
            "video_id": result.video_id,
            "chunk_count": result.chunk_count,
            "persisted_count": result.persisted_count,
            "model_id": result.model_id,
        }

    return handler


__all__ = [
    "IndexingAutomationError",
    "IndexingAutomationService",
    "IndexingResult",
    "InvalidIndexingInputError",
    "build_indexing_handler",
]
