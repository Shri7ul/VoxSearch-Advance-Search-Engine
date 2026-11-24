"""Goal 28 — context-aware matching (surrounding context included).

This module builds a bounded, deterministic text window around each candidate
from the neighbouring chunks of the SAME transcript asset (and model). The
target candidate's identity, timestamps, scores and provenance are untouched;
only the text sent to the reranker is enriched.

Boundaries:

* context comes exclusively from the provider (a read-only repository adapter),
  never from other videos, assets or models;
* the target chunk itself is never treated as context;
* neighbour count and total context characters are bounded;
* missing / empty context yields the verbatim target text (NOT an unrelated
  fallback ranking); malformed provider data fails closed.
* Goal 29 (frontend), endpoints, reranking quality metrics and writes are out of
  scope.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.reranking.scoring import InvalidRerankInputError
from app.services.rrf import FusedCandidate

DEFAULT_RADIUS = 1
MAX_RADIUS = 3
MAX_CONTEXT_CHARACTERS = 2000
MAX_TARGET_TEXT_CHARACTERS = 8192


@dataclass(frozen=True)
class ContextConfig:
    """Immutable, validated context-window settings."""

    radius: int = DEFAULT_RADIUS
    max_context_characters: int = MAX_CONTEXT_CHARACTERS

    def __post_init__(self) -> None:
        if isinstance(self.radius, bool) or not isinstance(self.radius, int):
            raise InvalidRerankInputError("radius must be an integer")
        if not 1 <= self.radius <= MAX_RADIUS:
            raise InvalidRerankInputError(
                f"radius must be between 1 and {MAX_RADIUS}"
            )
        if (
            isinstance(self.max_context_characters, bool)
            or not isinstance(self.max_context_characters, int)
            or not 1 <= self.max_context_characters <= MAX_CONTEXT_CHARACTERS
        ):
            raise InvalidRerankInputError("max_context_characters is out of bounds")


@dataclass(frozen=True)
class ContextChunk:
    """One verbatim neighbouring chunk used as context.

    Provenance is REQUIRED and checked against the target, so even a custom
    provider cannot silently mix chunks from another video, asset or model.
    """

    chunk_index: int
    text: str
    start_ms: int
    end_ms: int
    video_id: uuid.UUID
    transcript_asset_id: uuid.UUID
    model_id: str


@dataclass(frozen=True)
class ContextualCandidate:
    """Original candidate plus its deterministic context window."""

    candidate: FusedCandidate
    context: tuple[ContextChunk, ...]
    matching_text: str

    @property
    def chunk_id(self):
        return self.candidate.chunk_id


@runtime_checkable
class ContextProvider(Protocol):
    """Supply verbatim neighbouring chunks for one candidate."""

    def neighbors(
        self, candidate: FusedCandidate, *, radius: int
    ) -> Sequence[ContextChunk]: ...


class ContextExpansionService:
    """Expand candidates with bounded surrounding transcript context."""

    def __init__(
        self, provider: ContextProvider, *, config: ContextConfig | None = None
    ) -> None:
        self._provider = provider
        self._config = config if config is not None else ContextConfig()

    def expand(
        self, candidates: Sequence[FusedCandidate]
    ) -> list[ContextualCandidate]:
        """Return one context-enriched view per input candidate."""
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise InvalidRerankInputError("candidates must be a sequence")
        out: list[ContextualCandidate] = []
        for candidate in candidates:
            if not isinstance(candidate, FusedCandidate):
                raise InvalidRerankInputError("candidate must be a FusedCandidate")
            target_text = candidate.chunk.text
            if not isinstance(target_text, str) or not target_text.strip():
                raise InvalidRerankInputError("candidate chunk text must be non-empty")
            if len(target_text) > MAX_TARGET_TEXT_CHARACTERS:
                raise InvalidRerankInputError("candidate chunk text is too large")
            raw = self._provider.neighbors(candidate, radius=self._config.radius)
            chunks = self._validate_neighbors(raw, candidate)
            # The worker caps EACH scored text at 8192 chars, so the context
            # budget must shrink for long target chunks (including the newline).
            budget = min(
                self._config.max_context_characters,
                max(0, MAX_TARGET_TEXT_CHARACTERS - len(target_text) - 1),
            )
            context_text, used_chunks = self._join(chunks, budget)
            out.append(
                ContextualCandidate(
                    candidate=candidate,
                    context=tuple(used_chunks),
                    matching_text=(
                        f"{target_text}\n{context_text}" if context_text else target_text
                    ),
                )
            )
        return out

    def _validate_neighbors(
        self, raw: Sequence[ContextChunk], candidate: FusedCandidate
    ) -> list[ContextChunk]:
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise InvalidRerankInputError("context provider returned a non-sequence")
        target_index = candidate.chunk.chunk_index
        seen: set[int] = set()
        out: list[ContextChunk] = []
        for chunk in raw:
            if not isinstance(chunk, ContextChunk):
                raise InvalidRerankInputError("context provider returned invalid data")
            target = candidate.chunk
            if (
                chunk.video_id != target.video_id
                or chunk.transcript_asset_id != target.transcript_asset_id
                or chunk.model_id != target.model_id
            ):
                raise InvalidRerankInputError(
                    "context chunk belongs to a different video, asset or model"
                )
            if chunk.chunk_index == target_index or chunk.chunk_index in seen:
                raise InvalidRerankInputError("context contains the target or duplicates")
            if not isinstance(chunk.text, str) or not chunk.text.strip():
                raise InvalidRerankInputError("context text must be non-empty")
            if len(chunk.text) > MAX_TARGET_TEXT_CHARACTERS:
                raise InvalidRerankInputError("context text is too large")
            seen.add(chunk.chunk_index)
            out.append(chunk)
        out.sort(key=lambda chunk: chunk.chunk_index)
        return out[: 2 * self._config.radius]

    def _join(
        self, chunks: Sequence[ContextChunk], max_chars: int
    ) -> tuple[str, list[ContextChunk]]:
        pieces: list[str] = []
        used: list[ContextChunk] = []
        total = 0
        for chunk in chunks:
            if total + len(chunk.text) > max_chars:
                break
            pieces.append(chunk.text)
            used.append(chunk)
            total += len(chunk.text)
        return "\n".join(pieces), used


class ContextAwareRerankingService:
    """Compose bounded context expansion with the Goal 27 reranker."""

    def __init__(
        self,
        expansion: ContextExpansionService,
        reranker,
    ) -> None:
        self._expansion = expansion
        self._reranker = reranker

    def rerank(self, query: str, candidates: Sequence[FusedCandidate]):
        contextual = self._expansion.expand(candidates)
        texts = {
            item.chunk_id: item.matching_text
            for item in contextual
            if item.matching_text != item.candidate.chunk.text
        }
        # The ORIGINAL candidate objects are passed through; only their text is
        # enriched, and identity/rank mapping remain Goal 27's responsibility.
        return self._reranker.rerank(query, list(candidates), texts=texts)


class RepositoryContextProvider:
    """Read-only adapter translating repository rows into `ContextChunk`s."""

    def __init__(self, repository) -> None:
        self._repository = repository

    def neighbors(
        self, candidate: FusedCandidate, *, radius: int
    ) -> Sequence[ContextChunk]:
        chunk = candidate.chunk
        rows = self._repository.list_context_chunks(
            transcript_asset_id=chunk.transcript_asset_id,
            chunk_index=chunk.chunk_index,
            model_id=chunk.model_id,
            radius=radius,
        )
        return [
            ContextChunk(
                chunk_index=row[0],
                text=row[1],
                start_ms=row[2],
                end_ms=row[3],
                # The repository query is scoped to these same provenance keys.
                video_id=chunk.video_id,
                transcript_asset_id=chunk.transcript_asset_id,
                model_id=chunk.model_id,
            )
            for row in rows
        ]


__all__ = [
    "MAX_CONTEXT_CHARACTERS",
    "MAX_RADIUS",
    "ContextChunk",
    "ContextConfig",
    "ContextExpansionService",
    "ContextProvider",
    "ContextAwareRerankingService",
    "ContextualCandidate",
    "RepositoryContextProvider",
]
