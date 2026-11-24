"""Goal 27 — candidate reranking producing reordered results.

This service consumes the accepted deduplicated candidates (Goal 25) and uses
the isolated BGE reranker (Goal 26) to score ``(query, chunk_text)`` pairs. It
returns the SAME candidate objects with BGE relevance scores and 1-based ranks,
ordered by score descending. Input order is the deterministic tie-break, so
equivalent inputs always yield equivalent output.

Scope: reordering only. No surrounding-context expansion (Goal 28), no HTTP
endpoint, no re-fusion, no new dependency and no model load in the core process.
There is deliberately NO fallback ranking: if the isolated worker/model is
unavailable or returns malformed output, the failure propagates unchanged.
"""
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.reranking.contracts import RerankerConfig
from app.services.rrf import FusedCandidate

MAX_CANDIDATES = 50
MAX_QUERY_CHARACTERS = 8192


class RerankingError(Exception):
    """Credential-free error for invalid Goal 27 reranking inputs."""


class InvalidRerankInputError(RerankingError, ValueError):
    """Reranking input is missing, malformed or out of bounds."""


@dataclass(frozen=True)
class RerankedCandidate:
    """A deduplicated candidate with its BGE relevance score and final rank."""

    candidate: FusedCandidate
    score: float
    rank: int

    @property
    def chunk_id(self):
        return self.candidate.chunk_id


@runtime_checkable
class RerankScorer(Protocol):
    """Score query/text pairs in the isolated reranker process."""

    def score(
        self, pairs: Sequence[Sequence[str]], config: RerankerConfig
    ) -> Sequence[float]: ...


class CandidateRerankingService:
    """Reorder deduplicated candidates using isolated BGE reranker scores."""

    def __init__(
        self,
        scorer: RerankScorer | None = None,
        *,
        config: RerankerConfig | None = None,
        worker_python: str | None = None,
    ) -> None:
        self._config = config if config is not None else RerankerConfig()
        self._scorer = scorer
        self._worker_python = worker_python

    @property
    def config(self) -> RerankerConfig:
        return self._config

    def rerank(
        self,
        query: str,
        candidates: Sequence[FusedCandidate],
        *,
        texts: Mapping[uuid.UUID, str] | None = None,
    ) -> list[RerankedCandidate]:
        """Return ``candidates`` reordered by BGE score (best first).

        ``texts`` is an optional Goal 28 context-aware override keyed by the
        candidate's stable chunk id. It changes only the text sent to the
        reranker — never the candidate object, identity, ranking inputs or
        metadata. A missing id falls back to the verbatim chunk text.
        """
        validated_query = self._validate_query(query)
        validated_candidates = self._validate_candidates(candidates)
        if not validated_candidates:
            return []
        validated_texts = self._validate_texts(texts)

        pairs = [
            [
                validated_query,
                validated_texts.get(item.chunk_id, item.chunk.text),
            ]
            for item in validated_candidates
        ]
        scorer = self._scorer
        if scorer is None:
            from reranker_worker.client import SubprocessRerankerProvider

            from app.reranking.contracts import RERANKER_PYTHON_ENV
            import os

            python = self._worker_python or os.environ.get(RERANKER_PYTHON_ENV, "").strip()
            if not python:
                from app.reranking.contracts import RerankerUnavailableError

                raise RerankerUnavailableError("reranker worker is not configured")
            scorer = SubprocessRerankerProvider(
                worker_python=python, timeout_seconds=self._config.timeout_seconds
            )

        scores = scorer.score(pairs, self._config)
        return self._rank(validated_candidates, scores)

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise InvalidRerankInputError(
                f"query must be a str, got {type(query).__name__}"
            )
        if not query.strip():
            raise InvalidRerankInputError("query must not be empty or whitespace")
        if len(query) > MAX_QUERY_CHARACTERS:
            raise InvalidRerankInputError(
                f"query exceeds {MAX_QUERY_CHARACTERS} characters"
            )
        return query

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[FusedCandidate],
    ) -> list[FusedCandidate]:
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise InvalidRerankInputError("candidates must be a sequence")
        if len(candidates) > MAX_CANDIDATES:
            raise InvalidRerankInputError(
                f"at most {MAX_CANDIDATES} candidates are allowed, got {len(candidates)}"
            )
        return list(candidates)

    @staticmethod
    def _validate_texts(
        texts: Mapping[uuid.UUID, str] | None,
    ) -> dict[uuid.UUID, str]:
        if texts is None:
            return {}
        if not isinstance(texts, Mapping):
            raise InvalidRerankInputError("texts must be a mapping")
        validated: dict[uuid.UUID, str] = {}
        for chunk_id, text in texts.items():
            if not isinstance(chunk_id, uuid.UUID):
                raise InvalidRerankInputError("texts keys must be UUIDs")
            if not isinstance(text, str) or not text.strip():
                raise InvalidRerankInputError("texts values must be non-empty strings")
            if len(text) > MAX_QUERY_CHARACTERS:
                raise InvalidRerankInputError("texts value is too large")
            validated[chunk_id] = text  # verbatim: never normalized
        return validated

    @staticmethod
    def _rank(
        candidates: list[FusedCandidate], scores: Sequence[float]
    ) -> list[RerankedCandidate]:
        if len(scores) != len(candidates):
            raise InvalidRerankInputError(
                f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )
        cleaned: list[float] = []
        for score in scores:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise InvalidRerankInputError("reranker returned a non-numeric score")
            number = float(score)
            if not math.isfinite(number):
                raise InvalidRerankInputError("reranker returned a non-finite score")
            cleaned.append(number)

        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda pair: (-cleaned[pair[0]], pair[0]))
        return [
            RerankedCandidate(candidate=candidate, score=cleaned[index], rank=rank)
            for rank, (index, candidate) in enumerate(indexed, start=1)
        ]


__all__ = [
    "MAX_CANDIDATES",
    "MAX_QUERY_CHARACTERS",
    "CandidateRerankingService",
    "InvalidRerankInputError",
    "RerankScorer",
    "RerankedCandidate",
    "RerankingError",
]
