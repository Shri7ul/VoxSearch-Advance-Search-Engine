"""Goal 22 — keyword (lexical) search service.

Validates an exact-keyword query and delegates to the read-only Goal 22
PostgreSQL FTS repository. This service is the lexical analogue of Goal 17's
:class:`~app.services.video_chunks.VideoChunkSearchService`: it owns input
validation and the credential-free error taxonomy, performs no writes, opens no
transaction and never combines results with dense retrieval (Goal 23).
"""
from __future__ import annotations

import uuid
from typing import Any

from app.repositories.lexical_search import LexicalSearchRepository, LexicalSearchResult
from app.repositories.video_chunks import validate_top_k
from app.services import VoxSearchServiceError
from app.services.video_chunks import InvalidTopKError

MAX_QUERY_CHARACTERS = 2000


class KeywordSearchError(VoxSearchServiceError):
    """Base class for Goal 22 keyword search failures."""


class InvalidKeywordQueryError(KeywordSearchError, ValueError):
    """The keyword query is missing, malformed or too large."""


class InvalidKeywordFilterError(KeywordSearchError, ValueError):
    """A keyword-search filter has an invalid type or value."""


class KeywordSearchService:
    """Read-only exact-keyword retrieval over persisted video chunks."""

    def __init__(
        self,
        session: Any,
        repo: LexicalSearchRepository | None = None,
    ) -> None:
        self._session = session
        self._repo = repo if repo is not None else LexicalSearchRepository(session)

    def search(
        self,
        query: Any,
        *,
        top_k: int = 10,
        video_id: uuid.UUID | None = None,
        transcript_asset_id: uuid.UUID | None = None,
        model_id: str | None = None,
    ) -> list[LexicalSearchResult]:
        """Validate inputs and return ranked keyword matches."""
        validated_query = self._validate_query(query)
        validated_k = self._validate_top_k(top_k)
        self._validate_filters(
            video_id=video_id,
            transcript_asset_id=transcript_asset_id,
            model_id=model_id,
        )
        return self._repo.search_keywords(
            validated_query,
            top_k=validated_k,
            video_id=video_id,
            transcript_asset_id=transcript_asset_id,
            model_id=model_id,
        )

    @staticmethod
    def _validate_query(query: Any) -> str:
        if not isinstance(query, str):
            raise InvalidKeywordQueryError(
                f"query must be a str, got {type(query).__name__}"
            )
        if not query.strip():
            raise InvalidKeywordQueryError("query must not be empty or whitespace")
        if len(query) > MAX_QUERY_CHARACTERS:
            raise InvalidKeywordQueryError(
                f"query exceeds {MAX_QUERY_CHARACTERS} characters"
            )
        return query

    @staticmethod
    def _validate_top_k(top_k: Any) -> int:
        try:
            return validate_top_k(top_k)
        except ValueError as exc:
            raise InvalidTopKError(str(exc)) from exc

    @staticmethod
    def _validate_filters(
        *,
        video_id: Any,
        transcript_asset_id: Any,
        model_id: Any,
    ) -> None:
        if video_id is not None and not isinstance(video_id, uuid.UUID):
            raise InvalidKeywordFilterError("video_id must be a uuid.UUID")
        if transcript_asset_id is not None and not isinstance(
            transcript_asset_id, uuid.UUID
        ):
            raise InvalidKeywordFilterError(
                "transcript_asset_id must be a uuid.UUID"
            )
        if model_id is not None and (
            not isinstance(model_id, str) or not model_id.strip()
        ):
            raise InvalidKeywordFilterError("model_id must be a non-empty string")


__all__ = [
    "MAX_QUERY_CHARACTERS",
    "InvalidKeywordFilterError",
    "InvalidKeywordQueryError",
    "KeywordSearchError",
    "KeywordSearchService",
]
