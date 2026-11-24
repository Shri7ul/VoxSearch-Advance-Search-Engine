"""Goal 43 — deterministic 200-query labeled evaluation dataset.

The built-in dataset is a SYNTHETIC, reproducible fixture for retrieval
evaluation. It is NOT a set of human-annotated relevance judgments and must not
be presented as one. Real annotated datasets can be supplied through
:func:`validate_dataset` using the same immutable contracts.

Every query carries a stable id, text, closed category, language and the chunk
ids considered relevant (qrels). Generation is pure and deterministic: no
randomness, network, database or file access.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

DEFAULT_DATASET_SIZE = 200
_CATEGORY_SIZE = 40
_LANGUAGES = ("en", "bn")


class QueryCategory(str, Enum):
    """Closed, documented query categories used by later evaluation goals."""

    TECHNICAL_TERM = "technical_term"
    SEMANTIC_PARAPHRASE = "semantic_paraphrase"
    MULTILINGUAL = "multilingual"
    TIMESTAMP_SPECIFIC = "timestamp_specific"
    CONTEXTUAL = "contextual"


@dataclass(frozen=True)
class LabeledQuery:
    """One labeled query with explicit relevance judgments."""

    query_id: str
    text: str
    category: QueryCategory
    language: str
    relevant_chunk_ids: tuple[str, ...]
    primary_relevant_chunk_id: str

    def qrels(self) -> dict[str, int]:
        """Return graded relevance labels for this query."""
        labels = {chunk_id: 1 for chunk_id in self.relevant_chunk_ids}
        labels[self.primary_relevant_chunk_id] = 2
        return labels


@dataclass(frozen=True)
class EvaluationDataset:
    """Immutable labeled query collection."""

    name: str
    version: str
    queries: tuple[LabeledQuery, ...]

    def size(self) -> int:
        return len(self.queries)

    def by_category(self, category: QueryCategory) -> tuple[LabeledQuery, ...]:
        return tuple(query for query in self.queries if query.category == category)

    def qrels(self) -> dict[str, dict[str, int]]:
        return {query.query_id: query.qrels() for query in self.queries}


def validate_dataset(
    records: Any,
    *,
    name: str = "voxsearch-eval",
    version: str = "1.0",
    expected_size: int | None = DEFAULT_DATASET_SIZE,
) -> EvaluationDataset:
    """Validate an iterable of labeled-query mappings or dataclasses."""
    if isinstance(records, (str, bytes)) or not hasattr(records, "__iter__"):
        raise ValueError("dataset must be an iterable of labeled queries")
    queries: list[LabeledQuery] = []
    seen_ids: set[str] = set()
    for raw in records:
        if isinstance(raw, LabeledQuery):
            query = raw
        elif isinstance(raw, Mapping):
            for field in (
                "query_id",
                "text",
                "language",
                "relevant_chunk_ids",
                "primary_relevant_chunk_id",
            ):
                if not isinstance(raw.get(field), str) and not (
                    field == "relevant_chunk_ids"
                    and isinstance(raw.get(field), (list, tuple))
                ):
                    raise ValueError(f"malformed labeled query field: {field}")
            try:
                chunk_ids = tuple(raw["relevant_chunk_ids"])
                if any(not isinstance(item, str) for item in chunk_ids):
                    raise ValueError("relevance ids must be strings")
                query = LabeledQuery(
                    query_id=raw["query_id"],
                    text=raw["text"],
                    category=QueryCategory(raw["category"]),
                    language=raw["language"],
                    relevant_chunk_ids=chunk_ids,
                    primary_relevant_chunk_id=raw["primary_relevant_chunk_id"],
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError("malformed labeled query") from exc
        else:
            raise ValueError("malformed labeled query")
        if not query.query_id or query.query_id in seen_ids:
            raise ValueError("duplicate or empty query id")
        seen_ids.add(query.query_id)
        if not query.text.strip():
            raise ValueError("query text must not be blank")
        if query.language not in _LANGUAGES:
            raise ValueError("unsupported query language")
        if (
            not query.relevant_chunk_ids
            or any(not isinstance(item, str) for item in query.relevant_chunk_ids)
            or any(not item.strip() for item in query.relevant_chunk_ids)
        ):
            raise ValueError("relevance labels must be non-empty strings")
        if query.primary_relevant_chunk_id not in query.relevant_chunk_ids:
            raise ValueError("primary relevance id is not in the qrels")
        queries.append(query)
    if expected_size is not None and len(queries) != expected_size:
        raise ValueError("dataset size does not match the expected size")
    return EvaluationDataset(name=name, version=version, queries=tuple(queries))


_TEMPLATES: dict[QueryCategory, tuple[str, ...]] = {
    QueryCategory.TECHNICAL_TERM: (
        "exact term {topic}",
        "definition of {topic}",
        "what does {topic} mean",
        "technical details about {topic}",
    ),
    QueryCategory.SEMANTIC_PARAPHRASE: (
        "explain {topic} in simple words",
        "how would you describe {topic}",
        "tell me about the idea of {topic}",
        "what is the gist of {topic}",
    ),
    QueryCategory.MULTILINGUAL: (
        "{topic} কী",
        "{topic} সম্পর্কে বলুন",
        "ব্যাখ্যা করুন {topic}",
        "{topic} মানে কী",
    ),
    QueryCategory.TIMESTAMP_SPECIFIC: (
        "at what point does {topic} start",
        "when is {topic} discussed",
        "timestamp for {topic}",
        "where in the video is {topic}",
    ),
    QueryCategory.CONTEXTUAL: (
        "context around {topic}",
        "what was said before {topic}",
        "background of {topic} in the talk",
        "surrounding discussion of {topic}",
    ),
}

_TOPICS = (
    "latency",
    "embedding dimension",
    "cosine distance",
    "chunk overlap",
    "word timestamps",
    "sentence segmentation",
    "rights authorization",
    "reranking model",
    "queue retries",
    "indexing pipeline",
)


def build_default_dataset() -> EvaluationDataset:
    """Build the deterministic 200-query fixture (40 per category)."""
    records: list[dict[str, Any]] = []
    index = 0
    for category in QueryCategory:
        templates = _TEMPLATES[category]
        for number in range(_CATEGORY_SIZE):
            topic = _TOPICS[number % len(_TOPICS)]
            template = templates[(number // len(_TOPICS)) % len(templates)]
            primary = f"chunk-{category.value}-{number % 5:02d}"
            secondary = f"chunk-{category.value}-{(number + 1) % 5:02d}"
            records.append(
                {
                    "query_id": f"q{index + 1:03d}",
                    "text": template.format(topic=topic),
                    "category": category.value,
                    "language": "bn" if category is QueryCategory.MULTILINGUAL else "en",
                    "relevant_chunk_ids": [primary, secondary],
                    "primary_relevant_chunk_id": primary,
                }
            )
            index += 1
    return validate_dataset(records)


__all__ = [
    "DEFAULT_DATASET_SIZE",
    "EvaluationDataset",
    "LabeledQuery",
    "QueryCategory",
    "build_default_dataset",
    "validate_dataset",
]
