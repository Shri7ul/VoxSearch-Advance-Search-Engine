"""Tests for Goal 43 — deterministic 200-query labeled dataset."""
from __future__ import annotations

import pytest

from app.evaluation import (
    DEFAULT_DATASET_SIZE,
    QueryCategory,
    build_default_dataset,
    validate_dataset,
)


def test_default_dataset_has_exactly_200_unique_queries() -> None:
    dataset = build_default_dataset()
    assert dataset.size() == DEFAULT_DATASET_SIZE == 200
    ids = [query.query_id for query in dataset.queries]
    assert len(set(ids)) == 200
    assert ids[0] == "q001"
    assert ids[-1] == "q200"
    assert len({query.text for query in dataset.queries}) == 200


def test_default_dataset_is_deterministic() -> None:
    first = build_default_dataset()
    second = build_default_dataset()
    assert first == second
    assert first.qrels() == second.qrels()


def test_all_categories_are_represented_evenly() -> None:
    dataset = build_default_dataset()
    for category in QueryCategory:
        assert len(dataset.by_category(category)) == 40
    assert set(dataset.qrels()) == {query.query_id for query in dataset.queries}


def test_qrels_are_graded_and_primary_is_highest() -> None:
    dataset = build_default_dataset()
    for query in dataset.queries:
        labels = query.qrels()
        assert labels[query.primary_relevant_chunk_id] == 2
        assert all(value in (1, 2) for value in labels.values())
        assert query.primary_relevant_chunk_id in query.relevant_chunk_ids


def test_multilingual_queries_use_bengali() -> None:
    dataset = build_default_dataset()
    assert all(
        query.language == "bn"
        for query in dataset.by_category(QueryCategory.MULTILINGUAL)
    )
    assert all(
        query.language == "en"
        for query in dataset.queries
        if query.category is not QueryCategory.MULTILINGUAL
    )


def _records():
    return [
        {
            "query_id": "q001",
            "text": "latency",
            "category": "technical_term",
            "language": "en",
            "relevant_chunk_ids": ["c1"],
            "primary_relevant_chunk_id": "c1",
        }
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        {"query_id": ""},
        {"text": "   "},
        {"category": "unknown"},
        {"language": "xx"},
        {"relevant_chunk_ids": []},
        {"primary_relevant_chunk_id": "missing"},
        {"query_id": None},
        {"text": None},
        {"language": 123},
        {"relevant_chunk_ids": [123]},
        {"primary_relevant_chunk_id": None},
    ],
)
def test_invalid_records_are_rejected(mutation) -> None:
    records = _records()
    records[0].update(mutation)
    with pytest.raises(ValueError):
        validate_dataset(records, expected_size=1)


def test_duplicate_ids_are_rejected() -> None:
    records = _records()
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(records, expected_size=2)


def test_expected_size_is_enforced() -> None:
    with pytest.raises(ValueError, match="size"):
        validate_dataset(_records(), expected_size=2)
    assert validate_dataset(_records(), expected_size=1).size() == 1
