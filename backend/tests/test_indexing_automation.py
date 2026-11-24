"""Tests for Goal 42 — end-to-end indexing automation with injected boundaries."""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from app.embeddings.contracts import EmbeddingInferenceError
from app.transcripts.chunking import Chunk
from app.transcripts.sentence_segmentation import TimestampPrecision
from app.workers import (
    IndexingAutomationError,
    IndexingAutomationService,
    InvalidIndexingInputError,
    build_indexing_handler,
)
from app.workers.dispatcher import JobDispatcher
from app.workers.registry import TaskRegistry

VIDEO = uuid.uuid4()
ASSET = uuid.uuid4()
RIGHTS = uuid.uuid4()
UNIT = [1.0 / 32.0] * 1024

TRANSCRIPT = {
    "video_id": str(VIDEO),
    "content_right_id": str(RIGHTS),
    "source_type": "creator_provided",
    "language": "en",
    "segments": [
        {"start_time": 0.0, "end_time": 1.0, "text": "hello world"},
        {"start_time": 1.0, "end_time": 2.0, "text": "second segment"},
    ],
}
ALIGNED = [SimpleNamespace(index=0), SimpleNamespace(index=1)]


class FakeIngestion:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def ingest(self, payload):
        self.calls.append(payload)
        return SimpleNamespace(
            asset_id=ASSET, video_id=VIDEO, content_right_id=RIGHTS
        )


class FakeProvider:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[Any] = []
        self.fail = fail

    def embed(self, texts, *, config):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("model unavailable")
        return [list(UNIT) for _ in texts]


class FakePersistence:
    def __init__(self, count: int | None = None) -> None:
        self.calls: list[Any] = []
        self.count = count

    def persist(self, chunks, embeddings, *, model_id):
        self.calls.append((chunks, embeddings, model_id))
        return len(chunks) if self.count is None else self.count


def _chunks(aligned, parent):
    return tuple(
        Chunk(
            index=i,
            text=f"chunk {i}",
            sentences=(),
            start_ms=i * 1000,
            end_ms=(i + 1) * 1000,
            duration_ms=1000,
            timestamp_precision=TimestampPrecision.SEGMENT_LEVEL,
            word_count=2,
            evidence_start=0,
            boundary_reason="target_reached",
            source_segment_refs=(),
            overlap_sentence_indices=(),
            diagnostics=(),
            video_id=parent.video_id,
            transcript_asset_id=parent.transcript_asset_id,
            content_right_id=parent.content_right_id,
        )
        for i in range(2)
    )


def _service(
    ingestion=None, provider=None, persistence=None, chunker=None
):
    return IndexingAutomationService(
        ingestion=ingestion or FakeIngestion(),
        embedding_provider=provider or FakeProvider(),
        persistence=persistence or FakePersistence(),
        chunker=chunker or _chunks,
    )


# --- Automation -------------------------------------------------------------

def test_end_to_end_run_uses_real_contracts_and_preserves_order() -> None:
    ingestion, provider, persistence = FakeIngestion(), FakeProvider(), FakePersistence()
    result = _service(ingestion, provider, persistence).run(TRANSCRIPT, ALIGNED)
    assert result.asset_id == str(ASSET)
    assert result.video_id == str(VIDEO)
    assert result.chunk_count == 2
    assert result.persisted_count == 2
    assert result.model_id
    assert len(ingestion.calls) == 1
    assert provider.calls == [["chunk 0", "chunk 1"]]
    assert persistence.calls[0][2] == result.model_id


def test_invalid_transcript_payload_is_rejected_before_any_stage() -> None:
    ingestion, provider, persistence = FakeIngestion(), FakeProvider(), FakePersistence()
    with pytest.raises(Exception):
        _service(ingestion, provider, persistence).run(
            {"video_id": "bad"}, ALIGNED
        )
    assert ingestion.calls == []
    assert provider.calls == []
    assert persistence.calls == []


def test_empty_aligned_sentences_are_rejected() -> None:
    ingestion = FakeIngestion()
    with pytest.raises(InvalidIndexingInputError):
        _service(ingestion=ingestion).run(TRANSCRIPT, [])
    assert ingestion.calls == []


def test_empty_chunks_are_rejected() -> None:
    provider, persistence = FakeProvider(), FakePersistence()
    with pytest.raises(IndexingAutomationError):
        _service(provider=provider, persistence=persistence, chunker=lambda *a, **k: ()).run(
            TRANSCRIPT, ALIGNED
        )
    assert provider.calls == []
    assert persistence.calls == []


@pytest.mark.parametrize("count", [1, 3, True, "2"])
def test_persistence_count_mismatch_is_rejected(count) -> None:
    persistence = FakePersistence(count=count)
    with pytest.raises(IndexingAutomationError):
        _service(persistence=persistence).run(TRANSCRIPT, ALIGNED)


def test_malformed_ingestion_output_is_typed_and_stops_pipeline() -> None:
    class BadIngestion:
        def ingest(self, payload):
            return SimpleNamespace(
                asset_id="not-a-uuid", video_id=VIDEO, content_right_id=RIGHTS
            )

    persistence = FakePersistence()
    provider = FakeProvider()
    with pytest.raises(IndexingAutomationError, match="asset_id"):
        IndexingAutomationService(
            ingestion=BadIngestion(),
            embedding_provider=provider,
            persistence=persistence,
            chunker=_chunks,
        ).run(TRANSCRIPT, ALIGNED)
    assert provider.calls == []
    assert persistence.calls == []


def test_embedding_failure_propagates_without_persistence() -> None:
    persistence = FakePersistence()
    with pytest.raises(EmbeddingInferenceError):
        _service(provider=FakeProvider(fail=True), persistence=persistence).run(
            TRANSCRIPT, ALIGNED
        )
    assert persistence.calls == []


# --- Worker integration -----------------------------------------------------

def test_indexing_handler_registers_and_executes_as_background_job() -> None:
    service = _service()
    registry = TaskRegistry()
    registry.register(
        "index_transcript",
        build_indexing_handler(service, aligned_fetcher=lambda payload: ALIGNED),
    )
    dispatcher = JobDispatcher(registry)
    result = dispatcher.execute(
        {
            "job_id": str(uuid.uuid4()),
            "task": "index_transcript",
            "payload": {"transcript": TRANSCRIPT},
        }
    )
    assert result.output["chunk_count"] == 2
    assert result.output["persisted_count"] == 2


def test_indexing_handler_rejects_missing_payload_fields() -> None:
    handler = build_indexing_handler(_service())
    with pytest.raises(InvalidIndexingInputError):
        handler({"transcript": TRANSCRIPT})
