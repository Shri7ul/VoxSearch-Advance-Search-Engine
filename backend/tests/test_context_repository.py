"""Goal 28 repository adapter tests for bounded context retrieval."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from app.reranking.context import ContextChunk, RepositoryContextProvider
from app.repositories.video_chunks import ChunkSearchResult, VideoChunkRepository
from app.services.rrf import FusedCandidate


def _candidate() -> FusedCandidate:
    chunk = ChunkSearchResult(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        transcript_asset_id=uuid.uuid4(),
        content_right_id=uuid.uuid4(),
        chunk_index=5,
        text="target text",
        start_ms=500,
        end_ms=900,
        duration_ms=400,
        timestamp_precision="word",
        word_count=2,
        boundary_reason="sentence",
        source_segment_refs=("seg",),
        overlap_sentence_indices=(),
        diagnostics=(),
        model_id="BAAI/bge-m3",
        distance=0.2,
    )
    return FusedCandidate(
        source="dense", source_rank=1, rrf_score=0.016, chunk=chunk
    )


def _compiled(session: MagicMock):
    stmt = session.execute.call_args.args[0]
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_list_context_chunks_is_bounded_and_read_only() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    asset_id, model = uuid.uuid4(), "BAAI/bge-m3"
    VideoChunkRepository(session).list_context_chunks(
        transcript_asset_id=asset_id, chunk_index=5, model_id=model, radius=2
    )
    sql = _compiled(session)
    assert "WHERE" in sql
    assert "chunk_index >=" in sql
    assert "chunk_index <=" in sql
    assert "chunk_index !=" in sql
    assert "ORDER BY abs(" in sql
    assert "LIMIT" in sql
    assert "embedding" not in sql
    session.commit.assert_not_called()


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_radius_is_rejected_at_repository(bad) -> None:
    session = MagicMock()
    with pytest.raises(ValueError):
        VideoChunkRepository(session).list_context_chunks(
            transcript_asset_id=uuid.uuid4(), chunk_index=5, radius=bad
        )
    session.execute.assert_not_called()


def test_repository_provider_maps_rows_to_context_chunks() -> None:
    repo = MagicMock()
    rows = [(4, "before", 0, 100), (6, "after", 200, 300)]
    repo.list_context_chunks.return_value = rows
    provider = RepositoryContextProvider(repo)
    candidate = _candidate()
    result = provider.neighbors(candidate, radius=1)
    target = candidate.chunk
    assert result == [
        ContextChunk(
            4, "before", 0, 100,
            target.video_id, target.transcript_asset_id, target.model_id,
        ),
        ContextChunk(
            6, "after", 200, 300,
            target.video_id, target.transcript_asset_id, target.model_id,
        ),
    ]
    kwargs = repo.list_context_chunks.call_args.kwargs
    assert kwargs["radius"] == 1
    assert kwargs["chunk_index"] == 5
    assert kwargs["model_id"] == "BAAI/bge-m3"
