"""OPT-IN live PostgreSQL + pgvector integration test for Goal 16.

Status: **NOT VERIFIED in this environment.** There is no local/test
PostgreSQL with the pgvector extension available here (no ``psql``, no
``docker``, no ``pgvector`` Python package, no numpy, ports 5432/5433/5434
closed). Consequently this real-DB test is SKIPPED BY DEFAULT and is NEVER run
as part of Goal 16 validation. Offline DDL rendering + mocked repository /
service tests are the only verification performed in this goal.

Operator control
----------------
This test does NOTHING unless BOTH conditions hold:

1. the pytest marker is explicitly selected (``-m live_pgvector``), and
2. the operator sets ``VOXSEARCH_LIVE_PGVECTOR=1`` AND supplies the target URL
   in ``VOXSEARCH_TEST_DATABASE_URL``.

It reads ONLY those two operator-supplied environment variables. It NEVER reads
``.env``, never reads any ``SUPABASE_*`` / credential variable, never prints the
URL or any credential, and never connects to production. When enabled it must
be pointed at a disposable, dedicated test database with pgvector installed and
with ``voxsearch.video_chunks`` plus the required parent rows already created.

What it verifies (when a real DB is available)
---------------------------------------------
A genuine end-to-end INSERT through ``VideoChunkRepository.upsert_batch`` into
a ``vector(1024)`` column: the ``CAST(... AS vector(1024))`` bind guard, the
1024-dimension round-trip, and the idempotent ON CONFLICT upsert. These are the
bind/INSERT behaviours that offline DDL rendering can NOT prove.
"""
import math
import os
import uuid

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db.models import VideoChunk
from app.db.types import DEFAULT_DIM
from app.repositories.video_chunks import VideoChunkRepository

# Skip the entire module unless the operator EXPLICITLY opts in. This is the
# single gate; no connection is ever attempted at import/collection time.
pytestmark = pytest.mark.live_pgvector

_ENABLED = os.environ.get("VOXSEARCH_LIVE_PGVECTOR") == "1"


def _require_live_db() -> str:
    if not _ENABLED:
        pytest.skip(
            "live pgvector NOT VERIFIED here: set VOXSEARCH_LIVE_PGVECTOR=1 and "
            "VOXSEARCH_TEST_DATABASE_URL to opt in; no local PostgreSQL+pgvector "
            "instance is available in this environment."
        )
    url = os.environ.get("VOXSEARCH_TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "VOXSEARCH_TEST_DATABASE_URL not supplied by operator; "
            "never reading .env or any credential source."
        )
    return url


def _row(aid: uuid.UUID, vid: uuid.UUID, rid: uuid.UUID, vector) -> dict:
    return {
        "id": uuid.uuid4(),
        "video_id": vid,
        "transcript_asset_id": aid,
        "content_right_id": rid,
        "chunk_index": 0,
        "text": "live smoke",
        "start_ms": 0,
        "end_ms": 1000,
        "duration_ms": 1000,
        "timestamp_precision": "segment_level",
        "word_count": 2,
        "boundary_reason": "sentence_limit",
        "source_segment_refs": [],
        "overlap_sentence_indices": [],
        "diagnostics": [],
        "model_id": "BAAI/bge-m3",
        "embedding": vector,
    }


def test_live_pgvector_insert_roundtrip_and_idempotent_upsert() -> None:
    """Real-DB insert + idempotent upsert; SKIPPED by default here (NOT VERIFIED).

    A single opt-in test so the default suite reports exactly ONE live_pgvector
    skip. When a real DB is available it proves: (1) a valid 1024-dim unit
    vector round-trips through ``vector(1024)`` (exercising the CAST bind
    guard), and (2) re-inserting the same upsert key REFRESHES rather than
    duplicates.
    """
    url = _require_live_db()
    engine = create_engine(url)
    try:
        unit_a = (1.0,) + (0.0,) * (DEFAULT_DIM - 1)
        unit_b = (0.0, 1.0) + (0.0,) * (DEFAULT_DIM - 2)
        aid, vid, rid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        with Session(engine) as session:
            repo = VideoChunkRepository(session)
            repo.upsert_batch([_row(aid, vid, rid, unit_a)])
            session.commit()

        # (1) valid 1024-dim vector round-trips, L2 norm preserved.
        with Session(engine) as session:
            stored = session.execute(
                select(VideoChunk.embedding).where(
                    VideoChunk.transcript_asset_id == aid
                )
            ).scalar_one()
            assert len(stored) == DEFAULT_DIM
            assert math.isclose(math.hypot(*stored), 1.0, abs_tol=1e-3)

        # (2) repeated insertion of the same key refreshes, never duplicates.
        with Session(engine) as session:
            VideoChunkRepository(session).upsert_batch(
                [_row(aid, vid, rid, unit_b)]
            )
            session.commit()

        with Session(engine) as session:
            count = session.execute(
                select(func.count()).select_from(VideoChunk).where(
                    VideoChunk.transcript_asset_id == aid
                )
            ).scalar_one()
            assert count == 1  # refreshed, not duplicated
            stored = session.execute(
                select(VideoChunk.embedding).where(
                    VideoChunk.transcript_asset_id == aid
                )
            ).scalar_one()
            assert stored[1] == pytest.approx(1.0)  # second vector won
    finally:
        engine.dispose()


def test_live_pgvector_cosine_similarity_search() -> None:
    """Real-DB cosine distance retrieval; SKIPPED by default here (NOT VERIFIED).

    When a real DB is available it proves: (1) pgvector <=> operator computes
    distance correctly in SQL, (2) ordering by distance ascending ranks the
    closest vector first, and (3) filtering by video_id / asset_id isolates
    the candidate set.
    """
    url = _require_live_db()
    engine = create_engine(url)
    try:
        unit_target = (1.0,) + (0.0,) * (DEFAULT_DIM - 1)
        unit_orthogonal = (0.0, 1.0) + (0.0,) * (DEFAULT_DIM - 2)
        aid, vid, rid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        row_target = _row(aid, vid, rid, unit_target)
        row_target["id"] = uuid.uuid4()
        row_target["chunk_index"] = 0
        row_target["text"] = "target chunk"

        row_ortho = _row(aid, vid, rid, unit_orthogonal)
        row_ortho["id"] = uuid.uuid4()
        row_ortho["chunk_index"] = 1
        row_ortho["text"] = "orthogonal chunk"

        with Session(engine) as session:
            repo = VideoChunkRepository(session)
            repo.upsert_batch([row_target, row_ortho])
            session.commit()

        with Session(engine) as session:
            repo = VideoChunkRepository(session)
            results = repo.search_similar(unit_target, top_k=5, video_id=vid)
            assert len(results) == 2
            # Target should be distance ~0.0 (identical)
            assert results[0].text == "target chunk"
            assert results[0].distance == pytest.approx(0.0, abs=1e-4)
            # Orthogonal should be distance ~1.0
            assert results[1].text == "orthogonal chunk"
            assert results[1].distance == pytest.approx(1.0, abs=1e-4)
    finally:
        engine.dispose()
