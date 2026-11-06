"""Real-Session verification of duplicate-insert transaction recovery (Goal 08).

Why this file exists
--------------------
``test_services.py`` exercises duplicate handling with a ``MagicMock`` Session.
A synthetic ``IntegrityError`` raised by a mock never marks a real SQLAlchemy
``Session`` as "failed / needs rollback", so those tests CANNOT detect a
``PendingRollbackError`` on the post-failure query. These tests use a REAL
``Session`` bound to in-memory SQLite and a REAL UNIQUE-constraint violation on
``flush()`` to reproduce the exact SQLAlchemy transaction state machine the
Goal 08 verification is concerned with.

Scope / honesty
---------------
* The production ``Channel`` / ``Video`` models live in a dedicated
  ``voxsearch`` PostgreSQL schema and use ``gen_random_uuid()`` server defaults,
  which SQLite cannot create. We therefore inject a lightweight,
  SQLite-compatible repository stub (with an equivalent UNIQUE constraint) into
  the REAL, UNMODIFIED ``ChannelService`` / ``VideoService``. The service's
  ``transaction`` context manager and narrow ``IntegrityError`` re-read are the
  genuine code under test.
* "Concurrency" is simulated deterministically: a winning row is committed
  first, and the first existence lookup is forced to miss so the subsequent
  INSERT collides with that committed row on ``flush()``. This reproduces the
  failed-flush state without threads or a second connection.
* This is NOT a live PostgreSQL concurrency test. No isolated PostgreSQL is
  available here (production Supabase is off-limits and PostgreSQL/Docker must
  not be installed), so true multi-connection PostgreSQL racing remains
  unproven; see the Goal 08 final verification report.
"""
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import CheckConstraint, String, Text, create_engine, select
from sqlalchemy.exc import IntegrityError, PendingRollbackError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.services import transaction
from app.services.channel_service import ChannelService
from app.services.video_service import VideoService


class _Base(DeclarativeBase):
    pass


class _Row(_Base):
    """SQLite stand-in: a UNIQUE external id + a CHECK for non-duplicate errors."""

    __tablename__ = "race_rows"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    youtube_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint("length(title) > 0", name="ck_race_rows_title_nonempty"),
    )


class _RaceRepo:
    """Real-Session repository stub implementing both services' method names.

    ``create_*`` performs a real ``session.add()`` + ``session.flush()`` so a
    duplicate raises a REAL ``IntegrityError`` and leaves the Session in the
    failed/needs-rollback state. ``get_*`` on the first call can be forced to
    miss (simulating a racer not yet visible); later calls run a REAL select.
    """

    def __init__(self, session: Session, *, first_lookup_misses: bool = False) -> None:
        self.session = session
        self._first_lookup_misses = first_lookup_misses
        self.lookup_calls = 0
        self.create_calls = 0

    def _lookup(self, youtube_id: str) -> _Row | None:
        self.lookup_calls += 1
        if self._first_lookup_misses and self.lookup_calls == 1:
            return None
        return self.session.execute(
            select(_Row).where(_Row.youtube_id == youtube_id)
        ).scalar_one_or_none()

    def _insert(self, youtube_id: str, title: str) -> _Row:
        self.create_calls += 1
        row = _Row(youtube_id=youtube_id, title=title)
        self.session.add(row)
        self.session.flush()  # real flush -> real IntegrityError on duplicate
        return row

    # ChannelService-facing
    def get_channel_by_youtube_id(self, youtube_channel_id: str) -> _Row | None:
        return self._lookup(youtube_channel_id)

    def create_channel(
        self, *, youtube_channel_id: str, title: str, description=None
    ) -> _Row:
        return self._insert(youtube_channel_id, title)

    # VideoService-facing
    def get_video_by_youtube_id(self, youtube_video_id: str) -> _Row | None:
        return self._lookup(youtube_video_id)

    def create_video(
        self,
        *,
        channel_id,
        youtube_video_id: str,
        title: str,
        description=None,
        duration_seconds=None,
        published_at=None,
    ) -> _Row:
        return self._insert(youtube_video_id, title)


class _Harness:
    """Builds a real SQLite Session and injects the stub into a real service."""

    def __init__(
        self, kind: str, *, first_lookup_misses=False, seeded_id=None
    ) -> None:
        self.engine = create_engine("sqlite://")
        _Base.metadata.create_all(self.engine)
        if seeded_id is not None:
            with Session(self.engine) as seed:
                seed.add(_Row(youtube_id=seeded_id, title="winner"))
                seed.commit()
        self.session = Session(self.engine)
        self.repo = _RaceRepo(self.session, first_lookup_misses=first_lookup_misses)
        self.kind = kind
        if kind == "channel":
            self.service = ChannelService(self.session, channels=self.repo)
        else:
            self.service = VideoService(self.session, videos=self.repo)

    def call(self, youtube_id: str, **kw):
        kwargs = {"title": "mine"}
        kwargs.update(kw)
        if self.kind == "channel":
            return self.service.create_or_get_channel(
                youtube_channel_id=youtube_id, **kwargs
            )
        return self.service.create_or_get_video(
            channel_id=uuid.uuid4(), youtube_video_id=youtube_id, **kwargs
        )

    def rows_now(self):
        """Query through the (possibly recovered) real session."""
        return list(self.session.execute(select(_Row)).scalars().all())

    def fresh_rows(self):
        """Query committed state from a separate session."""
        with Session(self.engine) as s:
            return list(s.execute(select(_Row)).scalars().all())

    def close(self) -> None:
        self.session.close()
        self.engine.dispose()


@pytest.fixture(params=["channel", "video"], ids=["channel", "video"])
def kind(request):
    return request.param


# --- Control: prove the harness genuinely reproduces the failed-transaction state


def test_control_real_failed_flush_requires_rollback() -> None:
    """A real failed flush poisons the transaction; a bare re-query fails.

    This is the exact failure mode a MagicMock cannot reproduce and the whole
    point of this verification. It proves the test bed is genuine: only an
    explicit ``rollback()`` clears the "needs rollback" state.
    """
    engine = create_engine("sqlite://")
    _Base.metadata.create_all(engine)
    with Session(engine) as seed:
        seed.add(_Row(youtube_id="UC", title="winner"))
        seed.commit()

    s = Session(engine)
    s.add(_Row(youtube_id="UC", title="loser"))
    with pytest.raises(IntegrityError):
        s.flush()
    with pytest.raises(PendingRollbackError):
        s.execute(select(_Row)).all()
    s.rollback()  # explicit recovery
    assert [r.title for r in s.execute(select(_Row)).scalars().all()] == ["winner"]
    s.close()
    engine.dispose()


# --- 1. Normal create succeeds ------------------------------------------------


def test_normal_create_commits_and_persists(kind) -> None:
    h = _Harness(kind)
    try:
        row = h.call("UC", title="my channel")
        assert row.youtube_id == "UC"
        assert row.title == "my channel"
        assert h.repo.create_calls == 1
        # Committed: visible from a fresh session (transaction() committed).
        persisted = h.fresh_rows()
        assert [r.youtube_id for r in persisted] == ["UC"]
        assert [r.title for r in persisted] == ["my channel"]
    finally:
        h.close()


# --- 2. Existing YouTube ID returns existing record ---------------------------


def test_existing_youtube_id_returns_existing_record_no_insert(kind) -> None:
    h = _Harness(kind, seeded_id="UC")
    try:
        row = h.call("UC", title="mine")
        assert row.title == "winner"  # pre-existing row, not a new one
        assert h.repo.create_calls == 0
        assert h.repo.lookup_calls == 1
        assert [r.title for r in h.fresh_rows()] == ["winner"]
    finally:
        h.close()


# --- 3. Concurrent duplicate insert does not leave session unusable -----------


def test_concurrent_duplicate_does_not_leave_session_unusable(kind) -> None:
    h = _Harness(kind, seeded_id="UC", first_lookup_misses=True)
    try:
        # If the recovery re-query ran inside the failed transaction, this call
        # would raise PendingRollbackError and the test would fail here.
        row = h.call("UC", title="mine")
        assert row.title == "winner"  # winning row returned
        assert h.repo.create_calls == 1
        assert h.repo.lookup_calls == 2  # initial miss + recovery re-read
        assert [r.title for r in h.rows_now()] == ["winner"]
    finally:
        h.close()


# --- 4. Duplicate recovery does not commit transaction independently ----------


def test_duplicate_recovery_does_not_commit_independently(kind) -> None:
    h = _Harness(kind, seeded_id="UC", first_lookup_misses=True)
    try:
        with patch.object(h.session, "commit", wraps=h.session.commit) as commit_spy:
            row = h.call("UC", title="mine")
        assert row.title == "winner"
        # transaction() goes straight to rollback() on the IntegrityError, and
        # the recovery path only SELECTs. It must NOT commit on its own.
        assert commit_spy.call_count == 0
        # The losing INSERT was rolled back, never committed.
        assert [r.title for r in h.fresh_rows()] == ["winner"]
    finally:
        h.close()


def test_duplicate_path_recovers_via_explicit_rollback(kind) -> None:
    """The re-query succeeds because ``transaction()`` explicitly rolls back.

    On a REAL session this asserts exactly one ``rollback()`` happens on the
    duplicate path (the recovery that clears the failed-flush state), with no
    commit. This is the "explicit rollback and transaction restart" strategy —
    no ``begin_nested()`` / savepoint is involved.
    """
    h = _Harness(kind, seeded_id="UC", first_lookup_misses=True)
    try:
        with patch.object(
            h.session, "rollback", wraps=h.session.rollback
        ) as rollback_spy, patch.object(
            h.session, "commit", wraps=h.session.commit
        ) as commit_spy:
            row = h.call("UC", title="mine")
        assert row.title == "winner"
        assert rollback_spy.call_count == 1
        assert commit_spy.call_count == 0
        assert [r.title for r in h.fresh_rows()] == ["winner"]
    finally:
        h.close()


# --- 5. Non-duplicate IntegrityError is re-raised -----------------------------


def test_non_duplicate_integrity_error_is_reraised(kind) -> None:
    # Empty title violates ck_race_rows_title_nonempty -> a REAL IntegrityError
    # that is NOT a duplicate-id collision. No winner row exists for "NEW", so
    # the recovery re-read returns None and the service MUST re-raise.
    h = _Harness(kind)
    try:
        with pytest.raises(IntegrityError):
            h.call("NEW", title="")
        assert h.repo.create_calls == 1
        # Rollback still happened -> session usable, losing row not persisted.
        assert h.fresh_rows() == []
        assert h.rows_now() == []
    finally:
        h.close()


# --- 6. Subsequent queries work after duplicate recovery ----------------------


def test_subsequent_queries_work_after_duplicate_recovery(kind) -> None:
    h = _Harness(kind, seeded_id="UC", first_lookup_misses=True)
    try:
        h.call("UC", title="mine")
        # A brand-new query AND a follow-up write transaction both work after
        # the duplicate recovery (no lingering PendingRollbackError).
        assert len(h.rows_now()) == 1
        with transaction(h.session):
            h.session.add(_Row(youtube_id="SECOND", title="second"))
        assert sorted(r.youtube_id for r in h.fresh_rows()) == ["SECOND", "UC"]
    finally:
        h.close()


# --- 7. Outer transaction rollback still works --------------------------------


def test_outer_transaction_rollback_still_works(kind) -> None:
    h = _Harness(kind, seeded_id="UC", first_lookup_misses=True)
    try:
        h.call("UC", title="mine")  # duplicate recovered, session usable
        # A later transaction that fails must still roll back cleanly: the
        # duplicate recovery did NOT corrupt the transaction machinery.
        with pytest.raises(RuntimeError):
            with transaction(h.session):
                h.session.add(_Row(youtube_id="TEMP", title="temp"))
                raise RuntimeError("boom")
        assert [r.youtube_id for r in h.fresh_rows()] == ["UC"]
        assert [r.youtube_id for r in h.rows_now()] == ["UC"]
    finally:
        h.close()
