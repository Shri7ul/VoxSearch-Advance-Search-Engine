"""Goal 09 — transcript ingestion tests.

Two complementary styles, as required:

* **Mocked repositories** for precise orchestration assertions (which repository
  methods are called, with which arguments, and exactly one commit).
* **REAL SQLAlchemy ``Session``** + an isolated in-memory SQLite-compatible
  schema for genuine transaction / rollback / no-partial-persistence behavior.
  (The production models live in the ``voxsearch`` PostgreSQL schema with
  ``gen_random_uuid()`` server defaults and composite FKs, which SQLite cannot
  create; a faithful stand-in schema with the same relevant constraints is used
  instead — the same proven pattern as the Goal 08 race-recovery harness.)

No production Supabase connection is ever opened and no data is inserted into
it. No PostgreSQL/Docker is installed. Actual multi-connection PostgreSQL
concurrency is NOT tested here.
"""
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    create_engine,
    select,
)
from sqlalchemy.exc import IntegrityError, PendingRollbackError, SQLAlchemyError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from app.db.models import ContentRight
from app.repositories.content_rights_repository import ACTIVE_STATUS
from app.services import ContentRightsNotEligibleError
from app.services.transcript_ingestion_service import (
    COMPLETED_STATUS,
    PROCESSING_STATUS,
    TranscriptIngestionService,
    VideoNotFoundError,
)
from app.transcripts.validation import TranscriptValidationError

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
GRANTED = datetime(2020, 1, 1)  # naive -> treated as UTC by ContentRightsService


# --- Isolated SQLite-compatible stand-in schema -------------------------------


class _Base(DeclarativeBase):
    pass


class _Video(_Base):
    __tablename__ = "videos"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)


class _Right(_Base):
    __tablename__ = "content_rights"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    granted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class _Asset(_Base):
    __tablename__ = "transcript_assets"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    content_right_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class _Segment(_Base):
    __tablename__ = "transcript_segments"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    transcript_asset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    video_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    segment_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)

    __table_args__ = (
        # Mirrors the real uq_transcript_segments_asset_start and CHECKs.
        UniqueConstraint("transcript_asset_id", "start_seconds", name="uq_asset_start"),
        CheckConstraint("start_seconds >= 0 AND end_seconds > start_seconds", name="ck_times"),
        CheckConstraint("length(trim(segment_text)) > 0", name="ck_text_nonempty"),
    )


class _VideoRepo:
    def __init__(self, session):
        self.session = session

    def get_video_by_id(self, video_id):
        return self.session.get(_Video, video_id)


class _RightsRepo:
    def __init__(self, session):
        self.session = session

    def get_rights_by_id(self, rights_id):
        return self.session.get(_Right, rights_id)


class _AssetRepo:
    def __init__(self, session):
        self.session = session

    def create_transcript_asset(self, *, video_id, content_right_id, source_type, language, status):
        asset = _Asset(
            video_id=video_id, content_right_id=content_right_id,
            source_type=source_type, language=language, status=status,
        )
        self.session.add(asset)
        self.session.flush()  # real flush -> populates asset.id
        return asset

    def get_asset_by_id(self, asset_id):
        return self.session.get(_Asset, asset_id)

    def update_asset_status(self, asset_id, status):
        asset = self.get_asset_by_id(asset_id)
        if asset is None:
            return None
        asset.status = status
        self.session.flush()
        return asset


class _SegmentRepo:
    """Real-Session segment repository with optional failure injection.

    All inserts are REAL ``session.add()`` + ``session.flush()``. Failure
    injection is used only to simulate a mid-batch DB error; the resulting
    rollback is the genuine ``transaction()`` rollback on the real session.
    """

    def __init__(self, session, *, fail_on_index=None, failure=None,
                 duplicate_second_start=False):
        self.session = session
        self.fail_on_index = fail_on_index
        self.failure = failure
        self.duplicate_second_start = duplicate_second_start
        self.calls = 0
        self._first_start = None

    def create_segment(self, *, transcript_asset_id, video_id, start_seconds,
                       end_seconds, segment_text, language=None):
        self.calls += 1
        if self.calls == 1:
            self._first_start = start_seconds
        if self.fail_on_index == self.calls and self.failure is not None:
            raise self.failure
        effective_start = start_seconds
        if self.duplicate_second_start and self.calls == 2:
            effective_start = self._first_start  # real UNIQUE violation
        seg = _Segment(
            transcript_asset_id=transcript_asset_id, video_id=video_id,
            start_seconds=effective_start, end_seconds=end_seconds,
            segment_text=segment_text, language=language,
        )
        self.session.add(seg)
        self.session.flush()  # real flush -> real IntegrityError on violation
        return seg


class _Harness:
    def __init__(self, *, duration_seconds=60, rights=None, segment_repo_kwargs=None):
        self.engine = create_engine("sqlite://")
        _Base.metadata.create_all(self.engine)
        # Match the real SessionFactory: expire_on_commit=False, so the service
        # can read asset.id after commit without a lazy refresh.
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.video_id = uuid.uuid4()
        self.right_id = uuid.uuid4()
        rights = rights or {}
        with self.Session() as seed:
            seed.add(_Video(id=self.video_id, duration_seconds=duration_seconds))
            seed.add(_Right(
                id=self.right_id, video_id=rights.get("video_id", self.video_id),
                status=rights.get("status", "active"),
                granted_at=rights.get("granted_at", GRANTED),
                expires_at=rights.get("expires_at", None),
                revoked_at=rights.get("revoked_at", None),
            ))
            seed.commit()
        self.session = self.Session()
        self.videos = _VideoRepo(self.session)
        self.rights = _RightsRepo(self.session)
        self.assets = _AssetRepo(self.session)
        self.segments = _SegmentRepo(self.session, **(segment_repo_kwargs or {}))
        self.service = TranscriptIngestionService(
            self.session, videos=self.videos, rights=self.rights,
            assets=self.assets, segments=self.segments,
        )

    def payload(self, **overrides):
        base = {
            "video_id": str(self.video_id),
            "content_right_id": str(self.right_id),
            "source_type": "creator_provided",
            "language": "bn",
            "segments": [
                {"start_time": 0, "end_time": 2.0, "text": "  প্রথম  বাক্য  "},
                {"start_time": 2.0, "end_time": 4.5, "text": "Second one."},
            ],
        }
        base.update(overrides)
        return base

    def ingest(self, payload=None, **kwargs):
        return self.service.ingest(payload or self.payload(), now=NOW, **kwargs)

    def fresh_assets(self):
        with self.Session() as s:
            return list(s.execute(select(_Asset)).scalars().all())

    def fresh_segments(self):
        with self.Session() as s:
            return list(s.execute(select(_Segment)).scalars().all())

    def close(self):
        self.session.close()
        self.engine.dispose()


# --- Control: harness genuinely reproduces real failed-transaction semantics --


def test_control_real_failed_flush_requires_rollback() -> None:
    """A real failed flush poisons the transaction until an explicit rollback."""
    engine = create_engine("sqlite://")
    _Base.metadata.create_all(engine)
    aid = uuid.uuid4()
    with Session(engine) as seed:
        seed.add(_Segment(transcript_asset_id=aid, video_id=uuid.uuid4(),
                          start_seconds=0, end_seconds=2, segment_text="x"))
        seed.commit()
    s = Session(engine)
    s.add(_Segment(transcript_asset_id=aid, video_id=uuid.uuid4(),
                   start_seconds=0, end_seconds=2, segment_text="dup"))  # UNIQUE hit
    with pytest.raises(IntegrityError):
        s.flush()
    with pytest.raises(PendingRollbackError):
        s.execute(select(_Segment)).all()
    s.rollback()  # only explicit rollback clears the failed state
    assert len(s.execute(select(_Segment)).scalars().all()) == 1
    s.close()
    engine.dispose()


# --- Mocked orchestration assertions ------------------------------------------


def _mocked_service():
    session = MagicMock()
    videos, rights, assets, segments = (MagicMock() for _ in range(4))
    vid, rid = uuid.uuid4(), uuid.uuid4()
    video = MagicMock(); video.duration_seconds = 60
    videos.get_video_by_id.return_value = video
    # A real ContentRight instance so the fail-closed is_active() check passes.
    rights_row = ContentRight(
        video_id=vid, authorization_type="explicit_license",
        authorization_reference="ref", status=ACTIVE_STATUS,
        granted_at=GRANTED, expires_at=None, revoked_at=None,
    )
    rights.get_rights_by_id.return_value = rights_row
    asset = MagicMock(); asset.id = uuid.uuid4()
    assets.create_transcript_asset.return_value = asset
    segments.create_segment.side_effect = [MagicMock(), MagicMock()]
    assets.update_asset_status.return_value = asset
    service = TranscriptIngestionService(
        session, videos=videos, rights=rights, assets=assets, segments=segments,
    )
    return service, session, videos, rights, assets, segments, vid, rid


def test_orchestration_uses_one_transaction_and_status_lifecycle() -> None:
    service, session, _v, _r, assets, segments, vid, rid = _mocked_service()
    service.ingest({
        "video_id": str(vid), "content_right_id": str(rid),
        "source_type": "creator_provided", "language": "en",
        "segments": [
            {"start_time": 0, "end_time": 2, "text": "  hello  "},
            {"start_time": 2, "end_time": 4, "text": "world"},
        ],
    }, now=NOW)
    assert session.commit.call_count == 1  # exactly ONE transaction
    assert session.rollback.call_count == 0
    assert assets.create_transcript_asset.call_args.kwargs["status"] == PROCESSING_STATUS
    assert assets.update_asset_status.call_args.args[1] == COMPLETED_STATUS
    # normalized text is persisted; original is not lost (kept in the result).
    assert [c.kwargs["segment_text"] for c in segments.create_segment.call_args_list] == [
        "hello", "world",
    ]


# --- Real-session ingestion: success ------------------------------------------


def test_successful_ingestion_persists_asset_and_segments() -> None:
    h = _Harness()
    try:
        result = h.ingest()
        assert result.status == COMPLETED_STATUS
        assert result.segment_count == 2
        assert isinstance(result.asset_id, uuid.UUID)
        assets = h.fresh_assets()
        segments = h.fresh_segments()
        assert len(assets) == 1 and len(segments) == 2
        assert assets[0].status == COMPLETED_STATUS
        assert assets[0].video_id == h.video_id
        assert assets[0].content_right_id == h.right_id
        # Persisted text is the NORMALIZED form.
        texts = [s.segment_text for s in sorted(segments, key=lambda x: x.start_seconds)]
        assert texts == ["প্রথম বাক্য", "Second one."]
        assert all(s.video_id == h.video_id for s in segments)
        assert all(s.transcript_asset_id == assets[0].id for s in segments)
        assert all(s.language == "bn" for s in segments)
        # Ingestion result preserves BOTH original and normalized text.
        assert result.segments[0].original_text == "  প্রথম  বাক্য  "
        assert result.segments[0].normalized_text == "প্রথম বাক্য"
    finally:
        h.close()


def test_processing_status_is_transient_only_completed_persisted() -> None:
    h = _Harness()
    try:
        h.ingest()
        assert {a.status for a in h.fresh_assets()} == {COMPLETED_STATUS}
    finally:
        h.close()


def test_valid_overlap_is_persisted_verbatim() -> None:
    h = _Harness(duration_seconds=60)
    try:
        result = h.ingest(h.payload(segments=[
            {"start_time": 0, "end_time": 3, "text": "a"},
            {"start_time": 2, "end_time": 5, "text": "b"},
        ]))
        assert result.overlapping_pair_count == 1
        segs = sorted(h.fresh_segments(), key=lambda x: x.start_seconds)
        assert [(s.start_seconds, s.end_seconds) for s in segs] == [(0.0, 3.0), (2.0, 5.0)]
    finally:
        h.close()


# --- Real-session ingestion: authorization (fail closed) ---------------------


def test_missing_video_is_rejected_with_no_writes() -> None:
    h = _Harness()
    try:
        with pytest.raises(VideoNotFoundError):
            h.ingest(h.payload(video_id=str(uuid.uuid4())))
        assert h.fresh_assets() == [] and h.fresh_segments() == []
    finally:
        h.close()


def test_nonexistent_content_right_is_rejected_with_no_writes() -> None:
    h = _Harness()
    try:
        with pytest.raises(ContentRightsNotEligibleError):
            h.ingest(h.payload(content_right_id=str(uuid.uuid4())))
        assert h.fresh_assets() == [] and h.fresh_segments() == []
    finally:
        h.close()


@pytest.mark.parametrize("rights", [
    {"status": "revoked", "revoked_at": datetime(2025, 6, 1)},  # revoked
    {"status": "active", "expires_at": datetime(2025, 1, 1)},  # expired
    {"status": "pending"},                                       # inactive status
    {"status": "active", "granted_at": datetime(2030, 1, 1)},   # not yet granted
])
def test_unauthorized_rights_are_rejected_with_no_writes(rights) -> None:
    h = _Harness(rights=rights)
    try:
        with pytest.raises(ContentRightsNotEligibleError):
            h.ingest()
        assert h.fresh_assets() == [] and h.fresh_segments() == []
    finally:
        h.close()


def test_cross_video_rights_are_rejected_with_no_writes() -> None:
    h = _Harness(rights={"video_id": uuid.uuid4()})
    try:
        with pytest.raises(ContentRightsNotEligibleError):
            h.ingest()
        assert h.fresh_assets() == [] and h.fresh_segments() == []
    finally:
        h.close()


# --- Real-session ingestion: invalid transcript ------------------------------


def test_invalid_transcript_is_rejected_before_any_write() -> None:
    h = _Harness()
    try:
        with pytest.raises(ValidationError):
            h.ingest(h.payload(segments=[
                {"start_time": 0, "end_time": 2, "text": "   "},
            ]))
        assert h.fresh_assets() == [] and h.fresh_segments() == []
    finally:
        h.close()


def test_contextual_invalid_transcript_is_rejected_before_any_write() -> None:
    h = _Harness()
    try:
        with pytest.raises(TranscriptValidationError):
            h.ingest(h.payload(segments=[
                {"start_time": 5, "end_time": 6, "text": "a"},
                {"start_time": 2, "end_time": 3, "text": "b"},
            ]))
        assert h.fresh_assets() == [] and h.fresh_segments() == []
    finally:
        h.close()


def test_exceeding_known_duration_is_rejected_before_any_write() -> None:
    h = _Harness(duration_seconds=3)  # second segment ends at 4.5
    try:
        with pytest.raises(TranscriptValidationError):
            h.ingest()
        assert h.fresh_assets() == [] and h.fresh_segments() == []
    finally:
        h.close()


# --- Real-session ingestion: repeated ingestion ------------------------------


def test_repeated_ingestion_creates_distinct_assets_no_silent_dedup() -> None:
    h = _Harness()
    try:
        first = h.ingest()
        second = h.ingest()
        assert first.asset_id != second.asset_id
        assert len(h.fresh_assets()) == 2
        assert len(h.fresh_segments()) == 4
        assert {a.status for a in h.fresh_assets()} == {COMPLETED_STATUS}
    finally:
        h.close()


# --- Real-session ingestion: DB failure & atomic rollback --------------------


def test_db_failure_rolls_back_asset_and_all_segments() -> None:
    h = _Harness(segment_repo_kwargs={
        "fail_on_index": 2,
        "failure": SQLAlchemyError("simulated mid-batch DB failure"),
    })
    try:
        with pytest.raises(SQLAlchemyError):
            h.ingest()
        # The 1st segment WAS flushed, but the whole transaction rolled back.
        assert h.fresh_assets() == [] and h.fresh_segments() == []
    finally:
        h.close()


def test_no_partially_persisted_segments_after_failure() -> None:
    h = _Harness(segment_repo_kwargs={
        "fail_on_index": 2,
        "failure": RuntimeError("boom after first insert"),
    })
    try:
        with pytest.raises(RuntimeError):
            h.ingest()
        assert h.fresh_segments() == []  # not even the first segment survives
        assert h.fresh_assets() == []
    finally:
        h.close()


def test_real_integrity_error_rolls_back_everything() -> None:
    """A GENUINE DB constraint violation (no mock) also leaves zero rows."""
    h = _Harness(segment_repo_kwargs={"duplicate_second_start": True})
    try:
        with pytest.raises(IntegrityError):
            h.ingest()
        assert h.fresh_assets() == [] and h.fresh_segments() == []
    finally:
        h.close()


def test_session_is_usable_after_a_failed_ingestion() -> None:
    """Recovery (transaction rollback) leaves the real session usable."""
    h = _Harness(segment_repo_kwargs={
        "fail_on_index": 1,
        "failure": SQLAlchemyError("first insert failed"),
    })
    try:
        with pytest.raises(SQLAlchemyError):
            h.ingest()
        # A subsequent read through the SAME session works (no PendingRollbackError).
        assert h.session.execute(select(_Asset)).scalars().all() == []
        # And a subsequent successful ingestion works on the same session.
        h.segments.fail_on_index = None
        h.segments.failure = None
        result = h.ingest()
        assert result.status == COMPLETED_STATUS
        assert len(h.fresh_assets()) == 1 and len(h.fresh_segments()) == 2
    finally:
        h.close()
