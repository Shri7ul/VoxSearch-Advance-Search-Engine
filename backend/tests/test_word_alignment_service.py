"""Goal 12 — WordAlignmentService rights / transaction / retry tests.

Covers the section-16 "Rights and transaction tests":
missing / inactive / expired / revoked authorization, rights revoked between
attempt and persistence, atomic rollback, no orphan words, no partial word set
for a failed segment, failed retry preserving a previous success, successful
retry atomically replacing the previous current word set, and concurrent
attempt safety.

Harness (mirrors the proven Goal 08/09 pattern)
-----------------------------------------------
REAL SQLAlchemy ``Session`` (``expire_on_commit=False``, exactly like
production) over an isolated in-memory SQLite-compatible stand-in schema, with
faithful stand-in repositories injected into the REAL
:class:`WordAlignmentService`. The provider is ALWAYS a FAKE callable. This
gives genuine transaction / rollback / persistence behavior without a
production DB, torch, or whisperx.

Honesty boundaries (also reported in the Goal 12 final report):
* SQLite stand-ins are a BEHAVIORAL simulation of the composite ownership
  constraints; they are NOT proof of PostgreSQL composite-FK / partial-index
  correctness (see test_word_alignment_migration.py for offline DDL review).
* The concurrent-winner test injects a controlled ``IntegrityError`` to
  exercise the service's race-handling branch; it is NOT a true
  multi-connection PostgreSQL race (no isolated PostgreSQL is available).
* No live WhisperX / model weights are used anywhere.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    create_engine,
    delete,
    func,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from app.repositories.content_rights_repository import ACTIVE_STATUS
from app.services import (
    ContentRightsNotEligibleError,
    TranscriptAssetNotFoundError,
    TranscriptAssetOwnershipError,
)
from app.services.word_alignment_service import (
    AlignmentVideoNotFoundError,
    RegisteredAudioAsset,
    SourceAudioAssetNotFoundError,
    UnsupportedAlignmentLanguageError,
    WordAlignmentService,
)
from app.transcripts.word_contract import SourceAudioAssetInput

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
GRANTED = datetime(2020, 1, 1)  # naive -> treated as UTC by ContentRightsService
OFFSET_MS = 10_000            # video = audio + 10_000 ms
DURATION_MS = 60_000
_CHECKSUM = "a" * 64


def _audio_input(video_id, content_right_id, **overrides) -> SourceAudioAssetInput:
    """Build a spec-compliant SourceAudioAssetInput for registration tests."""
    base = dict(
        video_id=video_id,
        content_right_id=content_right_id,
        storage_uri="bucket/authorized.wav",
        checksum_sha256=_CHECKSUM,
        mime_type="audio/wav",
        rights_basis="explicit-license",
        timeline_offset_ms=OFFSET_MS,
        duration_ms=DURATION_MS,
    )
    base.update(overrides)
    return SourceAudioAssetInput(**base)


# --- Isolated SQLite-compatible stand-in schema -------------------------------

class _Base(DeclarativeBase):
    pass


class _Video(_Base):
    __tablename__ = "videos"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


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
    language: Mapped[str] = mapped_column(String(16), nullable=False)


class _Segment(_Base):
    __tablename__ = "transcript_segments"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    transcript_asset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    video_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    segment_text: Mapped[str] = mapped_column(Text, nullable=False)
    original_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)


class _Audio(_Base):
    __tablename__ = "source_audio_assets"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    content_right_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    # Spec-compliant (design-freeze §3.1): opaque storage_uri + provenance.
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False, default="bucket/clip.wav")
    checksum_sha256: Mapped[str] = mapped_column(Text, nullable=False, default="a" * 64)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False, default="audio/wav")
    rights_basis: Mapped[str] = mapped_column(Text, nullable=False, default="explicit-license")
    sample_rate_hz: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timeline_offset_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)


class _Run(_Base):
    __tablename__ = "transcript_alignment_runs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    transcript_asset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    source_audio_asset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    content_right_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Spec-compliant audit metadata (design-freeze §3.2). Defaults on the
    # NOT NULL columns keep direct competitor construction in the race test
    # working without specifying them.
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="whisperx")
    provider_version: Mapped[str] = mapped_column(String(32), nullable=False, default="3.8.6")
    align_model_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    segment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    aligned_segment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    skipped_segment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class _Word(_Base):
    __tablename__ = "transcript_words"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    alignment_run_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    segment_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    transcript_asset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    video_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    # Spec-compliant column name (§3.3 / §7 step 8: word_index = ordinal).
    word_index: Mapped[int] = mapped_column(Integer, nullable=False)
    word: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "alignment_run_id", "segment_id", "word_index",
            name="uq_run_segment_word_index",
        ),
    )


# The partial UNIQUE index (one current success per asset+audio pair) is created
# via raw DDL because SQLAlchemy's postgresql_where is dialect-specific.
_PARTIAL_INDEX_DDL = (
    "CREATE UNIQUE INDEX uq_one_current_success "
    "ON transcript_alignment_runs (transcript_asset_id, source_audio_asset_id) "
    "WHERE status IN ('completed', 'partial')"
)


# --- Faithful stand-in repositories (real flush, never commit) ---------------

class _VideoRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def get_video_by_id(self, vid: uuid.UUID) -> _Video | None:
        return self.s.get(_Video, vid)


class _RightsRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def get_rights_by_id(self, rid: uuid.UUID) -> _Right | None:
        return self.s.get(_Right, rid)

    def get_rights_by_id_fresh(self, rid: uuid.UUID) -> _Right | None:
        return self.s.scalars(
            select(_Right)
            .where(_Right.id == rid)
            .execution_options(populate_existing=True)
        ).first()


class _AssetRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def get_asset_by_id(self, aid: uuid.UUID) -> _Asset | None:
        return self.s.get(_Asset, aid)


class _SegmentRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def list_all_segments_by_asset(self, aid: uuid.UUID) -> list[_Segment]:
        return list(
            self.s.scalars(
                select(_Segment)
                .where(_Segment.transcript_asset_id == aid)
                .order_by(_Segment.start_seconds.asc(), _Segment.id.asc())
            ).all()
        )


class _AudioRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def get_audio_asset_by_id(self, aid: uuid.UUID) -> _Audio | None:
        return self.s.get(_Audio, aid)

    def create_source_audio_asset(
        self, *, video_id, content_right_id, storage_uri, checksum_sha256,
        mime_type, rights_basis, sample_rate_hz, timeline_offset_ms, duration_ms,
    ) -> _Audio:
        a = _Audio(
            video_id=video_id, content_right_id=content_right_id,
            storage_uri=storage_uri, checksum_sha256=checksum_sha256,
            mime_type=mime_type, rights_basis=rights_basis,
            sample_rate_hz=sample_rate_hz,
            timeline_offset_ms=timeline_offset_ms, duration_ms=duration_ms,
        )
        self.s.add(a)
        self.s.flush()
        return a


class _RunRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def next_attempt_number(
        self, transcript_asset_id, source_audio_asset_id, *, provider,
        provider_version,
    ) -> int:
        current = self.s.scalar(
            select(func.max(_Run.attempt_number))
            .where(_Run.transcript_asset_id == transcript_asset_id)
            .where(_Run.source_audio_asset_id == source_audio_asset_id)
            .where(_Run.provider == provider)
            .where(_Run.provider_version == provider_version)
        )
        return int(current or 0) + 1

    def create_run(
        self, *, video_id, transcript_asset_id, source_audio_asset_id,
        content_right_id, language, status, attempt_number, provider,
        provider_version, audio_duration_ms=None, align_model_name=None,
        started_at=None,
    ) -> _Run:
        r = _Run(
            video_id=video_id, transcript_asset_id=transcript_asset_id,
            source_audio_asset_id=source_audio_asset_id,
            content_right_id=content_right_id, language=language,
            status=status, word_count=0,
            attempt_number=attempt_number, provider=provider,
            provider_version=provider_version,
            audio_duration_ms=audio_duration_ms,
            align_model_name=align_model_name, started_at=started_at,
        )
        self.s.add(r)
        self.s.flush()
        return r

    def get_run_by_id(self, rid: uuid.UUID) -> _Run | None:
        return self.s.get(_Run, rid)

    def find_current_success(
        self, asset_id, audio_id, *, exclude_run_id=None
    ) -> _Run | None:
        stmt = (
            select(_Run)
            .where(_Run.transcript_asset_id == asset_id)
            .where(_Run.source_audio_asset_id == audio_id)
            .where(_Run.status.in_(("completed", "partial")))
        )
        if exclude_run_id is not None:
            stmt = stmt.where(_Run.id != exclude_run_id)
        return self.s.scalars(stmt.order_by(_Run.id.asc())).first()

    def set_run_outcome(
        self, rid, *, status, failure_reason=None, word_count=None,
        segment_count=None, aligned_segment_count=None,
        skipped_segment_count=None, finished_at=None,
    ) -> _Run | None:
        r = self.get_run_by_id(rid)
        if r is None:
            return None
        r.status = status
        r.failure_reason = failure_reason
        if word_count is not None:
            r.word_count = word_count
        if segment_count is not None:
            r.segment_count = segment_count
        if aligned_segment_count is not None:
            r.aligned_segment_count = aligned_segment_count
        if skipped_segment_count is not None:
            r.skipped_segment_count = skipped_segment_count
        if finished_at is not None:
            r.finished_at = finished_at
        self.s.flush()
        return r

    def supersede_run(self, rid) -> _Run | None:
        r = self.get_run_by_id(rid)
        if r is None:
            return None
        r.status = "superseded"
        self.s.flush()
        return r


class _WordRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def add_words(
        self, words, *, alignment_run_id, transcript_asset_id, video_id
    ) -> int:
        for w in words:
            self.s.add(
                _Word(
                    alignment_run_id=alignment_run_id,
                    segment_id=w.segment_id,
                    transcript_asset_id=transcript_asset_id,
                    video_id=video_id,
                    word_index=w.ordinal, word=w.text,
                    char_start=w.char_start, char_end=w.char_end,
                    start_ms=w.start_ms, end_ms=w.end_ms,
                )
            )
        self.s.flush()
        return len(words)

    def delete_words_by_run(self, rid: uuid.UUID) -> int:
        result = self.s.execute(
            delete(_Word).where(_Word.alignment_run_id == rid)
        )
        self.s.flush()
        return result.rowcount or 0

    def count_words_by_run(self, rid: uuid.UUID) -> int:
        return int(
            self.s.scalar(
                select(func.count()).select_from(_Word).where(
                    _Word.alignment_run_id == rid
                )
            )
            or 0
        )

    def list_words_by_run(self, rid: uuid.UUID) -> list[_Word]:
        return list(
            self.s.scalars(
                select(_Word)
                .where(_Word.alignment_run_id == rid)
                .order_by(_Word.segment_id.asc(), _Word.word_index.asc())
            ).all()
        )

    def list_words_by_asset(self, aid: uuid.UUID) -> list[_Word]:
        return list(
            self.s.scalars(
                select(_Word).where(_Word.transcript_asset_id == aid)
            ).all()
        )


# --- Fake provider ------------------------------------------------------------

def _success_words(seg: dict, *, bias: float = 0.0) -> list[dict]:
    """Build a fully valid word list in AUDIO-timeline seconds for a segment."""
    tokens = seg["tokens"]
    n = len(tokens)
    start, end = seg["start"], seg["end"]
    span = (end - start) / n
    out: list[dict] = []
    for i, tok in enumerate(tokens):
        ws = start + i * span + bias
        we = ws + span * 0.8
        out.append(
            {
                "word": tok["text"],
                "start": ws, "end": we,
                "char_start": ws, "char_end": we,
                "char_corroborated": True,
            }
        )
    return out


class FakeProvider:
    """Configurable FAKE provider; records every call for zero-call assertions."""

    def __init__(
        self,
        *,
        bias: float = 0.0,
        provider_error: str | None = None,
        drop_segment: bool = False,
        break_first_word: bool = False,
        on_call=None,
    ) -> None:
        self.bias = bias
        self.provider_error = provider_error
        self.drop_segment = drop_segment
        self.break_first_word = break_first_word
        self.on_call = on_call
        self.calls: list[dict] = []

    def __call__(self, request: dict) -> dict:
        self.calls.append(request)
        if self.on_call is not None:
            self.on_call()
        if self.provider_error:
            return {"segments": [], "provider_error": self.provider_error}
        segments = []
        for seg in request["segments"]:
            if self.drop_segment:
                words: list[dict] = []
            else:
                words = _success_words(seg, bias=self.bias)
            if self.break_first_word and words:
                # Remove the timing of the first word -> validation must skip
                # the WHOLE segment (no partial word set).
                words[0] = {
                    "word": words[0]["word"],
                    "char_corroborated": True,
                }
            segments.append(
                {"segment_id": seg["segment_id"], "words": words, "error": None}
            )
        return {"segments": segments, "provider_error": None}


# --- Harness ------------------------------------------------------------------

class Harness:
    def __init__(self) -> None:
        self.engine = create_engine("sqlite://")
        _Base.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            conn.execute(text(_PARTIAL_INDEX_DDL))
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.s: Session = self.Session()

    def service(self, *, runs: _RunRepo | None = None, words: _WordRepo | None = None) -> WordAlignmentService:
        return WordAlignmentService(
            self.s,
            videos=_VideoRepo(self.s),
            rights=_RightsRepo(self.s),
            assets=_AssetRepo(self.s),
            segments=_SegmentRepo(self.s),
            audio_assets=_AudioRepo(self.s),
            runs=runs or _RunRepo(self.s),
            words=words or _WordRepo(self.s),
        )

    # -- seeding --
    def seed(
        self,
        *,
        right_status: str = ACTIVE_STATUS,
        expires_at: datetime | None = None,
        revoked_at: datetime | None = None,
        granted_at: datetime | None = GRANTED,
        original_text: str | None = "Hello world. Second sentence!",
        with_video: bool = True,
        with_right: bool = True,
        with_asset: bool = True,
        with_audio: bool = True,
        audio_right_id: uuid.UUID | None = None,
        audio_video_id: uuid.UUID | None = None,
        asset_video_id: uuid.UUID | None = None,
        offset_ms: int = OFFSET_MS,
        duration_ms: int = DURATION_MS,
        extra_null_segment: bool = False,
    ) -> dict:
        vid = uuid.uuid4()
        rid = uuid.uuid4()
        aid = uuid.uuid4()
        audio_id = uuid.uuid4()
        seg_id = uuid.uuid4()
        if with_video:
            self.s.add(_Video(id=vid))
        if with_right:
            self.s.add(
                _Right(
                    id=rid, video_id=vid, status=right_status,
                    granted_at=granted_at, expires_at=expires_at,
                    revoked_at=revoked_at,
                )
            )
        if with_asset:
            self.s.add(
                _Asset(
                    id=aid, video_id=asset_video_id or vid,
                    content_right_id=rid, language="en",
                )
            )
        self.s.add(
            _Segment(
                id=seg_id, transcript_asset_id=aid, video_id=vid,
                start_seconds=10.0, end_seconds=12.0,
                segment_text="Hello world. Second sentence!",
                original_text=original_text, language="en",
            )
        )
        if extra_null_segment:
            self.s.add(
                _Segment(
                    id=uuid.uuid4(), transcript_asset_id=aid, video_id=vid,
                    start_seconds=12.0, end_seconds=14.0,
                    segment_text="no original", original_text=None,
                    language="en",
                )
            )
        if with_audio:
            self.s.add(
                _Audio(
                    id=audio_id, video_id=audio_video_id or vid,
                    content_right_id=audio_right_id or rid,
                    storage_uri="bucket/clip.wav",
                    checksum_sha256="a" * 64,
                    mime_type="audio/wav", rights_basis="explicit-license",
                    sample_rate_hz=None,
                    timeline_offset_ms=offset_ms, duration_ms=duration_ms,
                )
            )
        self.s.commit()
        return dict(
            video_id=vid, right_id=rid, asset_id=aid,
            audio_id=audio_id, segment_id=seg_id,
        )

    # -- introspection --
    def runs(self) -> list[_Run]:
        return list(self.s.scalars(select(_Run)).all())

    def words(self) -> list[_Word]:
        return list(self.s.scalars(select(_Word)).all())

    def count_runs(self) -> int:
        return len(self.runs())

    def count_words(self) -> int:
        return len(self.words())

    def words_for(self, rid: uuid.UUID) -> list[_Word]:
        return list(
            self.s.scalars(
                select(_Word).where(_Word.alignment_run_id == rid)
            ).all()
        )

    def revoke_out_of_band(self, rid: uuid.UUID) -> None:
        """Raw UPDATE bypassing the ORM identity map (simulates an external
        revocation during the out-of-transaction provider call)."""
        self.s.execute(
            update(_Right)
            .where(_Right.id == rid)
            .values(status="revoked", revoked_at=datetime(2026, 1, 1))
            .execution_options(synchronize_session=False)
        )
        self.s.commit()

    def close(self) -> None:
        self.s.close()
        self.engine.dispose()


@pytest.fixture()
def h() -> Harness:
    harness = Harness()
    yield harness
    harness.close()


def _align(h: Harness, ids: dict, provider, *, language: str = "en", **kw):
    return h.service(**kw).align(
        transcript_asset_id=ids["asset_id"],
        source_audio_asset_id=ids["audio_id"],
        video_id=ids["video_id"],
        language=language,
        provider=provider,
        now=NOW,
    )


# --- Initial gate rejections: NO run, NO words, NO provider calls -------------

class TestInitialGateRejections:
    """Every initial rejection creates ZERO alignment runs and ZERO word rows."""

    def _assert_no_writes(self, h: Harness, provider: FakeProvider) -> None:
        assert provider.calls == []       # zero provider calls
        assert h.count_runs() == 0        # zero alignment runs
        assert h.count_words() == 0       # zero word rows

    def test_missing_video(self, h: Harness) -> None:
        ids = h.seed(with_video=False)
        p = FakeProvider()
        with pytest.raises(AlignmentVideoNotFoundError):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    def test_missing_transcript_asset(self, h: Harness) -> None:
        ids = h.seed(with_asset=False)
        p = FakeProvider()
        with pytest.raises(Exception):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    def test_asset_belongs_to_other_video(self, h: Harness) -> None:
        ids = h.seed(asset_video_id=uuid.uuid4())
        p = FakeProvider()
        with pytest.raises(ContentRightsNotEligibleError.__mro__[1]):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    def test_missing_audio_asset(self, h: Harness) -> None:
        ids = h.seed(with_audio=False)
        p = FakeProvider()
        with pytest.raises(SourceAudioAssetNotFoundError):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    def test_audio_belongs_to_other_video(self, h: Harness) -> None:
        ids = h.seed(audio_video_id=uuid.uuid4())
        p = FakeProvider()
        with pytest.raises(Exception):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    def test_mismatched_rights_lineage(self, h: Harness) -> None:
        # Audio authorized by a DIFFERENT right than the transcript asset.
        ids = h.seed(audio_right_id=uuid.uuid4())
        p = FakeProvider()
        with pytest.raises(ContentRightsNotEligibleError):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    def test_missing_authorization(self, h: Harness) -> None:
        ids = h.seed(with_right=False)
        p = FakeProvider()
        with pytest.raises(ContentRightsNotEligibleError):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    def test_inactive_authorization(self, h: Harness) -> None:
        ids = h.seed(right_status="pending")
        p = FakeProvider()
        with pytest.raises(ContentRightsNotEligibleError):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    def test_expired_authorization(self, h: Harness) -> None:
        ids = h.seed(expires_at=datetime(2025, 1, 1))
        p = FakeProvider()
        with pytest.raises(ContentRightsNotEligibleError):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    def test_revoked_authorization(self, h: Harness) -> None:
        ids = h.seed(right_status="revoked", revoked_at=datetime(2025, 6, 1))
        p = FakeProvider()
        with pytest.raises(ContentRightsNotEligibleError):
            _align(h, ids, p)
        self._assert_no_writes(h, p)

    @pytest.mark.parametrize("code", ["bn", "ben", "bn-BD", "bn-IN", "BN-BD"])
    def test_bengali_zero_provider_calls_zero_rows(self, h: Harness, code: str) -> None:
        ids = h.seed()
        p = FakeProvider()
        with pytest.raises(UnsupportedAlignmentLanguageError):
            _align(h, ids, p, language=code)
        self._assert_no_writes(h, p)

    @pytest.mark.parametrize("code", ["xx", "unknown", "en-US"])
    def test_unknown_language_zero_provider_calls_zero_rows(
        self, h: Harness, code: str
    ) -> None:
        ids = h.seed()
        p = FakeProvider()
        with pytest.raises(UnsupportedAlignmentLanguageError):
            _align(h, ids, p, language=code)
        self._assert_no_writes(h, p)


# --- Successful / partial / failed alignment outcomes ------------------------

class TestAlignmentOutcomes:
    def test_success_persists_words_in_video_timeline(self, h: Harness) -> None:
        ids = h.seed()
        p = FakeProvider()
        result = _align(h, ids, p)
        assert result.status == "completed"
        assert result.word_count == 4
        assert result.failure_reason is None
        assert result.superseded_run_id is None
        assert len(p.calls) == 1
        assert h.count_runs() == 1
        run = h.runs()[0]
        assert run.status == "completed"
        assert run.word_count == 4
        words = h.words()
        assert len(words) == 4
        # Provider timings are AUDIO seconds within [0,2]; persisted start_ms
        # MUST include the 10_000 ms offset (video timeline).
        assert all(w.start_ms >= OFFSET_MS for w in words)
        assert all(w.start_ms < w.end_ms for w in words)
        assert all(w.alignment_run_id == run.id for w in words)
        assert sorted(w.word for w in words) == [
            "Hello", "Second", "sentence!", "world."
        ]

    def test_partial_when_some_segments_skipped(self, h: Harness) -> None:
        # One alignable segment + one segment with NULL original_text (skipped).
        ids = h.seed(extra_null_segment=True)
        result = _align(h, ids, FakeProvider())
        assert result.status == "partial"
        assert any(s.reason == "null_original_text" for s in result.skipped)
        assert h.runs()[0].word_count == 4

    def test_provider_error_yields_failed_zero_words(self, h: Harness) -> None:
        ids = h.seed()
        result = _align(h, ids, FakeProvider(provider_error="provider_error"))
        assert result.status == "failed"
        assert result.failure_reason == "provider_error"
        assert result.word_count == 0
        assert h.count_words() == 0

    def test_provider_language_error_classified(self, h: Harness) -> None:
        ids = h.seed()
        result = _align(
            h, ids,
            FakeProvider(provider_error="No align-model for language xx"),
        )
        assert result.status == "unsupported_language"
        assert result.failure_reason == "unsupported_language"
        assert h.count_words() == 0

    def test_provider_exception_fails_closed(self, h: Harness) -> None:
        ids = h.seed()

        def exploding(_request: dict) -> dict:
            raise RuntimeError("provider blew up")

        result = _align(h, ids, exploding)
        assert result.status == "failed"
        assert result.failure_reason == "provider_error"
        assert h.count_words() == 0

    def test_zero_word_result_is_failed(self, h: Harness) -> None:
        # Provider returns a segment with NO words -> token count mismatch ->
        # the whole segment is skipped -> zero-word result -> failed.
        ids = h.seed()
        result = _align(h, ids, FakeProvider(drop_segment=True))
        assert result.status == "failed"
        assert result.word_count == 0
        assert h.count_words() == 0

    def test_no_partial_word_set_for_failed_segment(self, h: Harness) -> None:
        # One word loses its timing -> validation skips the ENTIRE segment,
        # never persist a partial/re-indexed subset.
        ids = h.seed()
        result = _align(h, ids, FakeProvider(break_first_word=True))
        assert result.status == "failed"
        assert h.count_words() == 0

    def test_no_alignable_segments_no_provider_call(self, h: Harness) -> None:
        # A NULL original_text is the only segment -> skipped before provider.
        ids = h.seed(original_text=None)
        p = FakeProvider()
        result = _align(h, ids, p)
        assert result.status == "failed"
        assert result.failure_reason == "all_segments_skipped"
        assert p.calls == []              # NO provider call
        assert h.count_words() == 0


# --- Rights revoked DURING alignment (Phase B recheck) -----------------------

class TestRevocationDuringAlignment:
    def test_rights_revoked_preserves_previous_success(self, h: Harness) -> None:
        ids = h.seed()
        # First (successful) alignment.
        first = _align(h, ids, FakeProvider())
        assert first.status == "completed"
        first_id = first.run_id
        assert h.count_words() == 4

        # Second attempt: the FAKE provider revokes the right OUT OF BAND while
        # it runs (between the Phase A commit and Phase B persistence).
        revoker = FakeProvider(on_call=lambda: h.revoke_out_of_band(ids["right_id"]))
        second = _align(h, ids, revoker)
        assert revoker.calls, "provider should have been invoked before recheck"
        assert second.status == "failed"
        assert second.failure_reason == "rights_revoked"
        assert second.word_count == 0
        # No NEW word rows; the previous successful alignment is preserved.
        assert len(h.words_for(first_id)) == 4
        assert len(h.words_for(second.run_id)) == 0
        assert h.count_words() == 4
        # Previous run is STILL the one current success.
        assert h.runs()[0].status == "completed"

    def test_fresh_read_detects_revocation(self, h: Harness) -> None:
        # Direct proof that get_rights_by_id_fresh (populate_existing) sees an
        # out-of-band revocation that a plain identity-map get() would hide.
        ids = h.seed()
        svc = h.service()
        assert svc.rights.get_rights_by_id_fresh(ids["right_id"]) is not None
        # Load + cache the active object, then revoke out of band.
        cached = svc.rights.get_rights_by_id(ids["right_id"])
        assert cached is not None and cached.status == ACTIVE_STATUS
        h.revoke_out_of_band(ids["right_id"])
        assert svc.rights.get_rights_by_id(ids["right_id"]).status == "active"  # stale
        fresh = svc.rights.get_rights_by_id_fresh(ids["right_id"])
        assert fresh is not None and fresh.status == "revoked"  # fresh wins


# --- Retry safety -------------------------------------------------------------

class TestRetrySafety:
    def test_failed_retry_preserves_previous_success(self, h: Harness) -> None:
        ids = h.seed()
        first = _align(h, ids, FakeProvider())
        assert first.status == "completed"

        # A FAILED retry (provider error) must NOT destroy the earlier success.
        second = _align(h, ids, FakeProvider(provider_error="provider_error"))
        assert second.status == "failed"
        assert second.word_count == 0
        assert len(h.words_for(first.run_id)) == 4       # preserved
        assert len(h.words_for(second.run_id)) == 0      # no new words
        assert h.count_words() == 4
        assert h.runs()[0].status == "completed"          # NOT superseded

    def test_successful_retry_atomically_replaces(self, h: Harness) -> None:
        ids = h.seed()
        first = _align(h, ids, FakeProvider(bias=0.0))
        first_starts = sorted(w.start_ms for w in h.words_for(first.run_id))

        # Second, DIFFERENT valid result (bias shifts video-timeline ms).
        second = _align(h, ids, FakeProvider(bias=0.1))
        assert second.status == "completed"
        assert second.superseded_run_id == first.run_id

        old = next(r for r in h.runs() if r.id == first.run_id)
        new = next(r for r in h.runs() if r.id == second.run_id)
        assert old.status == "superseded"
        assert new.status == "completed"
        # Old word set deleted, only the new set remains (no orphans).
        assert h.words_for(first.run_id) == []
        second_words = h.words_for(second.run_id)
        assert len(second_words) == 4
        assert h.count_words() == 4
        assert sorted(w.start_ms for w in second_words) != first_starts

    def test_only_one_current_success_after_many_retries(self, h: Harness) -> None:
        ids = h.seed()
        for _ in range(3):
            _align(h, ids, FakeProvider(bias=0.05))
        current = [r for r in h.runs() if r.status in ("completed", "partial")]
        assert len(current) == 1
        # partial UNIQUE index also enforces this at the SQLite level.
        assert h.count_words() == 4  # only the winning run's words

    def test_zero_word_retry_preserves_previous_success(self, h: Harness) -> None:
        ids = h.seed()
        first = _align(h, ids, FakeProvider())
        second = _align(h, ids, FakeProvider(drop_segment=True))
        assert second.status == "failed"
        assert len(h.words_for(first.run_id)) == 4
        assert h.count_words() == 4
        assert h.runs()[0].status == "completed"


# --- Atomic rollback ----------------------------------------------------------

class TestAtomicRollback:
    def test_rollback_leaves_no_orphan_words(self, h: Harness) -> None:
        ids = h.seed()

        class ExplodingWordRepo(_WordRepo):
            def add_words(self, words, **kw):
                super().add_words(words, **kw)  # actually insert+flush
                raise RuntimeError("simulated DB failure after insert")

        svc = h.service(words=ExplodingWordRepo(h.s))
        with pytest.raises(RuntimeError):
            svc.align(
                transcript_asset_id=ids["asset_id"],
                source_audio_asset_id=ids["audio_id"],
                video_id=ids["video_id"],
                language="en", provider=FakeProvider(), now=NOW,
            )
        # transaction() rolled back -> NO word rows, NO orphan words.
        assert h.count_words() == 0
        # The run was committed in Phase A but never promoted to success.
        assert all(r.status == "running" for r in h.runs())


# --- Concurrent attempt safety (controlled IntegrityError simulation) --------

class TestConcurrentAttemptSafety:
    def test_concurrent_winner_finalizes_loser_as_failed(self, h: Harness) -> None:
        ids = h.seed()
        # A DIFFERENT run that already won the current-success slot.
        competitor = _Run(
            video_id=ids["video_id"],
            transcript_asset_id=ids["asset_id"],
            source_audio_asset_id=ids["audio_id"],
            content_right_id=ids["right_id"],
            language="en", status="completed", word_count=4,
        )
        h.s.add(competitor)
        h.s.commit()

        class ConcurrentRunRepo(_RunRepo):
            def __init__(self, s, competitor_id):
                super().__init__(s)
                self.competitor_id = competitor_id
                self.find_calls = 0
                self.outcome_raised = False

            def find_current_success(self, asset_id, audio_id, *, exclude_run_id=None):
                self.find_calls += 1
                if self.find_calls == 1:
                    return None  # service sees no current success on the first look
                return self.s.get(_Run, self.competitor_id)

            def set_run_outcome(self, rid, *, status, failure_reason=None,
                                word_count=None, **kwargs):
                if not self.outcome_raised:
                    self.outcome_raised = True
                    raise IntegrityError(
                        "simulated partial-index race", None, Exception("race")
                    )
                return super().set_run_outcome(
                    rid, status=status, failure_reason=failure_reason,
                    word_count=word_count, **kwargs,
                )

        runs_repo = ConcurrentRunRepo(h.s, competitor.id)
        svc = h.service(runs=runs_repo)
        result = svc.align(
            transcript_asset_id=ids["asset_id"],
            source_audio_asset_id=ids["audio_id"],
            video_id=ids["video_id"],
            language="en", provider=FakeProvider(), now=NOW,
        )
        assert result.status == "failed"
        assert result.failure_reason == "superseded_by_concurrent_run"
        assert result.word_count == 0
        # No word rows were persisted for the losing run (rolled back).
        assert h.count_words() == 0
        # The concurrent winner remains the one current success.
        assert h.s.get(_Run, competitor.id).status == "completed"


# --- Authorized audio asset registration -------------------------------------

class TestRegisterSourceAudioAsset:
    def test_registration_persists_authorized_asset(self, h: Harness) -> None:
        ids = h.seed(with_audio=False)
        svc = h.service()
        result = svc.register_source_audio_asset(
            _audio_input(ids["video_id"], ids["right_id"]),
            now=NOW,
        )
        assert isinstance(result, RegisteredAudioAsset)
        assert result.timeline_offset_ms == OFFSET_MS
        assert result.storage_uri == "bucket/authorized.wav"
        assert result.checksum_sha256 == _CHECKSUM
        assert result.mime_type == "audio/wav"
        assert result.rights_basis == "explicit-license"
        assert result.sample_rate_hz is None
        rows = list(h.s.scalars(select(_Audio)).all())
        assert len(rows) == 1
        assert rows[0].storage_uri == "bucket/authorized.wav"
        assert rows[0].checksum_sha256 == _CHECKSUM
        assert rows[0].mime_type == "audio/wav"
        assert rows[0].rights_basis == "explicit-license"

    def test_registration_rejects_inactive_rights(self, h: Harness) -> None:
        ids = h.seed(with_audio=False, right_status="revoked")
        svc = h.service()
        with pytest.raises(ContentRightsNotEligibleError):
            svc.register_source_audio_asset(
                _audio_input(
                    ids["video_id"], ids["right_id"],
                    storage_uri="bucket/x.wav", timeline_offset_ms=0,
                    duration_ms=1000,
                ),
                now=NOW,
            )
        assert list(h.s.scalars(select(_Audio)).all()) == []

    def test_registration_rejects_missing_video(self, h: Harness) -> None:
        ids = h.seed(with_audio=False, with_video=False)
        svc = h.service()
        with pytest.raises(AlignmentVideoNotFoundError):
            svc.register_source_audio_asset(
                _audio_input(
                    ids["video_id"], ids["right_id"],
                    storage_uri="bucket/x.wav", timeline_offset_ms=0,
                    duration_ms=1000,
                ),
                now=NOW,
            )


# --- Goal 13 integration contract (read-only) --------------------------------

class TestGetAlignedSentences:
    def test_fully_covered_sentences_are_word_aligned(self, h: Harness) -> None:
        from app.transcripts.sentence_segmentation import TimestampPrecision

        ids = h.seed()
        result = _align(h, ids, FakeProvider())
        aligned = h.service().get_aligned_sentences(result.run_id)
        assert aligned
        assert all(a.precision == TimestampPrecision.WORD_ALIGNED for a in aligned)
        assert all(a.exact_start_ms is not None and a.exact_end_ms is not None
                   for a in aligned)
        # Goal 10 Sentence reserved fields are NEVER populated.
        assert all(a.sentence.exact_start_seconds is None for a in aligned)
        assert all(a.sentence.exact_end_seconds is None for a in aligned)

    def test_unknown_run_returns_empty(self, h: Harness) -> None:
        h.seed()
        assert h.service().get_aligned_sentences(uuid.uuid4()) == ()
