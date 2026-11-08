"""Offline unit tests for the Goal 08 repository layer.

These tests use a mocked SQLAlchemy ``Session`` and inspect the generated
``Select`` statements (compiled against the PostgreSQL dialect). They NEVER
open a network / Supabase connection and never insert data into Supabase.
Live connectivity is verified separately (see the Goal 08 validation report).
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    SCHEMA,
    Base,
    Channel,
    ContentRight,
    TranscriptAsset,
    TranscriptSegment,
    Video,
)
from app.repositories import (
    ACTIVE_STATUS,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    REVOKED_STATUS,
    UNSET,
    BaseRepository,
    ChannelRepository,
    ContentRightsRepository,
    TranscriptAssetRepository,
    TranscriptSegmentRepository,
    VideoRepository,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def session() -> MagicMock:
    return MagicMock()


def compiled(stmt) -> str:
    """Compile a Select to PostgreSQL SQL (bind params stay as placeholders)."""
    return str(stmt.compile(dialect=postgresql.dialect()))


def scalar_statements(session: MagicMock) -> list:
    return [c.args[0] for c in session.scalars.call_args_list]


def last_scalar_stmt(session: MagicMock):
    return session.scalars.call_args.args[0]


def make_rights(**overrides) -> ContentRight:
    values = dict(
        video_id=uuid.uuid4(),
        authorization_type="explicit_license",
        authorization_reference="reference-001",
        status=ACTIVE_STATUS,
        granted_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        expires_at=None,
        revoked_at=None,
    )
    values.update(overrides)
    return ContentRight(**values)


# --- 1. BaseRepository: session, flush, pagination ---------------------------

def test_base_repository_holds_injected_session(session: MagicMock) -> None:
    assert BaseRepository(session).session is session


def test_flush_calls_session_flush_and_never_commits(session: MagicMock) -> None:
    BaseRepository(session).flush()
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0
    assert session.rollback.call_count == 0


def test_normalize_pagination_defaults() -> None:
    assert BaseRepository.normalize_pagination(None, None) == (DEFAULT_LIMIT, 0)


def test_normalize_pagination_clamps_to_max() -> None:
    assert BaseRepository.normalize_pagination(9999, 3) == (MAX_LIMIT, 3)


@pytest.mark.parametrize(
    "limit, offset",
    [(0, 0), (-1, 0), (10, -1), (10, -50)],
)
def test_normalize_pagination_rejects_invalid(limit: int, offset: int) -> None:
    with pytest.raises(ValueError):
        BaseRepository.normalize_pagination(limit, offset)


def test_paginate_applies_limit_and_offset() -> None:
    from sqlalchemy import select

    stmt = BaseRepository.paginate(select(Channel), limit=10, offset=5)
    assert stmt._limit_clause.value == 10
    assert stmt._offset_clause.value == 5


# --- 2. ChannelRepository ----------------------------------------------------

def test_create_channel_adds_and_flushes_never_commits(session: MagicMock) -> None:
    repo = ChannelRepository(session)
    channel = repo.create_channel(
        youtube_channel_id="UC_TEST", title="Test", description="desc"
    )
    assert isinstance(channel, Channel)
    assert channel.youtube_channel_id == "UC_TEST"
    assert channel.title == "Test"
    assert channel.description == "desc"
    session.add.assert_called_once_with(channel)
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_create_channel_description_defaults_to_none(session: MagicMock) -> None:
    channel = ChannelRepository(session).create_channel(
        youtube_channel_id="UC_TEST", title="Test"
    )
    assert channel.description is None


def test_create_channel_duplicate_raises_integrity_error_not_swallowed(
    session: MagicMock,
) -> None:
    session.flush.side_effect = IntegrityError("stmt", {}, Exception("duplicate"))
    with pytest.raises(IntegrityError):
        ChannelRepository(session).create_channel(
            youtube_channel_id="UC_DUP", title="Dup"
        )


def test_get_channel_by_id_uses_session_get(session: MagicMock) -> None:
    obj = object()
    session.get.return_value = obj
    cid = uuid.uuid4()
    assert ChannelRepository(session).get_channel_by_id(cid) is obj
    session.get.assert_called_once_with(Channel, cid)


def test_get_channel_by_youtube_id_filters_without_pagination(
    session: MagicMock,
) -> None:
    obj = object()
    session.scalars.return_value.first.return_value = obj
    assert (
        ChannelRepository(session).get_channel_by_youtube_id("UC_X") is obj
    )
    stmt = last_scalar_stmt(session)
    sql = compiled(stmt)
    assert "channels.youtube_channel_id" in sql
    assert stmt._limit_clause is None
    session.scalars.return_value.first.assert_called_once_with()


def test_list_channels_pagination_and_deterministic_order(session: MagicMock) -> None:
    rows = [Channel(title="A"), Channel(title="B")]
    session.scalars.return_value.all.return_value = rows
    result = ChannelRepository(session).list_channels(limit=10, offset=5)
    assert result == rows
    stmt = last_scalar_stmt(session)
    assert stmt._limit_clause.value == 10
    assert stmt._offset_clause.value == 5
    order = [str(c) for c in stmt._order_by_clauses]
    assert order == [
        f"{SCHEMA}.channels.created_at DESC",
        f"{SCHEMA}.channels.id DESC",
    ]
    assert session.scalars.call_count == 1  # single query, no N+1


def test_list_channels_uses_default_pagination(session: MagicMock) -> None:
    session.scalars.return_value.all.return_value = []
    ChannelRepository(session).list_channels()
    stmt = last_scalar_stmt(session)
    assert stmt._limit_clause.value == DEFAULT_LIMIT
    assert stmt._offset_clause.value == 0


def test_update_channel_only_changes_supplied_fields(session: MagicMock) -> None:
    channel = Channel(youtube_channel_id="UC", title="Old", description="keep")
    result = ChannelRepository(session).update_channel(channel, title="New")
    assert result is channel
    assert channel.title == "New"
    assert channel.description == "keep"  # UNSET -> unchanged
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_update_channel_can_clear_nullable_description(session: MagicMock) -> None:
    channel = Channel(youtube_channel_id="UC", title="T", description="keep")
    ChannelRepository(session).update_channel(channel, description=None)
    assert channel.description is None
    assert channel.title == "T"


def test_update_channel_unset_is_sentinel(session: MagicMock) -> None:
    channel = Channel(youtube_channel_id="UC", title="Old", description="keep")
    ChannelRepository(session).update_channel(
        channel, title=UNSET, description=UNSET
    )
    assert channel.title == "Old"
    assert channel.description == "keep"


def test_delete_channel_deletes_and_flushes(session: MagicMock) -> None:
    channel = Channel(youtube_channel_id="UC", title="T")
    ChannelRepository(session).delete_channel(channel)
    session.delete.assert_called_once_with(channel)
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


# --- 3. VideoRepository ------------------------------------------------------

def test_create_video_fields_and_no_commit(session: MagicMock) -> None:
    cid = uuid.uuid4()
    published = datetime(2024, 5, 1, tzinfo=timezone.utc)
    video = VideoRepository(session).create_video(
        channel_id=cid,
        youtube_video_id="VID_1",
        title="Title",
        description="d",
        duration_seconds=120,
        published_at=published,
    )
    assert isinstance(video, Video)
    assert video.channel_id == cid
    assert video.youtube_video_id == "VID_1"
    assert video.duration_seconds == 120
    assert video.published_at == published
    session.add.assert_called_once_with(video)
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_create_video_duplicate_raises_integrity_error(session: MagicMock) -> None:
    session.flush.side_effect = IntegrityError("stmt", {}, Exception("duplicate"))
    with pytest.raises(IntegrityError):
        VideoRepository(session).create_video(
            channel_id=uuid.uuid4(), youtube_video_id="VID_DUP", title="T"
        )


def test_get_video_by_id(session: MagicMock) -> None:
    obj = object()
    session.get.return_value = obj
    vid = uuid.uuid4()
    assert VideoRepository(session).get_video_by_id(vid) is obj
    session.get.assert_called_once_with(Video, vid)


def test_get_video_by_youtube_id_filters(session: MagicMock) -> None:
    obj = object()
    session.scalars.return_value.first.return_value = obj
    assert VideoRepository(session).get_video_by_youtube_id("VID_X") is obj
    assert "videos.youtube_video_id" in compiled(last_scalar_stmt(session))


def test_list_videos_by_channel_filters_orders_paginates(session: MagicMock) -> None:
    session.scalars.return_value.all.return_value = []
    cid = uuid.uuid4()
    result = VideoRepository(session).list_videos_by_channel(
        cid, limit=7, offset=3
    )
    assert result == []
    stmt = last_scalar_stmt(session)
    sql = compiled(stmt)
    assert "videos.channel_id" in sql
    assert stmt._limit_clause.value == 7
    assert stmt._offset_clause.value == 3
    assert [str(c) for c in stmt._order_by_clauses] == [
        f"{SCHEMA}.videos.created_at DESC",
        f"{SCHEMA}.videos.id DESC",
    ]
    assert session.scalars.call_count == 1


def test_update_video_fields(session: MagicMock) -> None:
    video = Video(
        channel_id=uuid.uuid4(), youtube_video_id="V", title="Old",
        description="keep", duration_seconds=10,
    )
    VideoRepository(session).update_video(video, duration_seconds=99)
    assert video.title == "Old"
    assert video.description == "keep"
    assert video.duration_seconds == 99
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_delete_video(session: MagicMock) -> None:
    video = Video(channel_id=uuid.uuid4(), youtube_video_id="V", title="T")
    VideoRepository(session).delete_video(video)
    session.delete.assert_called_once_with(video)
    assert session.commit.call_count == 0


# --- 4. ContentRightsRepository ---------------------------------------------

def test_status_constants_use_documented_values() -> None:
    assert ACTIVE_STATUS == "active"
    assert REVOKED_STATUS == "revoked"


def test_create_content_rights_uses_caller_supplied_status(
    session: MagicMock,
) -> None:
    vid = uuid.uuid4()
    granted = datetime(2020, 1, 1, tzinfo=timezone.utc)
    expires = datetime(2030, 1, 1, tzinfo=timezone.utc)
    rights = ContentRightsRepository(session).create_content_rights(
        video_id=vid,
        authorization_type="creator_permission",
        authorization_reference="ref-9",
        status="pending",  # caller decides; not invented by the repository
        granted_at=granted,
        expires_at=expires,
    )
    assert rights.video_id == vid
    assert rights.authorization_type == "creator_permission"
    assert rights.authorization_reference == "ref-9"
    assert rights.status == "pending"
    assert rights.granted_at == granted
    assert rights.expires_at == expires
    assert rights.revoked_at is None
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_get_rights_by_id(session: MagicMock) -> None:
    obj = object()
    session.get.return_value = obj
    rid = uuid.uuid4()
    assert ContentRightsRepository(session).get_rights_by_id(rid) is obj
    session.get.assert_called_once_with(ContentRight, rid)


def test_list_rights_by_video_filters_orders_paginates(session: MagicMock) -> None:
    session.scalars.return_value.all.return_value = []
    vid = uuid.uuid4()
    ContentRightsRepository(session).list_rights_by_video(vid, limit=4, offset=2)
    stmt = last_scalar_stmt(session)
    assert "content_rights.video_id" in compiled(stmt)
    assert stmt._limit_clause.value == 4
    assert stmt._offset_clause.value == 2
    assert [str(c) for c in stmt._order_by_clauses] == [
        f"{SCHEMA}.content_rights.created_at DESC",
        f"{SCHEMA}.content_rights.id DESC",
    ]


def test_current_rights_query_enforces_all_four_guards(session: MagicMock) -> None:
    obj = object()
    session.scalars.return_value.first.return_value = obj
    assert (
        ContentRightsRepository(session).get_current_rights_by_video(
            uuid.uuid4(), now=NOW
        )
        is obj
    )
    stmt = last_scalar_stmt(session)
    sql = compiled(stmt)
    assert "content_rights.video_id" in sql
    assert "content_rights.status" in sql
    assert "revoked_at IS NULL" in sql
    assert "granted_at IS NOT NULL" in sql
    assert "granted_at <=" in sql
    assert "expires_at IS NULL OR" in sql
    assert "expires_at >" in sql
    # Single most-recent record, deterministic tiebreaker, never a list scan.
    assert stmt._limit_clause is None
    assert [str(c) for c in stmt._order_by_clauses] == [
        f"{SCHEMA}.content_rights.granted_at DESC",
        f"{SCHEMA}.content_rights.id DESC",
    ]
    session.scalars.return_value.first.assert_called_once_with()


def test_current_rights_none_when_absent(session: MagicMock) -> None:
    session.scalars.return_value.first.return_value = None
    assert (
        ContentRightsRepository(session).get_current_rights_by_video(
            uuid.uuid4(), now=NOW
        )
        is None
    )


def test_revoke_rights_sets_status_and_timestamp(session: MagicMock) -> None:
    rights = make_rights()
    session.get.return_value = rights
    result = ContentRightsRepository(session).revoke_rights(
        uuid.uuid4(), now=NOW
    )
    assert result is rights
    assert rights.status == REVOKED_STATUS
    assert rights.revoked_at == NOW
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_revoke_rights_missing_returns_none_without_flush(session: MagicMock) -> None:
    session.get.return_value = None
    assert ContentRightsRepository(session).revoke_rights(uuid.uuid4()) is None
    assert session.flush.call_count == 0
    assert session.commit.call_count == 0


# --- 5. TranscriptAssetRepository -------------------------------------------

def test_create_asset_fields_include_composite_ownership_columns(
    session: MagicMock,
) -> None:
    vid, rid = uuid.uuid4(), uuid.uuid4()
    asset = TranscriptAssetRepository(session).create_transcript_asset(
        video_id=vid,
        content_right_id=rid,
        source_type="youtube_captions",
        language="en",
        status="pending",
    )
    # Both columns feed the Goal 07 composite FK (content_right_id, video_id).
    assert asset.video_id == vid
    assert asset.content_right_id == rid
    assert asset.source_type == "youtube_captions"
    assert asset.language == "en"
    assert asset.status == "pending"
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_get_asset_by_id(session: MagicMock) -> None:
    obj = object()
    session.get.return_value = obj
    aid = uuid.uuid4()
    assert TranscriptAssetRepository(session).get_asset_by_id(aid) is obj
    session.get.assert_called_once_with(TranscriptAsset, aid)


def test_list_assets_by_video_filters_orders_paginates(session: MagicMock) -> None:
    session.scalars.return_value.all.return_value = []
    TranscriptAssetRepository(session).list_assets_by_video(
        uuid.uuid4(), limit=6, offset=1
    )
    stmt = last_scalar_stmt(session)
    assert "transcript_assets.video_id" in compiled(stmt)
    assert stmt._limit_clause.value == 6
    assert stmt._offset_clause.value == 1
    assert f"{SCHEMA}.transcript_assets.id DESC" in [
        str(c) for c in stmt._order_by_clauses
    ]


def test_update_asset_status_changes_only_status(session: MagicMock) -> None:
    vid = uuid.uuid4()
    asset = TranscriptAsset(
        video_id=vid, content_right_id=uuid.uuid4(),
        source_type="upload", language="en", status="pending",
    )
    session.get.return_value = asset
    result = TranscriptAssetRepository(session).update_asset_status(
        uuid.uuid4(), "processed"
    )
    assert result is asset
    assert asset.status == "processed"
    assert asset.video_id == vid  # ownership columns untouched
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_update_asset_status_missing_returns_none(session: MagicMock) -> None:
    session.get.return_value = None
    assert (
        TranscriptAssetRepository(session).update_asset_status(
            uuid.uuid4(), "processed"
        )
        is None
    )
    assert session.flush.call_count == 0


# --- 6. TranscriptSegmentRepository -----------------------------------------

def test_create_segment_fields(session: MagicMock) -> None:
    aid, vid = uuid.uuid4(), uuid.uuid4()
    segment = TranscriptSegmentRepository(session).create_segment(
        transcript_asset_id=aid,
        video_id=vid,
        start_seconds=1.5,
        end_seconds=4.0,
        segment_text="hello world",
        language="en",
    )
    assert segment.transcript_asset_id == aid
    assert segment.video_id == vid
    assert segment.start_seconds == 1.5
    assert segment.end_seconds == 4.0
    assert segment.segment_text == "hello world"
    assert segment.language == "en"
    # The attribute maps to the physical "text" column.
    assert "text" in TranscriptSegment.__table__.c
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_create_segment_persists_original_text_separately(
    session: MagicMock,
) -> None:
    """Goal 09 corrective: verbatim original and normalized text are distinct."""
    segment = TranscriptSegmentRepository(session).create_segment(
        transcript_asset_id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        start_seconds=1.5,
        end_seconds=4.0,
        segment_text="normalized",
        original_text="  verbatim  original  ",
        language="en",
    )
    assert segment.segment_text == "normalized"
    assert segment.original_text == "  verbatim  original  "
    assert "original_text" in TranscriptSegment.__table__.c
    assert TranscriptSegment.__table__.c.original_text.nullable is True
    session.flush.assert_called_once_with()
    assert session.commit.call_count == 0


def test_create_segment_without_original_text_defaults_to_none(
    session: MagicMock,
) -> None:
    """Backward compatibility: raw/legacy callers get a NULL original."""
    segment = TranscriptSegmentRepository(session).create_segment(
        transcript_asset_id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        start_seconds=0,
        end_seconds=2,
        segment_text="normalized legacy",
    )
    assert segment.segment_text == "normalized legacy"
    assert segment.original_text is None


@pytest.mark.parametrize(
    "start, end, text",
    [
        (-1.0, 5.0, "ok"),
        (5.0, 5.0, "ok"),
        (5.0, 4.0, "ok"),
        (0.0, 5.0, ""),
        (0.0, 5.0, "   "),
    ],
)
def test_create_segment_validates_timestamps_and_text(
    session: MagicMock, start: float, end: float, text: str
) -> None:
    with pytest.raises(ValueError):
        TranscriptSegmentRepository(session).create_segment(
            transcript_asset_id=uuid.uuid4(),
            video_id=uuid.uuid4(),
            start_seconds=start,
            end_seconds=end,
            segment_text=text,
        )
    assert session.add.call_count == 0  # rejected before any write


def test_get_segment_by_id(session: MagicMock) -> None:
    obj = object()
    session.get.return_value = obj
    sid = uuid.uuid4()
    assert TranscriptSegmentRepository(session).get_segment_by_id(sid) is obj
    session.get.assert_called_once_with(TranscriptSegment, sid)


@pytest.mark.parametrize(
    "method, kwarg, expected_column",
    [
        ("list_segments_by_video", "video_id", "transcript_segments.video_id"),
        (
            "list_segments_by_asset",
            "asset_id",
            "transcript_segments.transcript_asset_id",
        ),
    ],
)
def test_segment_listings_filter_order_paginate(
    session: MagicMock, method: str, kwarg: str, expected_column: str
) -> None:
    session.scalars.return_value.all.return_value = []
    getattr(TranscriptSegmentRepository(session), method)(
        uuid.uuid4(), limit=8, offset=2
    )
    stmt = last_scalar_stmt(session)
    assert expected_column in compiled(stmt)
    assert stmt._limit_clause.value == 8
    assert stmt._offset_clause.value == 2
    assert [str(c) for c in stmt._order_by_clauses] == [
        f"{SCHEMA}.transcript_segments.start_seconds ASC",
        f"{SCHEMA}.transcript_segments.id ASC",
    ]


def test_search_segments_by_text_uses_basic_ilike(session: MagicMock) -> None:
    session.scalars.return_value.all.return_value = []
    result = TranscriptSegmentRepository(session).search_segments_by_text(
        "machine learning", limit=5, offset=1
    )
    assert result == []
    stmt = last_scalar_stmt(session)
    sql = compiled(stmt)
    assert "ILIKE" in sql
    # Basic substring filter only: no FTS / vector / embedding constructs.
    assert "to_tsquery" not in sql.lower()
    assert "tsvector" not in sql.lower()
    assert "@@" not in sql
    assert "embedding" not in sql.lower()
    assert stmt._limit_clause.value == 5
    assert stmt._offset_clause.value == 1
    assert [str(c) for c in stmt._order_by_clauses] == [
        f"{SCHEMA}.transcript_segments.start_seconds ASC",
        f"{SCHEMA}.transcript_segments.id ASC",
    ]


def test_search_segments_by_text_optional_video_scope(session: MagicMock) -> None:
    session.scalars.return_value.all.return_value = []
    TranscriptSegmentRepository(session).search_segments_by_text(
        "hello", video_id=uuid.uuid4()
    )
    assert "transcript_segments.video_id" in compiled(last_scalar_stmt(session))


def test_search_segments_by_text_blank_query_hits_no_database(
    session: MagicMock,
) -> None:
    assert TranscriptSegmentRepository(session).search_segments_by_text("  ") == []
    assert session.scalars.call_count == 0


# --- 7. Cross-cutting invariants --------------------------------------------

def test_all_write_repositories_never_commit(session: MagicMock) -> None:
    """Every create/update/delete flushes but never commits."""
    vid = uuid.uuid4()
    ChannelRepository(session).create_channel(youtube_channel_id="UC", title="T")
    VideoRepository(session).create_video(
        channel_id=vid, youtube_video_id="V", title="T"
    )
    ContentRightsRepository(session).create_content_rights(
        video_id=vid, authorization_type="license",
        authorization_reference="r", status=ACTIVE_STATUS,
    )
    TranscriptAssetRepository(session).create_transcript_asset(
        video_id=vid, content_right_id=uuid.uuid4(),
        source_type="upload", language="en", status="pending",
    )
    TranscriptSegmentRepository(session).create_segment(
        transcript_asset_id=uuid.uuid4(), video_id=vid,
        start_seconds=0, end_seconds=5, segment_text="text",
    )
    assert session.flush.call_count == 5
    assert session.commit.call_count == 0
    assert session.rollback.call_count == 0


def test_all_list_methods_have_id_tiebreaker(session: MagicMock) -> None:
    session.scalars.return_value.all.return_value = []
    vid = uuid.uuid4()
    ChannelRepository(session).list_channels()
    VideoRepository(session).list_videos_by_channel(vid)
    ContentRightsRepository(session).list_rights_by_video(vid)
    TranscriptAssetRepository(session).list_assets_by_video(vid)
    TranscriptSegmentRepository(session).list_segments_by_video(vid)
    stmts = scalar_statements(session)
    assert len(stmts) == 5
    for stmt in stmts:
        assert any("id" in str(c) for c in stmt._order_by_clauses)


def test_importing_repository_layer_does_not_change_schema() -> None:
    """Repositories add no schema: metadata matches the declared eight tables.

    Goal 08 added none; Goal 12 added exactly three additive word-alignment
    tables via the migration/models, not via the repository layer.
    """
    assert set(Base.metadata.tables) == {
        f"{SCHEMA}.{n}"
        for n in (
            "channels", "videos", "content_rights",
            "transcript_assets", "transcript_segments",
            "source_audio_assets", "transcript_alignment_runs",
            "transcript_words",
        )
    }
