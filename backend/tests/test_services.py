"""Offline unit tests for the Goal 08 service layer.

Services are tested with mocked repositories + a mocked ``Session`` so commit /
rollback boundaries and fail-closed rules are verified without any database.
No test opens a network / Supabase connection or inserts production data.

Also includes regression tests proving the existing 93 tests' endpoints,
metadata, schema, and migration history are unchanged by Goal 08.
"""
import ast
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.db.models import ContentRight, TranscriptAsset
from app.main import app
from app.repositories.content_rights_repository import ACTIVE_STATUS, REVOKED_STATUS
from app.services import (
    ContentRightsNotEligibleError,
    TranscriptAssetNotFoundError,
    TranscriptAssetOwnershipError,
    transaction,
)
from app.services.channel_service import ChannelService
from app.services.content_rights_service import ContentRightsService
from app.services.transcript_service import TranscriptService
from app.services.video_service import VideoService

BACKEND_DIR = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


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


# --- 1. transaction() boundary ------------------------------------------------

def test_transaction_commits_on_success(session: MagicMock) -> None:
    with transaction(session):
        session.add(object())
    session.commit.assert_called_once_with()
    assert session.rollback.call_count == 0


def test_transaction_rolls_back_and_reraises_on_error(session: MagicMock) -> None:
    with pytest.raises(RuntimeError):
        with transaction(session):
            raise RuntimeError("boom")
    session.rollback.assert_called_once_with()
    assert session.commit.call_count == 0


def test_transaction_rolls_back_when_commit_fails(session: MagicMock) -> None:
    session.commit.side_effect = RuntimeError("commit failed")
    with pytest.raises(RuntimeError):
        with transaction(session):
            session.add(object())
    session.rollback.assert_called_once_with()


def test_transaction_never_swallows_sqlalchemy_errors(session: MagicMock) -> None:
    failure = SQLAlchemyError("connection lost")
    with pytest.raises(SQLAlchemyError) as exc:
        with transaction(session):
            raise failure
    assert exc.value is failure
    session.rollback.assert_called_once_with()


# --- 2. ChannelService -------------------------------------------------------

def test_channel_create_or_get_existing_row_is_read_only(session: MagicMock) -> None:
    repo = MagicMock()
    existing = object()
    repo.get_channel_by_youtube_id.return_value = existing
    service = ChannelService(session, channels=repo)
    assert (
        service.create_or_get_channel(youtube_channel_id="UC", title="T")
        is existing
    )
    repo.create_channel.assert_not_called()
    assert session.commit.call_count == 0
    assert session.rollback.call_count == 0


def test_channel_create_or_get_creates_and_commits_once(session: MagicMock) -> None:
    repo = MagicMock()
    repo.get_channel_by_youtube_id.return_value = None
    created = object()
    repo.create_channel.return_value = created
    service = ChannelService(session, channels=repo)
    assert (
        service.create_or_get_channel(youtube_channel_id="UC", title="T")
        is created
    )
    repo.create_channel.assert_called_once()
    session.commit.assert_called_once_with()


def test_channel_create_or_get_resolves_concurrent_duplicate(
    session: MagicMock,
) -> None:
    repo = MagicMock()
    existing = object()
    repo.get_channel_by_youtube_id.side_effect = [None, existing]
    repo.create_channel.side_effect = IntegrityError(
        "stmt", {}, Exception("duplicate key")
    )
    service = ChannelService(session, channels=repo)
    assert (
        service.create_or_get_channel(youtube_channel_id="UC", title="T")
        is existing
    )
    session.rollback.assert_called_once_with()
    assert session.commit.call_count == 0


def test_channel_create_or_get_reraises_non_duplicate_db_error(
    session: MagicMock,
) -> None:
    repo = MagicMock()
    repo.get_channel_by_youtube_id.side_effect = [None, None]
    repo.create_channel.side_effect = IntegrityError(
        "stmt", {}, Exception("not a duplicate")
    )
    service = ChannelService(session, channels=repo)
    with pytest.raises(IntegrityError):
        service.create_or_get_channel(youtube_channel_id="UC", title="T")
    session.rollback.assert_called_once_with()


def test_channel_update_metadata(session: MagicMock) -> None:
    repo = MagicMock()
    channel = object()
    repo.get_channel_by_id.return_value = channel
    updated = object()
    repo.update_channel.return_value = updated
    service = ChannelService(session, channels=repo)
    cid = uuid.uuid4()
    assert service.update_channel_metadata(cid, title="New") is updated
    repo.update_channel.assert_called_once()
    session.commit.assert_called_once_with()


def test_channel_update_metadata_missing_row_skips_commit(
    session: MagicMock,
) -> None:
    repo = MagicMock()
    repo.get_channel_by_id.return_value = None
    assert (
        ChannelService(session, channels=repo).update_channel_metadata(
            uuid.uuid4(), title="New"
        )
        is None
    )
    repo.update_channel.assert_not_called()
    assert session.commit.call_count == 0


def test_channel_read_only_lookup_never_commits(session: MagicMock) -> None:
    repo = MagicMock()
    ChannelService(session, channels=repo).get_channel_by_youtube_id("UC")
    repo.get_channel_by_youtube_id.assert_called_once_with("UC")
    assert session.commit.call_count == 0


# --- 3. VideoService ----------------------------------------------------------

def test_video_create_or_get_existing_row(session: MagicMock) -> None:
    repo = MagicMock()
    existing = object()
    repo.get_video_by_youtube_id.return_value = existing
    service = VideoService(session, videos=repo)
    assert (
        service.create_or_get_video(
            channel_id=uuid.uuid4(), youtube_video_id="V", title="T"
        )
        is existing
    )
    repo.create_video.assert_not_called()
    assert session.commit.call_count == 0


def test_video_create_or_get_creates_and_commits(session: MagicMock) -> None:
    repo = MagicMock()
    repo.get_video_by_youtube_id.return_value = None
    created = object()
    repo.create_video.return_value = created
    service = VideoService(session, videos=repo)
    assert (
        service.create_or_get_video(
            channel_id=uuid.uuid4(), youtube_video_id="V", title="T"
        )
        is created
    )
    session.commit.assert_called_once_with()


def test_video_create_or_get_resolves_concurrent_duplicate(
    session: MagicMock,
) -> None:
    repo = MagicMock()
    existing = object()
    repo.get_video_by_youtube_id.side_effect = [None, existing]
    repo.create_video.side_effect = IntegrityError(
        "stmt", {}, Exception("duplicate key")
    )
    service = VideoService(session, videos=repo)
    assert (
        service.create_or_get_video(
            channel_id=uuid.uuid4(), youtube_video_id="V", title="T"
        )
        is existing
    )
    session.rollback.assert_called_once_with()


def test_video_list_by_channel_delegates_pagination(session: MagicMock) -> None:
    repo = MagicMock()
    repo.list_videos_by_channel.return_value = []
    cid = uuid.uuid4()
    assert (
        VideoService(session, videos=repo).list_videos_by_channel(
            cid, limit=3, offset=1
        )
        == []
    )
    repo.list_videos_by_channel.assert_called_once_with(cid, limit=3, offset=1)
    assert session.commit.call_count == 0


def test_video_update_metadata(session: MagicMock) -> None:
    repo = MagicMock()
    video = object()
    repo.get_video_by_id.return_value = video
    updated = object()
    repo.update_video.return_value = updated
    service = VideoService(session, videos=repo)
    assert service.update_video_metadata(uuid.uuid4(), duration_seconds=99) is updated
    session.commit.assert_called_once_with()


# --- 4. ContentRightsService: fail-closed eligibility -----------------------

@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, True),
        ({"expires_at": NOW + timedelta(days=1)}, True),
        ({"expires_at": NOW - timedelta(days=1)}, False),
        ({"expires_at": NOW}, False),  # exactly expired -> fail closed
        ({"revoked_at": NOW - timedelta(days=1)}, False),
        ({"status": REVOKED_STATUS}, False),
        ({"status": "pending"}, False),
        ({"status": "expired"}, False),
        ({"status": "unknown_value"}, False),
        ({"status": None}, False),
        ({"granted_at": NOW + timedelta(days=1)}, False),  # not yet granted
        ({"granted_at": None}, False),
        ({"granted_at": datetime(2020, 1, 1)}, True),  # naive -> assumed UTC
        ({"expires_at": datetime(2030, 1, 1)}, True),
    ],
)
def test_is_active_fail_closed_matrix(overrides: dict, expected: bool) -> None:
    assert ContentRightsService.is_active(make_rights(**overrides), now=NOW) is expected


def test_is_active_none_rights_is_not_authorized() -> None:
    assert ContentRightsService.is_active(None, now=NOW) is False


def test_is_active_is_pure_and_offline() -> None:
    """No session / DB is involved: a plain model instance is enough."""
    rights = make_rights()
    assert ContentRightsService.is_active(rights, now=NOW) is True
    assert rights.status == ACTIVE_STATUS  # unchanged


def test_get_current_rights_returns_active_record(session: MagicMock) -> None:
    repo = MagicMock()
    rights = make_rights()
    repo.get_current_rights_by_video.return_value = rights
    vid = uuid.uuid4()
    assert ContentRightsService(session, rights=repo).get_current_rights(
        vid, now=NOW
    ) is rights
    assert session.commit.call_count == 0  # read-only


def test_get_current_rights_rejects_ineligible_record_defense_in_depth(
    session: MagicMock,
) -> None:
    """Even if the repo returns a stale row, the service fails closed."""
    repo = MagicMock()
    repo.get_current_rights_by_video.return_value = make_rights(
        revoked_at=NOW - timedelta(days=1)
    )
    assert (
        ContentRightsService(session, rights=repo).get_current_rights(
            uuid.uuid4(), now=NOW
        )
        is None
    )


def test_get_current_rights_none_when_absent(session: MagicMock) -> None:
    repo = MagicMock()
    repo.get_current_rights_by_video.return_value = None
    assert (
        ContentRightsService(session, rights=repo).get_current_rights(
            uuid.uuid4(), now=NOW
        )
        is None
    )


@pytest.mark.parametrize(
    "row, expected", [(make_rights(), True), (None, False)]
)
def test_is_authorized(session: MagicMock, row, expected: bool) -> None:
    repo = MagicMock()
    repo.get_current_rights_by_video.return_value = row
    assert (
        ContentRightsService(session, rights=repo).is_authorized(
            uuid.uuid4(), now=NOW
        )
        is expected
    )


def test_service_does_not_expose_rights_creation() -> None:
    """No automatic authorization: services never create rights records."""
    assert not hasattr(ContentRightsService, "create_content_rights")
    assert not hasattr(ContentRightsService, "authorize_video")


def test_revoke_rights_commits(session: MagicMock) -> None:
    repo = MagicMock()
    rights = make_rights()
    repo.revoke_rights.return_value = rights
    assert (
        ContentRightsService(session, rights=repo).revoke_rights(
            uuid.uuid4(), now=NOW
        )
        is rights
    )
    session.commit.assert_called_once_with()


def test_revoke_rights_rolls_back_and_reraises(session: MagicMock) -> None:
    repo = MagicMock()
    repo.revoke_rights.side_effect = SQLAlchemyError("update failed")
    with pytest.raises(SQLAlchemyError):
        ContentRightsService(session, rights=repo).revoke_rights(uuid.uuid4())
    session.rollback.assert_called_once_with()
    assert session.commit.call_count == 0


# --- 5. TranscriptService -----------------------------------------------------

def _transcript_service(session: MagicMock):
    assets, segments, rights = MagicMock(), MagicMock(), MagicMock()
    service = TranscriptService(
        session, assets=assets, segments=segments, rights=rights
    )
    return service, assets, segments, rights


def test_create_asset_with_valid_active_rights_commits(session: MagicMock) -> None:
    service, assets, _segments, rights = _transcript_service(session)
    vid, rid = uuid.uuid4(), uuid.uuid4()
    rights.get_rights_by_id.return_value = make_rights(video_id=vid)
    created = object()
    assets.create_transcript_asset.return_value = created
    assert (
        service.create_transcript_asset(
            video_id=vid, content_right_id=rid,
            source_type="youtube_captions", language="en", status="pending",
            now=NOW,
        )
        is created
    )
    assets.create_transcript_asset.assert_called_once()
    session.commit.assert_called_once_with()


@pytest.mark.parametrize(
    "rights_row",
    [
        None,  # missing
        make_rights(expires_at=NOW - timedelta(days=1)),  # expired
        make_rights(revoked_at=NOW),  # revoked
        make_rights(status="pending"),  # inactive status
        make_rights(granted_at=NOW + timedelta(days=1)),  # not yet granted
    ],
)
def test_create_asset_rejects_ineligible_rights_no_commit(
    session: MagicMock, rights_row
) -> None:
    service, assets, _segments, rights = _transcript_service(session)
    vid = uuid.uuid4()
    if rights_row is not None:
        rights_row.video_id = vid
    rights.get_rights_by_id.return_value = rights_row
    with pytest.raises(ContentRightsNotEligibleError):
        service.create_transcript_asset(
            video_id=vid, content_right_id=uuid.uuid4(),
            source_type="upload", language="en", status="pending", now=NOW,
        )
    assets.create_transcript_asset.assert_not_called()
    assert session.commit.call_count == 0


def test_create_asset_rejects_cross_video_rights(session: MagicMock) -> None:
    service, assets, _segments, rights = _transcript_service(session)
    rights.get_rights_by_id.return_value = make_rights(video_id=uuid.uuid4())
    with pytest.raises(ContentRightsNotEligibleError):
        service.create_transcript_asset(
            video_id=uuid.uuid4(), content_right_id=uuid.uuid4(),
            source_type="upload", language="en", status="pending", now=NOW,
        )
    assets.create_transcript_asset.assert_not_called()
    assert session.commit.call_count == 0


def test_create_asset_rolls_back_on_db_error(session: MagicMock) -> None:
    service, assets, _segments, rights = _transcript_service(session)
    vid = uuid.uuid4()
    rights.get_rights_by_id.return_value = make_rights(video_id=vid)
    assets.create_transcript_asset.side_effect = SQLAlchemyError("fk violation")
    with pytest.raises(SQLAlchemyError):
        service.create_transcript_asset(
            video_id=vid, content_right_id=uuid.uuid4(),
            source_type="upload", language="en", status="pending", now=NOW,
        )
    session.rollback.assert_called_once_with()
    assert session.commit.call_count == 0


def test_update_processing_status_commits(session: MagicMock) -> None:
    service, assets, _segments, _rights = _transcript_service(session)
    updated = object()
    assets.update_asset_status.return_value = updated
    assert service.update_processing_status(uuid.uuid4(), "processed") is updated
    session.commit.assert_called_once_with()


def test_save_segment_with_owned_asset_commits(session: MagicMock) -> None:
    service, assets, segments, _rights = _transcript_service(session)
    vid, aid = uuid.uuid4(), uuid.uuid4()
    assets.get_asset_by_id.return_value = TranscriptAsset(
        video_id=vid, content_right_id=uuid.uuid4(),
        source_type="upload", language="en", status="pending",
    )
    created = object()
    segments.create_segment.return_value = created
    assert (
        service.save_segment(
            video_id=vid, transcript_asset_id=aid,
            start_seconds=0, end_seconds=5, segment_text="hi",
        )
        is created
    )
    # The video_id passed to the repository is the requested (validated) one.
    assert segments.create_segment.call_args.kwargs["video_id"] == vid
    session.commit.assert_called_once_with()


def test_save_segment_rejects_missing_asset(session: MagicMock) -> None:
    service, assets, segments, _rights = _transcript_service(session)
    assets.get_asset_by_id.return_value = None
    with pytest.raises(TranscriptAssetNotFoundError):
        service.save_segment(
            video_id=uuid.uuid4(), transcript_asset_id=uuid.uuid4(),
            start_seconds=0, end_seconds=5, segment_text="hi",
        )
    segments.create_segment.assert_not_called()
    assert session.commit.call_count == 0


def test_save_segment_rejects_cross_video_asset(session: MagicMock) -> None:
    service, assets, segments, _rights = _transcript_service(session)
    assets.get_asset_by_id.return_value = TranscriptAsset(
        video_id=uuid.uuid4(), content_right_id=uuid.uuid4(),
        source_type="upload", language="en", status="pending",
    )
    with pytest.raises(TranscriptAssetOwnershipError):
        service.save_segment(
            video_id=uuid.uuid4(), transcript_asset_id=uuid.uuid4(),
            start_seconds=0, end_seconds=5, segment_text="hi",
        )
    segments.create_segment.assert_not_called()
    assert session.commit.call_count == 0


def test_save_segments_batch_uses_single_transaction(session: MagicMock) -> None:
    service, assets, segments, _rights = _transcript_service(session)
    vid, aid = uuid.uuid4(), uuid.uuid4()
    assets.get_asset_by_id.return_value = TranscriptAsset(
        video_id=vid, content_right_id=uuid.uuid4(),
        source_type="upload", language="en", status="pending",
    )
    segments.create_segment.side_effect = [object(), object(), object()]
    payload = [
        {"start_seconds": 0, "end_seconds": 2, "segment_text": "a"},
        {"start_seconds": 2, "end_seconds": 4, "segment_text": "b", "language": "en"},
        {"start_seconds": 4, "end_seconds": 6, "segment_text": "c"},
    ]
    result = service.save_segments(
        video_id=vid, transcript_asset_id=aid, segments=payload
    )
    assert len(result) == 3
    assert segments.create_segment.call_count == 3
    session.commit.assert_called_once_with()  # one commit for the whole batch


def test_save_segments_rolls_back_batch_on_failure(session: MagicMock) -> None:
    service, assets, segments, _rights = _transcript_service(session)
    vid = uuid.uuid4()
    assets.get_asset_by_id.return_value = TranscriptAsset(
        video_id=vid, content_right_id=uuid.uuid4(),
        source_type="upload", language="en", status="pending",
    )
    segments.create_segment.side_effect = [object(), SQLAlchemyError("bad")]
    payload = [
        {"start_seconds": 0, "end_seconds": 2, "segment_text": "a"},
        {"start_seconds": 2, "end_seconds": 4, "segment_text": "b"},
    ]
    with pytest.raises(SQLAlchemyError):
        service.save_segments(
            video_id=vid, transcript_asset_id=uuid.uuid4(), segments=payload
        )
    session.rollback.assert_called_once_with()
    assert session.commit.call_count == 0


def test_get_timestamped_segments_by_asset(session: MagicMock) -> None:
    service, _assets, segments, _rights = _transcript_service(session)
    segments.list_segments_by_asset.return_value = []
    aid = uuid.uuid4()
    assert (
        service.get_timestamped_segments(asset_id=aid, limit=4, offset=2) == []
    )
    segments.list_segments_by_asset.assert_called_once_with(
        aid, limit=4, offset=2
    )
    assert session.commit.call_count == 0


def test_get_timestamped_segments_by_video(session: MagicMock) -> None:
    service, _assets, segments, _rights = _transcript_service(session)
    segments.list_segments_by_video.return_value = []
    vid = uuid.uuid4()
    assert service.get_timestamped_segments(video_id=vid) == []
    segments.list_segments_by_video.assert_called_once_with(
        vid, limit=50, offset=0
    )


def test_get_timestamped_segments_requires_scope(session: MagicMock) -> None:
    service, _assets, _segments, _rights = _transcript_service(session)
    with pytest.raises(ValueError):
        service.get_timestamped_segments()
    assert session.commit.call_count == 0


def test_transcript_service_never_creates_rights(session: MagicMock) -> None:
    """No automatic authorization anywhere in the transcript flow."""
    service, assets, _segments, rights = _transcript_service(session)
    vid = uuid.uuid4()
    rights.get_rights_by_id.return_value = make_rights(video_id=vid)
    service.create_transcript_asset(
        video_id=vid, content_right_id=uuid.uuid4(),
        source_type="upload", language="en", status="pending", now=NOW,
    )
    assert rights.create_content_rights.call_count == 0
    assets.create_transcript_asset.assert_called_once()


# --- 6. Transaction-ownership invariants (AST-based, no DB) ----------------

def _calls(path: Path, attr: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    )


def test_repository_source_never_commits_or_rolls_back() -> None:
    repo_dir = BACKEND_DIR / "app" / "repositories"
    for path in sorted(repo_dir.glob("*.py")):
        assert _calls(path, "commit") == 0, f"{path.name} must not commit"
        assert _calls(path, "rollback") == 0, f"{path.name} must not roll back"


def test_commit_lives_only_in_the_shared_transaction_helper() -> None:
    svc_dir = BACKEND_DIR / "app" / "services"
    for name in (
        "channel_service.py", "video_service.py",
        "content_rights_service.py", "transcript_service.py",
    ):
        assert _calls(svc_dir / name, "commit") == 0
        assert _calls(svc_dir / name, "rollback") == 0
    helper = svc_dir / "__init__.py"
    assert _calls(helper, "commit") == 1
    assert _calls(helper, "rollback") == 1


def test_every_service_module_uses_transaction() -> None:
    svc_dir = BACKEND_DIR / "app" / "services"
    for name in (
        "channel_service.py", "video_service.py",
        "content_rights_service.py", "transcript_service.py",
    ):
        tree = ast.parse((svc_dir / name).read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "transaction"
            for node in ast.walk(tree)
        ), f"{name} must use transaction()"


# --- 7. Regression: existing behaviour unchanged ----------------------------

def test_basic_health_endpoint_unchanged(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "voxsearch-backend",
        "version": "0.1.0",
    }


def test_db_health_endpoint_unchanged(client: TestClient) -> None:
    healthy = {
        "status": "healthy",
        "database": "connected",
        "vector_extension": "available",
    }
    with patch("app.db.health.check_database", return_value=dict(healthy)):
        response = client.get("/health/db")
    assert response.status_code == 200
    assert response.json() == healthy


def test_openapi_metadata_unchanged(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "voxsearch-backend"
    assert spec["info"]["version"] == "0.1.0"
    assert set(spec["paths"]) == {"/health", "/health/db"}


def test_migration_head_and_history_unchanged() -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    script = ScriptDirectory.from_config(cfg)
    assert script.get_heads() == ["fe7b2b8e5323"]
    assert [r.revision for r in script.walk_revisions()] == [
        "fe7b2b8e5323", "b3727ed7438a",
    ]


def test_no_new_migration_files_added() -> None:
    versions = BACKEND_DIR / "migrations" / "versions"
    revisions = sorted(p.name for p in versions.glob("*.py"))
    assert revisions == [
        "b3727ed7438a_initial_voxsearch_schema.py",
        "fe7b2b8e5323_enforce_transcript_ownership_integrity.py",
    ]
