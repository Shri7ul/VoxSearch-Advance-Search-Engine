"""enforce transcript ownership integrity

Revision ID: fe7b2b8e5323
Revises: b3727ed7438a
Create Date: 2026-11-06 03:57:01.645573

Goal 07 — harden the Goal 06 schema by enforcing cross-table video ownership
consistency. A transcript asset must reference a content-right record belonging
to the SAME video, and a transcript segment must reference a transcript asset
belonging to the SAME video. Composite foreign keys bind the child's video_id
to the parent's video_id; each referenced (id, video_id) pair is backed by a
matching unique constraint (required by PostgreSQL for composite FKs).

Safety properties:
* DDL-only: no INSERT / UPDATE / DELETE, no table or schema drops.
* Only the two single-column FKs being replaced are dropped; the video_id ->
  videos.id FKs and every other constraint / index / table are preserved.
* ``public`` and all Supabase-managed schemas are untouched.
* ``downgrade()`` only reverses the constraints introduced or replaced here;
  it is NEVER executed against the live database.

Note: these constraints enforce relational ownership consistency only. They do
NOT enforce whether a content-rights record is currently authorized, active,
or unexpired — that remains separate application-level authorization logic.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "fe7b2b8e5323"
down_revision: Union[str, None] = "b3727ed7438a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "voxsearch"

# New constraint names — kept byte-for-byte in sync with app/db/models.py.
UQ_CONTENT_RIGHTS_ID_VIDEO_ID = "uq_content_rights_id_video_id"
UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID = "uq_transcript_assets_id_video_id"
FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO = "fk_transcript_assets_content_right_id_video_id"
FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO = "fk_transcript_segments_transcript_asset_id_video_id"

# Exact names of the two FKs being replaced. Verified against the live
# database (pg_constraint) before this migration was written — never guessed.
FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_OLD = "fk_transcript_assets_content_right_id_content_rights"
FK_TRANSCRIPT_SEGMENTS_ASSET_OLD = "fk_transcript_segments_asset_id_transcript_assets"


def upgrade() -> None:
    # 1. Unique constraints on the referenced (id, video_id) pairs. PostgreSQL
    #    requires a unique constraint on exactly the referenced column set for
    #    a composite FK to be valid. ``id`` is already the primary key, but the
    #    pair must be declared explicitly. These also create the backing
    #    unique indexes that the composite FKs will use.
    op.create_unique_constraint(
        UQ_CONTENT_RIGHTS_ID_VIDEO_ID,
        "content_rights", ["id", "video_id"], schema=SCHEMA,
    )
    op.create_unique_constraint(
        UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID,
        "transcript_assets", ["id", "video_id"], schema=SCHEMA,
    )

    # 2. Drop ONLY the two single-column FKs being replaced. The
    #    video_id -> videos.id FKs and every other constraint/index stay.
    op.drop_constraint(
        FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_OLD,
        "transcript_assets", schema=SCHEMA, type_="foreignkey",
    )
    op.drop_constraint(
        FK_TRANSCRIPT_SEGMENTS_ASSET_OLD,
        "transcript_segments", schema=SCHEMA, type_="foreignkey",
    )

    # 3. Composite FKs: because video_id is part of the referenced pair, a
    #    child row can never point at another video's content_right / asset.
    op.create_foreign_key(
        FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO,
        "transcript_assets", "content_rights",
        ["content_right_id", "video_id"], ["id", "video_id"],
        source_schema=SCHEMA, referent_schema=SCHEMA,
    )
    op.create_foreign_key(
        FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO,
        "transcript_segments", "transcript_assets",
        ["transcript_asset_id", "video_id"], ["id", "video_id"],
        source_schema=SCHEMA, referent_schema=SCHEMA,
    )


def downgrade() -> None:
    """Reverse ONLY the constraints introduced/replaced by upgrade().

    Never executed against the live database. Drops the two composite FKs,
    restores the two original single-column FKs (by their exact original
    names), then drops the two unique constraints. No table, schema, index,
    or unrelated constraint is touched.
    """
    op.drop_constraint(
        FK_TRANSCRIPT_SEGMENTS_ASSET_VIDEO,
        "transcript_segments", schema=SCHEMA, type_="foreignkey",
    )
    op.drop_constraint(
        FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_VIDEO,
        "transcript_assets", schema=SCHEMA, type_="foreignkey",
    )

    op.create_foreign_key(
        FK_TRANSCRIPT_SEGMENTS_ASSET_OLD,
        "transcript_segments", "transcript_assets",
        ["transcript_asset_id"], ["id"],
        source_schema=SCHEMA, referent_schema=SCHEMA,
    )
    op.create_foreign_key(
        FK_TRANSCRIPT_ASSETS_CONTENT_RIGHT_OLD,
        "transcript_assets", "content_rights",
        ["content_right_id"], ["id"],
        source_schema=SCHEMA, referent_schema=SCHEMA,
    )

    op.drop_constraint(
        UQ_TRANSCRIPT_ASSETS_ID_VIDEO_ID,
        "transcript_assets", schema=SCHEMA, type_="unique",
    )
    op.drop_constraint(
        UQ_CONTENT_RIGHTS_ID_VIDEO_ID,
        "content_rights", schema=SCHEMA, type_="unique",
    )
