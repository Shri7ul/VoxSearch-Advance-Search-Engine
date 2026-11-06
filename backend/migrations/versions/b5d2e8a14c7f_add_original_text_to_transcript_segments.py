"""add original_text to transcript_segments

Revision ID: b5d2e8a14c7f
Revises: fe7b2b8e5323
Create Date: 2026-11-06 12:00:00.000000

Goal 09 corrective review (Issue 1) — persist the verbatim original transcript
text SEPARATELY from the normalized text.

The existing ``"text"`` column keeps its meaning: it still stores the normalized
text, stays NOT NULL, and keeps ``ck_transcript_segments_text_nonempty``. This
migration adds ONE nullable column and changes NOTHING else.

Safety properties:
* DDL-only: a single ADD COLUMN, with NO DEFAULT, NO UPDATE/backfill, and no
  INSERT/UPDATE/DELETE. No existing row is read or rewritten.
* ``original_text`` is NULLABLE. Rows created before this migration never
  captured an original, so they legitimately get NULL ("not captured"). NULL is
  NOT backfilled from ``"text"`` — copying normalized text into the original
  column would silently fabricate a false "original" and would violate the
  requirement to keep the two values separate.
* New rows written by the ingestion service populate BOTH columns: verbatim
  original in ``original_text``, normalized text in ``"text"``.
* Adding a nullable column with no default is a metadata-only operation on
  PostgreSQL 11+ (no table rewrite, no long lock).
* No new index (``original_text`` is not a query target), no new CHECK
  constraint (application-level validation covers non-blank originals), no
  table/schema drops, and ``public`` / Supabase-managed objects are untouched.
* ``downgrade()`` is NEVER executed against the live database. It drops ONLY
  the column added here; note it would discard any originals captured after
  this migration was applied.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b5d2e8a14c7f"
down_revision: Union[str, None] = "fe7b2b8e5323"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "voxsearch"


def upgrade() -> None:
    op.add_column(
        "transcript_segments",
        sa.Column("original_text", sa.Text(), nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("transcript_segments", "original_text", schema=SCHEMA)
