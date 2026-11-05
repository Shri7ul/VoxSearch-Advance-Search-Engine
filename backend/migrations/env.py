"""Alembic environment for VoxSearch.

Reuses the existing application SQLAlchemy configuration:

* ``DATABASE_URL`` is read from :func:`app.core.config.get_settings` (a
  pydantic ``SecretStr``) and unwrapped safely by ``database_url()``.
* The psycopg(v3) dialect is forced by ``_psycopg_url()`` — the same
  normalization used by :func:`app.db.session.get_engine`.

Credentials are never hardcoded, printed, or logged. Alembic's version table
is placed in the dedicated ``voxsearch`` schema so migration bookkeeping is
isolated alongside the application objects.
"""
from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
from sqlalchemy import create_engine, pool, text

# Make the backend package importable regardless of the invoking cwd, and load
# the local .env explicitly (by absolute path) so DATABASE_URL is available to
# get_settings() even when alembic is launched from the repository root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env", override=False)

from app.core.config import get_settings
from app.db.models import SCHEMA, Base
from app.db.session import _psycopg_url, database_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """Return the psycopg(v3) URL from application settings.

    ``database_url()`` unwraps the ``SecretStr`` and raises a clear,
    credential-free error when it is missing. The returned value is passed
    only to SQLAlchemy; it is never printed or logged.
    """
    return _psycopg_url(database_url(get_settings()))


def include_name(name, type_, parent_names):
    """Restrict autogenerate / inspection to the ``voxsearch`` schema only."""
    if type_ == "schema":
        return name == SCHEMA
    return True


def include_object(obj, name, type_, reflected, compare_to):
    """Exclude Alembic's own version table from autogenerate diffs."""
    if type_ == "table" and name == "alembic_version":
        return False
    return True


def run_migrations_offline() -> None:
    """Render migration SQL to stdout without opening a DB connection.

    Useful for reviewing exactly what DDL would be applied. No credentials are
    emitted — only SQL statements.
    """
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=SCHEMA,
        include_schemas=True,
        include_name=include_name,
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against the live database."""
    # NullPool: migrations are short-lived and should not retain connections.
    connectable = create_engine(get_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        # The version table lives in ``voxsearch`` (version_table_schema), so
        # the schema must exist before Alembic creates it. This statement is
        # idempotent and non-destructive; it is committed standalone so that
        # Alembic can begin its own migration transaction cleanly.
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=SCHEMA,
            include_schemas=True,
            include_name=include_name,
            include_object=include_object,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
