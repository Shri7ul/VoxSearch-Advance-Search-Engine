"""Offline tests for the Goal 16 dependency-free ``VectorType``.

These tests use NO database, no ``pgvector`` Python package, no numpy, and no
``.env``. They verify the pure stdlib literal formatting/parsing, offline
PostgreSQL DDL rendering, and the ``CAST(... AS vector(1024))`` bind guard.

Honesty boundary: rendering DDL / SQL proves SYNTAX only. The actual
INSERT/CAST/bind behaviour against a live PostgreSQL + pgvector server is NOT
VERIFIED here (no local instance); it is covered by the opt-in, skipped-by-
default ``live_pgvector`` test.
"""
import pytest
from sqlalchemy import Column, Integer, MetaData, Table, insert
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.db.types import DEFAULT_DIM, VectorType


def _dialect():
    return postgresql.dialect()


def _table(dim: int = 1024) -> Table:
    return Table(
        "vec_probe", MetaData(),
        Column("id", Integer, primary_key=True),
        Column("embedding", VectorType(dim), nullable=False),
    )


# --- 1. Construction / metadata --------------------------------------------

def test_default_dimension_is_1024() -> None:
    assert DEFAULT_DIM == 1024
    assert VectorType().dim == 1024


def test_cache_ok_is_true() -> None:
    assert VectorType(1024).cache_ok is True


@pytest.mark.parametrize("bad", [0, -1, 1.5, "1024", None, True, False])
def test_invalid_dimensions_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        VectorType(bad)


def test_invalid_validate_norm_and_tolerance_rejected() -> None:
    with pytest.raises(TypeError):
        VectorType(1024, validate_norm="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VectorType(1024, validate_norm=True, norm_tolerance=0)
    with pytest.raises(ValueError):
        VectorType(1024, validate_norm=True, norm_tolerance=-0.1)
    with pytest.raises(ValueError):
        VectorType(1024, validate_norm=True, norm_tolerance=True)  # type: ignore[arg-type]


def test_get_col_spec() -> None:
    assert VectorType(3).get_col_spec() == "vector(3)"
    assert VectorType(1024).get_col_spec() == "vector(1024)"


# --- 2. DDL rendering (offline) --------------------------------------------

def test_create_table_renders_vector_1024_not_null() -> None:
    ddl = str(CreateTable(_table()).compile(dialect=_dialect()))
    assert "embedding vector(1024) NOT NULL" in ddl


def test_insert_renders_cast_guard() -> None:
    stmt = insert(_table()).values(id=1, embedding=[0.1, 0.2])
    sql = str(stmt.compile(dialect=_dialect()))
    assert "CAST(" in sql and "AS vector(1024))" in sql


def test_bind_processor_is_registered_on_compiled_insert() -> None:
    """The execution path applies the bind processor (proven here offline)."""
    stmt = insert(_table()).values(id=1, embedding=[0.1, 0.2])
    compiled = stmt.compile(dialect=_dialect())
    processors = getattr(compiled, "_bind_processors", {})
    assert "embedding" in processors
    assert callable(processors["embedding"])


# --- 3. bind_processor ------------------------------------------------------

def test_bind_processor_formats_pgvector_literal() -> None:
    proc = VectorType(4).bind_processor(_dialect())
    out = proc([0.1, 0.2, -0.3, 1.0])
    assert out == "[0.1,0.2,-0.3,1.0]"


def test_bind_processor_accepts_none() -> None:
    assert VectorType(4).bind_processor(_dialect())(None) is None


def test_bind_processor_rejects_wrong_dimension() -> None:
    proc = VectorType(4).bind_processor(_dialect())
    with pytest.raises(ValueError):
        proc([0.1, 0.2, 0.3])


def test_bind_processor_rejects_nan_and_inf() -> None:
    proc = VectorType(4).bind_processor(_dialect())
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            proc([bad, 0.0, 0.0, 0.0])


def test_bind_processor_rejects_bool_and_non_real() -> None:
    proc = VectorType(4).bind_processor(_dialect())
    with pytest.raises(TypeError):
        proc([True, 0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        proc(["x", 0.0, 0.0, 0.0])


def test_bind_processor_rejects_non_sequence() -> None:
    proc = VectorType(4).bind_processor(_dialect())
    with pytest.raises(TypeError):
        proc("[0.1,0.2,0.3,0.4]")
    with pytest.raises(TypeError):
        proc(0.5)


def test_bind_processor_validates_normalization_when_enabled() -> None:
    proc = VectorType(4, validate_norm=True, norm_tolerance=1e-3).bind_processor(_dialect())
    # Unit vector passes
    assert proc([1.0, 0.0, 0.0, 0.0]) == "[1.0,0.0,0.0,0.0]"
    # Slightly off within tolerance (1e-3) passes
    assert proc([0.9995, 0.0, 0.0, 0.0]) == "[0.9995,0.0,0.0,0.0]"
    # Unnormalized vectors fail
    with pytest.raises(ValueError, match="L2 norm"):
        proc([0.5, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="L2 norm"):
        proc([2.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="L2 norm"):
        proc([0.0, 0.0, 0.0, 0.0])


def test_bind_processor_skips_normalization_by_default() -> None:
    proc = VectorType(4).bind_processor(_dialect())
    # Non-unit vectors succeed when validate_norm is False (default)
    assert proc([0.5, 0.0, 0.0, 0.0]) == "[0.5,0.0,0.0,0.0]"


# --- 4. result_processor ----------------------------------------------------

def test_result_processor_parses_pgvector_literal() -> None:
    proc = VectorType(4).result_processor(_dialect(), None)
    assert proc("[0.1,0.2,-0.3,1.0]") == (0.1, 0.2, -0.3, 1.0)


def test_result_processor_accepts_native_sequence() -> None:
    proc = VectorType(4).result_processor(_dialect(), None)
    assert proc([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)


def test_result_processor_accepts_none() -> None:
    assert VectorType(4).result_processor(_dialect(), None)(None) is None


def test_result_processor_rejects_bad_literal_and_wrong_dim() -> None:
    proc = VectorType(4).result_processor(_dialect(), None)
    with pytest.raises(ValueError):
        proc("not a vector")
    with pytest.raises(ValueError):
        proc("[0.1,0.2,0.3]")


@pytest.mark.parametrize("bad_literal", [
    "[0.1, nan, 0.2, 0.3]",
    "[0.1, inf, 0.2, 0.3]",
    "[0.1, -inf, 0.2, 0.3]",
])
def test_result_processor_rejects_nan_and_inf_string(bad_literal: str) -> None:
    proc = VectorType(4).result_processor(_dialect(), None)
    with pytest.raises(ValueError, match="not finite"):
        proc(bad_literal)


def test_result_processor_rejects_nan_and_inf_native_sequence() -> None:
    proc = VectorType(4).result_processor(_dialect(), None)
    with pytest.raises(ValueError, match="not finite"):
        proc([0.1, float("nan"), 0.2, 0.3])
    with pytest.raises(ValueError, match="not finite"):
        proc([0.1, float("inf"), 0.2, 0.3])


def test_result_processor_accepts_bytes_and_bytearray() -> None:
    proc = VectorType(4).result_processor(_dialect(), None)
    assert proc(b"[0.1,0.2,-0.3,1.0]") == (0.1, 0.2, -0.3, 1.0)
    assert proc(bytearray(b"[0.1,0.2,-0.3,1.0]")) == (0.1, 0.2, -0.3, 1.0)


def test_result_processor_validates_normalization_when_enabled() -> None:
    proc = VectorType(4, validate_norm=True, norm_tolerance=1e-3).result_processor(
        _dialect(), None
    )
    assert proc("[1.0, 0.0, 0.0, 0.0]") == (1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="L2 norm"):
        proc("[0.5, 0.0, 0.0, 0.0]")


def test_literal_round_trip_preserves_1024_floats() -> None:
    vt = VectorType(1024)
    bind = vt.bind_processor(_dialect())
    result = vt.result_processor(_dialect(), None)
    original = tuple(float(i) / 10000.0 for i in range(1024))
    restored = result(bind(list(original)))
    assert len(restored) == 1024
    assert all(isinstance(x, float) for x in restored)
    assert restored == pytest.approx(original)
