"""Dependency-free pgvector column type for SQLAlchemy (Goal 16).

Why this exists
---------------
Goal 16 persists Goal 15's 1024-dimensional dense embeddings into a pgvector
``vector(1024)`` column. The core backend ``.venv`` deliberately has NO
``pgvector`` Python package and NO ``numpy`` (Goal 15 isolation), and Goal 16
must NOT add either. A SQLAlchemy :class:`~sqlalchemy.types.UserDefinedType`
is enough: pgvector's on-the-wire / text representation is a plain
``[0.1,0.2,...]`` literal, which pure stdlib code can format and parse.

What it does
------------
* :meth:`VectorType.get_col_spec` renders ``vector(1024)`` so PostgreSQL DDL
  ``CREATE TABLE`` compiles the column correctly (verified offline in tests).
* :meth:`VectorType.bind_processor` serializes a Python sequence of finite
  floats into a pgvector text literal and validates the dimension + finiteness
  at the boundary (fail-closed; never silently coerces NaN/Inf or wrong dims).
* :meth:`VectorType.result_processor` parses that literal back into a
  ``tuple[float, ...]`` on read.
* :meth:`VectorType.bind_expression` wraps every bind in an explicit
  ``CAST(... AS vector(1024))``. psycopg3 sends a Python ``str`` with a
  ``text``/unknown OID; the explicit cast removes any ambiguity about the
  target type on INSERT/UPDATE.

Honesty boundary (Goal 16)
--------------------------
Only the OFFLINE DDL rendering and the pure literal round-trip are verified by
the automated tests. The actual bind/INSERT/CAST behaviour against a running
PostgreSQL + pgvector server is **NOT VERIFIED** here (no local instance is
available); it is covered by an opt-in ``live_pgvector`` test that is skipped by
default. This module imports no ML library, no ``pgvector`` package, no numpy,
no credentials, and no ``.env``.
"""
from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any, Callable

from sqlalchemy import cast
from sqlalchemy.types import UserDefinedType

# pgvector text literal: "[0.1,0.2,...]" (optional surrounding whitespace).
_LITERAL_RE = re.compile(r"^\s*\[(.*)\]\s*$", re.DOTALL)

# Default dense dimension, matching app.embeddings.contracts.DENSE_DIMENSION.
# Kept as a local default (not imported) so this DB layer stays decoupled from
# the embedding library and never pulls pydantic into a pure DB import.
DEFAULT_DIM = 1024


class VectorType(UserDefinedType):
    """A fixed-dimension pgvector ``vector(dim)`` column, pure stdlib.

    Parameters
    ----------
    dim:
        Exact vector dimension (default 1024). Must be a positive int; a
        ``bool`` is rejected (it is an ``int`` subclass but never a dimension).
    validate_norm:
        Whether to enforce unit L2 normalization (norm ≈ 1.0) on bind.
    norm_tolerance:
        Maximum allowed absolute deviation of L2 norm from 1.0 (default 1e-3).
    """

    cache_ok = True

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        *,
        validate_norm: bool = False,
        norm_tolerance: float = 1e-3,
    ) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        if not isinstance(validate_norm, bool):
            raise TypeError("validate_norm must be a bool")
        if (
            isinstance(norm_tolerance, bool)
            or not isinstance(norm_tolerance, Real)
            or norm_tolerance <= 0
        ):
            raise ValueError("norm_tolerance must be a positive number")
        self.dim = dim
        self.validate_norm = validate_norm
        self.norm_tolerance = float(norm_tolerance)

    def get_col_spec(self, **kw: Any) -> str:  # noqa: ARG002 - SQLAlchemy API
        return f"vector({self.dim})"

    def bind_processor(
        self, dialect: Any  # noqa: ARG002 - unused, part of the API
    ) -> Callable[[Any], Any]:
        dim = self.dim
        validate_norm = self.validate_norm
        norm_tolerance = self.norm_tolerance

        def process(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"vector expects a sequence of {dim} finite floats, "
                    f"got {type(value).__name__}"
                )
            if len(value) != dim:
                raise ValueError(
                    f"vector expects exactly {dim} values, got {len(value)}"
                )
            parts: list[str] = []
            floats: list[float] = []
            for pos, item in enumerate(value):
                if isinstance(item, bool) or not isinstance(item, Real):
                    raise TypeError(
                        f"vector[{pos}] expects a real number, "
                        f"got {type(item).__name__}"
                    )
                number = float(item)
                if not math.isfinite(number):
                    raise ValueError(f"vector[{pos}] is not finite (NaN/Inf)")
                if validate_norm:
                    floats.append(number)
                parts.append(repr(number))
            if validate_norm:
                norm = math.hypot(*floats)
                if not math.isfinite(norm) or abs(norm - 1.0) > norm_tolerance:
                    raise ValueError(
                        f"vector L2 norm {norm:.6f} deviates from 1.0 by more than "
                        f"tolerance {norm_tolerance}"
                    )
            return "[" + ",".join(parts) + "]"

        return process

    def result_processor(
        self, dialect: Any, coltype: Any  # noqa: ARG002 - part of the API
    ) -> Callable[[Any], Any]:
        dim = self.dim
        validate_norm = self.validate_norm
        norm_tolerance = self.norm_tolerance

        def process(value: Any) -> Any:
            if value is None:
                return None
            # Some drivers already return a native sequence.
            if isinstance(value, (list, tuple)):
                out = tuple(float(x) for x in value)
            elif isinstance(value, (bytes, bytearray)):
                value_str = value.decode("utf-8")
                match = _LITERAL_RE.match(value_str)
                if match is None:
                    raise ValueError("vector result is not a pgvector text literal")
                body = match.group(1).strip()
                raw = body.split(",") if body else []
                out = tuple(float(x) for x in raw)
            elif isinstance(value, str):
                match = _LITERAL_RE.match(value)
                if match is None:
                    raise ValueError("vector result is not a pgvector text literal")
                body = match.group(1).strip()
                raw = body.split(",") if body else []
                out = tuple(float(x) for x in raw)
            else:
                raise TypeError(
                    f"cannot parse vector result of type {type(value).__name__}"
                )
            if len(out) != dim:
                raise ValueError(
                    f"vector result has {len(out)} values, expected {dim}"
                )
            for pos, val in enumerate(out):
                if not math.isfinite(val):
                    raise ValueError(f"vector result[{pos}] is not finite (NaN/Inf)")
            if validate_norm:
                norm = math.hypot(*out)
                if not math.isfinite(norm) or abs(norm - 1.0) > norm_tolerance:
                    raise ValueError(
                        f"vector result L2 norm {norm:.6f} deviates from 1.0 by more than "
                        f"tolerance {norm_tolerance}"
                    )
            return out

        return process

    def bind_expression(self, bindvalue: Any) -> Any:
        """Emit ``CAST(bindvalue AS vector(dim))`` for every bound value.

        This is a defensive guard against psycopg3 inferring a plain ``text``
        OID for the string literal. The resulting behaviour on a live server is
        explicitly NOT VERIFIED in this goal (see module docstring).
        """
        return cast(bindvalue, self)


__all__ = ["DEFAULT_DIM", "VectorType"]
