"""A closed expression algebra for defining facial measurements.

A measurement is not a Python function. It is a small expression tree built
from this fixed set of nodes. Closing the algebra buys four things that an
arbitrary callable cannot give:

* **Serialisation.** A measurement round-trips through JSON, so the exact
  formula that produced a number lands in the run manifest.
* **Content hashing.** ``Expr.fingerprint`` is stable across processes, so a
  report can say *which version* of a definition it used, and a cache can be
  keyed on it.
* **Static dependency analysis.** ``Expr.landmarks()`` reports what the formula
  needs before anything is evaluated, so a backend that cannot supply
  ``gonion_l`` skips the gonial angle with a named reason instead of failing
  mid-evaluation.
* **Batched evaluation.** Every node broadcasts over leading axes, so a
  Monte-Carlo ensemble of 4096 perturbed point sets evaluates in one pass.

The type split between :class:`PointExpr` (yields ``(..., dim)``) and
:class:`ScalarExpr` (yields ``(...)``) is what stops a length being passed
where a point belongs.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from . import geometry as geo
from .landmarks import Landmark, PointSet

_REGISTRY: dict[str, type["Expr"]] = {}


def _register(cls: type["Expr"]) -> type["Expr"]:
    op = cls.op
    if op in _REGISTRY:
        raise ValueError(f"duplicate expression op {op!r}")
    _REGISTRY[op] = cls
    return cls


class Expr(ABC):
    """Base of the algebra. Subclasses are frozen dataclasses."""

    op: ClassVar[str]

    @abstractmethod
    def eval(self, ps: PointSet) -> NDArray[np.float64]: ...

    @abstractmethod
    def landmarks(self) -> frozenset[Landmark]:
        """Every landmark this subtree reads."""

    @abstractmethod
    def _args(self) -> dict[str, Any]:
        """Serialisable constructor arguments, excluding ``op``."""

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, **self._args()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """Stable 12-hex-character content hash of this formula."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Expr":
        try:
            cls = _REGISTRY[d["op"]]
        except KeyError as exc:
            raise ValueError(f"unknown expression op {d.get('op')!r}") from exc
        return cls._from_args({k: v for k, v in d.items() if k != "op"})

    @classmethod
    def _from_args(cls, args: dict[str, Any]) -> "Expr":
        raise NotImplementedError(f"{cls.__name__} does not implement _from_args")


class PointExpr(Expr):
    """Yields coordinates with shape ``(..., dim)``."""


class VectorExpr(Expr):
    """Yields a direction with shape ``(..., dim)``. Not necessarily unit length."""


class ScalarExpr(Expr):
    """Yields a scalar per batch element, shape ``(...)``."""

    def __truediv__(self, other: "ScalarExpr") -> "Ratio":
        return Ratio(self, other)

    def __add__(self, other: "ScalarExpr") -> "Sum":
        return Sum((self, other))

    def __sub__(self, other: "ScalarExpr") -> "Diff":
        return Diff(self, other)

    def __mul__(self, other: "ScalarExpr") -> "Product":
        return Product((self, other))


# --------------------------------------------------------------------------
# Point-valued nodes
# --------------------------------------------------------------------------


@_register
@dataclass(frozen=True)
class Pt(PointExpr):
    """A named landmark."""

    op: ClassVar[str] = "pt"
    name: Landmark

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return ps.get(self.name)

    def landmarks(self) -> frozenset[Landmark]:
        return frozenset({self.name})

    def _args(self) -> dict[str, Any]:
        return {"name": self.name.value}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Pt":
        return cls(Landmark(a["name"]))


@_register
@dataclass(frozen=True)
class Mid(PointExpr):
    """Midpoint of two points. The usual way to synthesise a median landmark."""

    op: ClassVar[str] = "mid"
    a: PointExpr
    b: PointExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return geo.midpoint(self.a.eval(ps), self.b.eval(ps))

    def landmarks(self) -> frozenset[Landmark]:
        return self.a.landmarks() | self.b.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"a": self.a.to_dict(), "b": self.b.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Mid":
        return cls(Expr.from_dict(a["a"]), Expr.from_dict(a["b"]))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Vector-valued nodes
# --------------------------------------------------------------------------


@_register
@dataclass(frozen=True)
class Axis(VectorExpr):
    """A canonical-frame axis: ``x`` right, ``y`` up, ``z`` toward the viewer.

    A leading ``-`` negates it (``"-x"`` points to the subject's left). The
    negated forms matter: one tilt formula serves both sides of the face only
    if each side can name the lateral direction that points away from the
    midline.

    Resolves to the point set's dimensionality at evaluation time, so a formula
    written against the canonical 3D frame degrades correctly when handed 2D
    image points -- except for ``z``, which has no 2D meaning and raises rather
    than quietly returning a zero vector.
    """

    op: ClassVar[str] = "axis"
    which: str

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        dim = ps.dim
        name = self.which.lstrip("+-")
        sign = -1.0 if self.which.startswith("-") else 1.0
        try:
            i = {"x": 0, "y": 1, "z": 2}[name]
        except KeyError as exc:
            raise ValueError(f"unknown axis {self.which!r}") from exc
        if i >= dim:
            raise ValueError(f"axis {self.which!r} is undefined for {dim}D points")
        v = np.zeros(dim)
        v[i] = sign
        return np.broadcast_to(v, (*ps.batch_shape, dim))

    def landmarks(self) -> frozenset[Landmark]:
        return frozenset()

    def _args(self) -> dict[str, Any]:
        return {"which": self.which}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Axis":
        return cls(a["which"])


@_register
@dataclass(frozen=True)
class Vec(VectorExpr):
    """The direction from ``a`` to ``b``."""

    op: ClassVar[str] = "vec"
    a: PointExpr
    b: PointExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return self.b.eval(ps) - self.a.eval(ps)

    def landmarks(self) -> frozenset[Landmark]:
        return self.a.landmarks() | self.b.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"a": self.a.to_dict(), "b": self.b.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Vec":
        return cls(Expr.from_dict(a["a"]), Expr.from_dict(a["b"]))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Scalar-valued nodes
# --------------------------------------------------------------------------


@_register
@dataclass(frozen=True)
class Const(ScalarExpr):
    op: ClassVar[str] = "const"
    value: float

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return np.broadcast_to(np.float64(self.value), ps.batch_shape)

    def landmarks(self) -> frozenset[Landmark]:
        return frozenset()

    def _args(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Const":
        return cls(float(a["value"]))


@_register
@dataclass(frozen=True)
class Dist(ScalarExpr):
    """Euclidean distance between two points."""

    op: ClassVar[str] = "dist"
    a: PointExpr
    b: PointExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return geo.distance(self.a.eval(ps), self.b.eval(ps))

    def landmarks(self) -> frozenset[Landmark]:
        return self.a.landmarks() | self.b.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"a": self.a.to_dict(), "b": self.b.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Dist":
        return cls(Expr.from_dict(a["a"]), Expr.from_dict(a["b"]))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class AngleAt(ScalarExpr):
    """Interior angle at ``vertex`` in the path a -> vertex -> c, in degrees."""

    op: ClassVar[str] = "angle_at"
    a: PointExpr
    vertex: PointExpr
    c: PointExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return geo.angle_at(self.a.eval(ps), self.vertex.eval(ps), self.c.eval(ps))

    def landmarks(self) -> frozenset[Landmark]:
        return self.a.landmarks() | self.vertex.landmarks() | self.c.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"a": self.a.to_dict(), "vertex": self.vertex.to_dict(), "c": self.c.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "AngleAt":
        return cls(*(Expr.from_dict(a[k]) for k in ("a", "vertex", "c")))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class AngleBetween(ScalarExpr):
    """Undirected angle between two directions, folded into [0, 90]."""

    op: ClassVar[str] = "angle_between"
    u: VectorExpr
    v: VectorExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        u, v = self.u.eval(ps), self.v.eval(ps)
        zero = np.zeros_like(u)
        return geo.angle_between_lines(zero, u, zero, v)

    def landmarks(self) -> frozenset[Landmark]:
        return self.u.landmarks() | self.v.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"u": self.u.to_dict(), "v": self.v.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "AngleBetween":
        return cls(Expr.from_dict(a["u"]), Expr.from_dict(a["v"]))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class SignedTilt(ScalarExpr):
    """Signed angle of the directed segment a -> b against an axis, in degrees.

    This is the canthal-tilt form: positive means ``b`` lies on the axis's
    positive-perpendicular side of ``a``.
    """

    op: ClassVar[str] = "signed_tilt"
    a: PointExpr
    b: PointExpr
    axis: VectorExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return geo.signed_angle_to_axis(self.a.eval(ps), self.b.eval(ps), self.axis.eval(ps))

    def landmarks(self) -> frozenset[Landmark]:
        return self.a.landmarks() | self.b.landmarks() | self.axis.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"a": self.a.to_dict(), "b": self.b.to_dict(), "axis": self.axis.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "SignedTilt":
        return cls(*(Expr.from_dict(a[k]) for k in ("a", "b", "axis")))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class ProjLength(ScalarExpr):
    """Extent of ``a`` -> ``b`` measured along ``axis``. Signed."""

    op: ClassVar[str] = "proj_length"
    a: PointExpr
    b: PointExpr
    axis: VectorExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return geo.project_on_axis(self.b.eval(ps), self.a.eval(ps), self.axis.eval(ps))

    def landmarks(self) -> frozenset[Landmark]:
        return self.a.landmarks() | self.b.landmarks() | self.axis.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"a": self.a.to_dict(), "b": self.b.to_dict(), "axis": self.axis.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "ProjLength":
        return cls(*(Expr.from_dict(a[k]) for k in ("a", "b", "axis")))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class LineOffset(ScalarExpr):
    """Signed offset of ``p`` from the line through ``a`` and ``b``, along ``normal``.

    The signed form matters: profile aesthetics are stated as "lower lip 2 mm
    behind the E-line", and an unsigned distance throws away the half of that
    statement that carries the meaning.
    """

    op: ClassVar[str] = "line_offset"
    p: PointExpr
    a: PointExpr
    b: PointExpr
    normal: VectorExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return geo.signed_point_to_line_offset(
            self.p.eval(ps), self.a.eval(ps), self.b.eval(ps), self.normal.eval(ps)
        )

    def landmarks(self) -> frozenset[Landmark]:
        return (
            self.p.landmarks() | self.a.landmarks() | self.b.landmarks() | self.normal.landmarks()
        )

    def _args(self) -> dict[str, Any]:
        return {
            "p": self.p.to_dict(),
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "normal": self.normal.to_dict(),
        }

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "LineOffset":
        return cls(*(Expr.from_dict(a[k]) for k in ("p", "a", "b", "normal")))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class Ratio(ScalarExpr):
    """Quotient of two scalars. Zero denominators propagate as NaN, not as an
    exception -- a degenerate sample in a Monte-Carlo ensemble should widen the
    interval, not abort the measurement."""

    op: ClassVar[str] = "ratio"
    num: ScalarExpr
    den: ScalarExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        n = np.asarray(self.num.eval(ps), dtype=float)
        d = np.asarray(self.den.eval(ps), dtype=float)
        out = np.full(np.broadcast_shapes(n.shape, d.shape), np.nan)
        np.divide(n, d, out=out, where=np.abs(d) > 1e-12)
        return out

    def landmarks(self) -> frozenset[Landmark]:
        return self.num.landmarks() | self.den.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"num": self.num.to_dict(), "den": self.den.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Ratio":
        return cls(Expr.from_dict(a["num"]), Expr.from_dict(a["den"]))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class Sum(ScalarExpr):
    op: ClassVar[str] = "sum"
    terms: tuple[ScalarExpr, ...]

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return np.sum([t.eval(ps) for t in self.terms], axis=0)

    def landmarks(self) -> frozenset[Landmark]:
        return frozenset().union(*(t.landmarks() for t in self.terms)) if self.terms else frozenset()

    def _args(self) -> dict[str, Any]:
        return {"terms": [t.to_dict() for t in self.terms]}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Sum":
        return cls(tuple(Expr.from_dict(t) for t in a["terms"]))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class Diff(ScalarExpr):
    op: ClassVar[str] = "diff"
    a: ScalarExpr
    b: ScalarExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return self.a.eval(ps) - self.b.eval(ps)

    def landmarks(self) -> frozenset[Landmark]:
        return self.a.landmarks() | self.b.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"a": self.a.to_dict(), "b": self.b.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Diff":
        return cls(Expr.from_dict(a["a"]), Expr.from_dict(a["b"]))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class Product(ScalarExpr):
    op: ClassVar[str] = "product"
    factors: tuple[ScalarExpr, ...]

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return np.prod([f.eval(ps) for f in self.factors], axis=0)

    def landmarks(self) -> frozenset[Landmark]:
        return (
            frozenset().union(*(f.landmarks() for f in self.factors))
            if self.factors
            else frozenset()
        )

    def _args(self) -> dict[str, Any]:
        return {"factors": [f.to_dict() for f in self.factors]}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Product":
        return cls(tuple(Expr.from_dict(f) for f in a["factors"]))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class Abs(ScalarExpr):
    op: ClassVar[str] = "abs"
    x: ScalarExpr

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return np.abs(self.x.eval(ps))

    def landmarks(self) -> frozenset[Landmark]:
        return self.x.landmarks()

    def _args(self) -> dict[str, Any]:
        return {"x": self.x.to_dict()}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Abs":
        return cls(Expr.from_dict(a["x"]))  # type: ignore[arg-type]


@_register
@dataclass(frozen=True)
class Mean(ScalarExpr):
    """Average of several scalars. Used to pool a bilateral pair into one
    reported value while the per-side values stay available."""

    op: ClassVar[str] = "mean"
    terms: tuple[ScalarExpr, ...]

    def eval(self, ps: PointSet) -> NDArray[np.float64]:
        return np.mean([t.eval(ps) for t in self.terms], axis=0)

    def landmarks(self) -> frozenset[Landmark]:
        return frozenset().union(*(t.landmarks() for t in self.terms)) if self.terms else frozenset()

    def _args(self) -> dict[str, Any]:
        return {"terms": [t.to_dict() for t in self.terms]}

    @classmethod
    def _from_args(cls, a: dict[str, Any]) -> "Mean":
        return cls(tuple(Expr.from_dict(t) for t in a["terms"]))  # type: ignore[arg-type]


def registered_ops() -> frozenset[str]:
    """Every op name the algebra understands. The algebra is closed by design."""
    return frozenset(_REGISTRY)
