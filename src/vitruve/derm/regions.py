"""Facial regions as polygons, derived from landmarks rather than from a parser.

The obvious way to cut a face into regions is a learned face parser. It is also
the way that quietly makes the whole pipeline non-commercial. Every widely used
parser -- the SegFormer port at ``jonathandinu/face-parsing``, every BiSeNet
checkpoint, every "face parsing" model on the hub with an MIT tag -- was trained
on CelebAMask-HQ, whose terms forbid reproducing, selling or trading any portion
of the derived data. The MIT tag covers the code. It does not cover the weights,
and it does not travel with them.

So Vitruve's default region set is built out of the landmarks it already has.
The construction is a rigid frame plus a table of offsets: the origin sits
between the pupils, ``u`` points to the subject's right, ``v`` points up, and
every offset is expressed in units of the interpupillary span. That buys three
things a pixel parser does not give you for free:

* **Roll invariance.** The frame rotates with the eye line, so a tilted
  photograph produces the same regions rather than the same rectangles.
* **Scale invariance.** Offsets in span units mean one table works at any
  resolution and any distance.
* **Named failure.** A region whose landmarks are missing is reported as
  unavailable with those landmarks named, which is the same discipline the
  measurement layer applies. A parser instead returns a confident mask over
  whatever it hallucinated.

What the geometric path gives up is boundary fidelity. A polygon does not know
where hair, eyebrow, nostril or vermilion actually are, so a region can include
pixels that are not skin. Two mitigations are provided: polygons carry holes
(the perioral region excludes the vermilion by construction), and
:func:`refine_with_parser` will intersect every region with a parser-derived
skin mask for callers who have accepted the non-commercial tier. The geometric
path must work alone, and it does.

Coordinate convention follows ``core.landmarks``: **+x is the subject's right in
image coordinates, +y is up.** Image arrays are indexed row-major from the top,
so :func:`rasterize` flips the vertical axis and says so in its signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.landmarks import Landmark, PointSet
from ..models.licensing import FACE_PARSING_SEGFORMER, Provenance, Tier, require


class Region(str, Enum):
    """Named facial regions.

    Membership is closed for the same reason the landmark vocabulary is: a
    finding that names a region not in this list has no defined geometry, and a
    report that prints one is printing a string rather than a measurement.
    """

    FOREHEAD = "forehead"
    GLABELLA = "glabella"
    T_ZONE = "t_zone"
    NASAL = "nasal"
    PERIORBITAL_L = "periorbital_l"
    PERIORBITAL_R = "periorbital_r"
    INFRAORBITAL_L = "infraorbital_l"
    INFRAORBITAL_R = "infraorbital_r"
    MALAR_L = "malar_l"
    MALAR_R = "malar_r"
    LATERAL_CHEEK_L = "lateral_cheek_l"
    LATERAL_CHEEK_R = "lateral_cheek_r"
    PERIORAL = "perioral"

    @property
    def side(self) -> str | None:
        if self.value.endswith("_l"):
            return "l"
        if self.value.endswith("_r"):
            return "r"
        return None


#: Regions paired with the reference region their colour is read against.
#:
#: The pairing is the whole point of the colorimetry module. An absolute a*
#: threshold for "red" depends on illuminant, camera, and constitutive skin
#: tone; a malar-minus-lateral-cheek difference in the same photograph cancels
#: all three to first order. Erythema pairs a region with adjacent cheek skin;
#: periorbital hyperpigmentation pairs the infraorbital band with the ipsilateral
#: malar region, which is the closest large patch of the same person's skin.
REFERENCE_PAIRS: Mapping[Region, Region] = {
    Region.MALAR_L: Region.LATERAL_CHEEK_L,
    Region.MALAR_R: Region.LATERAL_CHEEK_R,
    Region.INFRAORBITAL_L: Region.MALAR_L,
    Region.INFRAORBITAL_R: Region.MALAR_R,
}
#: Regions with no defensible paired reference. The nose and the glabella have
#: no adjacent patch of the same person's skin at the same illumination angle,
#: so a contrast against a cheek would report the lighting as much as the skin.
#: They are still measurable in absolute terms; they are simply not paired here,
#: and a caller who wants such a contrast has to ask for it explicitly through
#: :func:`colorimetry.paired_contrast` and own the confound.
UNPAIRED: frozenset[Region] = frozenset({Region.NASAL, Region.GLABELLA, Region.T_ZONE})


class RegionUnavailable(KeyError):
    """Raised when a region's landmarks are not present in the point set."""

    def __init__(self, region: Region, missing: Sequence[str]) -> None:
        super().__init__(f"{region.value}: missing {', '.join(missing)}")
        self.region = region
        self.missing = tuple(missing)


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionPolygon:
    """A closed polygon in image coordinates, optionally with holes.

    ``vertices`` is ``(k, 2)`` in the same frame as the point set it was built
    from. ``holes`` are subtracted at rasterisation time, which is how the
    perioral region excludes the vermilion without needing a parser.
    """

    region: Region
    vertices: NDArray[np.float64]
    holes: tuple[NDArray[np.float64], ...] = ()
    landmarks_used: tuple[Landmark, ...] = ()
    source: str = "geometric"
    rationale: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 3:
            raise ValueError(
                f"{self.region.value}: vertices must be (k>=3, 2), got {v.shape}"
            )
        object.__setattr__(self, "vertices", v)
        object.__setattr__(
            self, "holes", tuple(np.asarray(h, dtype=float) for h in self.holes)
        )

    @property
    def centroid(self) -> NDArray[np.float64]:
        return self.vertices.mean(axis=0)

    @property
    def area(self) -> float:
        """Signed-free polygon area with holes removed, in squared pixels."""
        return abs(_shoelace(self.vertices)) - sum(abs(_shoelace(h)) for h in self.holes)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` of the outer ring."""
        lo = self.vertices.min(axis=0)
        hi = self.vertices.max(axis=0)
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    def eroded(self, fraction: float) -> "RegionPolygon":
        """Shrink toward the centroid.

        Region boundaries are the least trustworthy part of a geometric
        construction, and a colour sample that straddles a boundary mixes two
        populations. Eroding by ten to twenty percent before sampling costs
        pixels and buys a cleaner mean.
        """
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"erosion fraction must be in [0, 1), got {fraction}")
        k = 1.0 - fraction
        c = self.centroid
        # Eroding a region shrinks its outer ring and *grows* its holes, each
        # about its own centre. Scaling a hole about the outer ring's centroid
        # would translate it, which for the perioral region means sliding the
        # vermilion cut-out off the lips.
        grown = tuple(h.mean(axis=0) + (h - h.mean(axis=0)) * (1.0 + fraction) for h in self.holes)
        return RegionPolygon(
            region=self.region,
            vertices=c + (self.vertices - c) * k,
            holes=grown,
            landmarks_used=self.landmarks_used,
            source=self.source,
            rationale=self.rationale,
        )

    def contains(self, xy: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Even-odd point-in-polygon test for points shaped ``(..., 2)``."""
        pts = np.asarray(xy, dtype=float)
        flat = pts.reshape(-1, 2)
        inside = _crossings(self.vertices, flat[:, 0], flat[:, 1])
        for h in self.holes:
            inside &= ~_crossings(h, flat[:, 0], flat[:, 1])
        return inside.reshape(pts.shape[:-1])


def _shoelace(poly: NDArray[np.float64]) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _crossings(poly: NDArray[np.float64], xs: NDArray, ys: NDArray) -> NDArray[np.bool_]:
    """Vectorised even-odd ray crossing test.

    Loops over edges (there are a handful) and vectorises over pixels (there are
    many), which is the right way round.
    """
    inside = np.zeros(xs.shape, dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for a, b, c, d in zip(x0, y0, x1, y1):
        if b == d:
            continue
        straddles = (b > ys) != (d > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (c - a) * (ys - b) / (d - b) + a
        inside ^= straddles & (xs < x_at)
    return inside


def rasterize(
    polygon: RegionPolygon,
    height: int,
    width: int,
    *,
    y_up: bool = True,
) -> NDArray[np.bool_]:
    """Boolean mask of ``polygon`` over an ``(height, width)`` image.

    ``y_up=True`` means the polygon lives in the canonical landmark frame, where
    +y points up, so image row ``r`` samples ``y = (height - 1) - r``. Passing a
    polygon built in raw array coordinates without flipping this flag mirrors
    every region vertically, and the failure is silent because the mask is still
    a plausible blob. Hence the explicit argument.
    """
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    mask = np.zeros((height, width), dtype=bool)
    x_lo, y_lo, x_hi, y_hi = polygon.bounds
    c0 = max(int(np.floor(x_lo)), 0)
    c1 = min(int(np.ceil(x_hi)) + 1, width)
    if y_up:
        r0 = max(int(np.floor(height - 1 - y_hi)), 0)
        r1 = min(int(np.ceil(height - 1 - y_lo)) + 1, height)
    else:
        r0 = max(int(np.floor(y_lo)), 0)
        r1 = min(int(np.ceil(y_hi)) + 1, height)
    if c0 >= c1 or r0 >= r1:
        return mask
    cols = np.arange(c0, c1, dtype=float)
    rows = np.arange(r0, r1, dtype=float)
    gx, gr = np.meshgrid(cols, rows)
    gy = (height - 1 - gr) if y_up else gr
    mask[r0:r1, c0:c1] = polygon.contains(np.stack([gx, gy], axis=-1))
    return mask


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceFrame:
    """Rigid in-plane frame: origin between the pupils, ``u`` subject-right.

    ``span`` is the interpupillary distance in pixels when the pupils are
    available and a scaled outer-canthal distance when they are not. Every
    region offset is a multiple of it, so the recipe table is resolution and
    distance independent.
    """

    origin: NDArray[np.float64]
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    span: float
    span_source: str

    def to_local(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        d = np.asarray(p, dtype=float) - self.origin
        return np.stack([d @ self.u, d @ self.v], axis=-1) / self.span

    def to_global(self, uv: NDArray[np.float64]) -> NDArray[np.float64]:
        uv = np.asarray(uv, dtype=float)
        return self.origin + self.span * (uv[..., 0:1] * self.u + uv[..., 1:2] * self.v)


#: Outer-canthal distance over interpupillary distance in adults. Farkas gives
#: roughly 91 mm outer canthal against 63 mm interpupillary, so an outer-canthal
#: fallback is divided by this to reach a comparable span.
OUTER_CANTHAL_OVER_IPD = 91.0 / 63.4


def build_frame(points: PointSet) -> FaceFrame:
    """Construct the in-plane frame, preferring pupils to canthi."""
    if points.batch_shape != ():
        raise ValueError(
            "regions are built from a single point estimate, not a Monte-Carlo "
            f"ensemble; got batch shape {points.batch_shape}"
        )
    if points.dim != 2:
        raise ValueError("region geometry is defined in the image plane; pass 2D points")

    if points.has(Landmark.PUPIL_L, Landmark.PUPIL_R):
        left = points.get(Landmark.PUPIL_L)
        right = points.get(Landmark.PUPIL_R)
        scale = 1.0
        source = "interpupillary distance"
    elif points.has(Landmark.EXOCANTHION_L, Landmark.EXOCANTHION_R):
        left = points.get(Landmark.EXOCANTHION_L)
        right = points.get(Landmark.EXOCANTHION_R)
        scale = 1.0 / OUTER_CANTHAL_OVER_IPD
        source = "outer-canthal distance, rescaled to an interpupillary equivalent"
    else:
        raise RegionUnavailable(
            Region.T_ZONE, ["pupil_l/pupil_r or exocanthion_l/exocanthion_r"]
        )

    axis = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    norm = float(np.hypot(axis[0], axis[1]))
    if norm <= 0:
        raise ValueError("the two eye landmarks coincide; the frame is undefined")
    u = axis / norm
    # +90 degrees from a subject-right axis is up, in a +y-up frame.
    v = np.array([-u[1], u[0]], dtype=float)
    origin = 0.5 * (np.asarray(left, dtype=float) + np.asarray(right, dtype=float))
    return FaceFrame(origin=origin, u=u, v=v, span=norm * scale, span_source=source)


# ---------------------------------------------------------------------------
# Derived anchor points
# ---------------------------------------------------------------------------

_Local = dict[str, NDArray[np.float64]]


@dataclass(frozen=True)
class _Derived:
    """An anchor built from landmarks, with ordered fallbacks.

    Backends differ in what they supply. Rather than declaring a region
    unavailable because one model calls the point between the brows ``glabella``
    and another does not emit it at all, an anchor lists the ways it can be
    built and takes the first that resolves. The requirement reported when none
    resolve is the *preferred* one, since that is the landmark a backend should
    add.
    """

    name: str
    options: tuple[tuple[tuple[Landmark, ...], Callable[[_Local], NDArray]], ...]


def _mid(a: NDArray, b: NDArray) -> NDArray:
    return 0.5 * (a + b)


_ANCHORS: tuple[_Derived, ...] = (
    _Derived(
        "brow_centre",
        (
            (
                (Landmark.SUPERCILIARE_L, Landmark.SUPERCILIARE_R),
                lambda L: _mid(L["superciliare_l"], L["superciliare_r"]),
            ),
            ((Landmark.GLABELLA,), lambda L: L["glabella"].copy()),
        ),
    ),
    _Derived(
        "glabella_pt",
        (
            ((Landmark.GLABELLA,), lambda L: L["glabella"].copy()),
            (
                (Landmark.SUPERCILIARE_L, Landmark.SUPERCILIARE_R),
                lambda L: _mid(L["superciliare_l"], L["superciliare_r"])
                + np.array([0.0, 0.02]),
            ),
        ),
    ),
    _Derived(
        "nose_top",
        (
            ((Landmark.SELLION,), lambda L: L["sellion"].copy()),
            ((Landmark.NASION,), lambda L: L["nasion"].copy()),
            (
                (Landmark.SUPERCILIARE_L, Landmark.SUPERCILIARE_R),
                lambda L: _mid(L["superciliare_l"], L["superciliare_r"])
                - np.array([0.0, 0.12]),
            ),
        ),
    ),
    _Derived(
        "eye_centre_l",
        (
            (
                (Landmark.ENDOCANTHION_L, Landmark.EXOCANTHION_L),
                lambda L: _mid(L["endocanthion_l"], L["exocanthion_l"]),
            ),
            ((Landmark.PUPIL_L,), lambda L: L["pupil_l"].copy()),
        ),
    ),
    _Derived(
        "eye_centre_r",
        (
            (
                (Landmark.ENDOCANTHION_R, Landmark.EXOCANTHION_R),
                lambda L: _mid(L["endocanthion_r"], L["exocanthion_r"]),
            ),
            ((Landmark.PUPIL_R,), lambda L: L["pupil_r"].copy()),
        ),
    ),
    _Derived(
        "mouth_centre",
        (
            ((Landmark.STOMION,), lambda L: L["stomion"].copy()),
            (
                (Landmark.CHEILION_L, Landmark.CHEILION_R),
                lambda L: _mid(L["cheilion_l"], L["cheilion_r"]),
            ),
        ),
    ),
    _Derived(
        "chin_top",
        (
            ((Landmark.SUBLABIALE,), lambda L: L["sublabiale"].copy()),
            (
                (Landmark.LABIALE_INFERIUS,),
                lambda L: L["labiale_inferius"] - np.array([0.0, 0.12]),
            ),
        ),
    ),
)


def _resolve_anchors(local: _Local) -> tuple[_Local, dict[str, tuple[str, ...]]]:
    """Add every derivable anchor to ``local``; report what could not be built."""
    out = dict(local)
    failed: dict[str, tuple[str, ...]] = {}
    for anchor in _ANCHORS:
        for needed, fn in anchor.options:
            if all(n.value in out for n in needed):
                try:
                    out[anchor.name] = np.asarray(fn(out), dtype=float)
                except KeyError:
                    continue
                break
        else:
            # Report every landmark that would have built this anchor by any
            # route, not only the preferred one. A backend reading the failure
            # needs to know its options, and naming a single landmark implies
            # that adding it is the only fix when it may not be.
            wanted: list[str] = []
            for needed, _ in anchor.options:
                wanted.extend(n.value for n in needed if n.value not in out)
            failed[anchor.name] = tuple(dict.fromkeys(wanted))
    return out, failed


# ---------------------------------------------------------------------------
# Region recipes
# ---------------------------------------------------------------------------


def _quad(u0: float, u1: float, v0: float, v1: float) -> NDArray[np.float64]:
    return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=float)


def _ellipse(centre: NDArray, ru: float, rv: float, n: int = 16) -> NDArray[np.float64]:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([centre[0] + ru * np.cos(t), centre[1] + rv * np.sin(t)], axis=-1)


@dataclass(frozen=True)
class _Recipe:
    region: Region
    requires: tuple[str, ...]
    build: Callable[[_Local], tuple[NDArray, tuple[NDArray, ...]]]
    rationale: str


def _forehead(L: _Local) -> tuple[NDArray, tuple[NDArray, ...]]:
    b = L["brow_centre"]
    top = b[1] + 0.75
    if "trichion" in L:
        top = min(top, L["trichion"][1] - 0.08)
    top = max(top, b[1] + 0.30)
    return _quad(-0.42, 0.42, b[1] + 0.18, top), ()


def _glabella(L: _Local) -> tuple[NDArray, tuple[NDArray, ...]]:
    g = L["glabella_pt"]
    return _quad(g[0] - 0.15, g[0] + 0.15, g[1] - 0.08, g[1] + 0.18), ()


def _t_zone(L: _Local) -> tuple[NDArray, tuple[NDArray, ...]]:
    b = L["brow_centre"]
    sub = L["subnasale"]
    top = b[1] + 0.72
    if "trichion" in L:
        top = min(top, L["trichion"][1] - 0.08)
    top = max(top, b[1] + 0.30)
    band = b[1] + 0.08
    w_face, w_nose = 0.42, 0.17
    if "alare_l" in L and "alare_r" in L:
        w_nose = max(0.13, 0.55 * abs(L["alare_r"][0] - L["alare_l"][0]))
    nose_bottom = sub[1] + 0.04
    return (
        np.array(
            [
                [-w_face, top],
                [w_face, top],
                [w_face, band],
                [w_nose, band],
                [w_nose, nose_bottom],
                [-w_nose, nose_bottom],
                [-w_nose, band],
                [-w_face, band],
            ],
            dtype=float,
        ),
        (),
    )


def _nasal(L: _Local) -> tuple[NDArray, tuple[NDArray, ...]]:
    top = L["nose_top"]
    sub = L["subnasale"]
    al_l, al_r = L["alare_l"], L["alare_r"]
    # Widen the alar points slightly: alare is the most lateral point of the
    # ala, so a polygon through it clips the alar rim itself.
    return (
        np.array(
            [
                [top[0], top[1]],
                [al_r[0] + 0.03, al_r[1] + 0.02],
                [sub[0] + 0.04, sub[1] - 0.01],
                [sub[0] - 0.04, sub[1] - 0.01],
                [al_l[0] - 0.03, al_l[1] + 0.02],
            ],
            dtype=float,
        ),
        (),
    )


def _periorbital(side: str) -> Callable[[_Local], tuple[NDArray, tuple[NDArray, ...]]]:
    def build(L: _Local) -> tuple[NDArray, tuple[NDArray, ...]]:
        en = L[f"endocanthion_{side}"]
        ex = L[f"exocanthion_{side}"]
        sup = L[f"palpebrale_superius_{side}"]
        inf = L[f"palpebrale_inferius_{side}"]
        ring = np.stack([ex, sup, en, inf], axis=0)
        c = ring.mean(axis=0)
        # 1.45x about the eye centre reaches the orbital rim skin -- lid, brow
        # fat pad and tear trough -- rather than only the palpebral aperture.
        return c + (ring - c) * 1.45, ()

    return build


def _infraorbital(side: str) -> Callable[[_Local], tuple[NDArray, tuple[NDArray, ...]]]:
    def build(L: _Local) -> tuple[NDArray, tuple[NDArray, ...]]:
        en = L[f"endocanthion_{side}"]
        ex = L[f"exocanthion_{side}"]
        inf = L[f"palpebrale_inferius_{side}"]
        u_lo, u_hi = sorted((en[0], ex[0]))
        # Inset horizontally to stay clear of the canthi themselves, and start
        # below the lash line so the band is skin and not eyelid margin.
        return (
            _quad(u_lo + 0.04, u_hi - 0.04, inf[1] - 0.26, inf[1] - 0.05),
            (),
        )

    return build


def _malar(side: str) -> Callable[[_Local], tuple[NDArray, tuple[NDArray, ...]]]:
    sign = -1.0 if side == "l" else 1.0

    def build(L: _Local) -> tuple[NDArray, tuple[NDArray, ...]]:
        pupil = L[f"pupil_{side}"]
        cheilion = L[f"cheilion_{side}"]
        centre = _mid(pupil, cheilion) + np.array([0.10 * sign, 0.02])
        return _ellipse(centre, 0.20, 0.17), ()

    return build


def _lateral_cheek(side: str) -> Callable[[_Local], tuple[NDArray, tuple[NDArray, ...]]]:
    sign = -1.0 if side == "l" else 1.0

    def build(L: _Local) -> tuple[NDArray, tuple[NDArray, ...]]:
        pupil = L[f"pupil_{side}"]
        ex = L[f"exocanthion_{side}"]
        centre = np.array([ex[0] + 0.09 * sign, pupil[1] - 0.45])
        return _ellipse(centre, 0.10, 0.17), ()

    return build


def _perioral(L: _Local) -> tuple[NDArray, tuple[NDArray, ...]]:
    ch_l, ch_r = L["cheilion_l"], L["cheilion_r"]
    sub = L["subnasale"]
    chin = L["chin_top"]
    outer = np.array(
        [
            [ch_l[0] - 0.12, ch_l[1]],
            [sub[0] - 0.10, sub[1] - 0.02],
            [sub[0] + 0.10, sub[1] - 0.02],
            [ch_r[0] + 0.12, ch_r[1]],
            [chin[0] + 0.14, chin[1] - 0.02],
            [chin[0] - 0.14, chin[1] - 0.02],
        ],
        dtype=float,
    )
    # The vermilion is not skin: its a* runs far above any cheek, and including
    # it would make every perioral erythema contrast read positive. It is cut
    # out as a hole rather than approximated away.
    if "labiale_superius" in L and "labiale_inferius" in L:
        ls, li = L["labiale_superius"], L["labiale_inferius"]
        lips = np.array([[ch_l[0], ch_l[1]], [ls[0], ls[1] + 0.02],
                         [ch_r[0], ch_r[1]], [li[0], li[1] - 0.02]], dtype=float)
        c = lips.mean(axis=0)
        holes: tuple[NDArray, ...] = (c + (lips - c) * 1.12,)
    else:
        holes = ()
    return outer, holes


RECIPES: tuple[_Recipe, ...] = (
    _Recipe(Region.FOREHEAD, ("brow_centre",), _forehead,
            "band above the brow line, capped at the hairline when trichion is known"),
    _Recipe(Region.GLABELLA, ("glabella_pt",), _glabella,
            "patch between the brows, the reference site for glabellar erythema"),
    _Recipe(Region.T_ZONE, ("brow_centre", "subnasale"), _t_zone,
            "forehead band joined to the nasal dorsum, the sebaceous distribution"),
    _Recipe(Region.NASAL, ("nose_top", "subnasale", "alare_l", "alare_r"), _nasal,
            "nasal dorsum and alae, from the nasal root to subnasale"),
    _Recipe(Region.PERIORBITAL_L,
            ("endocanthion_l", "exocanthion_l", "palpebrale_superius_l", "palpebrale_inferius_l"),
            _periorbital("l"), "orbital rim skin, the palpebral quad expanded 1.45x"),
    _Recipe(Region.PERIORBITAL_R,
            ("endocanthion_r", "exocanthion_r", "palpebrale_superius_r", "palpebrale_inferius_r"),
            _periorbital("r"), "orbital rim skin, the palpebral quad expanded 1.45x"),
    _Recipe(Region.INFRAORBITAL_L,
            ("endocanthion_l", "exocanthion_l", "palpebrale_inferius_l"),
            _infraorbital("l"), "tear-trough band below the lower lid margin"),
    _Recipe(Region.INFRAORBITAL_R,
            ("endocanthion_r", "exocanthion_r", "palpebrale_inferius_r"),
            _infraorbital("r"), "tear-trough band below the lower lid margin"),
    _Recipe(Region.MALAR_L, ("pupil_l", "cheilion_l"), _malar("l"),
            "cheek apex, midway from pupil to oral commissure"),
    _Recipe(Region.MALAR_R, ("pupil_r", "cheilion_r"), _malar("r"),
            "cheek apex, midway from pupil to oral commissure"),
    _Recipe(Region.LATERAL_CHEEK_L, ("pupil_l", "exocanthion_l"), _lateral_cheek("l"),
            "lateral cheek reference patch, placed inboard of the silhouette"),
    _Recipe(Region.LATERAL_CHEEK_R, ("pupil_r", "exocanthion_r"), _lateral_cheek("r"),
            "lateral cheek reference patch, placed inboard of the silhouette"),
    _Recipe(Region.PERIORAL, ("cheilion_l", "cheilion_r", "subnasale", "chin_top"), _perioral,
            "skin surrounding the mouth, with the vermilion cut out as a hole"),
)

BY_REGION: Mapping[Region, _Recipe] = {r.region: r for r in RECIPES}


# ---------------------------------------------------------------------------
# The region set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionSet:
    """Every region that could be built, plus a named reason for each that could not."""

    polygons: Mapping[Region, RegionPolygon]
    unavailable: Mapping[Region, tuple[str, ...]]
    frame: FaceFrame
    #: Optional parser-derived skin mask. ``None`` on the permissive path.
    skin_mask: NDArray[np.bool_] | None = None
    skin_mask_provenance: Provenance | None = None
    notes: tuple[str, ...] = ()

    def __contains__(self, region: object) -> bool:
        return region in self.polygons

    def __iter__(self):
        return iter(self.polygons)

    def __len__(self) -> int:
        return len(self.polygons)

    @property
    def available(self) -> frozenset[Region]:
        return frozenset(self.polygons)

    def get(self, region: Region) -> RegionPolygon:
        try:
            return self.polygons[region]
        except KeyError:
            raise RegionUnavailable(region, self.unavailable.get(region, ("unknown",))) from None

    def mask_for(
        self,
        region: Region,
        height: int,
        width: int,
        *,
        erode: float = 0.12,
        y_up: bool = True,
    ) -> NDArray[np.bool_]:
        """Rasterise a region, eroded, and intersected with the skin mask if present."""
        poly = self.get(region)
        if erode > 0:
            poly = poly.eroded(erode)
        mask = rasterize(poly, height, width, y_up=y_up)
        if self.skin_mask is not None:
            if self.skin_mask.shape != mask.shape:
                raise ValueError(
                    f"skin mask is {self.skin_mask.shape}, image is {mask.shape}"
                )
            mask &= self.skin_mask
        return mask

    def pair_for(self, region: Region) -> Region | None:
        """The reference region this one's colour is read against, if any."""
        ref = REFERENCE_PAIRS.get(region)
        return ref if ref in self.polygons else None


def build_regions(
    points: PointSet,
    *,
    only: Iterable[Region] | None = None,
) -> RegionSet:
    """Build every requestable region from a single 2D point set.

    Regions whose landmarks are absent are not silently skipped: they land in
    ``unavailable`` with the landmark names that would have built them, which is
    what a report needs in order to say *why* a finding is missing rather than
    leaving a hole.
    """
    frame = build_frame(points)
    local: _Local = {
        name.value: frame.to_local(points.get(name)) for name in points.available
    }
    local, anchor_failures = _resolve_anchors(local)

    wanted = set(only) if only is not None else {r.region for r in RECIPES}
    polygons: dict[Region, RegionPolygon] = {}
    unavailable: dict[Region, tuple[str, ...]] = {}

    for recipe in RECIPES:
        if recipe.region not in wanted:
            continue
        missing: list[str] = []
        for need in recipe.requires:
            if need in local:
                continue
            missing.extend(anchor_failures.get(need, (need,)))
        if missing:
            unavailable[recipe.region] = tuple(dict.fromkeys(missing))
            continue
        outer, holes = recipe.build(local)
        used = tuple(
            lm for lm in points.available if lm.value in recipe.requires
        )
        polygons[recipe.region] = RegionPolygon(
            region=recipe.region,
            vertices=frame.to_global(np.asarray(outer, dtype=float)),
            holes=tuple(frame.to_global(np.asarray(h, dtype=float)) for h in holes),
            landmarks_used=used,
            source="geometric",
            rationale=recipe.rationale,
        )

    return RegionSet(
        polygons=polygons,
        unavailable=unavailable,
        frame=frame,
        notes=(
            f"regions built geometrically from landmarks; span from {frame.span_source} "
            f"({frame.span:.1f} px)",
            "polygon boundaries are anatomical approximations, not segmented edges; "
            "a region may include hair, brow or nostril pixels near its border, which "
            "is why colour samples are eroded before they are taken",
        ),
    )


# ---------------------------------------------------------------------------
# Optional parser refinement, behind the non-commercial tier
# ---------------------------------------------------------------------------

#: CelebAMask-HQ class indices that count as facial skin in the SegFormer port.
#: Nose and the two ear classes are skin too, but ears are outside every region
#: here and including them only widens the mask where it cannot help.
PARSER_SKIN_CLASSES: tuple[int, ...] = (1, 10)

#: Classes that must be removed from every region even though they sit inside
#: the polygons: eyes, brows, glasses, lips, teeth and hair all have colour
#: statistics unlike skin, and a mean over them is not a skin measurement.
PARSER_NON_SKIN_CLASSES: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18)


def skin_mask_from_parse(parse: NDArray[np.integer]) -> NDArray[np.bool_]:
    """Reduce a CelebAMask-style label map to a skin mask.

    Pure array work, so it runs without any weights: a caller who has obtained a
    parse some other way, or is testing, does not need the model at all.
    """
    labels = np.asarray(parse)
    if labels.ndim != 2:
        raise ValueError(f"parse map must be 2D, got shape {labels.shape}")
    return np.isin(labels, np.asarray(PARSER_SKIN_CLASSES))


def refine_with_parser(
    regions: RegionSet,
    parse: NDArray[np.integer],
    *,
    allowed_tier: Tier = Tier.PERMISSIVE,
    provenance: Provenance = FACE_PARSING_SEGFORMER,
) -> RegionSet:
    """Intersect every region with a parser-derived skin mask.

    This is a refinement and never a requirement. It sharpens boundaries the
    geometric construction can only approximate, at the cost of the licence the
    parser's training data carries: every public face parser descends from
    CelebAMask-HQ, so this call is gated at :attr:`Tier.NONCOMMERCIAL` and the
    obligation is recorded in the returned set rather than in a README.
    """
    require(provenance, allowed_tier)
    mask = skin_mask_from_parse(parse)
    return RegionSet(
        polygons=regions.polygons,
        unavailable=regions.unavailable,
        frame=regions.frame,
        skin_mask=mask,
        skin_mask_provenance=provenance,
        notes=regions.notes
        + (
            "region boundaries refined by a face parser: " + provenance.describe(),
            "outputs derived with this refinement inherit the parser's training-data "
            "terms and are not commercially redistributable",
        ),
    )


def load_face_parser(
    *,
    allowed_tier: Tier = Tier.PERMISSIVE,
    model_id: str = "jonathandinu/face-parsing",
    device: str = "cpu",
):
    """Load the SegFormer face parser, if the tier permits it.

    The licence check happens before the import and before any weight touches
    the disk, which is the ordering ``models/licensing`` exists to enforce: a
    check that runs after ``from_pretrained`` has already downloaded a
    research-only checkpoint has not prevented anything.
    """
    require(FACE_PARSING_SEGFORMER, allowed_tier)
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "the face parser needs `transformers`; install the noncommercial extra"
        ) from exc
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


__all__ = [
    "Region",
    "RegionPolygon",
    "RegionSet",
    "RegionUnavailable",
    "FaceFrame",
    "REFERENCE_PAIRS",
    "UNPAIRED",
    "RECIPES",
    "build_frame",
    "build_regions",
    "rasterize",
    "skin_mask_from_parse",
    "refine_with_parser",
    "load_face_parser",
]
