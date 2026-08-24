"""Each half of the face doubled about the subject's own midline.

A mirror composite answers the question people are really asking when they ask
about symmetry, and it answers it without a number. The left-left picture is
what the subject would look like if the left half were the whole face; the
right-right picture is the same for the right. Set beside the photograph, the
two of them show where the halves differ and by how much, and the reader sees
it instead of taking a percentage on faith.

That is the whole of what this module does. It computes no symmetry score, no
percentage, no index, and no aggregate of any kind, and the reason is not
squeamishness. Every candidate for such a number is a weighted sum over
features whose weights nobody can defend: a millimetre at the mouth corner and
a degree at the canthus are not commensurable, and any rule for adding them
invents the exchange rate. Worse, the sum is dominated by whichever features
the landmark model localises worst, so the "score" ends up ranking photographs
rather than faces. What travels with the pictures instead is the set of
per-feature asymmetries the catalogue already measures, each carrying its own
interval and its own evidence tier, so a reader who wants to know which feature
differs is told which feature differs.

Two things about the geometry are easy to get wrong, and both produce output
that still looks like a face.

**The mirror axis is the subject's midline, not the image's.** A face is almost
never centred in a frame and almost never level in it. Reflecting about the
column of pixels down the middle of the picture mostly measures where the
photographer stood: it slides the composite sideways and, if the head is
rolled, shears the two halves against each other. So the axis here is derived
from named landmarks. Its direction is fixed by the interpupillary line, which
rotates with the head, so image roll is out of the geometry before the
reflection happens rather than being corrected for afterwards. Its position is
the mean of the midpoints of every bilateral pair the point set carries, which
is the construction that does not assume the face is symmetric.

**The frame is the image's, y downward.** Measurements in this project live in
the canonical frame of ``core.imaging``, where +y is up because that is how
anthropometric definitions are written. Image arrays index rows downward. This
module takes landmarks in **image pixels** and never converts, because the
conversion is the trap: canonical coordinates handed to a reflection built for
image coordinates give a mirror axis tipped the wrong way, and the composite
comes out looking like a plausible person who is not the subject. No helper is
offered for the conversion either, because two of them exist in this
repository, ``core.imaging.to_canonical`` flipping y alone and
``models.protocols.to_canonical`` negating both axes, and a helper that guessed
between them would be wrong half the time. The caller already holds image
pixels; that is what ``report.overlay`` draws with.

Pillow and numpy only, matching the rest of the report layer.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..core.landmarks import BILATERAL_PAIRS, Landmark
from ..measure.evaluate import Measured
from ..measure.registry import CATALOGUE
from .overlay import to_png

Point = tuple[float, float]

#: The catalogue's per-feature asymmetries, read off the catalogue rather than
#: listed here, so a measurement added there travels with the composites
#: without an edit in this file.
ASYMMETRY_IDS: tuple[str, ...] = tuple(
    s.id for s in CATALOGUE if s.id.endswith("_asymmetry")
)

#: Bilateral pairs that may fix the mirror direction, in order of preference.
#: Pupils first because they are the best-localised pair on any landmark model
#: and because the rest of Faciometry already measures inclination against them,
#: so the composite and the canthal tilt agree about which way is level. The
#: canthi are the fallback for a model that does not emit pupils.
AXIS_PAIRS: tuple[tuple[Landmark, Landmark], ...] = (
    (Landmark.PUPIL_L, Landmark.PUPIL_R),
    (Landmark.ENDOCANTHION_L, Landmark.ENDOCANTHION_R),
    (Landmark.EXOCANTHION_L, Landmark.EXOCANTHION_R),
)

#: Below this separation in pixels the pair is treated as coincident and the
#: direction it would give is noise. One pixel of landmark error on a two-pixel
#: baseline is thirty degrees of mirror axis.
_MIN_BASELINE_PX = 4.0


@dataclass(frozen=True)
class FacialMidline:
    """The line a face is symmetric about, in image pixels.

    ``toward_subject_left`` is the unit normal, pointing from the subject's
    right to the subject's left. It is derived from landmarks that name their
    own side, so the sign is anatomical rather than a convention about which
    way the photograph faces, and nothing downstream has to know whether the
    image was mirrored on the way in.
    """

    origin: Point
    toward_subject_left: Point
    #: Which bilateral pair fixed the direction, and how many pairs placed the
    #: origin. Both are provenance for a picture that otherwise looks like it
    #: came from nowhere.
    axis_pair: tuple[str, str]
    n_pairs: int

    @property
    def normal(self) -> NDArray[np.float64]:
        return np.asarray(self.toward_subject_left, dtype=float)

    @property
    def direction(self) -> NDArray[np.float64]:
        """Unit vector along the midline, running down the face in the image."""
        nx, ny = self.toward_subject_left
        return np.array([-ny, nx], dtype=float)

    @property
    def tilt_deg(self) -> float:
        """Tilt of the interocular axis in the image, in degrees.

        Zero when the two eyes sit at the same image row. Positive when the
        subject's left eye is lower in the frame, image y growing downward.
        This is a property of the picture, reported so a reader can see how far
        the mirror axis had to be leaned; the measurements are already
        independent of it.
        """
        nx, ny = self.toward_subject_left
        return math.degrees(math.atan2(ny, nx))

    def signed_offset(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Distance of each point from the line, positive on the subject's left."""
        p = np.asarray(points, dtype=float)
        return (p - np.asarray(self.origin, dtype=float)) @ self.normal

    def reflection(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The affine reflection ``(M, t)`` with ``p' = M p + t``.

        A reflection is its own inverse, which is why the same six numbers can
        be handed to Pillow, whose affine transform maps output pixels back to
        input pixels rather than forward.
        """
        n = self.normal
        m = np.eye(2) - 2.0 * np.outer(n, n)
        t = 2.0 * float(np.dot(np.asarray(self.origin, dtype=float), n)) * n
        return m, t


def facial_midline(points: Mapping[Landmark, Sequence[float]]) -> FacialMidline:
    """Derive the midline from named landmarks in image pixels.

    Raises ``ValueError`` naming what was missing rather than falling back to
    the middle of the picture. A composite drawn about the wrong axis is not a
    degraded result, it is a different face, and the caller has to know.
    """
    xy = {k: np.asarray(v, dtype=float)[:2] for k, v in points.items()}

    pair = None
    for left, right in AXIS_PAIRS:
        if left in xy and right in xy:
            baseline = float(np.linalg.norm(xy[left] - xy[right]))
            if baseline >= _MIN_BASELINE_PX:
                pair = (left, right)
                break
    if pair is None:
        wanted = ", ".join(f"{a.value}/{b.value}" for a, b in AXIS_PAIRS)
        raise ValueError(
            "no bilateral pair with a usable baseline, so the mirror axis "
            f"cannot be fixed; one of these is needed: {wanted}"
        )

    left, right = pair
    n = xy[left] - xy[right]
    n = n / float(np.linalg.norm(n))

    # The position comes from every pair available, not from the axis pair
    # alone. Midpoints of bilateral pairs are the standard construction because
    # they stay on the midline of an asymmetric face, where a median landmark
    # such as pronasale wanders off it and takes the axis with it.
    midpoints = [
        (xy[a] + xy[b]) / 2.0 for a, b in BILATERAL_PAIRS if a in xy and b in xy
    ]
    origin = np.mean(np.stack(midpoints), axis=0)

    return FacialMidline(
        origin=(float(origin[0]), float(origin[1])),
        toward_subject_left=(float(n[0]), float(n[1])),
        axis_pair=(left.value, right.value),
        n_pairs=len(midpoints),
    )


def reflect_points(
    points: Mapping[Landmark, Sequence[float]], midline: FacialMidline
) -> dict[Landmark, Point]:
    """Every point reflected across the midline, still in image pixels.

    Useful to a caller drawing the mirrored landmarks over a composite, and it
    is the same arithmetic the image transform uses, so a test that pins one
    pins the other.
    """
    m, t = midline.reflection()
    out: dict[Landmark, Point] = {}
    for name, value in points.items():
        p = np.asarray(value, dtype=float)[:2]
        q = m @ p + t
        out[name] = (float(q[0]), float(q[1]))
    return out


def mirror_image(
    image: Image.Image,
    midline: FacialMidline,
    *,
    resample: int = Image.BICUBIC,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """The whole photograph reflected about the midline.

    Pixels the reflection asks for from outside the frame come back as ``fill``
    rather than being clamped or wrapped. A face near the edge of its frame
    genuinely has no data for half of its own composite, and a smeared edge
    column would hide that.

    The half pixel below is not a fudge. Pillow resamples in a continuous frame
    whose origin is the outer corner of the top-left pixel, so the *centre* of
    pixel ``i`` sits at ``i + 0.5``, while a landmark at ``i`` means that same
    centre. Reflecting with the translation built from raw indices therefore
    lands the mirrored half one whole pixel off the mask, and a one-pixel
    misregistration down the middle of a face is a permanent faint seam that
    nobody attributes to arithmetic.
    """
    m, _ = midline.reflection()
    n = midline.normal
    centre = np.asarray(midline.origin, dtype=float) + 0.5
    t = 2.0 * float(centre @ n) * n
    data = (m[0, 0], m[0, 1], t[0], m[1, 0], m[1, 1], t[1])
    return image.convert("RGB").transform(
        image.size, Image.AFFINE, data, resample=resample, fillcolor=fill
    )


def half_mask(
    size: tuple[int, int],
    midline: FacialMidline,
    *,
    side: str,
    feather_px: float = 2.0,
) -> Image.Image:
    """An 8-bit mask, opaque over the named half of the face.

    The seam is a short linear ramp rather than a step. A hard edge along the
    midline reads as a scar down the middle of the composite, and a reader
    quite reasonably takes a visible line for a finding about the face.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    w, h = size
    nx, ny = midline.toward_subject_left
    ox, oy = midline.origin
    ys, xs = np.mgrid[0:h, 0:w]
    s = (xs - ox) * nx + (ys - oy) * ny
    if side == "right":
        s = -s
    ramp = max(float(feather_px), 1e-6)
    alpha = np.clip(0.5 + s / (2.0 * ramp), 0.0, 1.0)
    return Image.fromarray(np.round(alpha * 255.0).astype(np.uint8), mode="L")


def mirror_composite(
    image: Image.Image,
    midline: FacialMidline,
    *,
    side: str,
    feather_px: float = 2.0,
    resample: int = Image.BICUBIC,
) -> Image.Image:
    """The named half of the face, with its own reflection on the other side."""
    base = image.convert("RGB")
    mirrored = mirror_image(base, midline, resample=resample)
    mask = half_mask(base.size, midline, side=side, feather_px=feather_px)
    return Image.composite(base, mirrored, mask)


@dataclass(frozen=True)
class MirrorComposite:
    """One composite picture and the measurements that belong beside it.

    ``asymmetries`` are catalogue measurements, passed through untouched with
    their verdicts and their intervals. There is deliberately no field here
    that combines them: this object holds several measurements, and a number
    spanning several measurements is the thing this project does not produce.
    """

    side: str
    title: str
    caption: str
    png: bytes
    midline: FacialMidline
    asymmetries: tuple[Measured, ...] = ()

    @property
    def data_uri(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")


_CAPTION = (
    "The subject's {side} half reflected onto itself about the facial midline, "
    "which is fixed by the {pair} axis and the midpoints of {n} bilateral "
    "{pairs}, not by the middle of the photograph. The interocular axis leans "
    "{tilt:+.1f} deg in this frame and the mirror leans with it, so the picture "
    "does not inherit the camera tilt. It is a picture and not a measurement: "
    "the measured differences are listed beside it, one per paired feature."
)


def mirror_composites(
    image: Image.Image,
    points: Mapping[Landmark, Sequence[float]],
    *,
    measurements: Iterable[Measured] = (),
    feather_px: float = 2.0,
    resample: int = Image.BICUBIC,
) -> tuple[MirrorComposite, MirrorComposite]:
    """Both composites, in the order left-left then right-right.

    ``points`` are in image pixels. ``measurements`` is whatever the pipeline
    evaluated; the per-feature asymmetries are selected out of it by id and
    everything else is ignored, so the caller does not have to know which
    measurements those are.
    """
    midline = facial_midline(points)
    asymmetries = tuple(m for m in measurements if m.spec_id in ASYMMETRY_IDS)
    out = []
    for side in ("left", "right"):
        picture = mirror_composite(
            image, midline, side=side, feather_px=feather_px, resample=resample
        )
        out.append(
            MirrorComposite(
                side=side,
                title=f"Subject's {side} half, mirrored",
                caption=_CAPTION.format(
                    side=side,
                    pair=" to ".join(midline.axis_pair),
                    n=midline.n_pairs,
                    pairs="pair" if midline.n_pairs == 1 else "pairs",
                    tilt=midline.tilt_deg,
                ),
                png=to_png(picture),
                midline=midline,
                asymmetries=asymmetries,
            )
        )
    return out[0], out[1]
