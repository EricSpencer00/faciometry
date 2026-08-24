"""The one conversion between image arrays and the canonical frame.

Core measures in a canonical frame where **+y is up**, because that is how
every anthropometric definition in the literature is written: a positive
canthal tilt means the outer corner sits above the inner one. Image arrays
index rows downward, so **+y is down** there.

Every hand-rolled conversion between the two is a silent vertical mirroring,
and a mirrored face still looks like a face. The failure is not a crash; it is
a canthal tilt that reports the correct magnitude with the wrong sign, on every
subject, forever. So the conversion lives here once, it is an involution, and a
test pins the orientation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def to_canonical(points: Array, *, height: float, origin: Array | None = None) -> Array:
    """Image coordinates (x right, y down) to canonical (x right, y up).

    ``height`` is the image height in pixels, used to flip the vertical axis.
    ``origin`` optionally recentres afterwards, which is how a face is placed
    at the midpoint between the inner canthi.
    """
    pts = np.array(points, dtype=float, copy=True)
    pts[..., 1] = height - pts[..., 1]
    if origin is not None:
        pts[..., : len(origin)] -= np.asarray(origin, dtype=float)
    return pts


def to_image(points: Array, *, height: float, origin: Array | None = None) -> Array:
    """Canonical coordinates back to image coordinates.

    The exact inverse of :func:`to_canonical` under the same arguments, which
    is what lets a measurement taken in canonical space be drawn back onto the
    original photograph without accumulating a flip.
    """
    pts = np.array(points, dtype=float, copy=True)
    if origin is not None:
        pts[..., : len(origin)] += np.asarray(origin, dtype=float)
    pts[..., 1] = height - pts[..., 1]
    return pts


def rasterize_y_up(polygon: Array, *, height: float) -> Array:
    """A canonical-frame polygon expressed in array row/column order.

    Drawing libraries take (x, y) with y measured downward. Passing canonical
    points straight to one of them mirrors the shape about the horizontal
    midline, which for a facial region is subtle enough to survive review: the
    periorbital polygon lands on the cheek and still looks plausible.
    """
    return to_image(polygon, height=height)
