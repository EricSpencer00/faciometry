"""Vectorised geometric primitives.

Every function here accepts *batched* point arrays with shape ``(..., D)`` where
``D`` is 2 or 3, and broadcasts over the leading axes. The leading axes are how
Monte-Carlo uncertainty propagation gets its speed: one evaluation of a
measurement over ``(n_samples, n_points, D)`` costs about what one evaluation
over a single point set costs.

Angles are returned in degrees, because every downstream consumer -- the
report, the normative tables, the clinical literature -- speaks degrees.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_EPS = 1e-12


def _unit(v: Array) -> Array:
    """Normalise along the last axis, leaving zero-length vectors at zero."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > _EPS)


def distance(a: Array, b: Array) -> Array:
    """Euclidean distance between corresponding points."""
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)


def midpoint(a: Array, b: Array) -> Array:
    return (np.asarray(a) + np.asarray(b)) / 2.0


def angle_at(a: Array, vertex: Array, c: Array) -> Array:
    """Interior angle in degrees at ``vertex`` in the path a -> vertex -> c.

    Uses the atan2 form rather than ``arccos`` of a dot product: arccos loses
    all precision as the angle approaches 0 or 180 degrees, which is exactly
    where a nearly-straight profile contour lives.
    """
    u = np.asarray(a) - np.asarray(vertex)
    v = np.asarray(c) - np.asarray(vertex)
    dot = np.sum(u * v, axis=-1)
    if u.shape[-1] == 2:
        cross_mag = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    else:
        cross_mag = np.linalg.norm(np.cross(u, v), axis=-1)
    return np.degrees(np.arctan2(cross_mag, dot))


def angle_between_lines(p0: Array, p1: Array, q0: Array, q1: Array) -> Array:
    """Unsigned acute-or-obtuse angle in degrees between line p0p1 and q0q1.

    Undirected: the result is folded into [0, 180) so that reversing either
    segment's endpoints does not change the answer.
    """
    u = _unit(np.asarray(p1) - np.asarray(p0))
    v = _unit(np.asarray(q1) - np.asarray(q0))
    dot = np.sum(u * v, axis=-1)
    if u.shape[-1] == 2:
        cross_mag = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    else:
        cross_mag = np.linalg.norm(np.cross(u, v), axis=-1)
    ang = np.degrees(np.arctan2(cross_mag, dot))
    return np.where(ang > 90.0, 180.0 - ang, ang)


def signed_angle_to_axis(p0: Array, p1: Array, axis: Array) -> Array:
    """Signed angle in degrees of the directed segment p0->p1 relative to ``axis``.

    Sign is taken in the plane spanned by the segment and the axis, using the
    second component of the segment expressed in that plane. In the canonical
    face frame (x right, y up, z toward the viewer) with ``axis`` = +x, a
    positive result means p1 sits above p0 -- which is the sign convention the
    canthal-tilt literature uses.
    """
    d = np.asarray(p1) - np.asarray(p0)
    ax = _unit(np.broadcast_to(np.asarray(axis, dtype=float), d.shape))
    along = np.sum(d * ax, axis=-1)
    perp = d - along[..., None] * ax
    # The positive-perpendicular direction is the frame's up axis, projected
    # orthogonal to `axis`. Using up (rather than the axis rotated 90 degrees)
    # keeps the sign meaning "b is above a" for both +x and -x axes, which is
    # what lets a single tilt formula serve both sides of the face.
    up = np.zeros_like(d)
    up[..., 1] = 1.0
    n = _unit(up - np.sum(up * ax, axis=-1)[..., None] * ax)
    across = np.sum(perp * n, axis=-1)
    return np.degrees(np.arctan2(across, along))


def project_on_axis(p: Array, origin: Array, axis: Array) -> Array:
    """Scalar coordinate of ``p`` along ``axis`` measured from ``origin``."""
    d = np.asarray(p) - np.asarray(origin)
    ax = _unit(np.broadcast_to(np.asarray(axis, dtype=float), d.shape))
    return np.sum(d * ax, axis=-1)


def point_to_line_distance(p: Array, a: Array, b: Array) -> Array:
    """Perpendicular distance from ``p`` to the infinite line through a and b.

    Unsigned. For signed offsets relative to an aesthetic reference line (the
    Ricketts E-line, for instance) use :func:`signed_point_to_line_offset`.
    """
    d = _unit(np.asarray(b) - np.asarray(a))
    w = np.asarray(p) - np.asarray(a)
    proj = np.sum(w * d, axis=-1)[..., None] * d
    return np.linalg.norm(w - proj, axis=-1)


def signed_point_to_line_offset(p: Array, a: Array, b: Array, normal: Array) -> Array:
    """Offset of ``p`` from line ab, signed along ``normal``.

    Positive means ``p`` lies on the ``normal`` side. Profile aesthetics are
    stated as signed offsets ("lower lip 2 mm behind the E-line"), so the sign
    is the measurement, not a detail.
    """
    d = _unit(np.asarray(b) - np.asarray(a))
    w = np.asarray(p) - np.asarray(a)
    perp = w - np.sum(w * d, axis=-1)[..., None] * d
    nrm = _unit(np.broadcast_to(np.asarray(normal, dtype=float), perp.shape))
    return np.sum(perp * nrm, axis=-1)


def polygon_area(points: Array) -> Array:
    """Shoelace area of a closed 2D polygon given as ``(..., n, 2)``."""
    pts = np.asarray(points)
    if pts.shape[-1] != 2:
        raise ValueError("polygon_area requires 2D points")
    x, y = pts[..., 0], pts[..., 1]
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1)
    )


def rotation_matrix(yaw_deg: Array, pitch_deg: Array, roll_deg: Array) -> Array:
    """Right-handed intrinsic Y-X-Z rotation, matching the head-pose convention.

    Yaw about +y (turning left/right), pitch about +x (nodding), roll about +z
    (tilting toward a shoulder). Applied as ``R @ v`` for column vectors, or
    ``v @ R.T`` for the row-vector arrays used everywhere else in this module.
    """
    y, p, r = (np.radians(np.asarray(a, dtype=float)) for a in (yaw_deg, pitch_deg, roll_deg))
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return ry @ rx @ rz


def apply_rotation(points: Array, rot: Array) -> Array:
    """Rotate row-vector points ``(..., 3)`` by a ``(3, 3)`` matrix."""
    return np.asarray(points) @ np.asarray(rot).T
