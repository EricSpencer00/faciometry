"""A synthetic 3D face whose measurements are known before anything runs.

Everything in arms 1 through 5 stands on this file. The face is a fixed set of
45 landmark coordinates in millimetres in Vitruve's canonical frame (+x the
subject's right, +y up, +z toward the viewer), laid out so that a useful
subset of the catalogue is **exact by construction**:

* interpupillary distance is 63.36 mm because the pupils are placed at
  x = +/- 31.68 with identical y and z, so the distance is a coordinate
  difference and nothing else;
* intercanthal width 32, nose breadth 34, philtrum width 11, bizygomatic 141
  and bigonial 117 the same way;
* the two canthal tilts are 6.0 and 5.0 degrees exactly, because the outer
  canthi are placed by ``dy = dx * tan(theta)``;
* palpebral fissure heights are 11.0 (right) and 10.6 (left) exactly.

The rest of the face carries realistic depth -- a flat face would make the
pose sweep in arm 2 answer a question nobody asked -- so their ground truth is
computed in :mod:`evals.synth.truth` by a second, independent implementation
in pure Python ``math``, written from the anatomical definition rather than
from ``registry.py``.

Deliberate asymmetries, so that the three asymmetry measurements have a
non-zero known answer instead of a degenerate zero:

* right canthal tilt 6.0 deg, left 5.0 deg  -> canthal tilt asymmetry 1.0 deg
* left mouth corner 0.8 mm lower than right -> mouth corner asymmetry
* the outer canthi therefore differ in height -> ocular height asymmetry

The gonial angles are left mirror-symmetric on purpose: ``gonial_angle_l``
and ``gonial_angle_r`` returning different numbers on this face would be a
left/right transposition bug, and that check is free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vitruve.core.geometry import rotation_matrix
from vitruve.core.landmarks import Landmark as L
from vitruve.core.landmarks import PointSet

# --- construction parameters, the numbers the "exact by construction" claims
# --- above are made of.
IPD_MM = 63.36
INTERCANTHAL_MM = 32.0
BIOCULAR_X_MM = 90.0
NOSE_BREADTH_MM = 34.0
MOUTH_WIDTH_MM = 52.0
PHILTRUM_WIDTH_MM = 11.0
BIZYGOMATIC_MM = 141.0
BIGONIAL_MM = 117.0
CANTHAL_TILT_R_DEG = 6.0
CANTHAL_TILT_L_DEG = 5.0
FISSURE_HEIGHT_R_MM = 11.0
FISSURE_HEIGHT_L_MM = 10.6
MOUTH_CORNER_DROP_MM = 0.8

#: Horizontal run from inner to outer canthus, the ``dx`` of the tilt.
_CANTHUS_RUN = BIOCULAR_X_MM / 2.0 - INTERCANTHAL_MM / 2.0  # 29.0
_EN_Y, _EN_Z = -2.0, 12.0
_EX_Z = 2.0
_PUPIL_Y, _PUPIL_Z = -4.0, 10.0
_CHEILION_Y, _CHEILION_Z = -72.0, 26.0

FACE: dict[L, tuple[float, float, float]] = {
    # -- cranium
    L.TRICHION: (0.0, 70.0, 20.0),
    L.GLABELLA: (0.0, 18.0, 28.0),
    # -- orbital
    L.NASION: (0.0, 5.0, 22.0),
    L.SELLION: (0.0, 0.0, 20.0),
    L.ENDOCANTHION_R: (+INTERCANTHAL_MM / 2, _EN_Y, _EN_Z),
    L.ENDOCANTHION_L: (-INTERCANTHAL_MM / 2, _EN_Y, _EN_Z),
    L.EXOCANTHION_R: (
        +BIOCULAR_X_MM / 2,
        _EN_Y + _CANTHUS_RUN * math.tan(math.radians(CANTHAL_TILT_R_DEG)),
        _EX_Z,
    ),
    L.EXOCANTHION_L: (
        -BIOCULAR_X_MM / 2,
        _EN_Y + _CANTHUS_RUN * math.tan(math.radians(CANTHAL_TILT_L_DEG)),
        _EX_Z,
    ),
    L.PUPIL_R: (+IPD_MM / 2, _PUPIL_Y, _PUPIL_Z),
    L.PUPIL_L: (-IPD_MM / 2, _PUPIL_Y, _PUPIL_Z),
    L.PALPEBRALE_SUP_R: (+IPD_MM / 2, _PUPIL_Y + FISSURE_HEIGHT_R_MM / 2, _PUPIL_Z),
    L.PALPEBRALE_INF_R: (+IPD_MM / 2, _PUPIL_Y - FISSURE_HEIGHT_R_MM / 2, _PUPIL_Z),
    L.PALPEBRALE_SUP_L: (-IPD_MM / 2, _PUPIL_Y + FISSURE_HEIGHT_L_MM / 2, _PUPIL_Z),
    L.PALPEBRALE_INF_L: (-IPD_MM / 2, _PUPIL_Y - FISSURE_HEIGHT_L_MM / 2, _PUPIL_Z),
    L.ORBITALE_R: (+30.0, -14.0, 8.0),
    L.ORBITALE_L: (-30.0, -14.0, 8.0),
    # -- brow
    L.SUPERCILIARE_R: (+25.0, 12.0, 18.0),
    L.SUPERCILIARE_L: (-25.0, 12.0, 18.0),
    # -- nose
    L.PRONASALE: (0.0, -40.0, 52.0),
    L.SUBNASALE: (0.0, -53.0, 30.0),
    L.COLUMELLA: (0.0, -46.0, 41.0),
    L.ALARE_R: (+NOSE_BREADTH_MM / 2, -49.0, 22.0),
    L.ALARE_L: (-NOSE_BREADTH_MM / 2, -49.0, 22.0),
    L.SUBALARE_R: (+14.0, -52.0, 22.0),
    L.SUBALARE_L: (-14.0, -52.0, 22.0),
    # -- mouth
    L.LABIALE_SUPERIUS: (0.0, -68.0, 36.0),
    L.STOMION: (0.0, -72.0, 35.0),
    L.LABIALE_INFERIUS: (0.0, -80.0, 33.0),
    L.CRISTA_PHILTRI_R: (+PHILTRUM_WIDTH_MM / 2, -66.5, 34.0),
    L.CRISTA_PHILTRI_L: (-PHILTRUM_WIDTH_MM / 2, -66.5, 34.0),
    L.CHEILION_R: (+MOUTH_WIDTH_MM / 2, _CHEILION_Y, _CHEILION_Z),
    L.CHEILION_L: (-MOUTH_WIDTH_MM / 2, _CHEILION_Y - MOUTH_CORNER_DROP_MM, _CHEILION_Z),
    # -- chin and jaw
    L.SUBLABIALE: (0.0, -86.0, 24.0),
    L.POGONION: (0.0, -97.0, 26.0),
    L.GNATHION: (0.0, -113.0, 22.0),
    L.MENTON: (0.0, -120.0, 18.0),
    L.GONION_R: (+BIGONIAL_MM / 2, -95.0, -18.0),
    L.GONION_L: (-BIGONIAL_MM / 2, -95.0, -18.0),
    # -- lateral
    L.ZYGION_R: (+BIZYGOMATIC_MM / 2, -18.0, -12.0),
    L.ZYGION_L: (-BIZYGOMATIC_MM / 2, -18.0, -12.0),
    L.TRAGION_R: (+72.0, -12.0, -55.0),
    L.TRAGION_L: (-72.0, -12.0, -55.0),
    L.PORION_R: (+72.0, -8.0, -58.0),
    L.PORION_L: (-72.0, -8.0, -58.0),
    # -- neck
    L.CERVICALE: (0.0, -122.0, -52.0),
}

ORDER: tuple[L, ...] = tuple(FACE)
INDEX: dict[L, int] = {name: i for i, name in enumerate(ORDER)}
COORDS: np.ndarray = np.array([FACE[n] for n in ORDER], dtype=float)

#: Rotation is taken about the landmark centroid rather than the coordinate
#: origin. Under orthographic projection this is irrelevant (every measurement
#: is translation invariant), but under a perspective camera it decides which
#: part of the face swings toward the lens, so it must be physical.
CENTROID: np.ndarray = COORDS.mean(axis=0)

#: Depth of the pupil plane. The perspective cameras hold this plane fixed so
#: that "magnification relative to the eye plane" means what ICAO means by it.
EYE_PLANE_Z: float = _PUPIL_Z


def point_set(coords: np.ndarray | None = None) -> PointSet:
    return PointSet(index=dict(INDEX), coords=COORDS.copy() if coords is None else coords)


def rotated(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
            coords: np.ndarray | None = None) -> np.ndarray:
    """Rigidly rotate the face about its centroid. Still 3D, still metric."""
    c = COORDS if coords is None else coords
    r = rotation_matrix(yaw, pitch, roll)
    return (c - CENTROID) @ r.T + CENTROID


# ---------------------------------------------------------------------------
# Cameras.
#
# A photograph loses one axis. Rather than hand a 2D point set to a formula
# algebra whose profile measurements are written against the z axis -- which
# raises rather than silently returning zero, correctly -- each camera returns
# a 3D array with the lost axis zeroed. That is faithful: a frontal photograph
# knows x and y and nothing about depth, and a profile photograph knows z and y
# and nothing about the lateral axis.
# ---------------------------------------------------------------------------


def camera_frontal_ortho(coords: np.ndarray) -> np.ndarray:
    out = coords.copy()
    out[..., 2] = 0.0
    return out


def camera_profile_ortho(coords: np.ndarray) -> np.ndarray:
    out = coords.copy()
    out[..., 0] = 0.0
    return out


def camera_frontal_perspective(coords: np.ndarray, distance_m: float) -> np.ndarray:
    """Pinhole camera on the +z axis, ``distance_m`` from the eye plane.

    Scaled so that a landmark exactly on the eye plane is left where
    orthographic projection would put it. Everything nearer the lens is
    magnified, everything further shrinks, and that differential is the whole
    of the perspective error the ICAO guidance is about.
    """
    d_mm = distance_m * 1000.0
    depth = d_mm - (coords[..., 2] - EYE_PLANE_Z)
    if np.any(depth <= 1e-6):
        raise ValueError(f"camera at {distance_m} m is inside the face")
    m = d_mm / depth
    out = coords.copy()
    out[..., 0] = coords[..., 0] * m
    out[..., 1] = coords[..., 1] * m
    out[..., 2] = 0.0
    return out


def camera_profile_perspective(coords: np.ndarray, distance_m: float) -> np.ndarray:
    """Pinhole camera on the +x axis (the subject's right), looking at x = 0."""
    d_mm = distance_m * 1000.0
    depth = d_mm - coords[..., 0]
    if np.any(depth <= 1e-6):
        raise ValueError(f"camera at {distance_m} m is inside the face")
    m = d_mm / depth
    out = coords.copy()
    out[..., 0] = 0.0
    out[..., 1] = coords[..., 1] * m
    out[..., 2] = coords[..., 2] * m
    return out


@dataclass(frozen=True)
class Capture:
    """One synthetic photograph: the point set a landmarker would recover."""

    label: str
    coords: np.ndarray

    @property
    def points(self) -> PointSet:
        return PointSet(index=dict(INDEX), coords=self.coords)


def capture(view: str, *, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
            distance_m: float | None = None) -> Capture:
    """Rotate the face, then photograph it from the frontal or profile camera."""
    c = rotated(yaw, pitch, roll)
    if view == "frontal":
        c = camera_frontal_ortho(c) if distance_m is None else camera_frontal_perspective(c, distance_m)
    elif view == "profile":
        c = camera_profile_ortho(c) if distance_m is None else camera_profile_perspective(c, distance_m)
    elif view == "3d":
        pass
    else:  # pragma: no cover
        raise ValueError(f"unknown view {view!r}")
    return Capture(f"{view}/y{yaw:+.1f}/p{pitch:+.1f}/r{roll:+.1f}"
                   f"/{'ortho' if distance_m is None else f'{distance_m:g}m'}", c)
