"""Moving landmarks out of the photographer's frame and into the subject's.

Landmarks arrive in image coordinates: ``+x`` toward the right edge of the
frame, ``+y`` down. Measurements are declared in the canonical frame: ``+x`` the
*subject's* right, ``+y`` up, ``+z`` toward the viewer. In an unmirrored frontal
photograph the subject's right is the frame's left, so the mapping is a
half-turn, not a mirror. Getting that backwards does not produce an obviously
broken report; it produces a report in which every lateralised finding is
swapped and every canthal tilt has the wrong sign, which is far worse.

The other half of this module is roll correction, and it is worth being precise
about what it does and does not buy.

Canthal tilt is defined against the horizon. Tilt the camera by three degrees
and the measured tilt moves by three degrees, against a normal range of roughly
four to eight. That is the highest-leverage single correction in the pipeline,
and it is cheap: the interocular axis fixes image roll exactly, up to the
landmarks themselves. So the point estimate is corrected.

The *uncertainty* is not reduced, and this is the part that is easy to get
wrong. After rotating the points so the eyes are level, the residual roll is not
zero. It is however far the roll estimate itself was off, which is the pose
estimator's own standard deviation of roughly five degrees. Passing ``roll=0``
into :func:`~vitruve.measure.evaluate.evaluate` after correcting would let every
pose-critical measurement sail through a gate that exists precisely because the
correction is imperfect. So the residual handed downstream is the estimator's
spread, and canthal tilt still comes back caveated, which is the honest answer
for a single uncalibrated photograph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.landmarks import Landmark, PointSet
from ..core.spec import View
from ..measure.evaluate import LandmarkUncertainty
from .quality import PoseEstimate, yaw_deviation

#: Landmark pairs used to read the interocular axis, most trustworthy first.
#: Pupils are what ICAO's minimum face size is defined against; the canthi are
#: the fallback for a landmark model that does not emit pupil centres.
_INTEROCULAR_CANDIDATES: tuple[tuple[Landmark, Landmark], ...] = (
    (Landmark.PUPIL_R, Landmark.PUPIL_L),
    (Landmark.ENDOCANTHION_R, Landmark.ENDOCANTHION_L),
    (Landmark.EXOCANTHION_R, Landmark.EXOCANTHION_L),
)

#: Added to the diagonal of every canonical covariance. A backend that reports
#: a singular covariance -- a coordinate-regression model emitting zeros, or a
#: heatmap collapsed to a single pixel -- would otherwise crash the Cholesky
#: draw inside the Monte-Carlo sampler. A ten-thousandth of a pixel changes no
#: interval and turns a crash into a number.
_PD_FLOOR_PX2 = 1e-4

#: Image-to-canonical linear part before roll correction: a half turn.
#: ``x`` negates because the subject's right is the frame's left, ``y`` negates
#: because image ``y`` runs downward. The determinant is +1, so this is a
#: rotation and not a reflection, and handedness is preserved.
_HALF_TURN = np.array([[-1.0, 0.0], [0.0, -1.0]])


def _rotation(theta_deg: float) -> NDArray[np.float64]:
    """In-plane rotation for row vectors, applied as ``p @ R.T``."""
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def eye_pair(points: PointSet) -> tuple[Landmark, Landmark] | None:
    """The best available (subject's right, subject's left) ocular pair."""
    for right, left in _INTEROCULAR_CANDIDATES:
        if points.has(right, left):
            return right, left
    return None


def interocular_distance_px(points: PointSet) -> float | None:
    """Distance between the eyes in whatever units ``points`` carries."""
    pair = eye_pair(points)
    if pair is None:
        return None
    right, left = pair
    return float(np.linalg.norm(points.get(right) - points.get(left)))


def roll_from_eyes(points_image: PointSet) -> float | None:
    """Image roll read off the interocular axis, in the canonical convention.

    Positive means the subject's right eye sits above the subject's left eye in
    the photograph. Plane geometry fixes this exactly, so it is the tightest
    roll estimate available and it does not depend on a network having
    converged. It says nothing about yaw or pitch, which is why it supplements
    a pose estimator rather than replacing one.
    """
    pair = eye_pair(points_image)
    if pair is None:
        return None
    right, left = pair
    direction_image = np.asarray(points_image.get(right) - points_image.get(left), dtype=float)
    direction = direction_image[:2] @ _HALF_TURN.T
    if not np.any(np.abs(direction) > 1e-9):
        return None
    return float(math.degrees(math.atan2(direction[1], direction[0])))


@dataclass(frozen=True)
class CanonicalFrame:
    """Landmarks and covariances in the measurement frame, plus residual pose.

    ``yaw_deg`` / ``pitch_deg`` / ``roll_deg`` are what
    :func:`~vitruve.measure.evaluate.evaluate` should be handed, and they follow
    that function's contract: they are *point estimates* of the pose remaining
    in these coordinates, because ``assess_discriminability`` inflates them by
    the estimator's own spread itself and handing it a pre-inflated number would
    count that spread twice.

    Yaw and pitch are the estimates as reported, since undoing them needs depth
    this frame does not have. Roll is the interesting one. Its point estimate was
    rotated out, so what is left is a zero-mean error with the roll estimate's
    standard deviation, and the expected magnitude of a zero-mean normal is
    ``sd * sqrt(2/pi)`` -- which for the published pose spread comes back to the
    published mean absolute error, about four degrees. That is the number handed
    downstream. Handing over zero would tell the pose-critical gate that a
    correction it knows to be imperfect was perfect.
    """

    view: View
    points: PointSet
    uncertainty: LandmarkUncertainty
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    applied_roll_deg: float
    have_3d: bool
    interocular_px: float | None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _floor_covariances(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    dim = cov.shape[-1]
    return cov + np.eye(dim) * _PD_FLOOR_PX2


def to_canonical_frontal(
    points_image: PointSet,
    covariances: NDArray[np.float64],
    pose: PoseEstimate,
    *,
    correct_roll: bool = True,
) -> CanonicalFrame:
    """Frontal landmarks into the canonical measurement frame.

    The result stays two-dimensional. A z coordinate could be filled with zeros
    and every ``Axis("z")`` formula would then evaluate to something -- and that
    something would be meaningless. Leaving the frame 2D makes
    :class:`~vitruve.core.formula.Axis` raise instead, so a depth-dependent
    measurement asked of a frontal photograph fails loudly rather than
    returning a confident zero.

    The transform is a rigid motion: a half turn, a rotation, and a
    translation. No scaling, so a millimetres-per-pixel estimate measured in the
    original image applies unchanged here, and the scale ladder in
    :mod:`vitruve.core.scale` does not have to know this stage exists.
    """
    coords = np.asarray(points_image.coords, dtype=float)
    if coords.shape[-1] != 2:
        raise ValueError(f"frontal landmarks must be 2D, got {coords.shape[-1]}D")

    applied = pose.roll_deg if correct_roll else 0.0
    linear = _rotation(-applied) @ _HALF_TURN

    pair = eye_pair(points_image)
    if pair is not None:
        right, left = pair
        origin = (points_image.get(right) + points_image.get(left)) / 2.0
    else:
        origin = coords.mean(axis=-2)

    # Translation is presentational. Every formula in the catalogue is built
    # from differences, so putting the origin between the eyes changes no
    # number; it just makes a dumped coordinate readable.
    centred = coords - origin
    canonical = centred @ linear.T

    cov = np.asarray(covariances, dtype=float)
    canonical_cov = np.einsum("ij,njk,lk->nil", linear, cov, linear)

    notes: list[str] = []
    if correct_roll:
        notes.append(
            f"image roll of {applied:+.2f} deg was rotated out of the point estimate; "
            f"the residual passed to the measurement layer is "
            f"{residual_roll_deg(pose):.2f} deg, the expected size of what the "
            f"correction failed to remove, not zero"
        )
    else:
        notes.append("roll was not corrected, so the full estimate is passed downstream")
    notes.extend(pose.notes)

    return CanonicalFrame(
        view=View.FRONTAL,
        points=PointSet(index=dict(points_image.index), coords=canonical),
        uncertainty=LandmarkUncertainty(
            index=dict(points_image.index), covariances=_floor_covariances(canonical_cov)
        ),
        # Uncorrected axes pass their point estimates through; the measurement
        # layer inflates them by the estimator spread on its own.
        yaw_deg=pose.yaw_deg,
        pitch_deg=pose.pitch_deg,
        roll_deg=residual_roll_deg(pose) if correct_roll else pose.roll_deg,
        applied_roll_deg=applied,
        have_3d=False,
        interocular_px=interocular_distance_px(points_image),
        notes=tuple(notes),
    )


#: Mean absolute value of a zero-mean normal, as a multiple of its standard
#: deviation. Applied to the roll standard deviation after correction, it is
#: the expected size of the roll that correction failed to remove.
_MEAN_ABS_OVER_SD = math.sqrt(2.0 / math.pi)


def residual_roll_deg(pose: PoseEstimate) -> float:
    """Expected magnitude of the roll that survived correction.

    Not zero, and not the original estimate either. Rotating by the estimate
    leaves an error whose mean is zero and whose standard deviation is the
    estimate's own; its expected magnitude is ``sd * sqrt(2/pi)``. For the
    published pose spread that lands almost exactly on the published mean
    absolute error of about four degrees, which is a satisfying way to arrive at
    the same number from the other direction.
    """
    return pose.roll_sd_deg * _MEAN_ABS_OVER_SD


def to_canonical_profile(
    points_image: PointSet,
    covariances: NDArray[np.float64],
    pose: PoseEstimate,
) -> CanonicalFrame:
    """Profile landmarks into a three-dimensional sagittal frame.

    A profile photograph shows the sagittal plane, so the image's horizontal
    axis is the subject's anteroposterior direction -- canonical ``+z``, the
    axis the E-line offsets and the chin projection are declared against -- and
    the image's vertical axis is canonical ``-y``. The subject's right, ``+x``,
    points out of the photograph and is not observed at all.

    Two consequences are recorded rather than hidden.

    The frame is emitted as 3D because the profile formulas name ``Axis("z")``
    and a 2D point set has no such axis. It is *not* a 3D fit, so ``have_3d``
    stays false and every ``REQUIRES_3D`` measurement is still withheld. The
    third dimension here is a stipulation about where the landmarks lie, not a
    measurement of where they are.

    The unobserved ``x`` coordinate is given the same positional spread as the
    in-plane ones rather than zero variance. Setting it to zero would claim the
    landmarks lie exactly in the sagittal plane, which is true for pogonion and
    false for gonion, and would also hand a singular covariance to the
    Monte-Carlo sampler. Treating the out-of-plane direction as no better
    localised than the in-plane ones is the weakest defensible assumption.

    Image roll is *not* rotated out here. On a profile view a camera-axis
    rotation is indistinguishable from the subject having raised their chin, and
    there is no interocular axis to level against, so the full gated roll goes
    downstream instead of a correction that could as easily add error as remove
    it.
    """
    coords = np.asarray(points_image.coords, dtype=float)
    if coords.shape[-1] != 2:
        raise ValueError(f"profile landmarks must be 2D, got {coords.shape[-1]}D")

    anterior = _anterior_sign(points_image)
    origin = coords.mean(axis=-2)
    centred = coords - origin

    n = centred.shape[-2]
    canonical = np.zeros((n, 3))
    canonical[:, 1] = -centred[:, 1]
    canonical[:, 2] = anterior * centred[:, 0]

    # A, shaped (3, 2), taking image (x, y) to canonical (x, y, z).
    a = np.array([[0.0, 0.0], [0.0, -1.0], [anterior, 0.0]])
    cov = np.asarray(covariances, dtype=float)
    canonical_cov = np.einsum("ij,njk,lk->nil", a, cov, a)
    in_plane_var = np.trace(cov, axis1=-2, axis2=-1) / cov.shape[-1]
    canonical_cov[:, 0, 0] = in_plane_var

    notes = [
        "profile landmarks were placed in the sagittal plane, so the subject's "
        "right-left coordinate is a stipulation and carries the same spread as "
        "the observed ones",
        "image roll was not corrected on the profile view: with no interocular "
        "axis to level against, a camera-axis rotation and a raised chin are the "
        "same picture",
    ]
    notes.append(
        "yaw is reported as the departure from a true profile, not from frontal: "
        "ninety degrees of yaw is what this view is for, and gating it against "
        "zero would withhold every profile measurement for the pose the protocol "
        "asked the subject to adopt"
    )
    notes.extend(pose.notes)

    return CanonicalFrame(
        view=View.PROFILE,
        points=PointSet(index=dict(points_image.index), coords=canonical),
        uncertainty=LandmarkUncertainty(
            index=dict(points_image.index), covariances=_floor_covariances(canonical_cov)
        ),
        yaw_deg=yaw_deviation(View.PROFILE, pose.yaw_deg),
        pitch_deg=pose.pitch_deg,
        roll_deg=pose.roll_deg,
        applied_roll_deg=0.0,
        have_3d=False,
        interocular_px=interocular_distance_px(points_image),
        notes=tuple(notes),
    )


def _anterior_sign(points_image: PointSet) -> float:
    """Which way the subject faces in the frame: +1 for frame-right, -1 for left.

    Read from the landmarks rather than declared by the caller, because a user
    who turned to their other side has not made an error and should not have to
    say so. Pronasale against tragion is the most robust available pair: the
    nose is the most anterior soft-tissue point on the face and the ear canal is
    close to the most posterior.
    """
    for front, back in (
        (Landmark.PRONASALE, Landmark.TRAGION_R),
        (Landmark.PRONASALE, Landmark.TRAGION_L),
        (Landmark.PRONASALE, Landmark.CERVICALE),
        (Landmark.SUBNASALE, Landmark.CERVICALE),
    ):
        if points_image.has(front, back):
            delta = float(points_image.get(front)[0] - points_image.get(back)[0])
            if abs(delta) > 1e-9:
                return 1.0 if delta > 0 else -1.0
    # Nothing to read it from. Assume the subject faces frame-right, which is
    # the convention every clinical profile protocol asks for, and let the sign
    # of the E-line offsets be the thing a reader notices if it is wrong.
    return 1.0


def to_canonical(
    view: View,
    points_image: PointSet,
    covariances: NDArray[np.float64],
    pose: PoseEstimate,
    *,
    correct_roll: bool = True,
) -> CanonicalFrame:
    """Dispatch to the frontal or profile mapping."""
    if view is View.PROFILE:
        return to_canonical_profile(points_image, covariances, pose)
    return to_canonical_frontal(points_image, covariances, pose, correct_roll=correct_roll)
