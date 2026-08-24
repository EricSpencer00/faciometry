"""The canonical frame: orientation, rigidity, and what roll correction costs.

The orientation assertions look trivial and are not. If canonical ``+x`` stops
meaning the subject's right, nothing crashes: every lateralised finding swaps
sides and every canthal tilt flips sign, and the report stays plausible. So the
frame's handedness is pinned down here against landmarks whose side is known by
construction.

The rest of the file is about the trade roll correction makes. Rotating the
points so the eyes are level removes the roll from the *value* exactly, and
leaves the roll estimate's own error behind. Both halves are asserted, because
a change that dropped the second half would look like an improvement -- tighter
numbers, fewer caveats -- and would be the single worst regression available in
this codebase.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from faciometry.core.formula import Axis, Pt, SignedTilt
from faciometry.core.landmarks import Landmark as L
from faciometry.core.landmarks import PointSet
from faciometry.core.sensitivity import POSE_ESTIMATOR_SD_DEG
from faciometry.core.spec import View
from faciometry.measure.evaluate import evaluate
from faciometry.measure.registry import BY_ID
from faciometry.pipeline.canonical import (
    eye_pair,
    interocular_distance_px,
    residual_roll_deg,
    roll_from_eyes,
    to_canonical,
    to_canonical_frontal,
    to_canonical_profile,
)
from faciometry.pipeline.quality import PoseEstimate

#: A handful of landmarks in the canonical frame, in millimetres: ``+x`` the
#: subject's right, ``+y`` up, ``+z`` anterior. Every expected value below is
#: derived from these rather than written down, so a change to the geometry
#: cannot leave a stale constant behind.
TRUTH_MM: dict[L, tuple[float, float, float]] = {
    L.GLABELLA: (0.0, 25.0, 18.0),
    L.PUPIL_R: (31.0, 0.0, 0.0),
    L.PUPIL_L: (-31.0, 0.0, 0.0),
    L.ENDOCANTHION_R: (16.0, 1.0, -4.0),
    L.ENDOCANTHION_L: (-16.0, 1.0, -4.0),
    L.EXOCANTHION_R: (46.0, 3.6, -10.0),
    L.EXOCANTHION_L: (-46.0, 3.6, -10.0),
    L.PALPEBRALE_SUP_R: (31.0, 6.0, -2.0),
    L.PALPEBRALE_INF_R: (31.0, -4.0, -2.0),
    L.PRONASALE: (0.0, -30.0, 30.0),
    L.SUBNASALE: (0.0, -40.0, 15.0),
    L.POGONION: (0.0, -85.0, 12.0),
    L.MENTON: (0.0, -95.0, 4.0),
    L.TRAGION_R: (72.0, 2.0, -75.0),
}

PX_PER_MM = 3.0
IMAGE_ORIGIN = np.array([800.0, 1000.0])

RIGHT_SIDE = (L.PUPIL_R, L.ENDOCANTHION_R, L.EXOCANTHION_R, L.TRAGION_R)
LEFT_SIDE = (L.PUPIL_L, L.ENDOCANTHION_L, L.EXOCANTHION_L)


def _rot(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def frontal_image_points(roll_deg: float = 0.0) -> PointSet:
    """Orthographic frontal projection into image coordinates, with a known roll.

    The inverse of the mapping under test, written independently: canonical
    ``+x`` is the subject's right and image ``+x`` runs to the frame's right, so
    the linear part is a half turn, and a positive roll is the subject's right
    eye rising in the frame.
    """
    names = list(TRUTH_MM)
    canonical = np.array([[TRUTH_MM[n][0], TRUTH_MM[n][1]] for n in names])
    linear = -_rot(roll_deg)
    image = IMAGE_ORIGIN + PX_PER_MM * canonical @ linear.T
    return PointSet(index={n: i for i, n in enumerate(names)}, coords=image)


def profile_image_points(facing: int = -1) -> PointSet:
    """Sagittal projection into image coordinates, facing frame-left by default."""
    names = list(TRUTH_MM)
    image = np.array(
        [
            [
                IMAGE_ORIGIN[0] + facing * PX_PER_MM * TRUTH_MM[n][2],
                IMAGE_ORIGIN[1] - PX_PER_MM * TRUTH_MM[n][1],
            ]
            for n in names
        ]
    )
    return PointSet(index={n: i for i, n in enumerate(names)}, coords=image)


def isotropic(points: PointSet, sd: float = 1.2) -> np.ndarray:
    n = points.coords.shape[-2]
    return np.broadcast_to(np.eye(2) * sd**2, (n, 2, 2)).copy()


def level_pose(roll_deg: float = 0.0) -> PoseEstimate:
    return PoseEstimate(0.0, 0.0, roll_deg, sources=("test",))


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def test_canonical_x_is_the_subjects_right():
    points = frontal_image_points()
    frame = to_canonical_frontal(points, isotropic(points), level_pose())
    for name in RIGHT_SIDE:
        assert frame.points.get(name)[0] > 0, f"{name.value} should sit at positive x"
    for name in LEFT_SIDE:
        assert frame.points.get(name)[0] < 0, f"{name.value} should sit at negative x"
    # And the subject's right is the frame's left, which is the half of this
    # that a mirrored input would break.
    assert points.get(L.PUPIL_R)[0] < points.get(L.PUPIL_L)[0]


def test_canonical_y_points_up_while_image_y_points_down():
    points = frontal_image_points()
    frame = to_canonical_frontal(points, isotropic(points), level_pose())
    assert points.get(L.GLABELLA)[1] < points.get(L.MENTON)[1]
    assert frame.points.get(L.GLABELLA)[1] > frame.points.get(L.MENTON)[1]


def test_the_frontal_transform_is_rigid():
    """Distances must survive, or a pixel scale measured in the original image
    stops applying in the frame the measurements are taken in."""
    points = frontal_image_points(roll_deg=7.0)
    frame = to_canonical_frontal(points, isotropic(points), level_pose(7.0))
    names = list(TRUTH_MM)
    for a, b in pairwise(names):
        before = np.linalg.norm(points.get(a) - points.get(b))
        after = np.linalg.norm(frame.points.get(a) - frame.points.get(b))
        assert after == pytest.approx(before, rel=1e-9)


def test_the_frontal_frame_stays_two_dimensional():
    points = frontal_image_points()
    frame = to_canonical_frontal(points, isotropic(points), level_pose())
    assert frame.points.dim == 2
    # A depth-dependent formula asked of a frontal photograph must fail loudly
    # rather than return a confident zero.
    with pytest.raises(ValueError):
        Axis("z").eval(frame.points)


def test_interocular_distance_falls_back_through_the_ocular_pairs():
    points = frontal_image_points()
    assert eye_pair(points) == (L.PUPIL_R, L.PUPIL_L)
    assert interocular_distance_px(points) == pytest.approx(62.0 * PX_PER_MM)

    without_pupils = PointSet(
        index={n: i for i, n in enumerate(n for n in points.index if "pupil" not in n.value)},
        coords=np.stack(
            [points.get(n) for n in points.index if "pupil" not in n.value]
        ),
    )
    assert eye_pair(without_pupils) == (L.ENDOCANTHION_R, L.ENDOCANTHION_L)


# ---------------------------------------------------------------------------
# Roll
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("roll", [-11.0, -3.0, 0.0, 3.0, 11.0])
def test_roll_from_eyes_recovers_the_roll_that_was_applied(roll):
    points = frontal_image_points(roll_deg=roll)
    assert roll_from_eyes(points) == pytest.approx(roll, abs=1e-9)


def _canthal_tilt(frame) -> float:
    formula = SignedTilt(Pt(L.ENDOCANTHION_R), Pt(L.EXOCANTHION_R), Axis("x"))
    return float(formula.eval(frame.points))


def test_roll_correction_removes_roll_from_the_point_estimate():
    level = to_canonical_frontal(
        frontal_image_points(), isotropic(frontal_image_points()), level_pose()
    )
    rolled_points = frontal_image_points(roll_deg=8.0)
    corrected = to_canonical_frontal(
        rolled_points, isotropic(rolled_points), level_pose(8.0)
    )
    assert _canthal_tilt(corrected) == pytest.approx(_canthal_tilt(level), abs=1e-9)


def test_without_correction_canthal_tilt_absorbs_roll_one_for_one():
    """The finding the correction exists for: three degrees of camera tilt moves
    a quantity whose whole normal range spans about four degrees."""
    level = to_canonical_frontal(
        frontal_image_points(), isotropic(frontal_image_points()), level_pose()
    )
    rolled_points = frontal_image_points(roll_deg=8.0)
    uncorrected = to_canonical_frontal(
        rolled_points, isotropic(rolled_points), level_pose(8.0), correct_roll=False
    )
    shift = _canthal_tilt(uncorrected) - _canthal_tilt(level)
    assert shift == pytest.approx(8.0, abs=1e-6)


def test_correction_does_not_reduce_the_pose_uncertainty():
    rolled_points = frontal_image_points(roll_deg=8.0)
    pose = level_pose(8.0)
    frame = to_canonical_frontal(rolled_points, isotropic(rolled_points), pose)

    assert frame.applied_roll_deg == pytest.approx(8.0)
    # Not zero. The residual is the expected magnitude of a zero-mean error
    # with the roll estimate's own standard deviation.
    assert frame.roll_deg == pytest.approx(
        POSE_ESTIMATOR_SD_DEG * math.sqrt(2.0 / math.pi)
    )
    assert frame.roll_deg == pytest.approx(residual_roll_deg(pose))
    assert frame.roll_deg > 2.0, "the pose-critical gate must still see a residual"


def test_a_wider_roll_estimate_leaves_a_wider_residual():
    points = frontal_image_points(roll_deg=2.0)
    tight = to_canonical_frontal(points, isotropic(points), PoseEstimate(0, 0, 2.0))
    loose = to_canonical_frontal(
        points, isotropic(points), PoseEstimate(0, 0, 2.0, roll_sd_deg=18.0)
    )
    assert loose.roll_deg > tight.roll_deg * 3


def test_residual_roll_still_caveats_a_pose_critical_measurement():
    """The end of the chain: correction happened, and the gate still knows."""
    points = frontal_image_points(roll_deg=6.0)
    frame = to_canonical_frontal(points, isotropic(points), level_pose(6.0))
    spec = BY_ID["canthal_tilt_r"]
    result = evaluate(
        spec,
        frame.points,
        frame.uncertainty,
        yaw_deg=frame.yaw_deg,
        pitch_deg=frame.pitch_deg,
        roll_deg=frame.roll_deg,
        have_3d=frame.have_3d,
        n_samples=256,
        seed=0,
    )
    assert not hasattr(result, "missing_landmarks")
    assert any("roll" in reason for reason in result.verdict.reasons)


# ---------------------------------------------------------------------------
# Covariances
# ---------------------------------------------------------------------------


def test_covariances_rotate_with_the_points():
    points = frontal_image_points(roll_deg=30.0)
    n = points.coords.shape[-2]
    # Strongly anisotropic: well localised along image x, poorly along y. This
    # is the jaw-contour case, and it is the whole reason covariances are kept.
    cov = np.broadcast_to(np.diag([1.0, 25.0]), (n, 2, 2)).copy()
    frame = to_canonical_frontal(points, cov, level_pose(30.0))

    for i in range(n):
        before = np.linalg.eigvalsh(cov[i])
        after = np.linalg.eigvalsh(frame.uncertainty.covariances[i])
        # A rigid motion preserves the spectrum; only the axes move.
        assert after == pytest.approx(before, abs=1e-3)

    principal = np.linalg.eigh(frame.uncertainty.covariances[0])[1][:, -1]
    expected = _rot(-30.0) @ np.array([0.0, -1.0])
    assert abs(abs(float(principal @ expected)) - 1.0) < 1e-6


def test_a_singular_covariance_is_made_positive_definite():
    """A backend that reports zero uncertainty would otherwise crash the sampler
    inside the Monte-Carlo draw, and a crash is a worse answer than a number."""
    points = frontal_image_points()
    n = points.coords.shape[-2]
    frame = to_canonical_frontal(points, np.zeros((n, 2, 2)), level_pose())
    for cov in frame.uncertainty.covariances:
        np.linalg.cholesky(cov)  # raises if not positive definite


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def test_profile_frame_is_sagittal_and_three_dimensional():
    points = profile_image_points()
    frame = to_canonical_profile(points, isotropic(points), level_pose())
    assert frame.points.dim == 3
    assert frame.view is View.PROFILE
    # Three dimensions, but not a 3D fit: REQUIRES_3D must still be withheld.
    assert frame.have_3d is False
    xs = frame.points.coords[:, 0]
    assert np.allclose(xs, 0.0)


@pytest.mark.parametrize("facing", [-1, 1])
def test_profile_z_points_anterior_whichever_way_the_subject_faces(facing):
    points = profile_image_points(facing=facing)
    frame = to_canonical_profile(points, isotropic(points), level_pose())
    z = {n: float(frame.points.get(n)[2]) for n in (L.PRONASALE, L.SUBNASALE, L.TRAGION_R)}
    assert z[L.PRONASALE] > z[L.SUBNASALE] > z[L.TRAGION_R]
    assert frame.points.get(L.GLABELLA)[1] > frame.points.get(L.MENTON)[1]


def test_profile_preserves_sagittal_angles_exactly():
    """The mapping is a rigid motion of the y-z plane, so an angle measured in
    the photograph is the angle in the subject's midline."""
    points = profile_image_points()
    frame = to_canonical_profile(points, isotropic(points), level_pose())
    from faciometry.core.geometry import angle_at

    world = np.array([TRUTH_MM[n] for n in (L.GLABELLA, L.SUBNASALE, L.POGONION)])
    measured = angle_at(
        frame.points.get(L.GLABELLA),
        frame.points.get(L.SUBNASALE),
        frame.points.get(L.POGONION),
    )
    assert float(measured) == pytest.approx(float(angle_at(*world)), abs=1e-9)


def test_profile_out_of_plane_uncertainty_is_stated_not_assumed_away():
    points = profile_image_points()
    frame = to_canonical_profile(points, isotropic(points, sd=1.2), level_pose())
    for cov in frame.uncertainty.covariances:
        # Not zero: the sagittal-plane assumption is true of pogonion and false
        # of gonion, and claiming otherwise would also hand the sampler a
        # singular matrix.
        assert cov[0, 0] == pytest.approx(1.2**2, rel=1e-3)
        np.linalg.cholesky(cov)


def test_profile_roll_is_passed_through_rather_than_corrected():
    points = profile_image_points()
    pose = PoseEstimate(0.0, 0.0, 5.0, sources=("test",))
    frame = to_canonical_profile(points, isotropic(points), pose)
    assert frame.applied_roll_deg == 0.0
    assert frame.roll_deg == pytest.approx(5.0)
    assert any("no interocular axis" in n for n in frame.notes)


def test_dispatch_picks_the_frame_the_view_needs():
    frontal = to_canonical(
        View.FRONTAL,
        frontal_image_points(),
        isotropic(frontal_image_points()),
        level_pose(),
    )
    profile = to_canonical(
        View.PROFILE,
        profile_image_points(),
        isotropic(profile_image_points()),
        level_pose(),
    )
    assert frontal.points.dim == 2
    assert profile.points.dim == 3
