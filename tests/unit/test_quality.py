"""The quality gate: what it measures, what it refuses, and what it tells the user.

Two properties are asserted here that are easy to lose in a refactor and
expensive to lose in production.

Every issue must carry a remedy the photographer can act on. A gate that only
grades photographs pushes the work back onto the person least able to do it,
and the photograph is the one part of this pipeline a user can actually change.

Disagreement between the two pose sources must widen the pose interval. It is
the only observable that separates a pose estimate that failed from one that was
merely imprecise, and if it stops propagating, every downstream discriminability
ratio silently becomes optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from faciometry.core.scale import magnification_distortion
from faciometry.core.sensitivity import POSE_ESTIMATOR_SD_DEG, gated_pose
from faciometry.core.spec import View
from faciometry.pipeline.ingest import SubjectDistance
from faciometry.pipeline.quality import (
    AGREEMENT_FAIL_DEG,
    BLUR_WARN_VARIANCE,
    FACE_HEIGHT_MINIMUM_PX,
    ICAO_INTEROCULAR_MINIMUM_PX,
    ICAO_INTEROCULAR_PREFERRED_PX,
    INTEROCULAR_UNUSABLE_PX,
    UNKNOWN_POSE_SD_DEG,
    PoseEstimate,
    Severity,
    assess,
    clipped_fractions,
    laplacian_variance,
    nominal_yaw_deg,
    reconcile_pose,
    to_grayscale,
    yaw_deviation,
)


@dataclass
class FakePose:
    """The narrowest thing that satisfies the HeadPose protocol."""

    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def sharp_face(seed: int = 0, level: float = 128.0, noise: float = 8.0) -> np.ndarray:
    """A 512-pixel crop with plenty of high-frequency detail and no clipping."""
    rng = np.random.default_rng(seed)
    img = np.full((512, 512, 3), level) + rng.normal(0.0, noise, (512, 512, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def box_blur(pixels: np.ndarray, radius: int = 4) -> np.ndarray:
    """Crude separable box blur, which is all a focus test needs."""
    out = np.asarray(pixels, dtype=float)
    k = 2 * radius + 1
    for axis in (0, 1):
        cum = np.cumsum(np.pad(out, [(radius + 1, radius) if a == axis else (0, 0) for a in range(3)],
                               mode="edge"), axis=axis)
        lo = np.take(cum, range(0, out.shape[axis]), axis=axis)
        hi = np.take(cum, range(k, k + out.shape[axis]), axis=axis)
        out = (hi - lo) / k
    return np.clip(out, 0, 255).astype(np.uint8)


def good_report(**overrides):
    kwargs = dict(
        view=View.FRONTAL,
        aligned_pixels=sharp_face(),
        interocular_px=200.0,
        pose=reconcile_pose(FakePose(0.0, 0.0, 0.0), FakePose(0.5, -0.5, 0.2)),
        subject_distance=SubjectDistance(1.6, 0.1, "test"),
        detection_score=0.99,
    )
    kwargs.update(overrides)
    return assess(**kwargs)


# ---------------------------------------------------------------------------
# Image measures
# ---------------------------------------------------------------------------


def test_laplacian_variance_separates_sharp_from_blurred():
    sharp = to_grayscale(sharp_face())
    blurred = to_grayscale(box_blur(sharp_face()))
    assert laplacian_variance(sharp) > BLUR_WARN_VARIANCE
    assert laplacian_variance(blurred) < laplacian_variance(sharp) / 10


def test_laplacian_variance_of_a_flat_field_is_zero():
    assert laplacian_variance(np.full((64, 64), 128.0)) == pytest.approx(0.0)


def test_clipped_fractions_count_both_ends_separately():
    img = np.full((100, 100), 128.0)
    img[:10, :] = 0.0
    img[10:30, :] = 255.0
    dark, bright = clipped_fractions(img)
    assert dark == pytest.approx(0.10)
    assert bright == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Pose reconciliation
# ---------------------------------------------------------------------------


def test_two_agreeing_sources_keep_the_published_spread():
    pose = reconcile_pose(FakePose(2.0, 1.0, 0.0), FakePose(3.0, 0.0, 1.0))
    assert pose.agreement is not None
    assert pose.agreement.max_delta_deg == pytest.approx(1.0)
    # Averaging two reads of the same image cannot beat the published figure,
    # so the floor holds.
    assert pose.yaw_sd_deg == pytest.approx(POSE_ESTIMATOR_SD_DEG)
    assert pose.yaw_deg == pytest.approx(2.5)


def test_disagreement_widens_the_pose_interval():
    pose = reconcile_pose(FakePose(0.0, 0.0, 0.0), FakePose(24.0, 0.0, 0.0))
    assert pose.yaw_sd_deg == pytest.approx(12.0)
    assert pose.yaw_sd_deg > POSE_ESTIMATOR_SD_DEG
    assert pose.gated_yaw_deg > gated_pose(pose.yaw_deg)


def test_a_single_source_is_recorded_as_uncrosschecked():
    pose = reconcile_pose(FakePose(1.0, 2.0, 3.0), None)
    assert pose.agreement is None
    assert pose.sources == ("landmarker",)
    assert any("only one pose source" in n for n in pose.notes)
    report = good_report(pose=pose)
    codes = {i.code for i in report.issues}
    assert "pose_uncrosschecked" in codes


def test_no_pose_source_is_unknown_rather_than_zero():
    pose = reconcile_pose(None, None)
    assert pose.yaw_sd_deg == pytest.approx(UNKNOWN_POSE_SD_DEG)
    assert pose.gated_yaw_deg > POSE_ESTIMATOR_SD_DEG
    assert any("treated as unknown" in n for n in pose.notes)


def test_geometric_roll_supplies_roll_when_no_network_does():
    pose = reconcile_pose(None, None, geometric_roll_deg=-6.0)
    assert pose.roll_deg == pytest.approx(-6.0)
    assert pose.sources == ("interocular axis",)


def test_geometric_roll_that_contradicts_the_network_widens_roll():
    close = reconcile_pose(FakePose(0.0, 0.0, 0.0), None, geometric_roll_deg=0.2)
    far = reconcile_pose(FakePose(0.0, 0.0, 0.0), None, geometric_roll_deg=30.0)
    assert close.roll_sd_deg == pytest.approx(POSE_ESTIMATOR_SD_DEG)
    assert far.roll_sd_deg == pytest.approx(15.0)


def test_gated_pose_is_the_floor_not_the_rule():
    pose = PoseEstimate(0.0, 0.0, 0.0)
    assert pose.gated_yaw_deg == pytest.approx(gated_pose(0.0))
    wide = PoseEstimate(0.0, 0.0, 0.0, yaw_sd_deg=20.0)
    assert wide.gated_yaw_deg == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_a_good_photograph_produces_no_failure():
    report = good_report()
    assert not report.failed
    assert report.failure_message() == ""
    assert report.worst_severity in (None, Severity.INFO)


def test_every_issue_carries_an_actionable_remedy():
    reports = [
        good_report(),
        good_report(interocular_px=40.0),
        good_report(aligned_pixels=box_blur(sharp_face())),
        good_report(subject_distance=SubjectDistance(0.35, 0.2, "test")),
        good_report(pose=reconcile_pose(FakePose(0.0, 0.0, 0.0), FakePose(30.0, 0.0, 0.0))),
        good_report(interocular_px=None, face_height_px=150.0),
    ]
    issues = [i for r in reports for i in r.issues]
    assert issues, "the fixtures should between them trip several issues"
    for issue in issues:
        assert issue.remedy.strip(), f"{issue.code} has no remedy"
        assert issue.message.strip()
        assert issue.code


def test_a_face_below_the_usable_size_fails_the_run():
    report = good_report(interocular_px=INTEROCULAR_UNUSABLE_PX - 1)
    assert report.failed
    assert "face_too_small" in {i.code for i in report.issues}
    assert str(int(ICAO_INTEROCULAR_MINIMUM_PX)) in report.failure_message()


def test_the_icao_band_is_warn_then_info_then_silence():
    warn = good_report(interocular_px=ICAO_INTEROCULAR_MINIMUM_PX - 5)
    info = good_report(interocular_px=ICAO_INTEROCULAR_PREFERRED_PX - 5)
    quiet = good_report(interocular_px=ICAO_INTEROCULAR_PREFERRED_PX + 5)
    assert {i.severity for i in warn.issues if i.code.startswith("face_")} == {Severity.WARN}
    assert {i.severity for i in info.issues if i.code.startswith("face_")} == {Severity.INFO}
    assert not [i for i in quiet.issues if i.code.startswith("face_")]


def test_a_blurred_photograph_is_reported_with_its_focus_measure():
    report = good_report(aligned_pixels=box_blur(sharp_face(), radius=6))
    blur_issues = [i for i in report.issues if "blur" in i.code or "focus" in i.code]
    assert blur_issues
    assert report.blur_laplacian_variance < BLUR_WARN_VARIANCE


def test_clipping_is_reported_at_each_end_independently():
    bright = sharp_face()
    bright[:200] = 255
    report = good_report(aligned_pixels=bright)
    codes = {i.code for i in report.issues}
    assert "clipped_highlights" in codes
    assert "clipped_shadows" not in codes
    assert report.clipped_bright_fraction > report.clipped_dark_fraction


def test_pose_source_disagreement_is_itself_a_failure():
    pose = reconcile_pose(
        FakePose(0.0, 0.0, 0.0), FakePose(AGREEMENT_FAIL_DEG + 5, 0.0, 0.0)
    )
    report = good_report(pose=pose)
    assert report.failed
    codes = {i.code for i in report.issues}
    assert "pose_sources_disagree" in codes


def test_close_camera_reports_the_icao_magnification_it_implies():
    report = good_report(subject_distance=SubjectDistance(0.4, 0.2, "test"))
    assert report.magnification_distortion == pytest.approx(magnification_distortion(0.4))
    assert report.failed
    assert "camera_too_close" in {i.code for i in report.issues}


def test_unknown_distance_is_an_info_not_a_silence():
    report = good_report(subject_distance=None)
    assert report.magnification_distortion is None
    assert "distance_unknown" in {i.code for i in report.issues}
    assert not report.failed


def _profile_report(**overrides):
    kwargs = dict(
        view=View.PROFILE,
        aligned_pixels=sharp_face(),
        interocular_px=None,
        face_height_px=FACE_HEIGHT_MINIMUM_PX + 50,
        # A profile photograph is *meant* to be at ninety degrees of yaw.
        pose=reconcile_pose(FakePose(90.0, 0.0, 0.0), FakePose(90.0, 0.0, 0.0)),
    )
    kwargs.update(overrides)
    return assess(**kwargs)


def test_a_profile_view_is_sized_against_the_face_box_not_the_eyes():
    ok = _profile_report()
    assert not ok.failed
    assert ok.interocular_px is None
    assert ok.face_height_px == pytest.approx(FACE_HEIGHT_MINIMUM_PX + 50)

    assert _profile_report(face_height_px=100.0).failed


def test_a_profile_is_gated_against_a_profile_not_against_frontal():
    """Ninety degrees of yaw is what this view is for, so it is the reference."""
    assert nominal_yaw_deg(View.PROFILE, 88.0) == pytest.approx(90.0)
    assert nominal_yaw_deg(View.PROFILE, -88.0) == pytest.approx(-90.0)
    assert nominal_yaw_deg(View.FRONTAL, 88.0) == pytest.approx(0.0)
    assert yaw_deviation(View.PROFILE, -84.0) == pytest.approx(6.0)

    assert not _profile_report().failed
    # A frontal photograph submitted as a profile is a mislabelled photograph,
    # and saying so beats measuring a sagittal plane that is not in the frame.
    mislabelled = _profile_report(
        pose=reconcile_pose(FakePose(0.0, 0.0, 0.0), FakePose(0.0, 0.0, 0.0))
    )
    assert mislabelled.failed
    assert "a true profile" in mislabelled.failure_message()


def test_severity_ordering_is_total_and_worst_wins():
    assert Severity.FAIL.rank > Severity.WARN.rank > Severity.INFO.rank
    report = good_report(interocular_px=10.0, aligned_pixels=box_blur(sharp_face(), radius=6))
    assert report.worst_severity is Severity.FAIL
    assert len(report.issues_at(Severity.FAIL)) >= 1


def test_subject_distance_is_exposed_in_the_shape_evaluate_wants():
    report = good_report(subject_distance=SubjectDistance(1.8, 0.1, "test"))
    assert report.subject_distance_m == pytest.approx(1.8)
    assert good_report(subject_distance=None).subject_distance_m is None


def test_gated_magnitudes_never_understate_the_estimator():
    pose = reconcile_pose(FakePose(4.0, 0.0, 0.0), FakePose(4.0, 0.0, 0.0))
    # The whole reason the gate exists: a 4-degree estimate with a 5-degree
    # spread must not pass a 5-degree tolerance.
    assert pose.gated_yaw_deg > 4.0 + math.sqrt(math.pi / 2) * 3.0
