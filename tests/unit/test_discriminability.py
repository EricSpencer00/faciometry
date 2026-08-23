"""The gate that decides whether a number may be printed.

These tests encode published findings as executable assertions. If a change to
the sensitivity model or the gate breaks one of them, the model has drifted
away from the literature it was built on.
"""

from __future__ import annotations

import pytest

from vitruve.core.sensitivity import (
    CANTHAL_TILT,
    KLEINBERG_WORST,
    POSE_ESTIMATOR_SD_DEG,
    TRANSVERSE_RATIO,
    TRANSVERSE_WIDTH,
    discriminability,
    gated_pose,
)
from vitruve.core.spec import (
    Evidence,
    Reportability,
    Unit,
    assess_discriminability,
    decide_reportability,
)
from vitruve.measure.registry import BY_ID, CATALOGUE


def test_reproduces_kleinberg_and_vanezis():
    """The finding the whole project is organised around.

    Kleinberg and Vanezis (2007) measured facial indices shifting 8 to 19
    percent at ten degrees of yaw, against a between-subject relative spread of
    1.2 percent for the tightest index. The pose artifact exceeded the entire
    spread between different people, so such an index is uninformative.
    """
    d = discriminability(
        between_subject_sd=0.012,
        pose_error=KLEINBERG_WORST.error_at(10.0, 0.0, 0.0),
        landmark_error=0.016,
    )
    assert d.ratio < 1.0
    assert not d.informative
    assert "photograph contributes more variance" in d.verdict


def test_a_same_plane_ratio_survives_the_same_pose():
    """Cosine cancels between two transverse terms, which is why ratios are the
    most robust thing the pipeline produces."""
    d = discriminability(
        between_subject_sd=0.05,
        pose_error=TRANSVERSE_RATIO.error_at(10.0, 0.0, 0.0),
        landmark_error=0.016,
    )
    assert d.informative
    assert d.ratio > 2.0


def test_pose_gate_accounts_for_the_estimator_uncertainty():
    """Head pose estimators sit near 4 degrees mean absolute error, so gating on
    the point estimate would pass a face whose pose is barely known."""
    assert POSE_ESTIMATOR_SD_DEG > 4.0
    assert gated_pose(4.0) > 8.0
    assert gated_pose(0.0) == pytest.approx(POSE_ESTIMATOR_SD_DEG)


def test_canthal_tilt_absorbs_roll_one_for_one():
    """Vaca et al. (2022) swept the Frankfort plane and watched apparent canthal
    tilt collapse from 8.3 to 0.2 degrees over thirty degrees of pitch."""
    assert CANTHAL_TILT.roll == pytest.approx(1.0)
    assert CANTHAL_TILT.pitch == pytest.approx(0.27)
    assert CANTHAL_TILT.error_at(0.0, 0.0, 5.0) == pytest.approx(5.0)


def test_errors_combine_in_quadrature_not_linearly():
    s = TRANSVERSE_WIDTH
    combined = s.error_at(10.0, 10.0, 10.0)
    linear = abs(s.yaw * 10.0) + abs(s.pitch * 10.0) + abs(s.roll * 10.0)
    assert combined <= linear + 1e-12


def test_fwhr_is_withheld_because_expression_beats_identity():
    """Kramer (2016) decomposed fWHR variance and found posed expression
    accounting for more of it than identity did. No projection model predicts
    that, so the measured within-person spread must override the derived one.
    """
    spec = BY_ID["facial_width_height_ratio"]
    assert spec.measured_within_person_rsd is not None
    assert "Kramer" in spec.within_person_source
    d = assess_discriminability(
        spec, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, relative_landmark_error=0.005
    )
    assert not d.informative
    v = decide_reportability(
        spec,
        max_pose_error_deg=0.0,
        roll_deg=0.0,
        have_3d=True,
        subject_distance_m=1.5,
        relative_ci_width=0.01,
        disc=d,
    )
    assert v.reportability is Reportability.WITHHOLD


def test_a_perfect_photograph_cannot_rescue_fwhr():
    """Even with zero pose and zero landmark error, the measured within-person
    spread stands. This is the point of the override."""
    spec = BY_ID["facial_width_height_ratio"]
    d = assess_discriminability(
        spec,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        relative_landmark_error=0.0,
        inflate_for_pose_uncertainty=False,
    )
    assert not d.informative


def test_requires_3d_measurements_are_withheld_from_two_dimensions():
    """Lim et al. (2022) measured bigonial breadth at a mean difference of 9.3 mm
    from direct measurement, with limits of agreement spanning -0.9 to 19.6 mm."""
    v = decide_reportability(
        BY_ID["bigonial_width"],
        max_pose_error_deg=0.0,
        roll_deg=0.0,
        have_3d=False,
        subject_distance_m=1.5,
        relative_ci_width=0.01,
        disc=None,
    )
    assert v.reportability is Reportability.WITHHOLD
    assert any("self-occluding" in r for r in v.reasons)


def test_close_range_photography_is_flagged_then_refused():
    spec = BY_ID["nose_breadth"]
    caveat = decide_reportability(
        spec, max_pose_error_deg=0.0, roll_deg=0.0, have_3d=True,
        subject_distance_m=0.8, relative_ci_width=0.01, disc=None,
    )
    refused = decide_reportability(
        spec, max_pose_error_deg=0.0, roll_deg=0.0, have_3d=True,
        subject_distance_m=0.35, relative_ci_width=0.01, disc=None,
    )
    assert caveat.reportability is Reportability.CAVEAT
    assert refused.reportability is Reportability.WITHHOLD
    assert any("perspective magnification" in r for r in refused.reasons)


def test_an_unknown_between_person_spread_is_stated_not_guessed():
    spec = BY_ID["nasofrontal_angle"]
    d = assess_discriminability(
        spec, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, relative_landmark_error=0.01
    )
    v = decide_reportability(
        spec, max_pose_error_deg=0.0, roll_deg=0.0, have_3d=True,
        subject_distance_m=1.5, relative_ci_width=0.01, disc=d,
    )
    if d is None:
        assert any("no published between-person spread" in r for r in v.reasons)


def test_every_withheld_measurement_gives_a_reason():
    """A withheld number without a stated reason is indistinguishable from a
    bug, and the report has nothing to print in its place."""
    for spec in CATALOGUE:
        err = 0.03 if spec.unit is not Unit.DEGREES else 0.9
        d = assess_discriminability(
            spec, yaw_deg=8.0, pitch_deg=8.0, roll_deg=8.0, relative_landmark_error=err
        )
        v = decide_reportability(
            spec, max_pose_error_deg=8.0, roll_deg=8.0, have_3d=False,
            subject_distance_m=0.5, relative_ci_width=0.2, disc=d,
        )
        if v.reportability is not Reportability.REPORT:
            assert v.reasons, spec.id


def test_a_casual_photograph_withholds_a_substantial_share():
    """Printing all forty-five numbers off a selfie would be the bug. Every
    existing tool in this space does exactly that."""
    withheld = 0
    for spec in CATALOGUE:
        err = 0.03 if spec.unit is not Unit.DEGREES else 0.9
        d = assess_discriminability(
            spec, yaw_deg=8.0, pitch_deg=8.0, roll_deg=8.0, relative_landmark_error=err
        )
        v = decide_reportability(
            spec, max_pose_error_deg=8.0, roll_deg=8.0, have_3d=True,
            subject_distance_m=0.4, relative_ci_width=0.1, disc=d,
        )
        withheld += v.reportability is Reportability.WITHHOLD
    assert withheld >= 8, f"only {withheld} withheld on a casual photograph"
