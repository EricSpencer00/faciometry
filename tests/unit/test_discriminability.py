"""The gate that decides whether a number may be printed.

These tests encode published findings as executable assertions. If a change to
the sensitivity model or the gate breaks one of them, the model has drifted
away from the literature it was built on.
"""

from __future__ import annotations

import pytest

from faciometry.core.sensitivity import (
    CANTHAL_TILT,
    KLEINBERG_WORST,
    POSE_ESTIMATOR_SD_DEG,
    TRANSVERSE_RATIO,
    TRANSVERSE_WIDTH,
    discriminability,
    gated_pose,
)
from faciometry.core.spec import (
    Evidence,
    Reportability,
    Unit,
    assess_discriminability,
    decide_reportability,
)
from faciometry.measure.registry import BY_ID, CATALOGUE


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


def test_canthal_tilt_no_longer_absorbs_roll():
    """Roll used to enter one-for-one, because the tilt was measured against the
    image horizon and a tilted camera is indistinguishable from a tilted face.

    It is now measured against the interpupillary line, which rotates with the
    head, so roll cancels exactly. Pitch is untouched by that change and still
    enters at about 0.27 degrees per degree: Vaca et al. (2022) swept the
    Frankfort plane and watched apparent tilt fall from 8.3 to 0.2 degrees over
    thirty degrees.
    """
    assert CANTHAL_TILT.pitch == pytest.approx(0.27)
    assert CANTHAL_TILT.roll < 0.05
    assert CANTHAL_TILT.error_at(0.0, 0.0, 5.0) < 0.1
    assert CANTHAL_TILT.error_at(0.0, 5.0, 0.0) == pytest.approx(1.35)


def test_roll_cancels_exactly_in_the_geometry_not_merely_in_the_table(upright_face):
    """The constant was corrected because the measurement was redefined. If the
    formula regresses to an image-horizon axis this fails, even though the
    declared sensitivity would still read low.

    Two degrees of roll used to manufacture four degrees of canthal tilt
    asymmetry on a perfectly symmetric face.
    """
    import numpy as np

    from faciometry.core import geometry as geo
    from faciometry.core.landmarks import PointSet

    face = {n: upright_face.get(n) for n in upright_face.available}
    ids = (
        "canthal_tilt_l",
        "canthal_tilt_r",
        "canthal_tilt_asymmetry",
        "ocular_height_asymmetry",
        "mouth_corner_asymmetry",
    )
    upright = PointSet.from_mapping(face)
    baseline = {i: float(BY_ID[i].formula.eval(upright)) for i in ids}
    for roll in (1.0, 2.0, 5.0, 10.0):
        rot = geo.rotation_matrix(0.0, 0.0, roll)
        rolled = PointSet.from_mapping(
            {k: geo.apply_rotation(v, rot) for k, v in face.items()}
        )
        for i in ids:
            assert float(BY_ID[i].formula.eval(rolled)) == pytest.approx(
                baseline[i], abs=1e-9
            ), f"{i} moved under {roll} deg of roll"


def test_a_real_asymmetry_still_registers_under_roll(upright_face):
    """Roll invariance is worthless if it were achieved by measuring nothing."""
    import numpy as np

    from faciometry.core import geometry as geo
    from faciometry.core.landmarks import Landmark, PointSet

    skewed = {n: upright_face.get(n) for n in upright_face.available}
    skewed[Landmark.EXOCANTHION_L] = np.array([-46.0, 7.0, -4.0])
    flat = float(BY_ID["canthal_tilt_asymmetry"].formula.eval(PointSet.from_mapping(skewed)))
    assert flat > 3.0

    rot = geo.rotation_matrix(0.0, 0.0, 5.0)
    rolled = PointSet.from_mapping({k: geo.apply_rotation(v, rot) for k, v in skewed.items()})
    assert float(
        BY_ID["canthal_tilt_asymmetry"].formula.eval(rolled)
    ) == pytest.approx(flat, abs=1e-9)


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


def test_self_occlusion_is_keyed_off_landmarks_not_the_evidence_tier():
    """The gonial angle is tagged POSE_CRITICAL for its pose sensitivity, but it
    reads gonion and tragion, both of which sit on a self-occluding surface.

    Keying the 3D requirement off the evidence tier let it through, and it was
    briefly the only measurement a frontal photograph reported.
    """
    from faciometry.core.landmarks import SELF_OCCLUDING

    spec = BY_ID["gonial_angle_l"]
    assert spec.evidence is Evidence.POSE_CRITICAL
    assert spec.landmarks & SELF_OCCLUDING

    v = decide_reportability(
        spec, max_pose_error_deg=1.0, roll_deg=0.5, have_3d=False,
        subject_distance_m=1.5, relative_ci_width=0.05, disc=None,
    )
    assert v.reportability is Reportability.WITHHOLD
    assert any("self-occluding" in r for r in v.blocking)


def test_every_measurement_touching_a_self_occluding_landmark_needs_3d():
    from faciometry.core.landmarks import SELF_OCCLUDING

    checked = 0
    for spec in CATALOGUE:
        if not spec.landmarks & SELF_OCCLUDING:
            continue
        checked += 1
        v = decide_reportability(
            spec, max_pose_error_deg=0.0, roll_deg=0.0, have_3d=False,
            subject_distance_m=1.5, relative_ci_width=0.01, disc=None,
        )
        assert v.reportability is Reportability.WITHHOLD, spec.id
    assert checked >= 6
