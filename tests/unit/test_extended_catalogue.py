"""The measurements added for parity with a commercial facial-analysis report.

Every value here is checked against geometry computed independently of the
expression algebra, on the synthetic face in ``conftest.py`` whose coordinates
were chosen so the answers come out in closed form. A failure therefore means a
formula is wired to the wrong landmark, the wrong axis or the wrong sign, and
not that somebody re-baselined a recorded number.

The pose tests earn their place separately. Most of these measurements claim to
be free of image roll because they are read against the interpupillary line
rather than against the image horizon, and that claim has been wrong in this
project before: the catalogue once asserted that roll cancelled in a difference
when it in fact added, by a factor of a thousand. So the claim is measured
here, on projected two-dimensional points, which is what a photograph actually
supplies, and the horizon-referenced form is run alongside as a control to show
the test can tell the two apart.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vitruve.core import geometry as geo
from vitruve.core.formula import Axis, Pt, SignedTilt
from vitruve.core.landmarks import Landmark as L
from vitruve.core.landmarks import PointSet
from vitruve.core.spec import Evidence, Unit, View
from vitruve.measure.registry import BY_ID, CATALOGUE

#: Everything this file is responsible for. Listed rather than derived so that
#: deleting a measurement breaks a test instead of shrinking a set silently.
NEW_IDS: tuple[str, ...] = (
    # orbital and periocular
    "brow_apex_lateral_offset_l",
    "brow_apex_lateral_offset_r",
    "margin_reflex_distance_1_l",
    "margin_reflex_distance_1_r",
    "margin_reflex_distance_2_l",
    "margin_reflex_distance_2_r",
    "medial_canthal_angle_l",
    "medial_canthal_angle_r",
    # lips
    "upper_vermilion_height",
    "lower_vermilion_height",
    "cupids_bow_peak_height_l",
    "cupids_bow_peak_height_r",
    "commissure_height_l",
    "commissure_height_r",
    "upper_lip_projection",
    # nose
    "nasal_dorsal_deviation",
    "alar_base_intercanthal_ratio",
    "nasal_tip_rotation",
    # midface, mandible and chin
    "midface_projection",
    "chin_height",
    "labiomental_sulcus_depth",
    "ramus_body_ratio_l",
    "ramus_body_ratio_r",
)


def val(spec_id: str, ps: PointSet) -> float:
    return float(BY_ID[spec_id].formula.eval(ps))


def rotated(ps: PointSet, *, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> PointSet:
    return ps.with_coords(geo.apply_rotation(ps.coords, geo.rotation_matrix(yaw, pitch, roll)))


def projected(ps: PointSet) -> PointSet:
    """Orthographic projection onto the image plane, which drops depth.

    Rotating in three dimensions and then measuring in three dimensions hides
    the whole problem: a camera does not keep z, and it is the loss of z that
    turns a turned head into a changed measurement.
    """
    return ps.with_coords(ps.coords[..., :2])


def perpendicular_offset(p, a, b, normal) -> float:
    """Offset of ``p`` from the line ab, signed along ``normal``, written out."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d / np.linalg.norm(d)
    w = np.asarray(p, float) - np.asarray(a, float)
    perp = w - (w @ d) * d
    n = np.asarray(normal, float)
    return float(perp @ (n / np.linalg.norm(n)))


@pytest.fixture
def asymmetric(face: PointSet) -> PointSet:
    """The synthetic face with three deliberate asymmetries.

    The symmetric face answers zero for a Cupid's bow height, a commissure
    height and a nasal deviation, and zero is the one value that would pass a
    broken formula as readily as a correct one. These three displacements are
    each exactly known, so the measurements have something to find.
    """
    coords = face.coords.copy()
    coords[face.index[L.CRISTA_PHILTRI_L]] += np.array([0.0, 2.0, 0.0])
    coords[face.index[L.CHEILION_L]] += np.array([0.0, 3.0, 0.0])
    coords[face.index[L.PRONASALE]] += np.array([2.0, 0.0, 0.0])
    return face.with_coords(coords)


# ---------------------------------------------------------------------------
# The catalogue grew, and grew by exactly this
# ---------------------------------------------------------------------------


def test_every_new_measurement_is_in_the_catalogue():
    missing = [i for i in NEW_IDS if i not in BY_ID]
    assert missing == []
    assert len(NEW_IDS) == len(set(NEW_IDS)) == 23


def test_the_catalogue_still_has_unique_ids_and_formulas():
    assert len({s.id for s in CATALOGUE}) == len(CATALOGUE)
    assert len({s.fingerprint for s in CATALOGUE}) == len(CATALOGUE)


def test_no_new_measurement_stands_in_for_a_landmark_the_vocabulary_lacks():
    """The refusals, asserted rather than only described.

    Tarsal platform show, inferior scleral show, eyebrow tail taper, malar
    prominence, paranasal hollowing, the maxillary dental midline, philtral
    column definition and the supratip break each need a point the `Landmark`
    enum does not carry. Adding any of them means adding the landmark first;
    adding it by aiming a formula at the nearest available point would produce
    a number that looks exactly like the real one.
    """
    forbidden = (
        "tarsal",
        "scleral",
        "brow_tail",
        "malar",
        "paranasal",
        "dental_midline",
        "philtral_column",
        "supratip",
    )
    hits = [s.id for s in CATALOGUE if any(f in s.id for f in forbidden)]
    assert hits == []


def test_every_new_measurement_carries_its_evidence_and_its_sensitivity():
    for i in NEW_IDS:
        spec = BY_ID[i]
        assert spec.references, i
        assert spec.description, i
        assert spec.sensitivity.source, i
        assert spec.view in (View.FRONTAL, View.PROFILE), i
        assert 0.0 < spec.pose_tolerance_deg <= 12.0, i
        assert spec.needs_metric_scale == (spec.unit is Unit.MILLIMETRES), i


def test_the_mandibular_ratio_still_refuses_a_flat_photograph():
    """Being dimensionless does not rescue a self-occluding endpoint."""
    for side in ("l", "r"):
        spec = BY_ID[f"ramus_body_ratio_{side}"]
        assert spec.evidence is Evidence.REQUIRES_3D
        assert {L.GONION_L, L.GONION_R} & spec.landmarks


def test_the_only_new_pose_invariant_ratio_is_the_one_that_earns_it():
    """Both terms of the alar ratio join a bilateral pair, so the yaw cosine
    cancels exactly. Nothing else added here can say that."""
    invariant = [i for i in NEW_IDS if BY_ID[i].evidence is Evidence.POSE_INVARIANT_RATIO]
    assert invariant == ["alar_base_intercanthal_ratio"]


# ---------------------------------------------------------------------------
# Orbital and periocular
# ---------------------------------------------------------------------------


def test_brow_apex_offset_is_the_lateral_shift_over_the_fissure_width(face: PointSet):
    sci, ex, en = (face.get(n) for n in (L.SUPERCILIARE_L, L.EXOCANTHION_L, L.ENDOCANTHION_L))
    lateral = face.get(L.PUPIL_L) - face.get(L.PUPIL_R)
    lateral = lateral / np.linalg.norm(lateral)
    expected = float((sci - ex) @ lateral / np.linalg.norm(en - ex))

    assert expected == pytest.approx(-16.0 / math.sqrt(932.0))
    assert val("brow_apex_lateral_offset_l", face) == pytest.approx(expected, abs=1e-12)
    # Negative is the apex sitting medial to the outer canthus, which is where
    # the convention puts it. A sign flip here would say the opposite.
    assert expected < 0.0
    assert val("brow_apex_lateral_offset_r", face) == pytest.approx(expected, abs=1e-12)


def test_margin_reflex_distances_are_measured_from_the_interpupillary_line(face: PointSet):
    pr, pl = face.get(L.PUPIL_R), face.get(L.PUPIL_L)
    up = np.array([0.0, 1.0, 0.0])
    for side, sup, inf in (
        ("l", L.PALPEBRALE_SUP_L, L.PALPEBRALE_INF_L),
        ("r", L.PALPEBRALE_SUP_R, L.PALPEBRALE_INF_R),
    ):
        mrd1 = perpendicular_offset(face.get(sup), pr, pl, up)
        mrd2 = perpendicular_offset(face.get(inf), pr, pl, -up)
        assert (mrd1, mrd2) == pytest.approx((5.0, 5.0))
        assert val(f"margin_reflex_distance_1_{side}", face) == pytest.approx(mrd1, abs=1e-12)
        # Both are positive: the upper margin is above the line and the lower
        # one below it. A shared sign convention would put one of them at -5.
        assert val(f"margin_reflex_distance_2_{side}", face) == pytest.approx(mrd2, abs=1e-12)


def test_medial_canthal_angle_is_subtended_by_the_two_lid_margins(face: PointSet):
    expected = math.degrees(2.0 * math.atan(5.0 / 15.5))
    assert expected == pytest.approx(35.7574, abs=1e-3)
    for side in ("l", "r"):
        assert val(f"medial_canthal_angle_{side}", face) == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Lips
# ---------------------------------------------------------------------------


def test_vermilion_heights_are_the_two_halves_of_the_ratio_already_carried(face: PointSet):
    upper, lower = math.sqrt(20.0), math.sqrt(26.0)
    assert val("upper_vermilion_height", face) == pytest.approx(upper, abs=1e-12)
    assert val("lower_vermilion_height", face) == pytest.approx(lower, abs=1e-12)
    # The pre-existing ratio has to be the quotient of the two new heights, or
    # the catalogue is carrying two definitions of the same lip.
    assert val("lip_vermilion_ratio", face) == pytest.approx(upper / lower, abs=1e-12)


def test_cupids_bow_height_is_zero_on_a_flat_bow_and_the_displacement_otherwise(
    face: PointSet, asymmetric: PointSet
):
    assert val("cupids_bow_peak_height_l", face) == pytest.approx(0.0, abs=1e-12)
    assert val("cupids_bow_peak_height_l", asymmetric) == pytest.approx(2.0, abs=1e-12)
    # The untouched side stays flat, so the measurement is per side and not a
    # pooled value wearing a side label.
    assert val("cupids_bow_peak_height_r", asymmetric) == pytest.approx(0.0, abs=1e-12)


def test_commissure_height_is_the_corner_rise_over_the_mouth_width(
    face: PointSet, asymmetric: PointSet
):
    assert val("commissure_height_l", face) == pytest.approx(0.0, abs=1e-12)
    width = math.sqrt(50.0**2 + 3.0**2)
    assert val("commissure_height_l", asymmetric) == pytest.approx(3.0 / width, abs=1e-12)
    assert val("commissure_height_r", asymmetric) == pytest.approx(0.0, abs=1e-12)


def test_upper_lip_projection_is_signed_against_the_subnasale_pogonion_line(face: PointSet):
    expected = perpendicular_offset(
        face.get(L.LABIALE_SUPERIUS),
        face.get(L.SUBNASALE),
        face.get(L.POGONION),
        np.array([0.0, 0.0, 1.0]),
    )
    assert val("upper_lip_projection", face) == pytest.approx(expected, abs=1e-12)
    # Negative is behind the line. On this face the lip sits fractionally
    # behind it, and the sign is the half of the statement that carries the
    # meaning, so it is asserted rather than taken on the magnitude.
    assert expected == pytest.approx(-0.3674, abs=1e-3)


# ---------------------------------------------------------------------------
# Nose
# ---------------------------------------------------------------------------


def test_nasal_dorsal_deviation_is_zero_on_a_straight_dorsum(
    face: PointSet, asymmetric: PointSet
):
    assert val("nasal_dorsal_deviation", face) == pytest.approx(0.0, abs=1e-12)
    dorsum = asymmetric.get(L.PRONASALE) - asymmetric.get(L.SELLION)
    expected = math.degrees(math.asin(2.0 / float(np.linalg.norm(dorsum))))
    assert expected == pytest.approx(2.7889, abs=1e-3)
    assert val("nasal_dorsal_deviation", asymmetric) == pytest.approx(expected, abs=1e-10)


def test_alar_base_ratio_is_the_subalare_span_over_the_intercanthal_one(face: PointSet):
    assert val("alar_base_intercanthal_ratio", face) == pytest.approx(24.0 / 32.0, abs=1e-12)
    # Not the same measurement as the alare-based rule, which is already in the
    # catalogue inverted. If these ever coincide, one of them is redundant.
    assert val("eye_spacing_ratio", face) != pytest.approx(
        1.0 / val("alar_base_intercanthal_ratio", face)
    )


def test_nasal_tip_rotation_is_the_columellar_inclination_above_the_horizontal(
    face: PointSet,
):
    expected = math.degrees(math.atan2(5.0, 6.0))
    assert val("nasal_tip_rotation", face) == pytest.approx(expected, abs=1e-10)
    assert expected > 0.0  # tip rotated upward


# ---------------------------------------------------------------------------
# Midface, mandible and chin
# ---------------------------------------------------------------------------


def test_midface_projection_is_the_subnasale_ahead_of_the_nasion(face: PointSet):
    assert val("midface_projection", face) == pytest.approx(10.0, abs=1e-12)


def test_chin_height_is_the_sublabiale_to_gnathion_span(face: PointSet):
    assert val("chin_height", face) == pytest.approx(18.0, abs=1e-12)


def test_labiomental_sulcus_depth_is_negative_behind_the_lip_chin_line(face: PointSet):
    expected = perpendicular_offset(
        face.get(L.SUBLABIALE),
        face.get(L.LABIALE_INFERIUS),
        face.get(L.POGONION),
        np.array([0.0, 0.0, 1.0]),
    )
    assert expected == pytest.approx(-4.0, abs=1e-12)
    assert val("labiomental_sulcus_depth", face) == pytest.approx(expected, abs=1e-12)


def test_ramus_body_ratio_is_the_two_arms_of_the_mandible(face: PointSet):
    ramus = float(np.linalg.norm(face.get(L.TRAGION_L) - face.get(L.GONION_L)))
    body = float(np.linalg.norm(face.get(L.GONION_L) - face.get(L.GNATHION)))
    assert (ramus, body) == pytest.approx((math.sqrt(3800.0), math.sqrt(5484.0)))
    assert val("ramus_body_ratio_l", face) == pytest.approx(ramus / body, abs=1e-12)
    assert val("ramus_body_ratio_r", face) == pytest.approx(ramus / body, abs=1e-12)


# ---------------------------------------------------------------------------
# Pose: what the tiers and the sensitivity table claim, measured
# ---------------------------------------------------------------------------

ROLL_DEG = 5.0

#: Measurements whose value a rolled camera cannot touch at all, because every
#: term in them rotates with the head.
ROLL_EXACT = (
    "medial_canthal_angle_l",
    "medial_canthal_angle_r",
    "brow_apex_lateral_offset_l",
    "brow_apex_lateral_offset_r",
    "alar_base_intercanthal_ratio",
    "nasal_dorsal_deviation",
    "upper_vermilion_height",
    "lower_vermilion_height",
)

#: Measurements a rolled camera foreshortens by cos(roll) and no more. They are
#: perpendicular offsets from the interpupillary line, so the distance itself
#: is exact and only its projection onto the frame's vertical is not: 0.4% at
#: five degrees, against the one-for-one a horizon-referenced form would take.
ROLL_COSINE = (
    "margin_reflex_distance_1_l",
    "margin_reflex_distance_2_l",
    "cupids_bow_peak_height_l",
    "commissure_height_l",
)


def test_roll_leaves_the_head_referenced_measurements_exactly_alone(asymmetric: PointSet):
    flat = projected(asymmetric)
    tilted = projected(rotated(asymmetric, roll=ROLL_DEG))
    for i in ROLL_EXACT:
        before, after = val(i, flat), val(i, tilted)
        assert abs(before) > 1e-9, f"{i} is zero here, so this proves nothing"
        assert after == pytest.approx(before, rel=1e-12), i


def test_roll_only_foreshortens_the_perpendicular_offsets(asymmetric: PointSet):
    flat = projected(asymmetric)
    tilted = projected(rotated(asymmetric, roll=ROLL_DEG))
    cos = math.cos(math.radians(ROLL_DEG))
    for i in ROLL_COSINE:
        before, after = val(i, flat), val(i, tilted)
        assert abs(before) > 1e-9, f"{i} is zero here, so this proves nothing"
        assert after == pytest.approx(before * cos, rel=1e-12), i
        assert abs(after - before) / abs(before) < 0.005


def test_the_roll_control_shows_the_test_can_see_roll(asymmetric: PointSet):
    """Without this, the two tests above would pass on a rotation that did not
    happen. The horizon-referenced form of a canthal tilt is the measurement
    this catalogue rejected; it takes the roll one for one."""
    horizon = SignedTilt(Pt(L.ENDOCANTHION_L), Pt(L.EXOCANTHION_L), Axis("x"))
    flat = projected(asymmetric)
    tilted = projected(rotated(asymmetric, roll=ROLL_DEG))
    before = float(horizon.eval(flat))
    after = float(horizon.eval(tilted))
    assert after - before == pytest.approx(ROLL_DEG, abs=1e-9)


def test_the_alar_ratio_survives_yaw_and_the_brow_offset_does_not(asymmetric: PointSet):
    """The POSE_INVARIANT_RATIO tier, and its control.

    A bilateral pair projects to 2*x*cos(yaw) whatever depth it sits at, so a
    ratio of two such pairs is exactly free of yaw. The brow apex offset is
    dimensionless too and is not free of it, because superciliare and
    exocanthion sit at different depths: that is the difference the tier is
    supposed to encode, and it is measured here rather than asserted.
    """
    flat = projected(asymmetric)
    turned = projected(rotated(asymmetric, yaw=10.0))

    ratio_before, ratio_after = (val("alar_base_intercanthal_ratio", p) for p in (flat, turned))
    assert ratio_after == pytest.approx(ratio_before, rel=1e-12)

    brow_before, brow_after = (val("brow_apex_lateral_offset_l", p) for p in (flat, turned))
    assert abs(brow_after - brow_before) / abs(brow_before) > 0.01


def test_every_new_measurement_evaluates_to_a_finite_number(face: PointSet):
    for i in NEW_IDS:
        assert np.isfinite(val(i, face)), i
