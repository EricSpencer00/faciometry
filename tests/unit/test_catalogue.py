"""Invariants over the measurement catalogue as a whole."""

from __future__ import annotations

import pytest

from faciometry.core.landmarks import Landmark
from faciometry.core.spec import Evidence, Unit, View
from faciometry.measure.registry import BY_ID, CATALOGUE, for_view, satisfiable
from faciometry.norms import niosh


def test_ids_and_fingerprints_are_unique():
    assert len({s.id for s in CATALOGUE}) == len(CATALOGUE)
    assert len({s.fingerprint for s in CATALOGUE}) == len(CATALOGUE)


def test_every_spec_has_a_reference():
    """A measurement with no literature behind it is somebody's invention."""
    for s in CATALOGUE:
        assert s.references, s.id


def test_every_spec_declares_a_pose_sensitivity():
    for s in CATALOGUE:
        assert s.sensitivity.source, s.id


def test_reference_ranges_are_increasing():
    for s in CATALOGUE:
        if s.reference_range:
            lo, hi, src = s.reference_range
            assert lo < hi and src, s.id


def test_lateral_measurements_require_three_dimensions():
    """Zygion and gonion sit on a self-occluding surface, so their apparent 2D
    position is a silhouette artifact rather than the anatomical point."""
    lateral = {Landmark.ZYGION_L, Landmark.ZYGION_R, Landmark.GONION_L, Landmark.GONION_R}
    for s in CATALOGUE:
        if s.landmarks & lateral and s.unit is not Unit.DEGREES:
            assert s.evidence in (Evidence.REQUIRES_3D, Evidence.POSE_CRITICAL), s.id


def test_millimetre_measurements_know_they_need_scale():
    for s in CATALOGUE:
        assert s.needs_metric_scale == (s.unit is Unit.MILLIMETRES), s.id


def test_pose_critical_measurements_have_tight_tolerances():
    for s in CATALOGUE:
        if s.evidence is Evidence.POSE_CRITICAL:
            assert s.pose_tolerance_deg <= 4.0, s.id


def test_pose_invariant_ratios_tolerate_more_pose_than_raw_widths():
    ratio = BY_ID["intercanthal_biocular_ratio"]
    width = BY_ID["bizygomatic_width"]
    assert ratio.pose_tolerance_deg > width.pose_tolerance_deg
    assert ratio.sensitivity.yaw < width.sensitivity.yaw


def test_caliper_measured_spreads_are_preferred_over_photogrammetric_ones():
    """Using a photogrammetric spread as the discriminability numerator would
    put the same errors on both sides of the ratio."""
    for mid in niosh.available():
        if mid in BY_ID:
            assert BY_ID[mid].between_subject_rsd == pytest.approx(
                niosh.spread(mid), rel=1e-9
            ), mid


def test_niosh_covers_the_two_measurements_photogrammetry_cannot_recover():
    assert niosh.covers("bigonial_width")
    assert niosh.covers("bizygomatic_width")


def test_views_partition_the_catalogue():
    assert len(for_view(View.FRONTAL)) + len(for_view(View.PROFILE)) == len(CATALOGUE)


def test_satisfiable_shrinks_with_a_poorer_landmark_model():
    """A backend that cannot see gonion must simply lose those measurements,
    never receive an approximated point."""
    full = frozenset(Landmark)
    without_jaw = full - {Landmark.GONION_L, Landmark.GONION_R}
    assert len(satisfiable(without_jaw)) < len(satisfiable(full))
    assert all(
        Landmark.GONION_L not in s.landmarks for s in satisfiable(without_jaw)
    )


def test_catalogue_declares_no_aggregate():
    """There is no overall score, and there is no field that could become one."""
    for s in CATALOGUE:
        assert "score" not in s.id
        assert "overall" not in s.id
        assert "harmony" not in s.id


def test_measured_slopes_override_the_first_order_model():
    """The projection model is a prediction; the sweep is a measurement.

    `commissure_height_l` declared 0.010 per degree of pitch and measured
    1.189, a factor of 119. A measurement that moves that much under pitch was
    being gated as though pitch barely touched it.
    """
    from faciometry.measure.registry import _measured_slopes

    slopes = _measured_slopes()
    assert slopes, "the generated slope table is missing"
    assert BY_ID["commissure_height_l"].sensitivity.pitch > 1.0


def test_the_override_can_only_widen():
    """The sweep runs on one synthetic face, so a slope it happens to find
    small is not thereby small on every face. Taking the larger of the two can
    withhold a number or widen an interval and never the reverse."""
    from faciometry.measure.registry import _measured_slopes, _sensitivity_for

    slopes = _measured_slopes()
    for spec in CATALOGUE:
        modelled = _sensitivity_for(spec)
        final = spec.sensitivity
        for axis in ("yaw", "pitch", "roll"):
            assert getattr(final, axis) >= abs(getattr(modelled, axis)) - 1e-12, (
                spec.id,
                axis,
            )
            if spec.id in slopes:
                measured = abs(slopes[spec.id].get(axis, 0.0))
                assert getattr(final, axis) >= measured - 1e-12, (spec.id, axis)


def test_a_widened_sensitivity_says_where_it_came_from():
    assert "evals/" in BY_ID["commissure_height_l"].sensitivity.source
