"""The scale ladder and its uncertainty."""

from __future__ import annotations

import math

import pytest

from faciometry.core.scale import (
    CUE_CORRELATION,
    IPD_PRIORS,
    IRIS_DIAMETER_MM,
    NEAR_FIXATION_FACTOR,
    ScaleSource,
    from_interpupillary,
    from_iris,
    from_ruler,
    fuse,
    magnification_distortion,
)


def test_ruler_beats_every_prior():
    """A ruler is a direct observation. Averaging it against a population prior
    would only add the prior's error back in."""
    r = from_ruler(px_per_known_mm=500.0, known_mm=100.0)
    p = from_interpupillary(250.0)
    fused = fuse(r, p)
    assert fused.source is ScaleSource.RULER
    assert fused.mm_per_px == pytest.approx(r.mm_per_px)
    assert fused.relative_sd == pytest.approx(r.relative_sd)


def test_fusing_correlated_cues_does_not_claim_independence():
    """Both cues are read by the same landmark model from the same image.
    Treating them as independent understates the fused uncertainty, which is
    the exact failure this module exists to avoid."""
    i, p = from_iris(48.0), from_interpupillary(255.0)
    fused = fuse(i, p)
    independent = 1.0 / math.sqrt(i.relative_sd**-2 + p.relative_sd**-2)
    assert fused.relative_sd > independent
    assert fused.relative_sd < min(i.relative_sd, p.relative_sd)
    assert 0.0 < CUE_CORRELATION < 1.0


def test_declaring_sex_narrows_the_interval():
    pooled = from_interpupillary(250.0)
    declared = from_interpupillary(250.0, declared_sex="female")
    assert declared.relative_sd < pooled.relative_sd
    assert any("pooled over sex" in n for n in pooled.notes)


def test_undeclared_sex_widens_rather_than_inferring():
    """Faciometry never infers demographics. An undeclared subject gets the pooled
    prior and a wider interval, which is the correct trade."""
    pooled = from_interpupillary(250.0)
    assert pooled.mm_per_px == pytest.approx(IPD_PRIORS[None][0] / 250.0)
    assert any("was not declared" in n for n in pooled.notes)


def test_ancestry_shift_is_disclosed_not_silently_absorbed():
    est = from_interpupillary(250.0, declared_sex="male")
    assert any("ancestry" in n and "does not cover" in n for n in est.notes)


def test_near_fixation_correction_applies_only_at_close_range():
    far = from_interpupillary(250.0, declared_sex="female", subject_distance_m=1.5)
    near = from_interpupillary(250.0, declared_sex="female", subject_distance_m=0.35)
    assert near.mm_per_px / far.mm_per_px == pytest.approx(NEAR_FIXATION_FACTOR)
    assert near.mm_per_px < far.mm_per_px
    assert any("near fixation" in n for n in near.notes)


def test_iris_prior_is_age_invariant_and_stated():
    est = from_iris(50.0)
    assert est.mm_per_px == pytest.approx(IRIS_DIAMETER_MM / 50.0)
    assert any("age-invariant" in n for n in est.notes)


@pytest.mark.parametrize(
    "distance_m,expected_pct",
    [(0.3, 16.7), (0.5, 10.0), (1.0, 5.0), (1.5, 3.3), (2.0, 2.5), (3.0, 1.7)],
)
def test_magnification_distortion_matches_the_icao_table(distance_m, expected_pct):
    assert magnification_distortion(distance_m) * 100 == pytest.approx(expected_pct, abs=0.1)


def test_rejects_impossible_inputs():
    for bad in (lambda: from_iris(0.0), lambda: from_interpupillary(-1.0),
                lambda: from_ruler(0.0, 100.0), lambda: magnification_distortion(0.0)):
        with pytest.raises(ValueError):
            bad()


def test_fuse_requires_at_least_one_estimate():
    with pytest.raises(ValueError, match="at least one"):
        fuse()


def test_a_ruler_removes_the_scale_caveat_rather_than_reciting_the_prior():
    """Blaming an interpupillary prior when a ruler was photographed would be
    simply false, and it made every millimetre value in a clinical capture
    carry a caveat it had not earned."""
    from faciometry.core.spec import Reportability, decide_reportability
    from faciometry.measure.registry import BY_ID

    spec = BY_ID["nose_breadth"]
    common = dict(
        max_pose_error_deg=0.25, roll_deg=0.25, have_3d=True,
        subject_distance_m=1.5, relative_ci_width=0.02, disc=None,
    )
    measured = decide_reportability(**common, spec=spec, scale_is_measured=True)
    prior = decide_reportability(**common, spec=spec, scale_relative_sd=0.058)

    assert not any("scale prior" in r for r in measured.reasons)
    assert any("scale prior" in r for r in prior.reasons)
    assert any("5.8%" in r for r in prior.reasons)
    assert any("ruler" in r for r in prior.reasons)
