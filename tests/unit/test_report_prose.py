"""The prose layer: what it says, and what it refuses to say.

The synthetic face here is built once and shared by the other two report test
modules. It is deliberately a whole face evaluated through the real
``evaluate`` path rather than a handful of hand-written ``Measured`` objects:
the sentences that matter most are the ones written about measurements the gate
refused, and only the gate knows which those are.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vitruve.core.landmarks import Landmark as L
from vitruve.core.landmarks import PointSet
from vitruve.core.scale import from_interpupillary
from vitruve.core.spec import Verdict, decide_reportability
from vitruve.measure.evaluate import (
    LandmarkUncertainty,
    Measured,
    Unavailable,
    evaluate,
)
from vitruve.measure.registry import BY_ID, CATALOGUE
from vitruve.report import model, prose
from vitruve.report.model import QualityIssue, ReportInput, niosh_stratum

# ---------------------------------------------------------------------------
# A synthetic face, in the canonical frame: +x subject's right, +y up, +z
# toward the viewer, in pixels, with an interpupillary distance of 180.
# ---------------------------------------------------------------------------

FACE: dict[L, tuple[float, float, float]] = {
    L.TRICHION: (0, 230, 0),
    L.GLABELLA: (0, 78, 55),
    L.NASION: (0, 40, 40),
    L.SELLION: (0, 18, 30),
    L.PUPIL_R: (90, 0, 20),
    L.PUPIL_L: (-90, 0, 20),
    L.ENDOCANTHION_R: (47, -2, 10),
    L.ENDOCANTHION_L: (-47, -2, 10),
    L.EXOCANTHION_R: (128, 6, -18),
    L.EXOCANTHION_L: (-128, 3, -18),
    L.PALPEBRALE_SUP_R: (90, 22, 16),
    L.PALPEBRALE_SUP_L: (-90, 22, 16),
    L.PALPEBRALE_INF_R: (90, -20, 16),
    L.PALPEBRALE_INF_L: (-90, -19, 16),
    L.ORBITALE_R: (88, -42, 4),
    L.ORBITALE_L: (-88, -42, 4),
    L.SUPERCILIARE_R: (92, 62, 22),
    L.SUPERCILIARE_L: (-92, 62, 22),
    L.PRONASALE: (0, -105, 120),
    L.SUBNASALE: (0, -140, 62),
    L.COLUMELLA: (0, -125, 85),
    L.ALARE_R: (52, -125, 45),
    L.ALARE_L: (-52, -125, 45),
    L.SUBALARE_R: (44, -140, 50),
    L.SUBALARE_L: (-44, -140, 50),
    L.LABIALE_SUPERIUS: (0, -180, 62),
    L.STOMION: (0, -198, 58),
    L.LABIALE_INFERIUS: (0, -222, 55),
    L.CHEILION_R: (78, -200, 20),
    L.CHEILION_L: (-78, -202, 20),
    L.CRISTA_PHILTRI_R: (16, -176, 62),
    L.CRISTA_PHILTRI_L: (-16, -176, 62),
    L.SUBLABIALE: (0, -250, 40),
    L.POGONION: (0, -290, 56),
    L.GNATHION: (0, -320, 40),
    L.MENTON: (0, -330, 20),
    L.GONION_R: (140, -230, -70),
    L.GONION_L: (-140, -230, -70),
    L.ZYGION_R: (175, -30, -40),
    L.ZYGION_L: (-175, -30, -40),
    L.TRAGION_R: (190, 20, -130),
    L.TRAGION_L: (-190, 20, -130),
    L.PORION_R: (188, 22, -140),
    L.PORION_L: (-188, 22, -140),
    # CERVICALE is left out on purpose, so two profile measurements come back
    # Unavailable and the report has to say why.
}

#: Landmarks whose uncertainty runs along a contour rather than around a point.
#: A gonion is well placed across the jawline and badly placed along it, and
#: that anisotropy is the thing the overlay exists to show.
_CONTOUR = {
    L.GONION_R: 40.0,
    L.GONION_L: 140.0,
    L.ZYGION_R: 70.0,
    L.ZYGION_L: 110.0,
}


def _cov(sx: float, sy: float, sz: float, theta_deg: float = 0.0) -> np.ndarray:
    t = math.radians(theta_deg)
    rot = np.array([[math.cos(t), -math.sin(t), 0.0],
                    [math.sin(t), math.cos(t), 0.0],
                    [0.0, 0.0, 1.0]])
    return rot @ np.diag([sx**2, sy**2, sz**2]) @ rot.T


def point_set() -> PointSet:
    return PointSet.from_mapping(
        {name: np.asarray(xyz, dtype=float) for name, xyz in FACE.items()}
    )


def uncertainty(ps: PointSet) -> LandmarkUncertainty:
    covs = np.empty((len(ps.index), 3, 3))
    for name, i in ps.index.items():
        if name in _CONTOUR:
            covs[i] = _cov(6.0, 1.6, 6.0, _CONTOUR[name])
        else:
            covs[i] = _cov(1.2, 1.2, 2.5)
    return LandmarkUncertainty(index=dict(ps.index), covariances=covs)


def synthetic_measurements(
    *, yaw: float = 4.0, pitch: float = 2.0, roll: float = 1.5, seed: int = 7
) -> tuple[tuple[Measured, ...], tuple[Unavailable, ...]]:
    """Every catalogue measurement this face supports, through the real gate.

    A 3D fit is assumed available, because the withholding path is exercised
    plentifully without also suppressing every measurement that reads a
    self-occluding landmark. With ``have_3d`` False the fixture produced no
    reported measurement at all, and the tests below would then have passed
    over an empty set. The dedicated 3D-refusal case is asserted separately.
    """
    ps = point_set()
    unc = uncertainty(ps)
    scale = from_interpupillary(420.0, declared_sex="female", subject_distance_m=1.6)
    measured: list[Measured] = []
    unavailable: list[Unavailable] = []
    for spec in CATALOGUE:
        out = evaluate(
            spec,
            ps,
            unc,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            have_3d=True,
            scale=scale,
            subject_distance_m=1.6,
            n_samples=256,
            seed=seed,
        )
        (measured if isinstance(out, Measured) else unavailable).append(out)
    return tuple(measured), tuple(unavailable)


def synthetic_report(**kw) -> ReportInput:
    """A whole report, built without a model, a photograph or a network."""
    measured, unavailable = synthetic_measurements(**kw)
    strata = {}
    for m in measured:
        s = niosh_stratum(m.spec_id, sex="female", ancestry="Black")
        if s is not None:
            strata[m.spec_id] = s
    return ReportInput(
        measurements=measured,
        unavailable=unavailable,
        quality=(
            QualityIssue(
                code="pose_yaw",
                detail=(
                    "The head is turned slightly away from the camera, and the "
                    "pose estimate carries its own spread of about 5 degrees."
                ),
                severity="caveat",
                reading="yaw 4.0 deg, pitch 2.0 deg, roll 1.5 deg",
            ),
            QualityIssue(
                code="distance",
                detail="Camera about 1.6 m from the subject, inside the range "
                "ICAO portrait guidance asks for.",
                severity="note",
                reading="1.6 m",
            ),
        ),
        manifest={
            "seed": 7,
            "n_samples": 256,
            "landmark_backend": "synthetic fixture",
            "vitruve_version": "0.0.0+test",
        },
        strata=strata,
        obligations=(
            "SPIGA landmark model: BSD-3-Clause (permissive)",
            "6DRepNet head pose: MIT (permissive) -- inherits 300W-LP, which is "
            "derived from the Basel Face Model",
        ),
        references=(
            "Kleinberg & Vanezis 2007, J Forensic Sci 52:679",
            "Lim et al. 2022, Front Public Health 9:813058",
        ),
        subject_label="Fixture face 01",
        declared_sex="female",
        declared_ancestry="Black",
        generated_at="2026-08-23 00:00 UTC",
    )


@pytest.fixture(scope="module")
def report() -> ReportInput:
    return synthetic_report()


# ---------------------------------------------------------------------------
# Coverage of the catalogue
# ---------------------------------------------------------------------------


def test_every_catalogue_measurement_has_a_region():
    """A measurement with no region would render into no section at all."""
    orphans = [s.id for s in CATALOGUE if s.id not in model.REGION_OF]
    assert orphans == []


def test_regions_are_declared_once():
    keys = [r.key for r in model.REGIONS]
    assert len(keys) == len(set(keys))
    assert set(model.REGION_OF.values()) <= set(keys)


def test_the_fixture_exercises_all_three_outcomes(report: ReportInput):
    """Without all three, the tests below would pass on an empty set."""
    assert report.n_reported > 0
    assert report.n_caveated > 0
    assert report.n_withheld > 0
    assert report.n_unavailable > 0


# ---------------------------------------------------------------------------
# Rule 1: a value never appears without its interval
# ---------------------------------------------------------------------------


def test_every_shown_value_carries_its_interval(report: ReportInput):
    for m in report.shown:
        sentence = prose.statement(m)
        assert "95% interval" in sentence, m.spec_id
        assert " to " in sentence, m.spec_id


def test_withheld_measurements_never_print_the_number():
    """The gate computed a value. Printing it with the refusal would undo it."""
    base, _ = synthetic_measurements()
    victim = next(m for m in base if not m.shown)
    loud = Measured(
        spec_id=victim.spec_id,
        label=victim.label,
        unit=victim.unit,
        value=987.654,
        ci_low=900.0,
        ci_high=1100.0,
        sd=50.0,
        verdict=victim.verdict,
        discriminability=victim.discriminability,
        formula_fingerprint=victim.formula_fingerprint,
        landmarks_used=victim.landmarks_used,
        n_samples=victim.n_samples,
        n_valid=victim.n_valid,
    )
    text = " ".join(prose.describe(loud).sentences())
    assert "987" not in text
    assert "1100" not in text
    assert "is withheld" in text


def test_no_helper_returns_a_bare_value():
    """There is no formatter in the module that yields a value on its own.

    A caller cannot print an uncaveated number by accident if the only
    formatter available emits the interval alongside it.
    """
    measured, _ = synthetic_measurements()
    m = measured[0]
    assert "95% interval" in prose.value_phrase(m)


# ---------------------------------------------------------------------------
# Rule 2: a withheld measurement names its cause
# ---------------------------------------------------------------------------


def test_withheld_sentences_name_a_cause(report: ReportInput):
    names = {c.name for c in (
        prose.CAUSE_DISCRIMINABILITY, prose.CAUSE_UNKNOWN_SPREAD, prose.CAUSE_MARGINAL,
        prose.CAUSE_SELF_OCCLUDING, prose.CAUSE_ROLL, prose.CAUSE_POSE, prose.CAUSE_SCALE,
        prose.CAUSE_LANDMARK, prose.CAUSE_PERSPECTIVE, prose.CAUSE_REPEATABILITY,
        prose.CAUSE_NO_AGREEMENT,
    )}
    assert report.withheld
    for m in report.withheld:
        text = prose.withheld_paragraph(m)
        assert text.startswith(f"{m.label} is withheld. Cause: ")
        assert any(n in text for n in names), text


def test_the_four_named_causes_reach_the_reader():
    """Pose, scale, a self-occluding landmark, and losing to the photograph.

    These are the four the design commits to naming in plain language, so each
    one is provoked and read back.
    """
    fired = {
        prose.cause_of(r).key
        for verdict in _verdict_sweep()
        for r in verdict.reasons
    }
    assert {"pose", "scale", "self_occluding", "discriminability"} <= fired


def _verdict_sweep() -> list[Verdict]:
    """Reason strings from every branch of the core gate that can fire."""
    from vitruve.core.sensitivity import discriminability

    spec_3d = BY_ID["bigonial_width"]
    spec_roll = BY_ID["canthal_tilt_l"]
    spec_conv = BY_ID["philtrum_length"]
    dead = discriminability(
        between_subject_sd=0.01, pose_error=0.4, landmark_error=0.2
    )
    return [
        decide_reportability(
            spec_3d, max_pose_error_deg=30.0, roll_deg=0.0, have_3d=False,
            subject_distance_m=0.4, relative_ci_width=0.5, repeatability_cv=0.4,
            disc=dead,
        ),
        decide_reportability(
            spec_roll, max_pose_error_deg=1.0, roll_deg=7.0, have_3d=True,
            subject_distance_m=None, relative_ci_width=0.02, disc=None,
        ),
        decide_reportability(
            spec_conv, max_pose_error_deg=1.0, roll_deg=0.0, have_3d=True,
            subject_distance_m=None, relative_ci_width=0.02, disc=None,
        ),
    ]


def test_no_reason_falls_through_unclassified(report: ReportInput):
    """An unrecognised reason still reaches the reader, verbatim.

    A new rule in the core gate must not be able to produce a refusal whose
    explanation is silently dropped on the way to the page.
    """
    unmatched = prose.cause_of("some rule nobody has written yet")
    assert unmatched.key == "other"
    assert "some rule nobody has written yet" in unmatched.plain

    seen = [
        prose.cause_of(r)
        for m in report.measurements
        for r in m.verdict.reasons
    ]
    assert seen
    assert all(c.plain for c in seen)


def test_unavailable_is_not_the_same_statement_as_withheld(report: ReportInput):
    assert report.unavailable
    for u in report.unavailable:
        text = prose.unavailable_sentence(u)
        assert "was not attempted" in text
        assert "withheld" not in text
        for name in u.missing_landmarks:
            assert name in text


# ---------------------------------------------------------------------------
# Rule 3: a normative comparison names its stratum
# ---------------------------------------------------------------------------


def test_stratum_sentence_names_the_sample_and_its_size():
    stratum = niosh_stratum("bigonial_width", sex="female", ancestry="Black")
    assert stratum is not None
    assert stratum.n == 589
    measured, _ = synthetic_measurements()
    m = next(x for x in measured if x.spec_id == "bigonial_width")
    sentence = prose.stratum_sentence(m, stratum)
    assert "589 Black female respirator users" in sentence
    assert "caliper" in sentence
    assert "not a general population" in sentence


def test_pooled_stratum_says_it_is_pooled():
    stratum = niosh_stratum("nose_breadth")
    assert stratum is not None
    assert "pooled over ancestry" in stratum.label
    assert "both sexes" in stratum.label


def test_every_stratum_in_the_report_carries_an_n(report: ReportInput):
    assert report.strata
    for stratum in report.strata.values():
        assert stratum.n > 0
        assert str(stratum.n) in stratum.label or f"{stratum.n:,}" in stratum.label


# ---------------------------------------------------------------------------
# Rule 4: context, never a target, and no prescriptions
# ---------------------------------------------------------------------------


def test_reference_ranges_are_framed_as_context():
    spec = BY_ID["nasofrontal_angle"]
    sentence = prose.reference_range_sentence(spec)
    assert sentence is not None
    assert "not a target" in sentence
    assert "where a cited sample fell" in sentence


def test_no_prescriptive_language_anywhere(report: ReportInput):
    text = prose.report_text(report).lower()
    hits = [term for term in prose.PRESCRIPTIVE_TERMS if term in text]
    assert hits == [], f"prescriptive language in the report: {hits}"


def test_no_recommendations(report: ReportInput):
    text = prose.report_text(report).lower()
    for phrase in ("you could", "you should", "treatment", "procedure",
                   "skincare", "surgery", "filler", "routine"):
        assert phrase not in text, phrase


def test_prose_avoids_the_house_style_tells(report: ReportInput):
    """No em dashes, no en dashes, no rule-of-three padding by way of a list."""
    text = prose.report_text(report)
    assert "—" not in text
    assert "–" not in text  # noqa: RUF001
    for phrase in ("it's not just", "it is not just", "delve", "tapestry",
                   "testament", "showcase", "vibrant"):
        assert phrase not in text.lower(), phrase


# ---------------------------------------------------------------------------
# The summary leads with what is missing
# ---------------------------------------------------------------------------


def test_summary_opens_with_the_count(report: ReportInput):
    first = prose.summary(report)[0]
    assert first.startswith(f"{report.n_shown} of {report.n_attempted} attempted")
    assert str(report.n_withheld) in first
    assert str(report.n_unavailable) in first


def test_summary_explains_why_the_rest_are_missing(report: ReportInput):
    text = " ".join(prose.summary(report))
    assert "withheld measurements break down by cause" in text
    assert "landmark model does not supply" in text
    assert "A withheld measurement is a result." in text


def test_summary_states_the_declared_attributes(report: ReportInput):
    text = " ".join(prose.summary(report))
    assert "declared female, Black" in text
    assert "never infers them" in text


def test_undeclared_subject_gets_the_pooled_statement():
    plain = ReportInput(measurements=(), unavailable=())
    text = " ".join(prose.summary(plain))
    assert "No sex or ancestry was declared" in text
    assert "does not infer either attribute" in text


def test_cause_counts_are_counts_of_refusals(report: ReportInput):
    counts = prose.cause_counts(report)
    assert counts
    assert all(isinstance(n, int) and n > 0 for _, n in counts)
    assert [n for _, n in counts] == sorted([n for _, n in counts], reverse=True)


def test_report_text_covers_every_measurement(report: ReportInput):
    text = prose.report_text(report)
    for m in report.measurements:
        assert m.label in text
    for u in report.unavailable:
        assert u.label in text


# ---------------------------------------------------------------------------
# The error budget: what was responsible, and what would move it
# ---------------------------------------------------------------------------


def test_every_budget_decomposes_its_own_measurement(report: ReportInput):
    """The budget is the gate's arithmetic re-read, not a second estimate.

    If these two ever diverge the report is explaining a refusal that did not
    happen, which is worse than printing no explanation at all.
    """
    by_id = {m.spec_id: m for m in report.withheld}
    budgets = report.budgets()
    assert budgets
    for b in budgets:
        d = by_id[b.spec_id].discriminability
        assert d is not None
        assert b.total == pytest.approx(d.total_error_sd)
        assert b.ratio == pytest.approx(d.ratio)
        assert b.spread == pytest.approx(d.between_subject_sd)


def test_the_budget_names_the_dominant_term_across_the_report(report: ReportInput):
    sentence = prose.error_terms_sentence(report)
    assert sentence is not None
    assert sentence.startswith(
        f"Across {len(report.budgets())} withheld measurements the largest error term was"
    )
    assert "share of the variance" in sentence


def test_the_levers_are_ranked_by_measurements_recovered(report: ReportInput):
    levers = prose.lever_lines(report)
    assert levers
    counts = [line.n_recovered for line in levers]
    assert counts == sorted(counts, reverse=True)
    assert all(isinstance(n, int) and n > 0 for n in counts)


def test_a_counterfactual_is_only_offered_when_it_changes_the_verdict(report: ReportInput):
    """The rule that keeps the section from becoming a wish list."""
    offered = 0
    for b in report.budgets():
        sentence = prose.counterfactual_sentence(b)
        if sentence is None:
            assert not b.sufficient
            continue
        offered += 1
        assert b.sufficient
        assert b.sufficient[-1].ratio > 1.0
    assert offered, "no counterfactual was offered at all, so the rule is untested"


def test_the_smallest_sufficient_change_is_the_one_offered(report: ReportInput):
    """Not the largest. Pricing a bigger change than the arithmetic needs would
    ask for work the numbers do not."""
    for b in report.budgets():
        if len(b.sufficient) < 2:
            continue
        chosen = b.sufficient[-1]
        assert chosen.error == max(c.error for c in b.sufficient)
        return
    pytest.skip("this fixture offers no measurement with two sufficient changes")


def test_repeats_are_priced_with_the_shared_fraction_not_root_n():
    """The one sentence that could quietly overstate what averaging buys."""
    from vitruve.measure.budget import repeat_factor
    from vitruve.measure.multishot import DEFAULT_SHARED_FRACTION
    from vitruve.report.model import REPEATS_PRICED

    sentence = prose.repeats_sentence()
    assert f"{repeat_factor(REPEATS_PRICED):.2f}" in sentence
    assert f"{DEFAULT_SHARED_FRACTION:.0%}" in sentence
    assert "does not average away" in sentence
    naive = 1.0 / math.sqrt(REPEATS_PRICED)
    assert f"{naive:.2f}" not in sentence


def test_the_budget_section_precedes_the_withheld_measurements(report: ReportInput):
    text = prose.report_text(report)
    assert prose.BUDGET_TITLE in text
    first_withheld = min(
        text.index(m.label) for m in report.withheld if m.label in text
    )
    assert text.index(prose.BUDGET_TITLE) < first_withheld


def test_the_budget_section_is_empty_when_nothing_was_withheld():
    """No section rather than a section saying there is nothing to say."""
    kept = tuple(m for m in synthetic_report().measurements if m.shown)
    assert prose.budget_section(ReportInput(measurements=kept)) == ()


def test_a_run_that_used_a_ruler_is_not_offered_one():
    from vitruve.measure.budget import Lever

    withheld = synthetic_report().withheld
    measured = ReportInput(measurements=withheld, scale_is_measured=True)
    levers = {
        lever
        for b in measured.budgets()
        for c in b.counterfactuals
        for lever in (c.lever,)
    }
    assert Lever.RULER not in levers


def test_the_capture_note_reaches_the_summary():
    note = "3 of 3 photographs averaged. Landmark error falls to 0.66 of a single capture"
    text = " ".join(prose.summary(ReportInput(capture_note=note)))
    assert note in text
    assert "Several frontal photographs were pooled" in text
