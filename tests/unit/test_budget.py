"""The error budget and its counterfactuals."""

from __future__ import annotations

import math

import pytest

from faciometry.measure.budget import Lever, budget_for, ranked_levers, repeat_factor
from faciometry.measure.multishot import DEFAULT_SHARED_FRACTION


def _real_case():
    """The numbers off an actual withheld measurement from a real photograph."""
    return budget_for(
        spec_id="biocular_width",
        label="Biocular width (ex-ex)",
        spread=0.0423,
        pose_error=0.0076,
        landmark_error=0.0465,
        scale_error=0.053,
        scale_is_measured=False,
        sex_declared=False,
    )


def test_terms_combine_in_quadrature():
    b = _real_case()
    assert b.total == pytest.approx(0.0709, abs=1e-4)
    assert b.ratio == pytest.approx(0.60, abs=0.01)


def test_variance_share_not_magnitude_share():
    """A term at half the size of another contributes a quarter as much, which
    is why the loudest sentence in a report is often not the problem."""
    b = _real_case()
    shares = {t.name: t.share_of(b.total) for t in b.terms}
    assert shares["head pose"] < 0.02
    assert shares["scale prior"] > 0.5
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)


def test_the_dominant_term_is_named():
    assert _real_case().dominant.name == "scale prior"


def test_averaging_uses_the_correlated_model_not_root_n():
    """`budget` and `multishot` must agree, or the report promises a reduction
    the feature cannot deliver."""
    for n in (2, 4, 9, 16):
        assert repeat_factor(n) > n**-0.5
        expected = math.sqrt(DEFAULT_SHARED_FRACTION + (1 - DEFAULT_SHARED_FRACTION) / n)
        assert repeat_factor(n) == pytest.approx(expected)


def test_averaging_has_a_floor():
    """More captures cannot drive landmark error to zero, because a model makes
    much the same mistake on much the same photograph."""
    assert repeat_factor(1000) > math.sqrt(DEFAULT_SHARED_FRACTION) - 1e-9


def test_no_single_lever_rescues_this_photograph():
    b = _real_case()
    singles = [c for c in b.counterfactuals if "and" not in c.detail]
    assert singles, "expected single-lever offers"
    assert not any(c.reports for c in singles)


def test_the_combination_does_rescue_it():
    b = _real_case()
    assert b.sufficient
    assert b.sufficient[0].ratio > 1.0


def test_a_lever_that_would_not_help_is_not_offered():
    """Suggesting a ruler to someone who already used one is noise."""
    b = budget_for(
        spec_id="x", label="X", spread=0.05, pose_error=0.001,
        landmark_error=0.02, scale_error=0.0,
        scale_is_measured=True, sex_declared=True,
    )
    assert Lever.RULER not in {c.lever for c in b.counterfactuals}
    assert Lever.DECLARE_SEX not in {c.lever for c in b.counterfactuals}


def test_counterfactual_ratios_use_the_same_arithmetic_as_the_gate():
    b = _real_case()
    for c in b.counterfactuals:
        assert c.ratio == pytest.approx(b.spread / c.error)
        assert c.reports == (c.ratio > 1.0)


def test_ranked_levers_answers_what_do_i_change():
    budgets = [_real_case() for _ in range(3)]
    ranking = ranked_levers(budgets)
    assert ranking
    assert all(n == 3 for _, n, _ in ranking)


def test_an_already_reporting_measurement_contributes_no_lever():
    good = budget_for(
        spec_id="y", label="Y", spread=0.20, pose_error=0.001,
        landmark_error=0.01, scale_error=0.0,
        scale_is_measured=True, sex_declared=True,
    )
    assert good.ratio > 1.0
    assert ranked_levers([good]) == []


# ---------------------------------------------------------------------------
# The budget decomposes the gate's own arithmetic, not a second opinion
# ---------------------------------------------------------------------------


def test_the_budget_total_is_the_gate_total():
    """The identity that makes this a decomposition rather than a re-estimate.

    ``budget_for`` is handed the three components the gate recorded, so the
    quadrature it does has to land back on the number the gate divided by. If
    it does not, the report is explaining a refusal that never happened.
    """
    from faciometry.core.sensitivity import discriminability

    terms = dict(pose_error=0.0076, landmark_error=0.0465, scale_error=0.053)
    d = discriminability(between_subject_sd=0.0423, **terms)
    b = budget_for(
        spec_id="biocular_width",
        label="Biocular width (ex-ex)",
        spread=d.between_subject_sd,
        pose_error=d.pose_component,
        landmark_error=d.landmark_component,
        scale_error=d.scale_component,
        scale_is_measured=False,
        sex_declared=False,
    )
    assert b.total == pytest.approx(d.total_error_sd)
    assert b.ratio == pytest.approx(d.ratio)
    assert sum(t.share_of(b.total) for t in b.terms) == pytest.approx(1.0)


def test_a_capture_already_taken_is_not_offered_back():
    """A run that pooled six photographs is not told to take four."""
    b = budget_for(
        spec_id="x", label="X", spread=0.05, pose_error=0.004,
        landmark_error=0.06, scale_error=0.0,
        scale_is_measured=True, sex_declared=True, repeats=6,
    )
    details = [c.detail for c in b.counterfactuals if c.lever is Lever.REPEATS]
    assert not any("4 photographs" in d for d in details)
    assert any("9 photographs" in d for d in details)
