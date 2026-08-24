"""Pooling several captures of one face."""

from __future__ import annotations

import numpy as np
import pytest

from vitruve.core.landmarks import Landmark, PointSet
from vitruve.measure.evaluate import LandmarkUncertainty
from vitruve.measure.multishot import DEFAULT_SHARED_FRACTION, combine


def _capture(truth, rng, sd=3.0):
    ps = PointSet.from_mapping({k: v + rng.normal(0, sd, size=3) for k, v in truth.items()})
    return ps, LandmarkUncertainty.isotropic(ps, sd=sd)


@pytest.fixture
def truth(face):
    return {n: face.get(n) for n in face.available}


def test_a_single_capture_is_returned_unchanged(truth):
    rng = np.random.default_rng(0)
    ps, unc = _capture(truth, rng)
    c = combine([(ps, unc)])
    assert c.n_used == 1
    assert c.error_factor == pytest.approx(1.0)
    assert "no averaging" in c.note()


def test_pooling_moves_the_estimate_toward_truth(truth):
    rng = np.random.default_rng(1)
    one = combine([_capture(truth, rng)])
    many = combine([_capture(truth, rng) for _ in range(16)])
    err = lambda c: np.mean(
        [np.linalg.norm(c.points.get(n) - truth[n]) for n in c.points.available]
    )
    assert err(many) < err(one)


def test_the_reduction_is_capped_by_the_shared_fraction(truth):
    """Averaging cannot remove an error the model makes every time."""
    rng = np.random.default_rng(2)
    c = combine([_capture(truth, rng) for _ in range(100)])
    assert c.error_factor > np.sqrt(DEFAULT_SHARED_FRACTION) - 1e-6
    assert c.effective_n < 4.0


def test_effective_n_is_reported_not_the_raw_count(truth):
    """Printing 'nine photographs' beside a reduction worth four overstates it."""
    rng = np.random.default_rng(3)
    c = combine([_capture(truth, rng) for _ in range(9)])
    assert c.n_used == 9
    assert c.effective_n < 9
    assert "worth" in c.note()


def test_an_inconsistent_capture_is_discarded(truth):
    rng = np.random.default_rng(4)
    good = [_capture(truth, rng) for _ in range(6)]
    bad_truth = {k: v + np.array([80.0, 60.0, 0.0]) for k, v in truth.items()}
    bad = _capture(bad_truth, rng)
    c = combine(good + [bad])
    assert c.dropped == (6,)
    assert c.n_used == 6
    assert "discarded as inconsistent" in c.note()


def test_only_landmarks_present_in_every_capture_are_pooled(truth):
    """A model that could not see a point in one photograph does not vote on it."""
    rng = np.random.default_rng(5)
    full = _capture(truth, rng)
    partial_truth = {k: v for k, v in truth.items() if k is not Landmark.GONION_L}
    partial = _capture(partial_truth, rng)
    c = combine([full, partial, full])
    assert Landmark.GONION_L not in c.points.available
    assert Landmark.PUPIL_L in c.points.available


def test_rejects_captures_with_nothing_in_common(truth):
    rng = np.random.default_rng(6)
    a = _capture({Landmark.PUPIL_L: truth[Landmark.PUPIL_L]}, rng)
    b = _capture({Landmark.PUPIL_R: truth[Landmark.PUPIL_R]}, rng)
    with pytest.raises(ValueError, match="share no landmark"):
        combine([a, b])


def test_rejects_an_impossible_shared_fraction(truth):
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="shared_fraction"):
        combine([_capture(truth, rng) for _ in range(2)], shared_fraction=1.0)


def test_outlier_rejection_never_leaves_fewer_than_two(truth):
    """Over-aggressive rejection would silently turn a pool into one capture."""
    rng = np.random.default_rng(8)
    spread = [_capture({k: v * (1 + 0.3 * i) for k, v in truth.items()}, rng) for i in range(3)]
    c = combine(spread)
    assert c.n_used >= 2


def test_the_same_capture_repeated_earns_nothing(truth):
    """The bug this guard exists for.

    A landmark model is deterministic, so the same photograph submitted twice
    yields identical points and identical covariances, and averaging them
    changes nothing about the estimate. A count-based reduction handed out a
    smaller interval for free: three copies of one real photograph took a real
    report from 1 of 68 reportable to 3 of 68, and claimed 1.76 effective
    captures. Someone submitting one photograph nine times would have received
    measurements that are not real.
    """
    rng = np.random.default_rng(11)
    one = _capture(truth, rng)
    for n in (2, 3, 9):
        c = combine([one] * n)
        assert c.shared_fraction == pytest.approx(1.0)
        assert c.error_factor == pytest.approx(1.0)
        assert c.effective_n == pytest.approx(1.0)
        assert "100% of the error does not average away" in c.note()


def test_near_identical_captures_earn_almost_nothing(truth):
    """The reduction degrades smoothly rather than falling off a cliff."""
    rng = np.random.default_rng(12)
    barely = [_capture(truth, rng, sd=3.0) for _ in range(3)]
    barely = [
        (
            PointSet.from_mapping(
                {n: truth[n] + (ps.get(n) - truth[n]) * 0.1 for n in ps.available}
            ),
            unc,
        )
        for ps, unc in barely
    ]
    c = combine(barely)
    assert c.shared_fraction > 0.9
    assert c.error_factor > 0.95


def test_genuinely_independent_captures_still_earn_the_reduction(truth):
    """The guard must not be so strict that honest repeats stop helping."""
    rng = np.random.default_rng(13)
    c = combine([_capture(truth, rng) for _ in range(9)])
    assert c.shared_fraction == pytest.approx(DEFAULT_SHARED_FRACTION)
    assert c.error_factor < 0.7
    assert c.effective_n > 2.0


def test_the_estimate_can_only_tighten_the_assumption(truth):
    """Scatter matching the model's claim does not prove independence, because
    a systematic bias on a face is invisible to this test precisely because it
    repeats. So the observed estimate may raise the shared fraction and never
    lower it below the caller's floor."""
    from vitruve.measure.multishot import observed_shared_fraction

    rng = np.random.default_rng(14)
    wild = np.stack([rng.normal(0, 100.0, size=(30, 3)) for _ in range(5)])
    assert observed_shared_fraction(wild, claimed_sd=1.0, floor=0.35) == pytest.approx(0.35)
    identical = np.repeat(rng.normal(0, 5.0, size=(1, 30, 3)), 5, axis=0)
    assert observed_shared_fraction(identical, claimed_sd=1.0, floor=0.35) == pytest.approx(1.0)
