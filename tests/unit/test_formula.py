"""The formula algebra: serialisation, dependency analysis, batching."""

from __future__ import annotations

import numpy as np
import pytest

from vitruve.core.formula import (
    Abs,
    AngleAt,
    Axis,
    Const,
    Diff,
    Dist,
    Expr,
    Mid,
    Pt,
    Ratio,
    SignedTilt,
    registered_ops,
)
from vitruve.core.landmarks import Landmark as L
from vitruve.core.landmarks import PointSet
from vitruve.measure.registry import CATALOGUE


def test_round_trip_preserves_every_catalogue_formula():
    """Serialisation is what puts the exact formula into the run manifest."""
    for spec in CATALOGUE:
        restored = Expr.from_dict(spec.formula.to_dict())
        assert restored.to_json() == spec.formula.to_json(), spec.id
        assert restored.fingerprint == spec.fingerprint, spec.id


def test_fingerprints_are_stable_across_construction_paths():
    a = Ratio(Dist(Pt(L.PUPIL_L), Pt(L.PUPIL_R)), Const(2.0))
    b = Expr.from_dict(a.to_dict())
    assert a.fingerprint == b.fingerprint
    assert len(a.fingerprint) == 12


def test_dependency_analysis_is_static(face):
    """A backend that cannot supply a landmark must learn so before evaluating."""
    f = AngleAt(Pt(L.GLABELLA), Pt(L.NASION), Pt(L.PRONASALE))
    assert f.landmarks() == frozenset({L.GLABELLA, L.NASION, L.PRONASALE})


def test_unknown_op_is_rejected():
    with pytest.raises(ValueError, match="unknown expression op"):
        Expr.from_dict({"op": "definitely_not_an_op"})


def test_algebra_is_closed():
    """Seventeen ops and no escape hatch. An arbitrary callable could not be
    serialised, hashed, or statically analysed."""
    assert len(registered_ops()) == 17
    assert "eval" not in registered_ops()


def test_ratio_yields_nan_rather_than_raising(face):
    """A degenerate Monte-Carlo sample must widen the interval, not abort the
    measurement."""
    r = Ratio(Dist(Pt(L.PUPIL_L), Pt(L.PUPIL_L)), Dist(Pt(L.PUPIL_L), Pt(L.PUPIL_L)))
    assert np.isnan(r.eval(face))


def test_negated_axis_points_the_other_way(face):
    ps = PointSet.from_mapping({L.PUPIL_L: np.array([0.0, 0.0, 0.0])})
    assert np.allclose(Axis("x").eval(ps), [1.0, 0.0, 0.0])
    assert np.allclose(Axis("-x").eval(ps), [-1.0, 0.0, 0.0])


def test_z_axis_is_undefined_in_two_dimensions():
    """Silently returning a zero vector would make a profile formula evaluate to
    a plausible-looking number on a frontal image."""
    ps = PointSet.from_mapping({L.PUPIL_L: np.array([0.0, 0.0])})
    with pytest.raises(ValueError, match="undefined for 2D"):
        Axis("z").eval(ps)


def test_canthal_tilt_matches_closed_form(face):
    expected = np.degrees(np.arctan2(4.0, 30.0))
    for side, en, ex, ax in (
        ("l", L.ENDOCANTHION_L, L.EXOCANTHION_L, "-x"),
        ("r", L.ENDOCANTHION_R, L.EXOCANTHION_R, "x"),
    ):
        got = SignedTilt(Pt(en), Pt(ex), Axis(ax)).eval(face)
        assert float(got) == pytest.approx(expected, abs=1e-9), side


def test_symmetric_face_has_zero_asymmetry(face):
    """A synthetic face that is exactly symmetric must measure as exactly
    symmetric, or the asymmetry formulas carry a constant bias."""
    from vitruve.measure.registry import BY_ID

    for mid in ("canthal_tilt_asymmetry", "ocular_height_asymmetry", "mouth_corner_asymmetry"):
        assert float(BY_ID[mid].formula.eval(face)) == pytest.approx(0.0, abs=1e-9), mid


def test_evaluation_broadcasts_over_a_monte_carlo_ensemble(face, rng):
    ens = face.coords[None] + rng.normal(0.0, 0.5, size=(512, *face.coords.shape))
    batched = PointSet(index=face.index, coords=ens)
    out = Dist(Pt(L.PUPIL_L), Pt(L.PUPIL_R)).eval(batched)
    assert out.shape == (512,)
    assert out.mean() == pytest.approx(63.0, abs=0.2)


def test_mid_and_diff_compose(face):
    m = Mid(Pt(L.ENDOCANTHION_L), Pt(L.ENDOCANTHION_R))
    assert np.allclose(m.eval(face), [0.0, 0.0, 0.0])
    d = Abs(Diff(Dist(Pt(L.PUPIL_L), m), Dist(Pt(L.PUPIL_R), m)))
    assert float(d.eval(face)) == pytest.approx(0.0, abs=1e-9)
