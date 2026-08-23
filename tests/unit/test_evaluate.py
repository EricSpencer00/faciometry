"""The Monte-Carlo evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from vitruve.core.landmarks import Landmark, PointSet
from vitruve.core.scale import from_interpupillary, from_ruler
from vitruve.core.spec import Reportability
from vitruve.measure.evaluate import (
    LandmarkUncertainty,
    Measured,
    Unavailable,
    evaluate,
)
from vitruve.measure.registry import BY_ID, CATALOGUE


def _run(face, mid, sd=0.5, **kw):
    unc = LandmarkUncertainty.isotropic(face, sd=sd)
    kw.setdefault("yaw_deg", 0.5)
    kw.setdefault("pitch_deg", 0.5)
    kw.setdefault("roll_deg", 0.5)
    kw.setdefault("have_3d", True)
    kw.setdefault("subject_distance_m", 1.5)
    kw.setdefault("scale", from_ruler(px_per_known_mm=100.0, known_mm=100.0))
    kw.setdefault("seed", 11)
    return evaluate(BY_ID[mid], face, unc, **kw)


def test_recovers_a_known_distance(face):
    r = _run(face, "interpupillary_distance", sd=0.2)
    assert isinstance(r, Measured)
    assert r.value == pytest.approx(63.0, abs=0.5)
    assert r.ci_low < 63.0 < r.ci_high


def test_recovers_a_known_ratio(face):
    r = _run(face, "intercanthal_biocular_ratio", sd=0.2)
    assert r.value == pytest.approx(32.0 / 92.0, rel=0.02)


def test_more_landmark_noise_widens_the_interval(face):
    tight = _run(face, "interpupillary_distance", sd=0.2)
    loose = _run(face, "interpupillary_distance", sd=2.0)
    assert loose.relative_ci_width > tight.relative_ci_width


def test_a_missing_landmark_is_unavailable_not_withheld(face):
    """Two different outcomes: the model cannot see the landmark, versus the
    number would not mean anything. The report must distinguish them."""
    partial = PointSet.from_mapping(
        {Landmark.PUPIL_L: np.array([-31.5, 0.0, 0.0]),
         Landmark.PUPIL_R: np.array([31.5, 0.0, 0.0])}
    )
    unc = LandmarkUncertainty.isotropic(partial, sd=0.5)
    r = evaluate(
        BY_ID["gonial_angle_l"], partial, unc, yaw_deg=0.0, pitch_deg=0.0,
        roll_deg=0.0, have_3d=True, scale=None, subject_distance_m=1.5,
    )
    assert isinstance(r, Unavailable)
    assert "tragion_l" in r.missing_landmarks
    assert "does not supply" in r.reason


def test_a_millimetre_measurement_without_scale_is_unavailable(face):
    r = _run(face, "interpupillary_distance", scale=None)
    assert isinstance(r, Unavailable)
    assert "no scale reference" in r.missing_landmarks[0]


def test_scale_uncertainty_propagates_into_the_interval(face):
    """The scale prior is often the dominant error term for a millimetre value.
    Applying it as a constant would make the interval look far tighter than it
    is."""
    ruler = _run(face, "nose_breadth", scale=from_ruler(100.0, 100.0, reading_error_mm=0.5))
    prior = _run(face, "nose_breadth", scale=from_interpupillary(63.0))
    assert prior.relative_ci_width > ruler.relative_ci_width * 2


def test_provenance_is_recorded_for_every_measurement(face):
    r = _run(face, "nose_mouth_width_ratio")
    assert len(r.formula_fingerprint) == 12
    assert "alare_l" in r.landmarks_used
    assert r.n_valid > 0


def test_evaluation_is_deterministic_under_a_seed(face):
    a = _run(face, "nose_breadth", seed=3)
    b = _run(face, "nose_breadth", seed=3)
    c = _run(face, "nose_breadth", seed=4)
    assert a.value == b.value
    assert a.value != c.value


def test_format_never_hides_the_interval(face):
    r = _run(face, "nose_breadth")
    text = r.format()
    assert "[" in text and "to" in text


def test_a_withheld_measurement_formats_as_withheld(face):
    r = _run(face, "facial_width_height_ratio")
    assert r.verdict.reportability is Reportability.WITHHOLD
    assert r.format() == "Facial width-to-height ratio (fWHR): withheld"
    assert str(r.value) not in r.format()


def test_the_whole_catalogue_evaluates_without_raising(face):
    """Some measurements will be unavailable and many will be withheld. None
    may throw."""
    unc = LandmarkUncertainty.isotropic(face, sd=0.5)
    scale = from_ruler(100.0, 100.0)
    kinds = {"measured": 0, "unavailable": 0}
    for spec in CATALOGUE:
        r = evaluate(
            spec, face, unc, yaw_deg=0.5, pitch_deg=0.5, roll_deg=0.5,
            have_3d=True, scale=scale, subject_distance_m=1.5, seed=5, n_samples=256,
        )
        kinds["measured" if isinstance(r, Measured) else "unavailable"] += 1
    assert kinds["measured"] + kinds["unavailable"] == len(CATALOGUE)
    assert kinds["measured"] > 30


def test_anisotropic_uncertainty_is_representable(face):
    """A point on a jaw contour is well localised across the contour and poorly
    along it. Isotropy is a fallback, not the model."""
    n, dim = face.coords.shape[-2], face.dim
    cov = np.zeros((n, dim, dim))
    for i in range(n):
        cov[i] = np.diag([0.2, 4.0, 4.0][:dim])
    unc = LandmarkUncertainty(index=dict(face.index), covariances=cov)
    r = evaluate(
        BY_ID["interpupillary_distance"], face, unc, yaw_deg=0.5, pitch_deg=0.5,
        roll_deg=0.5, have_3d=True, scale=from_ruler(100.0, 100.0),
        subject_distance_m=1.5, seed=2,
    )
    horizontal = evaluate(
        BY_ID["face_height_sellion_menton"], face, unc, yaw_deg=0.5, pitch_deg=0.5,
        roll_deg=0.5, have_3d=True, scale=from_ruler(100.0, 100.0),
        subject_distance_m=1.5, seed=2,
    )
    # The interpupillary span lies along x, which this covariance localises
    # tightly; the vertical face height lies along y, which it does not.
    assert r.sd < horizontal.sd
