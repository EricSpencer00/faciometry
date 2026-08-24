"""The image-to-canonical conversion, pinned by orientation."""

from __future__ import annotations

import numpy as np
import pytest

from vitruve.core.imaging import rasterize_y_up, to_canonical, to_image


def test_round_trip_is_exact():
    pts = np.array([[10.0, 20.0], [300.5, 411.25]])
    back = to_image(to_canonical(pts, height=512.0), height=512.0)
    assert np.allclose(back, pts)


def test_round_trip_is_exact_with_an_origin():
    pts = np.array([[10.0, 20.0], [300.5, 411.25]])
    origin = np.array([256.0, 128.0])
    out = to_canonical(pts, height=512.0, origin=origin)
    assert np.allclose(to_image(out, height=512.0, origin=origin), pts)


def test_a_point_higher_in_the_image_is_higher_in_canonical_space():
    """The whole point. A row index nearer zero is nearer the top of the
    photograph, and must come out with a larger canonical y."""
    higher_in_image = np.array([[0.0, 10.0]])
    lower_in_image = np.array([[0.0, 400.0]])
    assert (
        to_canonical(higher_in_image, height=512.0)[0, 1]
        > to_canonical(lower_in_image, height=512.0)[0, 1]
    )


def test_the_horizontal_axis_is_untouched():
    pts = np.array([[-40.0, 5.0], [40.0, 5.0]])
    out = to_canonical(pts, height=512.0)
    assert np.allclose(out[:, 0], pts[:, 0])


def test_canthal_tilt_keeps_its_sign_through_the_conversion():
    """A mirrored face still looks like a face, so this bug does not crash. It
    reports the right magnitude with the wrong sign on every subject."""
    from vitruve.core.formula import Axis, Pt, SignedTilt
    from vitruve.core.landmarks import Landmark as L
    from vitruve.core.landmarks import PointSet

    # In image coordinates the outer canthus sits at a smaller row index,
    # meaning it is higher up the photograph: a positive tilt.
    image_pts = {L.ENDOCANTHION_R: np.array([200.0, 260.0]),
                 L.EXOCANTHION_R: np.array([240.0, 250.0])}
    canonical = {
        k: to_canonical(v[None], height=512.0)[0] for k, v in image_pts.items()
    }
    ps = PointSet.from_mapping(canonical)
    tilt = float(SignedTilt(Pt(L.ENDOCANTHION_R), Pt(L.EXOCANTHION_R), Axis("x")).eval(ps))
    assert tilt > 0
    assert tilt == pytest.approx(np.degrees(np.arctan2(10.0, 40.0)))


def test_rasterize_matches_the_image_transform():
    poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert np.allclose(rasterize_y_up(poly, height=512.0), to_image(poly, height=512.0))
