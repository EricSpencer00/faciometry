"""Geometry primitives against closed-form answers."""

from __future__ import annotations

import numpy as np
import pytest

from faciometry.core import geometry as geo


def test_distance_is_pythagorean():
    assert geo.distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(5.0)


def test_angle_at_right_angle():
    o, a, c = np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert geo.angle_at(a, o, c) == pytest.approx(90.0)


@pytest.mark.parametrize("deg", [0.5, 1.0, 5.0, 179.0, 179.5])
def test_angle_at_stays_precise_near_degenerate(deg):
    """arccos of a dot product loses precision near 0 and 180 degrees.

    A nearly straight profile contour lives exactly there, so the atan2 form is
    not a stylistic preference. This test fails on an arccos implementation.
    """
    r = np.radians(deg)
    a, o, c = np.array([1.0, 0.0]), np.zeros(2), np.array([np.cos(r), np.sin(r)])
    assert geo.angle_at(a, o, c) == pytest.approx(deg, abs=1e-9)


def test_angle_between_lines_is_undirected():
    p0, p1, q0, q1 = (np.array(v, float) for v in ([0, 0], [1, 0], [0, 0], [1, 1]))
    forward = geo.angle_between_lines(p0, p1, q0, q1)
    reversed_ = geo.angle_between_lines(p1, p0, q0, q1)
    assert forward == pytest.approx(45.0)
    assert reversed_ == pytest.approx(forward)


def test_signed_tilt_agrees_on_both_lateral_axes():
    """One tilt formula must serve both sides of the face with a matching sign.

    Subject-left points lateral in -x and subject-right in +x. If the sign
    convention depended on the axis direction, every lateralised finding would
    silently invert on one side.
    """
    left = geo.signed_angle_to_axis(
        np.array([-16.0, 0.0]), np.array([-46.0, 4.0]), np.array([-1.0, 0.0])
    )
    right = geo.signed_angle_to_axis(
        np.array([16.0, 0.0]), np.array([46.0, 4.0]), np.array([1.0, 0.0])
    )
    assert left == pytest.approx(np.degrees(np.arctan2(4.0, 30.0)))
    assert left == pytest.approx(right)


def test_signed_tilt_inverts_with_the_offset():
    down = geo.signed_angle_to_axis(
        np.array([0.0, 0.0]), np.array([10.0, -10.0]), np.array([1.0, 0.0])
    )
    assert down == pytest.approx(-45.0)


def test_line_offset_sign_follows_the_normal():
    """Profile aesthetics are stated as signed offsets, so the sign is the
    measurement rather than a detail."""
    a, b = np.array([0.0, 10.0, 20.0]), np.array([0.0, -10.0, 20.0])
    infront = geo.signed_point_to_line_offset(
        np.array([0.0, 0.0, 25.0]), a, b, np.array([0.0, 0.0, 1.0])
    )
    behind = geo.signed_point_to_line_offset(
        np.array([0.0, 0.0, 15.0]), a, b, np.array([0.0, 0.0, 1.0])
    )
    assert infront == pytest.approx(5.0)
    assert behind == pytest.approx(-5.0)


def test_polygon_area_of_unit_square():
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert geo.polygon_area(sq) == pytest.approx(1.0)


def test_rotation_is_orthonormal_and_composes():
    r = geo.rotation_matrix(17.0, -9.0, 4.0)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_yaw_preserves_distances():
    pts = np.array([[10.0, 0.0, 0.0], [-10.0, 5.0, 3.0]])
    rotated = geo.apply_rotation(pts, geo.rotation_matrix(35.0, 0.0, 0.0))
    assert geo.distance(pts[0], pts[1]) == pytest.approx(geo.distance(rotated[0], rotated[1]))


def test_yaw_shrinks_a_projected_transverse_width_by_cosine():
    """The first-order model the sensitivity table is built on."""
    left, right = np.array([-70.0, 0.0, 0.0]), np.array([70.0, 0.0, 0.0])
    for yaw in (5.0, 10.0, 20.0):
        r = geo.rotation_matrix(yaw, 0.0, 0.0)
        proj = geo.apply_rotation(np.stack([left, right]), r)[:, :2]
        assert geo.distance(proj[0], proj[1]) / 140.0 == pytest.approx(
            np.cos(np.radians(yaw)), abs=1e-9
        )


def test_everything_broadcasts_over_leading_axes():
    a = np.random.default_rng(0).normal(size=(64, 7, 3))
    b = np.random.default_rng(1).normal(size=(64, 7, 3))
    assert geo.distance(a, b).shape == (64, 7)
    assert geo.angle_at(a, b, a + b).shape == (64, 7)
