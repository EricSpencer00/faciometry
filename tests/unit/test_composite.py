"""Mirror composites: the axis, the frame, and the number that is not there.

Three properties are worth testing and one is worth testing hard.

The easy two are that the composite is symmetric about the axis it claims, and
that the two halves of an asymmetric face give visibly different pictures.

The hard one is roll. A composite built about the image's vertical looks
entirely convincing on a rolled photograph: it is still a face, it is still
symmetric about something, and nothing about it announces that the something
was the camera. So the test rotates the whole scene, image and landmarks
together, builds the composite from the rotated scene, rotates it back, and
requires the same picture. The control alongside it builds a composite about
the middle of the frame and requires that the comparison notices, because
without that control a roll test passes on any function of two images.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image

from vitruve.core.landmarks import Landmark as L
from vitruve.core.spec import Reportability, Unit, Verdict
from vitruve.measure.evaluate import Measured
from vitruve.report import composite as comp

W, H = 240, 300

#: A face whose midline sits sixteen pixels off the centre of the frame, which
#: is an ordinary amount of off-centre for a portrait. The offset is the whole
#: point: a composite built about the middle of the picture would move the face
#: sideways by twice that, which is plainly visible and which no symmetry
#: number would ever report.
MIDLINE_X = 136.0

#: Bilateral pairs as (lateral offset, image row). The subject's left is at the
#: larger image x, matching `models.protocols`: in a mirror-free frontal
#: photograph the subject's right side is at the smaller column.
_PAIRS: tuple[tuple[L, L, float, float], ...] = (
    (L.PUPIL_L, L.PUPIL_R, 20.0, 120.0),
    (L.ENDOCANTHION_L, L.ENDOCANTHION_R, 10.0, 122.0),
    (L.EXOCANTHION_L, L.EXOCANTHION_R, 30.0, 119.0),
    (L.ALARE_L, L.ALARE_R, 10.0, 165.0),
    (L.CHEILION_L, L.CHEILION_R, 20.0, 200.0),
    (L.ZYGION_L, L.ZYGION_R, 40.0, 150.0),
)

POINTS: dict[L, tuple[float, float]] = {}
for _left, _right, _dx, _y in _PAIRS:
    POINTS[_left] = (MIDLINE_X + _dx, _y)
    POINTS[_right] = (MIDLINE_X - _dx, _y)


def _blob(cx: float, cy: float, sx: float, sy: float, amp: float) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    return amp * np.exp(-((((xs - cx) / sx) ** 2) + (((ys - cy) / sy) ** 2)))


def _face_array(*, asymmetric: bool) -> np.ndarray:
    """A smooth synthetic portrait.

    Smooth on purpose: every comparison below survives one or two resamplings,
    and a test image with hard edges would fail on interpolation ringing rather
    than on anything about the composite.
    """
    a = np.full((H, W), 24.0)
    a += _blob(MIDLINE_X, 160.0, 58.0, 92.0, 150.0)
    for x in (MIDLINE_X - 20.0, MIDLINE_X + 20.0):
        a -= _blob(x, 120.0, 9.0, 6.0, 90.0)
    a -= _blob(MIDLINE_X, 200.0, 22.0, 7.0, 60.0)
    a += _blob(MIDLINE_X, 165.0, 10.0, 16.0, 30.0)
    if asymmetric:
        # A bright patch on the subject's left cheek, which is the only thing
        # telling the two composites apart.
        a += _blob(MIDLINE_X + 32.0, 170.0, 16.0, 20.0, 70.0)
    return np.clip(a, 0.0, 255.0)


def _image(*, asymmetric: bool = True) -> Image.Image:
    a = _face_array(asymmetric=asymmetric).astype(np.uint8)
    return Image.fromarray(np.stack([a, a, a], axis=-1), mode="RGB")


def _rotation(theta_deg: float, centre: tuple[float, float]):
    """Forward map ``p -> R p + t`` for a rotation about ``centre``."""
    t = math.radians(theta_deg)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    c = np.asarray(centre, dtype=float)
    return r, c - r @ c


def _warp(img: Image.Image, r: np.ndarray, t: np.ndarray) -> Image.Image:
    """Apply a forward affine map to the picture.

    Pillow's transform runs backwards, from output pixel to input pixel, so the
    inverse goes in.
    """
    inv = np.linalg.inv(r)
    off = -inv @ t
    data = (inv[0, 0], inv[0, 1], off[0], inv[1, 0], inv[1, 1], off[1])
    return img.transform(img.size, Image.AFFINE, data, resample=Image.BICUBIC)


def _move(points, r: np.ndarray, t: np.ndarray) -> dict[L, tuple[float, float]]:
    out = {}
    for name, p in points.items():
        q = r @ np.asarray(p, dtype=float) + t
        out[name] = (float(q[0]), float(q[1]))
    return out


def _disc(centre: tuple[float, float], radius: float) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    return ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) <= radius**2


def _difference(a: Image.Image, b: Image.Image, mask: np.ndarray) -> float:
    """Mean absolute difference in grey levels over the masked region."""
    x = np.asarray(a.convert("RGB"), dtype=float)
    y = np.asarray(b.convert("RGB"), dtype=float)
    return float(np.mean(np.abs(x - y)[mask]))


def _decode(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


def _measured(spec_id: str, value: float) -> Measured:
    return Measured(
        spec_id=spec_id,
        label=spec_id,
        unit=Unit.DEGREES,
        value=value,
        ci_low=value - 1.0,
        ci_high=value + 1.0,
        sd=0.5,
        verdict=Verdict(Reportability.REPORT),
        discriminability=None,
        formula_fingerprint="0" * 12,
        landmarks_used=(),
        n_samples=8,
        n_valid=8,
    )


# ---------------------------------------------------------------------------
# The axis
# ---------------------------------------------------------------------------


def test_the_midline_is_the_face_and_not_the_frame():
    midline = comp.facial_midline(POINTS)
    assert midline.origin[0] == pytest.approx(MIDLINE_X)
    assert abs(midline.origin[0] - W / 2.0) > 2.0
    assert midline.axis_pair == ("pupil_l", "pupil_r")
    assert midline.n_pairs == 6


def test_the_normal_points_to_the_subjects_left():
    """Sides are anatomical here, taken from landmarks that name their own.

    A convention read off the picture instead would swap every left and right
    finding on a mirrored photograph and nothing downstream would notice.
    """
    midline = comp.facial_midline(POINTS)
    left = midline.signed_offset(np.array(POINTS[L.PUPIL_L]))
    right = midline.signed_offset(np.array(POINTS[L.PUPIL_R]))
    assert float(left) > 0.0 > float(right)


def test_the_midline_rotates_with_the_face():
    """Exact, because this is geometry and not resampling."""
    r, t = _rotation(9.0, (110.0, 150.0))
    base = comp.facial_midline(POINTS)
    turned = comp.facial_midline(_move(POINTS, r, t))

    expected_origin = r @ np.asarray(base.origin) + t
    assert np.allclose(np.asarray(turned.origin), expected_origin, atol=1e-9)
    assert np.allclose(turned.normal, r @ base.normal, atol=1e-12)
    assert turned.tilt_deg == pytest.approx(base.tilt_deg + 9.0, abs=1e-9)


def test_a_missing_pair_refuses_rather_than_guessing_the_centre():
    with pytest.raises(ValueError, match="mirror axis"):
        comp.facial_midline({L.SUBNASALE: (126.0, 160.0), L.MENTON: (126.0, 230.0)})


def test_a_collapsed_pair_refuses_too():
    """Two pixels of baseline is one pixel of landmark error away from thirty
    degrees of mirror axis, so it is refused rather than used."""
    with pytest.raises(ValueError):
        comp.facial_midline({L.PUPIL_L: (126.0, 120.0), L.PUPIL_R: (124.0, 120.0)})


def test_reflecting_a_point_twice_returns_it():
    midline = comp.facial_midline(POINTS)
    once = comp.reflect_points(POINTS, midline)
    twice = comp.reflect_points(once, midline)
    for name, p in POINTS.items():
        assert twice[name] == pytest.approx(p, abs=1e-9)
    # A bilateral pair lands on its partner, because the axis was built from
    # their midpoints.
    assert once[L.PUPIL_L] == pytest.approx(POINTS[L.PUPIL_R], abs=1e-9)


# ---------------------------------------------------------------------------
# The pictures
# ---------------------------------------------------------------------------


def test_the_composite_is_symmetric_about_the_midline_it_claims():
    img = _image()
    midline = comp.facial_midline(POINTS)
    left = comp.mirror_composite(img, midline, side="left")
    folded = comp.mirror_image(left, midline)
    # Restricted to a disc whose reflection also lies inside the frame; outside
    # it the reflection asks for pixels the photograph does not have.
    assert _difference(left, folded, _disc((MIDLINE_X, 160.0), 60.0)) < 0.5


def test_the_two_halves_of_an_asymmetric_face_give_different_pictures():
    img = _image(asymmetric=True)
    left, right = comp.mirror_composites(img, POINTS)
    diff = _difference(
        _decode(left.png), _decode(right.png), _disc((MIDLINE_X, 170.0), 70.0)
    )
    assert diff > 5.0


def test_a_symmetric_face_composites_back_to_itself():
    """The control for the test above: without it, any two pictures differ.

    The tolerance is half a grey level rather than a comfortable few, because
    on this axis the reflection lands on the pixel grid and there is nothing
    left to resample. A tolerance loose enough to absorb a one-pixel shift
    would absorb the registration bug that a mirror composite is most likely to
    have, which is the half-pixel offset between Pillow's sampling frame and
    the landmark grid.
    """
    img = _image(asymmetric=False)
    left, right = comp.mirror_composites(img, POINTS)
    mask = _disc((MIDLINE_X, 170.0), 70.0)
    assert _difference(_decode(left.png), _decode(right.png), mask) < 0.5
    assert _difference(_decode(left.png), img, mask) < 0.5


def test_the_image_reflection_lands_where_the_point_reflection_says():
    """The picture and the points have to move together, to the pixel.

    They are computed by different code, one in Pillow's continuous sampling
    frame and one in landmark indices, and the two frames differ by half a
    pixel. Nothing about a face shows a one-pixel disagreement; a bright spot
    with a known position does.
    """
    a = np.full((H, W), 20.0) + _blob(96.0, 150.0, 5.0, 5.0, 200.0)
    img = Image.fromarray(
        np.stack([a.astype(np.uint8)] * 3, axis=-1), mode="RGB"
    )
    midline = comp.facial_midline(POINTS)
    reflected = np.asarray(comp.mirror_image(img, midline), dtype=float)[..., 0]
    y, x = np.unravel_index(int(np.argmax(reflected)), reflected.shape)

    expected = comp.reflect_points({L.PUPIL_L: (96.0, 150.0)}, midline)[L.PUPIL_L]
    assert (x, y) == (round(expected[0]), round(expected[1]))
    assert expected[0] == pytest.approx(2 * MIDLINE_X - 96.0)


def test_the_composite_is_roll_invariant():
    """Rotate the scene, composite, rotate back, and require the same picture."""
    theta, centre = 8.0, (118.0, 155.0)
    img = _image()
    r, t = _rotation(theta, centre)

    upright = comp.mirror_composite(img, comp.facial_midline(POINTS), side="left")
    rolled_scene = _warp(img, r, t)
    rolled = comp.mirror_composite(
        rolled_scene, comp.facial_midline(_move(POINTS, r, t)), side="left"
    )
    inverse_r, inverse_t = _rotation(-theta, centre)
    recovered = _warp(rolled, inverse_r, inverse_t)

    mask = _disc((MIDLINE_X, 165.0), 55.0)
    assert _difference(upright, recovered, mask) < 3.0


def test_the_roll_control_shows_the_comparison_has_power():
    """A composite about the frame's vertical, which is the mistake this module
    exists to avoid. If the check above cannot see this, it sees nothing."""
    img = _image()
    honest = comp.mirror_composite(img, comp.facial_midline(POINTS), side="left")
    frame_axis = comp.FacialMidline(
        origin=(W / 2.0, H / 2.0),
        toward_subject_left=(1.0, 0.0),
        axis_pair=("frame", "frame"),
        n_pairs=0,
    )
    naive = comp.mirror_composite(img, frame_axis, side="left")
    assert _difference(honest, naive, _disc((MIDLINE_X, 165.0), 55.0)) > 6.0


def test_a_rolled_scene_defeats_the_frame_axis_and_not_the_face_axis():
    """The same control under roll, which is where the frame axis fails worst:
    it shears the two halves against each other instead of sliding them."""
    theta, centre = 8.0, (118.0, 155.0)
    img = _image()
    r, t = _rotation(theta, centre)
    rolled_scene = _warp(img, r, t)
    frame_axis = comp.FacialMidline(
        origin=(W / 2.0, H / 2.0),
        toward_subject_left=(1.0, 0.0),
        axis_pair=("frame", "frame"),
        n_pairs=0,
    )
    naive = comp.mirror_composite(rolled_scene, frame_axis, side="left")
    honest = comp.mirror_composite(
        rolled_scene, comp.facial_midline(_move(POINTS, r, t)), side="left"
    )
    inverse_r, inverse_t = _rotation(-theta, centre)
    mask = _disc((MIDLINE_X, 165.0), 55.0)
    upright = comp.mirror_composite(img, comp.facial_midline(POINTS), side="left")
    kept = _difference(upright, _warp(honest, inverse_r, inverse_t), mask)
    lost = _difference(upright, _warp(naive, inverse_r, inverse_t), mask)
    # Stated as a ratio so the claim is about separation rather than about a
    # threshold that happens to sit between two numbers on this machine. The
    # residual on the honest side is resampling, three passes of it.
    assert kept < 3.0
    assert lost > 4.0 * kept


def test_the_seam_is_ramped_rather_than_stepped():
    midline = comp.facial_midline(POINTS)
    mask = np.asarray(comp.half_mask((W, H), midline, side="left", feather_px=3.0), dtype=float)
    row = mask[160]
    partial = np.sum((row > 5) & (row < 250))
    assert partial >= 3, "a hard edge along the midline reads as a scar"
    assert row[0] == 0 and row[-1] == 255


def test_the_two_sides_keep_opposite_halves():
    midline = comp.facial_midline(POINTS)
    left = np.asarray(comp.half_mask((W, H), midline, side="left", feather_px=1.0), dtype=int)
    right = np.asarray(comp.half_mask((W, H), midline, side="right", feather_px=1.0), dtype=int)
    # One grey level of slack, and one only: the two ramps sum to exactly 1
    # before they are quantised to bytes, and a pixel that lands on a half
    # level rounds down in both masks.
    assert np.all(np.abs(left + right - 255) <= 1)


# ---------------------------------------------------------------------------
# What travels with the pictures, and what does not
# ---------------------------------------------------------------------------


def test_the_asymmetries_are_the_catalogue_ones_passed_through_untouched():
    given = (
        _measured("canthal_tilt_asymmetry", 1.4),
        _measured("mouth_corner_asymmetry", 0.6),
        _measured("interpupillary_distance", 63.0),
    )
    left, right = comp.mirror_composites(_image(), POINTS, measurements=given)
    assert [m.spec_id for m in left.asymmetries] == [
        "canthal_tilt_asymmetry",
        "mouth_corner_asymmetry",
    ]
    assert left.asymmetries == right.asymmetries
    # Passed through, not recomputed: same objects, same intervals.
    assert left.asymmetries[0] is given[0]
    assert comp.ASYMMETRY_IDS and all(i.endswith("_asymmetry") for i in comp.ASYMMETRY_IDS)


def test_nothing_here_returns_an_aggregate_number():
    """Rule 1, checked where the temptation is strongest.

    A composite is the natural home for a symmetry percentage, so the object
    that carries one is asserted to expose no number at all beyond the counts
    on its axis. `composite` in this module means a picture assembled from two
    halves, in the photographic sense, and never a composite score.
    """
    left, _ = comp.mirror_composites(_image(), POINTS, measurements=())
    numbers = [
        name
        for name in dir(left)
        if not name.startswith("_")
        and isinstance(getattr(left, name), (int, float))
        and not isinstance(getattr(left, name), bool)
    ]
    assert numbers == []

    forbidden = ("score", "harmony", "attractiv", "beauty", "rating", "percentile", "index")
    names = [n for n in dir(comp) if not n.startswith("_")]
    assert [n for n in names if any(f in n.lower() for f in forbidden)] == []
    prose = (left.title + " " + left.caption).lower()
    assert [f for f in (*forbidden, "percent", "symmetry score") if f in prose] == []


def test_the_caption_says_what_the_picture_is_and_is_not():
    left, right = comp.mirror_composites(_image(), POINTS)
    assert "picture and not a measurement" in left.caption
    assert "not by the middle of the photograph" in left.caption
    assert left.side == "left" and right.side == "right"
    assert left.png.startswith(b"\x89PNG") and left.data_uri.startswith("data:image/png;base64,")
