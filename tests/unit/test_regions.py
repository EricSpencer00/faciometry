"""Regions are geometry, so they are tested against geometry.

Every assertion here is either a closed-form property of the construction
(rotating the face rotates the regions and nothing else; a square polygon has
the area of a square) or a statement about what happens when a landmark is
missing. Nothing here needs weights, an image, or a network.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from faciometry.core.landmarks import Landmark as L
from faciometry.core.landmarks import PointSet
from faciometry.derm.regions import (
    REFERENCE_PAIRS,
    Region,
    RegionPolygon,
    RegionUnavailable,
    build_frame,
    build_regions,
    rasterize,
    refine_with_parser,
    skin_mask_from_parse,
)
from faciometry.models.licensing import LicenseViolation, Tier

IPD = 100.0
CX, CY = 400.0, 500.0


def face_points(rotate_deg: float = 0.0, scale: float = 1.0) -> PointSet:
    """A synthetic frontal face in the canonical frame: +x subject right, +y up."""
    raw = {
        L.PUPIL_L: (-50, 0),
        L.PUPIL_R: (50, 0),
        L.ENDOCANTHION_L: (-25, 0),
        L.ENDOCANTHION_R: (25, 0),
        L.EXOCANTHION_L: (-72, 2),
        L.EXOCANTHION_R: (72, 2),
        L.PALPEBRALE_SUP_L: (-50, 14),
        L.PALPEBRALE_SUP_R: (50, 14),
        L.PALPEBRALE_INF_L: (-50, -12),
        L.PALPEBRALE_INF_R: (50, -12),
        L.SUPERCILIARE_L: (-48, 40),
        L.SUPERCILIARE_R: (48, 40),
        L.GLABELLA: (0, 42),
        L.NASION: (0, 30),
        L.SELLION: (0, 22),
        L.TRICHION: (0, 165),
        L.ALARE_L: (-19, -78),
        L.ALARE_R: (19, -78),
        L.SUBNASALE: (0, -88),
        L.PRONASALE: (0, -70),
        L.LABIALE_SUPERIUS: (0, -118),
        L.LABIALE_INFERIUS: (0, -148),
        L.STOMION: (0, -132),
        L.CHEILION_L: (-33, -130),
        L.CHEILION_R: (33, -130),
        L.SUBLABIALE: (0, -165),
        L.POGONION: (0, -195),
        L.MENTON: (0, -215),
    }
    t = math.radians(rotate_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    origin = np.array([CX, CY])
    return PointSet.from_mapping(
        {k: origin + scale * (rot @ np.array(v, dtype=float)) for k, v in raw.items()}
    )


# ---------------------------------------------------------------------------
# Polygon primitives
# ---------------------------------------------------------------------------


def test_polygon_area_matches_closed_form():
    square = RegionPolygon(Region.MALAR_L, np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float))
    assert square.area == pytest.approx(100.0)
    assert square.centroid == pytest.approx(np.array([5.0, 5.0]))


def test_polygon_area_subtracts_holes():
    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    hole = np.array([[4, 4], [6, 4], [6, 6], [4, 6]], float)
    poly = RegionPolygon(Region.PERIORAL, outer, holes=(hole,))
    assert poly.area == pytest.approx(96.0)


def test_contains_is_even_odd_with_holes():
    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    hole = np.array([[4, 4], [6, 4], [6, 6], [4, 6]], float)
    poly = RegionPolygon(Region.PERIORAL, outer, holes=(hole,))
    pts = np.array([[1.0, 1.0], [5.0, 5.0], [-1.0, 5.0], [9.5, 9.5]])
    assert list(poly.contains(pts)) == [True, False, False, True]


def test_rasterize_respects_the_y_up_flag():
    # A polygon occupying the top of the canonical frame (high y) must land in
    # the *low* row indices of the array, and the flag flips exactly that.
    poly = RegionPolygon(Region.FOREHEAD, np.array([[2, 12], [8, 12], [8, 18], [2, 18]], float))
    up = rasterize(poly, 20, 20, y_up=True)
    down = rasterize(poly, 20, 20, y_up=False)
    assert up.any() and down.any()
    assert np.nonzero(up)[0].max() < 10
    assert np.nonzero(down)[0].min() >= 10
    assert up.sum() == down.sum()


def test_rasterized_area_converges_to_polygon_area():
    poly = RegionPolygon(Region.MALAR_L, np.array([[10, 10], [70, 10], [70, 50], [10, 50]], float))
    mask = rasterize(poly, 100, 100)
    assert mask.sum() == pytest.approx(poly.area, rel=0.05)


def test_erosion_shrinks_the_ring_and_grows_the_hole():
    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    hole = np.array([[4, 4], [6, 4], [6, 6], [4, 6]], float)
    poly = RegionPolygon(Region.PERIORAL, outer, holes=(hole,))
    small = poly.eroded(0.2)
    assert abs(_shoelace_area(small.vertices)) < abs(_shoelace_area(poly.vertices))
    assert abs(_shoelace_area(small.holes[0])) > abs(_shoelace_area(hole))
    # The hole grows about its own centre, so it does not slide off the lips.
    assert small.holes[0].mean(axis=0) == pytest.approx(hole.mean(axis=0))


def _shoelace_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_erosion_fraction_is_bounded():
    poly = RegionPolygon(Region.MALAR_L, np.array([[0, 0], [1, 0], [1, 1]], float))
    with pytest.raises(ValueError):
        poly.eroded(1.0)


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


def test_frame_axes_are_orthonormal_and_right_is_subject_right():
    frame = build_frame(face_points())
    assert np.dot(frame.u, frame.v) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(frame.u) == pytest.approx(1.0)
    assert frame.span == pytest.approx(IPD)
    # u points from the subject's left pupil to the right one, which is +x.
    assert frame.u[0] > 0.9
    # v is +90 degrees from u, which in a +y-up frame is up.
    assert frame.v[1] > 0.9


def test_frame_falls_back_to_outer_canthi_and_rescales():
    pts = face_points()
    reduced = PointSet.from_mapping(
        {k: pts.get(k) for k in pts.available if k not in (L.PUPIL_L, L.PUPIL_R)}
    )
    frame = build_frame(reduced)
    assert "outer-canthal" in frame.span_source
    # 144 px outer canthal distance divided by the 91/63.4 population ratio.
    assert frame.span == pytest.approx(144.0 * 63.4 / 91.0, rel=1e-9)


def test_frame_refuses_a_monte_carlo_ensemble():
    pts = face_points()
    batched = PointSet(index=pts.index, coords=np.stack([pts.coords] * 4))
    with pytest.raises(ValueError, match="ensemble"):
        build_frame(batched)


def test_frame_round_trips_local_and_global():
    frame = build_frame(face_points(rotate_deg=17.0))
    uv = np.array([[0.3, -0.4], [-1.0, 0.9]])
    assert frame.to_local(frame.to_global(uv)) == pytest.approx(uv)


# ---------------------------------------------------------------------------
# Region construction
# ---------------------------------------------------------------------------


def test_every_region_builds_from_a_complete_face():
    regions = build_regions(face_points())
    assert regions.available == frozenset(Region)
    assert regions.unavailable == {}
    assert len(regions) == len(Region)


def test_regions_are_where_anatomy_says_they_are():
    regions = build_regions(face_points())
    # Regions live in the canonical frame, so higher y is higher on the face.
    y = {r: regions.get(r).centroid[1] for r in regions.available}
    assert y[Region.FOREHEAD] > y[Region.GLABELLA] > y[Region.PERIORBITAL_L]
    assert y[Region.PERIORBITAL_L] > y[Region.INFRAORBITAL_L] > y[Region.MALAR_L]
    assert y[Region.MALAR_L] > y[Region.PERIORAL]
    x = {r: regions.get(r).centroid[0] for r in regions.available}
    # Subject-left regions sit at negative x relative to the midline at CX.
    assert x[Region.MALAR_L] < CX < x[Region.MALAR_R]
    assert x[Region.LATERAL_CHEEK_L] < x[Region.MALAR_L]
    assert x[Region.MALAR_R] < x[Region.LATERAL_CHEEK_R]
    for midline in (Region.GLABELLA, Region.NASAL, Region.T_ZONE, Region.PERIORAL, Region.FOREHEAD):
        assert x[midline] == pytest.approx(CX, abs=1.0)


def test_regions_are_mirror_symmetric_on_a_symmetric_face():
    regions = build_regions(face_points())
    for left, right in (
        (Region.MALAR_L, Region.MALAR_R),
        (Region.INFRAORBITAL_L, Region.INFRAORBITAL_R),
        (Region.PERIORBITAL_L, Region.PERIORBITAL_R),
        (Region.LATERAL_CHEEK_L, Region.LATERAL_CHEEK_R),
    ):
        assert regions.get(left).area == pytest.approx(regions.get(right).area, rel=1e-9)
        cl, cr = regions.get(left).centroid, regions.get(right).centroid
        assert (CX - cl[0]) == pytest.approx(cr[0] - CX, abs=1e-9)
        assert cl[1] == pytest.approx(cr[1], abs=1e-9)


def test_construction_is_roll_invariant():
    """A tilted head must give the same regions, rotated -- not the same rectangles."""
    upright = build_regions(face_points())
    tilted = build_regions(face_points(rotate_deg=20.0))
    for region in Region:
        a = upright.get(region)
        b = tilted.get(region)
        assert a.area == pytest.approx(b.area, rel=1e-9)
        # Both expressed in their own frame must be identical.
        assert upright.frame.to_local(a.vertices) == pytest.approx(
            tilted.frame.to_local(b.vertices), abs=1e-9
        )


def test_construction_is_scale_invariant():
    small = build_regions(face_points(scale=1.0))
    large = build_regions(face_points(scale=3.0))
    ratio = large.get(Region.MALAR_L).area / small.get(Region.MALAR_L).area
    assert ratio == pytest.approx(9.0, rel=1e-9)


def test_a_missing_landmark_names_itself_rather_than_vanishing():
    pts = face_points()
    reduced = PointSet.from_mapping(
        {k: pts.get(k) for k in pts.available if k is not L.CHEILION_L}
    )
    regions = build_regions(reduced)
    assert Region.MALAR_L not in regions
    assert regions.unavailable[Region.MALAR_L] == ("cheilion_l",)
    # The contralateral region is unaffected: one missing point is not a total loss.
    assert Region.MALAR_R in regions
    with pytest.raises(RegionUnavailable):
        regions.get(Region.MALAR_L)


def test_derived_anchors_fall_back_before_giving_up():
    pts = face_points()
    # No glabella landmark: the anchor falls back to the brow midpoint.
    reduced = PointSet.from_mapping(
        {k: pts.get(k) for k in pts.available if k is not L.GLABELLA}
    )
    regions = build_regions(reduced)
    assert Region.GLABELLA in regions
    # Now remove the brows too, and the anchor has nothing left.
    stripped = PointSet.from_mapping(
        {
            k: pts.get(k)
            for k in pts.available
            if k not in (L.GLABELLA, L.SUPERCILIARE_L, L.SUPERCILIARE_R)
        }
    )
    regions = build_regions(stripped)
    assert Region.GLABELLA not in regions
    assert "superciliare_l" in regions.unavailable[Region.GLABELLA]


def test_only_filter_builds_a_subset():
    regions = build_regions(face_points(), only=[Region.MALAR_L, Region.MALAR_R])
    assert regions.available == {Region.MALAR_L, Region.MALAR_R}


def test_perioral_excludes_the_vermilion():
    regions = build_regions(face_points())
    poly = regions.get(Region.PERIORAL)
    assert poly.holes, "the perioral region must cut out the lips"
    lips_centre = np.array([CX, CY - 132.0])  # stomion
    assert not bool(poly.contains(lips_centre))
    # A point just outside the commissures is still perioral skin.
    assert bool(poly.contains(np.array([CX - 40.0, CY - 130.0])))


def test_reference_pairs_resolve_on_a_complete_face():
    regions = build_regions(face_points())
    for target, reference in REFERENCE_PAIRS.items():
        assert regions.pair_for(target) is reference
    assert regions.pair_for(Region.T_ZONE) is None


def test_mask_for_erodes_and_stays_inside_the_image():
    regions = build_regions(face_points())
    tight = regions.mask_for(Region.MALAR_R, 900, 800, erode=0.0)
    eroded = regions.mask_for(Region.MALAR_R, 900, 800, erode=0.2)
    assert 0 < eroded.sum() < tight.sum()
    assert eroded.shape == (900, 800)


def test_regions_note_how_the_span_was_recovered():
    regions = build_regions(face_points())
    assert any("interpupillary" in n for n in regions.notes)
    assert any("not segmented edges" in n for n in regions.notes)


# ---------------------------------------------------------------------------
# Parser refinement, and its licence
# ---------------------------------------------------------------------------


def test_skin_mask_from_parse_keeps_skin_and_nose_only():
    parse = np.array([[0, 1, 2], [10, 17, 1]], dtype=int)
    mask = skin_mask_from_parse(parse)
    assert mask.tolist() == [[False, True, False], [True, False, True]]


def test_parser_refinement_is_refused_at_the_default_tier():
    regions = build_regions(face_points())
    parse = np.ones((900, 800), dtype=int)
    with pytest.raises(LicenseViolation, match="noncommercial"):
        refine_with_parser(regions, parse, allowed_tier=Tier.PERMISSIVE)
    with pytest.raises(LicenseViolation):
        refine_with_parser(regions, parse, allowed_tier=Tier.COPYLEFT)


def test_parser_refinement_intersects_and_records_the_obligation():
    regions = build_regions(face_points())
    parse = np.ones((900, 800), dtype=int)
    parse[:, :400] = 17  # hair, on the subject's left half of the image
    refined = refine_with_parser(regions, parse, allowed_tier=Tier.NONCOMMERCIAL)
    assert refined.skin_mask is not None
    assert refined.skin_mask_provenance is not None
    assert any("not commercially redistributable" in n for n in refined.notes)
    assert refined.mask_for(Region.MALAR_L, 900, 800).sum() == 0
    assert refined.mask_for(Region.MALAR_R, 900, 800).sum() > 0
    # The geometry is untouched; only the sampling mask changed.
    assert refined.get(Region.MALAR_L).area == pytest.approx(
        regions.get(Region.MALAR_L).area
    )


def test_refined_set_rejects_a_mismatched_image_size():
    regions = build_regions(face_points())
    refined = refine_with_parser(
        regions, np.ones((100, 100), dtype=int), allowed_tier=Tier.NONCOMMERCIAL
    )
    with pytest.raises(ValueError, match="skin mask"):
        refined.mask_for(Region.MALAR_R, 900, 800)
