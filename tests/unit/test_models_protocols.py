"""The backend boundary, and the left/right transposition that hides in it.

A transposed landmark map produces a perfectly plausible face. Every symmetric
measurement is unchanged, every asymmetry measurement has its sign flipped, and
nothing in the report looks wrong. So it gets its own set of tests, built on
three independent sources of evidence rather than on the map restating itself:

1. **SPIGA's own packaged reference shape.** ``mean_face_3D_98.txt`` ships
   inside the `spiga` wheel and is not derived from anything in this
   repository. Every index assignment is checked against it, including
   identities that do not involve sides at all -- the nose tip is the maximum-z
   point, the chin is the minimum-y point, the Cupid's bow peaks sit above the
   midline of the upper lip.
2. **The synthetic face in `conftest`**, authored by the core work in canonical
   millimetres, which the mapping round-trip must reproduce.
3. **Two independently trained models on the same rendered face**, which is the
   only evidence that can tell which physical eye a backend calls the right
   one. That test needs weights and skips without them.

Each check that could be satisfied by a map merely agreeing with itself is
paired with a control: the same assertion is run against a deliberately
transposed map and must fail. A test for a silent error is worth nothing
without evidence that it can fail.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vitruve.core.landmarks import BILATERAL_PAIRS, Landmark, PointSet
from vitruve.measure.evaluate import LandmarkUncertainty
from vitruve.models import dense_mediapipe as MP
from vitruve.models import landmark_spiga as SP
from vitruve.models import registry, weights
from vitruve.models.licensing import LicenseViolation, Provenance, Tier
from vitruve.models.protocols import (
    KEYPOINT_ORDER,
    DenseResult,
    FaceBox,
    HeadPose,
    LandmarkResult,
    canonicalise,
    heatmap_moments,
    scale_covariances,
    to_canonical,
    uncertainty_from_covariances,
)

TOY = Provenance(name="toy", tier=Tier.PERMISSIVE, license_id="MIT", source_url="https://x.invalid")

#: In SPIGA's packaged reference shape, negative x is the subject's **right**.
#:
#: Established by inference rather than by reading the axis convention off the
#: file, which states none: on a frontal portrait, reference index 96 (negative
#: x) landed at image x 364 and index 97 (positive x) at 450, and index 96 is
#: the same physical eye that YuNet's documented right-eye keypoint and
#: MediaPipe's right iris centre both land on to within two pixels. Three
#: independently trained models agreeing is the evidence; this constant records
#: the conclusion so the map can be checked without re-running any of them.
REFERENCE_RIGHT_IS_NEGATIVE_X = True

#: Detection threshold for the drawn stimulus below. It scores about 0.57,
#: under the default tuned for photographs, which is unsurprising for a face
#: made of five ellipses.
DRAWN_FACE_THRESHOLD = 0.4

#: Landmarks no frontal 2D model can honestly locate. Absence is the correct
#: behaviour and `evaluate` turns it into a named `Unavailable`, so a backend
#: that started supplying one of these would be approximating.
NEVER_FROM_FRONTAL_2D = (
    Landmark.TRICHION,
    Landmark.GONION_L,
    Landmark.GONION_R,
    Landmark.ZYGION_L,
    Landmark.ZYGION_R,
    Landmark.PORION_L,
    Landmark.PORION_R,
    Landmark.CERVICALE,
)

BACKEND_MAPS = {
    "spiga": (SP.WFLW_INDEX, SP.UNSUPPORTED, 98),
    "mediapipe": (MP.MESH_INDEX, MP.UNSUPPORTED, MP.N_POINTS),
}


def transposed(index_map: dict[Landmark, int]) -> dict[Landmark, int]:
    """The same map with every bilateral pair swapped: the bug under test."""
    out = dict(index_map)
    for left, right in BILATERAL_PAIRS:
        if left in out and right in out:
            out[left], out[right] = out[right], out[left]
    return out


def spiga_reference_shape() -> np.ndarray:
    """The 98-point canonical shape that ships inside the `spiga` wheel."""
    spec = importlib.util.find_spec("spiga")
    if spec is None or spec.origin is None:
        pytest.skip("spiga is not installed, so its reference shape is unavailable")
    path = Path(spec.origin).parent / "data" / "models3D" / "mean_face_3D_98.txt"
    if not path.exists():
        pytest.skip(f"{path} is missing from the installed spiga package")
    rows = [line.strip().split("|") for line in path.read_text().splitlines() if line.strip()]
    return np.array([[float(v) for v in row[1:]] for row in rows], dtype=float)


def weights_available() -> bool:
    try:
        return all(r.ok for r in weights.verify_all())
    except weights.WeightsError:  # pragma: no cover - depends on the checkout
        return False


needs_weights = pytest.mark.skipif(
    not weights_available(), reason="run `vitruve weights fetch` first"
)


def make_box(**over) -> FaceBox:
    base = dict(
        x=100.0,
        y=50.0,
        w=200.0,
        h=260.0,
        score=0.9,
        right_eye=np.array([160.0, 130.0]),
        left_eye=np.array([240.0, 130.0]),
        nose=np.array([200.0, 180.0]),
        right_mouth=np.array([170.0, 230.0]),
        left_mouth=np.array([230.0, 230.0]),
        provenance=TOY,
    )
    base.update(over)
    return FaceBox(**base)


# ---------------------------------------------------------------------------
# FaceBox
# ---------------------------------------------------------------------------


def test_keypoints_come_back_as_rows_in_the_declared_order():
    box = make_box()
    assert KEYPOINT_ORDER == ("eye_r", "eye_l", "nose", "mouth_r", "mouth_l")
    assert box.keypoints.shape == (5, 2)
    assert np.allclose(box.keypoints[0], box.right_eye)
    assert np.allclose(box.keypoints[1], box.left_eye)
    assert box.named_keypoints["eye_r"][0] < box.named_keypoints["eye_l"][0]


def test_bbox_and_xywh_are_the_same_four_numbers():
    box = make_box()
    assert box.bbox == box.xywh == (100.0, 50.0, 200.0, 260.0)
    assert box.xyxy == (100.0, 50.0, 300.0, 310.0)
    assert box.area == 52000.0


def test_expanding_a_box_keeps_its_centre():
    box = make_box()
    grown = box.expanded(1.5)
    assert np.allclose(grown.centre, box.centre)
    assert grown.w == pytest.approx(300.0)
    with pytest.raises(ValueError, match="must be positive"):
        box.expanded(0.0)


def test_a_degenerate_box_is_refused_at_construction():
    with pytest.raises(ValueError, match="positive extent"):
        make_box(w=0.0)
    with pytest.raises(ValueError, match="2-vector"):
        make_box(nose=np.zeros(3))


def test_looks_mirrored_catches_a_selfie_or_a_misread_keypoint_order():
    """Nothing downstream can see this from the landmarks alone."""
    assert not make_box().looks_mirrored
    swapped = make_box(right_eye=np.array([240.0, 130.0]), left_eye=np.array([160.0, 130.0]))
    assert swapped.looks_mirrored


def test_from_detection_accepts_a_foreign_structural_detection():
    """The pipeline is written against a protocol and its tests supply fakes."""

    class Foreign:
        bbox = (10.0, 20.0, 30.0, 40.0)
        score = 0.5
        keypoints = np.array([[12.0, 25.0], [30.0, 25.0], [21.0, 32.0], [15.0, 38.0], [27.0, 38.0]])

    box = FaceBox.from_detection(Foreign(), TOY)
    assert box.bbox == (10.0, 20.0, 30.0, 40.0)
    assert np.allclose(box.right_eye, [12.0, 25.0])
    assert FaceBox.from_detection(box, TOY) is box

    class Wrong(Foreign):
        keypoints = np.zeros((3, 2))

    with pytest.raises(ValueError, match=r"\(5, 2\)"):
        FaceBox.from_detection(Wrong(), TOY)


# ---------------------------------------------------------------------------
# The canonical frame
# ---------------------------------------------------------------------------


def test_the_canonical_frame_negates_both_axes():
    """+x is the subject's right, +y is up; image x runs right and y runs down.

    In an unmirrored frontal photograph the subject's right is at the *smaller*
    image x, so both axes turn over. Half a rotation, not a mirror.
    """
    origin = np.array([100.0, 100.0])
    # A point to the left of centre in the image is on the subject's right.
    assert to_canonical(np.array([80.0, 100.0]), origin_px=origin)[0] > 0
    assert to_canonical(np.array([120.0, 100.0]), origin_px=origin)[0] < 0
    # A point higher up the image has a smaller row index and positive y.
    assert to_canonical(np.array([100.0, 60.0]), origin_px=origin)[1] > 0
    assert to_canonical(np.array([100.0, 140.0]), origin_px=origin)[1] < 0


def test_the_canonical_transform_is_its_own_inverse_up_to_scale():
    origin = np.array([37.0, 91.0])
    pts = np.array([[10.0, 20.0], [300.0, 5.0]])
    there = to_canonical(pts, origin_px=origin)
    back = to_canonical(there, origin_px=np.zeros(2))
    assert np.allclose(back + origin, pts)


def test_to_canonical_rejects_three_dimensional_input():
    with pytest.raises(ValueError, match="2D points"):
        to_canonical(np.zeros((4, 3)), origin_px=np.zeros(2))


# ---------------------------------------------------------------------------
# Heatmap second moments
# ---------------------------------------------------------------------------


def gaussian_map(size: int, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    return np.exp(-0.5 * (((xs - cx) / sx) ** 2 + ((ys - cy) / sy) ** 2))


def test_second_moments_recover_a_known_gaussian():
    field = gaussian_map(41, 20.0, 24.0, 2.0, 2.0)[None]
    centres, covs = heatmap_moments(field, window=15)
    assert centres[0] == pytest.approx([20.0, 24.0], abs=0.05)
    # The window truncates the tails, so the recovered variance is a lower
    # bound. It should be within a fifth of the truth, not exact.
    assert covs[0][0, 0] == pytest.approx(4.0, rel=0.25)
    assert abs(covs[0][0, 1]) < 0.05


def test_second_moments_carry_anisotropy():
    """The whole reason for keeping a covariance rather than a scalar."""
    field = gaussian_map(61, 30.0, 30.0, 1.0, 5.0)[None]
    _centres, covs = heatmap_moments(field, window=21)
    vxx, vyy = covs[0][0, 0], covs[0][1, 1]
    assert vyy > 4 * vxx


def test_a_uniform_background_does_not_dominate_the_moment():
    """Sigmoid attention maps have a small response over the whole face.

    Without the window and the floor subtraction, the second moment would
    measure the width of the face and report the same covariance for every
    landmark.
    """
    field = gaussian_map(61, 30.0, 30.0, 1.5, 1.5) + 0.3
    _c, covs = heatmap_moments(field[None], window=11)
    assert covs[0][0, 0] < 4.0


def test_a_perfectly_sharp_peak_still_carries_the_pixel_quantisation_floor():
    """A zero covariance would make the Monte-Carlo draw degenerate."""
    field = np.zeros((1, 21, 21))
    field[0, 10, 10] = 1.0
    _c, covs = heatmap_moments(field, window=5)
    assert covs[0][0, 0] == pytest.approx(1.0 / 12.0)


def test_a_dead_channel_reports_window_sized_uncertainty_not_certainty():
    field = np.zeros((1, 21, 21))
    centres, covs = heatmap_moments(field, window=11)
    assert centres[0].tolist() == [0.0, 0.0]
    assert covs[0][0, 0] == 25.0


def test_heatmap_moments_validates_its_arguments():
    with pytest.raises(ValueError, match="n_landmarks, h, w"):
        heatmap_moments(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="odd integer"):
        heatmap_moments(np.zeros((1, 8, 8)), window=10)


def test_scaling_a_covariance_scales_the_cross_term_by_the_product():
    cov = np.array([[[4.0, 1.0], [1.0, 9.0]]])
    out = scale_covariances(cov, sx=2.0, sy=3.0)
    assert out[0][0, 0] == pytest.approx(16.0)
    assert out[0][1, 1] == pytest.approx(81.0)
    assert out[0][0, 1] == pytest.approx(6.0)
    assert out[0][0, 1] == out[0][1, 0]


def test_covariances_are_rekeyed_onto_the_named_points_not_the_backend_rows():
    """A covariance attached to the wrong landmark would be invisible."""
    backend_index = {Landmark.PUPIL_R: 7, Landmark.PUPIL_L: 3}
    raw = np.zeros((10, 2, 2))
    raw[7] = np.eye(2) * 4.0
    raw[3] = np.eye(2) * 9.0
    compact = {Landmark.PUPIL_R: 0, Landmark.PUPIL_L: 1}
    unc = uncertainty_from_covariances(compact, raw, backend_index)
    assert unc.covariances[0][0, 0] == 4.0
    assert unc.covariances[1][0, 0] == 9.0


# ---------------------------------------------------------------------------
# The landmark maps: structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND_MAPS))
def test_a_map_never_puts_two_landmarks_on_one_vertex(backend):
    """Two names for one observation would let a measurement count it twice."""
    index_map, _unsupported, n_rows = BACKEND_MAPS[backend]
    assert len(set(index_map.values())) == len(index_map)
    assert all(0 <= i < n_rows for i in index_map.values())


@pytest.mark.parametrize("backend", sorted(BACKEND_MAPS))
def test_every_landmark_is_either_mapped_or_declared_unsupported(backend):
    """A landmark that is neither is one somebody forgot about.

    The declared list is what lets the report say "this model cannot see the
    gonion" rather than leaving a gap the reader has to interpret.
    """
    index_map, unsupported, _n = BACKEND_MAPS[backend]
    mapped = set(index_map)
    declared = set(unsupported)
    assert not (mapped & declared), sorted(m.value for m in mapped & declared)
    assert mapped | declared == set(Landmark)


@pytest.mark.parametrize("backend", sorted(BACKEND_MAPS))
def test_landmarks_no_frontal_photograph_can_locate_stay_absent(backend):
    """Trichion needs a hairline, gonion and zygion need bone under skin, porion
    needs an ear canal. A silhouette is not a skeleton, and the standard move in
    this product category is to read bizygomatic width off the widest contour
    point, which measures hair."""
    index_map, unsupported, _n = BACKEND_MAPS[backend]
    for name in NEVER_FROM_FRONTAL_2D:
        assert name not in index_map, f"{backend} approximates {name.value}"
        assert name in unsupported


def test_mediapipe_maps_the_iris_centres_and_keeps_the_rings_apart():
    """The iris ring is the only permissively-licensed metric cue there is."""
    assert MP.MESH_INDEX[Landmark.PUPIL_R] == MP.IRIS_RIGHT[0]
    assert MP.MESH_INDEX[Landmark.PUPIL_L] == MP.IRIS_LEFT[0]
    assert not set(MP.IRIS_RIGHT) & set(MP.IRIS_LEFT)
    assert len(MP.IRIS_RIGHT) == len(MP.IRIS_LEFT) == 5


def test_spiga_does_not_emit_the_chin_under_two_names():
    """Menton and gnathion are the same pixel in a frontal silhouette."""
    assert Landmark.MENTON in SP.WFLW_INDEX
    assert Landmark.GNATHION not in SP.WFLW_INDEX
    assert Landmark.GNATHION in SP.UNSUPPORTED


# ---------------------------------------------------------------------------
# The landmark maps: checked against SPIGA's own packaged reference shape
# ---------------------------------------------------------------------------


def test_the_reference_shape_identifies_the_landmarks_the_map_claims():
    """Identity checks that involve no sides at all.

    If these hold, the map is reading the right rows; the side checks below then
    only have to settle left from right.
    """
    shape = spiga_reference_shape()
    assert shape.shape == (98, 3)
    x, y, z = shape[:, 0], shape[:, 1], shape[:, 2]

    # The nose tip is the origin and the closest point to the camera.
    assert SP.WFLW_INDEX[Landmark.PRONASALE] == int(np.argmax(z))
    # The chin is the lowest point on the face.
    assert SP.WFLW_INDEX[Landmark.MENTON] == int(np.argmin(y))
    # Midline landmarks are on the midline.
    for name in (Landmark.SELLION, Landmark.PRONASALE, Landmark.SUBNASALE,
                 Landmark.LABIALE_SUPERIUS, Landmark.LABIALE_INFERIUS, Landmark.MENTON):
        assert abs(x[SP.WFLW_INDEX[name]]) < 0.02, name.value
    # Vertical order down the face.
    order = [Landmark.SELLION, Landmark.PRONASALE, Landmark.SUBNASALE,
             Landmark.LABIALE_SUPERIUS, Landmark.LABIALE_INFERIUS, Landmark.MENTON]
    heights = [y[SP.WFLW_INDEX[n]] for n in order]
    assert heights == sorted(heights, reverse=True), list(zip(order, heights, strict=True))
    # The Cupid's bow: both peaks sit above the midline of the upper lip.
    for peak in (Landmark.CRISTA_PHILTRI_L, Landmark.CRISTA_PHILTRI_R):
        assert y[SP.WFLW_INDEX[peak]] > y[SP.WFLW_INDEX[Landmark.LABIALE_SUPERIUS]]
    # Eyelids straddle the pupil on each side.
    for sup, inf, pupil in (
        (Landmark.PALPEBRALE_SUP_L, Landmark.PALPEBRALE_INF_L, Landmark.PUPIL_L),
        (Landmark.PALPEBRALE_SUP_R, Landmark.PALPEBRALE_INF_R, Landmark.PUPIL_R),
    ):
        assert y[SP.WFLW_INDEX[sup]] > y[SP.WFLW_INDEX[pupil]] > y[SP.WFLW_INDEX[inf]]
        assert abs(x[SP.WFLW_INDEX[sup]] - x[SP.WFLW_INDEX[pupil]]) < 0.1
    # The outer canthus is further from the midline than the inner one.
    for outer, inner in (
        (Landmark.EXOCANTHION_L, Landmark.ENDOCANTHION_L),
        (Landmark.EXOCANTHION_R, Landmark.ENDOCANTHION_R),
    ):
        assert abs(x[SP.WFLW_INDEX[outer]]) > abs(x[SP.WFLW_INDEX[inner]])


def side_check(index_map: dict[Landmark, int], shape: np.ndarray) -> list[str]:
    """Bilateral pairs whose sides contradict the reference shape."""
    x = shape[:, 0]
    wrong = []
    for left, right in BILATERAL_PAIRS:
        if left not in index_map or right not in index_map:
            continue
        if not x[index_map[right]] < 0 < x[index_map[left]]:
            wrong.append(f"{right.value}/{left.value}")
    return wrong


def test_no_bilateral_pair_in_the_spiga_map_is_transposed():
    """The check that a transposition has to survive, against packaged data."""
    assert REFERENCE_RIGHT_IS_NEGATIVE_X
    shape = spiga_reference_shape()
    assert side_check(SP.WFLW_INDEX, shape) == []


def test_the_side_check_can_actually_fail():
    """Positive control. A test for a silent error is worth nothing without
    evidence that it detects the error."""
    shape = spiga_reference_shape()
    wrong = side_check(transposed(SP.WFLW_INDEX), shape)
    assert len(wrong) == 9, wrong


# ---------------------------------------------------------------------------
# The landmark maps: round trip through the synthetic face
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND_MAPS))
def test_mapping_a_synthetic_face_reproduces_its_canonical_coordinates(backend, face):
    """Scatter the `conftest` face into a backend's raw layout and read it back.

    This exercises the frame conversion rather than the index semantics: it is
    what catches a missing negation in `to_canonical`, a transposed x and y, or
    an index that landed outside the array. The side assignment itself is
    settled by the reference-shape test above and by the two-model test below,
    because a map that both writes and reads a raw array cancels its own
    transposition.
    """
    index_map, _unsupported, n_rows = BACKEND_MAPS[backend]
    centre = np.array([320.0, 400.0])
    px_per_mm = 3.0

    raw = np.full((n_rows, 2), np.nan)
    for name, row in index_map.items():
        canonical = face.get(name)[:2]
        raw[row] = centre - canonical * px_per_mm

    index = {name: i for i, name in enumerate(index_map)}
    coords = np.stack([raw[index_map[name]] for name in index], axis=0)
    assert np.isfinite(coords).all()

    recovered = canonicalise(
        PointSet(index=index, coords=coords), origin_px=centre, scale=1.0 / px_per_mm
    )
    for name in index:
        assert np.allclose(recovered.get(name), face.get(name)[:2], atol=1e-9), name.value

    for left, right in BILATERAL_PAIRS:
        if recovered.has(left, right):
            assert recovered.get(left)[0] < 0 < recovered.get(right)[0], (left.value, right.value)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def small_pointset() -> PointSet:
    return PointSet.from_mapping(
        {Landmark.PUPIL_R: np.array([10.0, 10.0]), Landmark.PUPIL_L: np.array([30.0, 10.0])}
    )


def test_a_landmark_result_refuses_a_point_without_a_covariance():
    ps = small_pointset()
    partial = LandmarkUncertainty(index={Landmark.PUPIL_R: 0}, covariances=np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="pupil_l"):
        LandmarkResult(points=ps, uncertainty=partial, provenance=TOY)


def test_a_landmark_result_exposes_covariances_under_the_name_the_pipeline_reads():
    ps = small_pointset()
    unc = LandmarkUncertainty.isotropic(ps, sd=2.0)
    res = LandmarkResult(points=ps, uncertainty=unc, provenance=TOY)
    assert res.covariances.shape == (2, 2, 2)
    assert res.covariances is unc.covariances
    assert not res.has_pose
    assert LandmarkResult(points=ps, uncertainty=unc, provenance=TOY, yaw_deg=0.0).has_pose


def test_a_landmark_result_refuses_three_dimensional_points():
    ps = PointSet.from_mapping({Landmark.PUPIL_R: np.zeros(3)})
    unc = LandmarkUncertainty.isotropic(ps, sd=1.0)
    with pytest.raises(ValueError, match="2D points"):
        LandmarkResult(points=ps, uncertainty=unc, provenance=TOY)


def iris_ring(cx: float, cy: float, r: float) -> np.ndarray:
    return np.array([[cx, cy], [cx + r, cy], [cx, cy - r], [cx - r, cy], [cx, cy + r]])


def make_dense(**over) -> DenseResult:
    base = dict(
        points=small_pointset(),
        all_points_px=np.zeros((478, 2)),
        iris_right_px=iris_ring(100.0, 100.0, 6.0),
        iris_left_px=iris_ring(180.0, 100.0, 7.0),
        transform=np.eye(4),
        provenance=TOY,
    )
    base.update(over)
    return DenseResult(**base)


def test_the_iris_diameter_is_horizontal_not_averaged():
    """The 11.84 mm prior is a horizontal corneal diameter, and the eyelids
    occlude the iris top and bottom in most photographs."""
    tall = iris_ring(100.0, 100.0, 6.0)
    tall[2] = [100.0, 80.0]  # a lid pushing the top point far out
    tall[4] = [100.0, 120.0]
    dense = make_dense(iris_right_px=tall)
    assert dense.iris_diameter_px[0] == pytest.approx(12.0)
    assert dense.iris_diameter_px_r == pytest.approx(12.0)
    assert dense.iris_diameter_px_l == pytest.approx(14.0)


def test_a_dense_result_insists_on_the_iris_points():
    with pytest.raises(ValueError, match="478 dense points"):
        make_dense(all_points_px=np.zeros((468, 2)))
    with pytest.raises(ValueError, match="centre plus four ring points"):
        make_dense(iris_left_px=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="4x4"):
        make_dense(transform=np.eye(3))


def test_head_pose_reports_the_largest_per_axis_disagreement():
    """Two estimators disagreeing by more than their scatter is the signal."""
    a = HeadPose(yaw_deg=2.0, pitch_deg=-4.0, roll_deg=0.5, sd_deg=5.0, provenance=TOY)
    b = HeadPose(yaw_deg=1.0, pitch_deg=9.0, roll_deg=0.0, sd_deg=5.0, provenance=TOY)
    assert a.disagreement(b) == pytest.approx(13.0)
    assert a.max_abs_deg == pytest.approx(4.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        HeadPose(0.0, 0.0, 0.0, -1.0, TOY)


# ---------------------------------------------------------------------------
# The registry and the licence tier
# ---------------------------------------------------------------------------


def test_the_documented_import_works():
    from vitruve.models.registry import build_detector  # noqa: F401


def test_a_copyleft_backend_is_refused_at_the_permissive_tier():
    with pytest.raises(LicenseViolation, match=r"AGPL-3\.0"):
        registry.build_detector("yolo_face", tier=Tier.PERMISSIVE)


def test_the_refusal_happens_before_anything_is_imported(monkeypatch):
    """A check that runs after loading has already let the obligation attach."""
    entry = registry.DETECTORS["yolo_face"]
    monkeypatch.setattr(
        type(entry),
        "factory",
        lambda self: (_ for _ in ()).throw(AssertionError("factory was reached")),
    )
    with pytest.raises(LicenseViolation):
        registry.build_detector("yolo_face", tier=Tier.PERMISSIVE)


def test_raising_the_tier_gets_past_the_licence_and_stops_at_the_missing_backend():
    with pytest.raises(registry.BackendUnavailable, match="Ultralytics-lineage"):
        registry.build_detector("yolo_face", tier=Tier.COPYLEFT)


def test_an_unlicensed_backend_is_refused_at_every_tier():
    for tier in (Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL):
        with pytest.raises(LicenseViolation):
            registry.build_landmarker("star", tier=tier)


def test_an_unknown_name_lists_the_registered_ones():
    with pytest.raises(registry.UnknownBackend, match="spiga"):
        registry.build_landmarker("nonexistent")


def test_available_reflects_the_tier():
    permissive = {e.name for e in registry.available(Tier.PERMISSIVE)}
    assert permissive == {"yunet", "spiga", "mediapipe", "sixdrepnet"}
    assert "threeddfa_v2" not in {e.name for e in registry.available(Tier.NONCOMMERCIAL)}


def test_the_selection_manifest_records_the_obligations_the_tier_carries():
    manifest = registry.selection_manifest()
    assert manifest["detector"] == "yunet"
    assert manifest["landmarker"] == "spiga"
    assert "Basel Face Model" in manifest["obligations"]
    assert "BSD-3-Clause" in manifest["landmarker_provenance"]


def test_describe_says_why_each_backend_is_or_is_not_usable():
    lines = "\n".join(registry.describe(Tier.PERMISSIVE))
    assert "detector/yunet: available" in lines
    assert "detector/yolo_face: refused at tier permissive" in lines
    assert "landmarker/star: refused" in lines


# ---------------------------------------------------------------------------
# Two models on one face: the only evidence about which eye is which
# ---------------------------------------------------------------------------


def rendered_face(size: int = 512) -> np.ndarray:
    """A crude RGB face, drawn rather than photographed.

    Good enough for both detectors to fire, which is all this test needs: the
    question is which *side* each backend calls right, and that is answered by
    where the eyes are in the frame, not by how realistic they look.
    """
    cv2 = pytest.importorskip("cv2")
    img = np.full((size, size, 3), 210, np.uint8)
    cx, cy = size // 2, size // 2
    cv2.ellipse(img, (cx, cy), (130, 170), 0, 0, 360, (150, 168, 196), -1)
    for side in (-1, 1):
        eye = (cx + side * 55, cy - 40)
        cv2.ellipse(img, eye, (30, 16), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(img, eye, 12, (30, 45, 70), -1)
        cv2.circle(img, eye, 5, (10, 10, 10), -1)
        cv2.ellipse(img, (cx + side * 55, cy - 70), (34, 12), 0, 180, 360, (30, 40, 60), 6)
    cv2.ellipse(img, (cx, cy + 30), (22, 40), 0, 0, 360, (132, 150, 178), -1)
    cv2.ellipse(img, (cx, cy + 95), (48, 20), 0, 0, 360, (90, 90, 150), -1)
    cv2.line(img, (cx - 48, cy + 95), (cx + 48, cy + 95), (60, 60, 110), 2)
    return cv2.GaussianBlur(img, (5, 5), 0)


@needs_weights
def test_three_models_agree_on_which_eye_is_the_subjects_right():
    """The transposition test that no amount of internal consistency can give.

    YuNet's keypoint order, SPIGA's WFLW indices and MediaPipe's mesh indices
    were all fixed by their authors, independently. If any one of the three maps
    in this repository had left and right the wrong way round, the three would
    disagree about which physical eye they are looking at.
    """
    image = rendered_face()
    # A drawn face scores around 0.57, below the threshold tuned for
    # photographs. Lowering it here is a property of the stimulus, not of the
    # detector, and the assertions below do not depend on the score.
    detector = registry.build_detector(score_threshold=DRAWN_FACE_THRESHOLD)
    box = detector.detect_largest(image)
    assert box is not None, "the synthetic face was not detected at all"
    assert not box.looks_mirrored

    landmarker = registry.build_landmarker()
    located = landmarker.locate(image, box)
    dense = registry.build_dense()
    meshed = dense.locate(image, box)

    for result in (located.points, meshed.points):
        right = result.get(Landmark.PUPIL_R)
        left = result.get(Landmark.PUPIL_L)
        assert right[0] < left[0], "subject's right eye must sit at the smaller image x"
        # And each agrees with the detector about which eye that is.
        assert abs(right[0] - box.right_eye[0]) < abs(right[0] - box.left_eye[0])

    for left, right in BILATERAL_PAIRS:
        for result in (located.points, meshed.points):
            if result.has(left, right):
                assert result.get(right)[0] < result.get(left)[0], (left.value, right.value)


@needs_weights
def test_the_two_landmarkers_land_on_the_same_points():
    """A cross-model check on the maps as a whole, not only on the sides.

    Disagreement here means one of the two index tables is reading a different
    anatomical point under the same name, which no single-model test can see.
    """
    image = rendered_face()
    box = registry.build_detector(score_threshold=DRAWN_FACE_THRESHOLD).detect_largest(image)
    located = registry.build_landmarker().locate(image, box)
    meshed = registry.build_dense().locate(image, box)

    shared = [n for n in located.points.index if meshed.points.has(n)]
    assert len(shared) >= 20
    face_height = box.h
    distances = {
        n.value: float(np.linalg.norm(located.points.get(n) - meshed.points.get(n)))
        for n in shared
    }
    worst = max(distances.items(), key=lambda kv: kv[1])
    assert np.median(list(distances.values())) < 0.05 * face_height, distances
    assert worst[1] < 0.12 * face_height, worst


@needs_weights
def test_the_landmarker_returns_covariances_that_can_be_sampled():
    """The uncertainty has to be usable by `evaluate`, not merely present.

    A singular or non-positive-definite covariance fails inside the Cholesky
    draw, several stages later, with an error that names neither the landmark
    nor the backend.
    """
    image = rendered_face()
    box = registry.build_detector(score_threshold=DRAWN_FACE_THRESHOLD).detect_largest(image)
    located = registry.build_landmarker().locate(image, box)

    covs = located.covariances
    assert covs.shape == (len(located.points.index), 2, 2)
    eigenvalues = np.linalg.eigvalsh(covs)
    assert (eigenvalues > 0).all(), "a covariance with a zero eigenvalue cannot be sampled"

    drawn = located.uncertainty.sample(located.points, 64, np.random.default_rng(0))
    assert drawn.coords.shape == (64, len(located.points.index), 2)
    assert np.isfinite(drawn.coords).all()


@needs_weights
def test_the_dense_backend_measures_both_irides():
    image = rendered_face()
    box = registry.build_detector(score_threshold=DRAWN_FACE_THRESHOLD).detect_largest(image)
    meshed = registry.build_dense().locate(image, box)
    right, left = meshed.iris_diameter_px
    assert right > 0 and left > 0
    # The drawn face is symmetric, so the two should be within a few percent.
    assert abs(right - left) / max(right, left) < 0.1

    estimate = MP.scale_from_dense(meshed)
    assert estimate.mm_per_px > 0
    # Fusing two correlated cues must not claim the independence gain.
    from vitruve.core.scale import IRIS_DIAMETER_MM, IRIS_DIAMETER_SD

    single = IRIS_DIAMETER_SD / IRIS_DIAMETER_MM
    assert single / np.sqrt(2) < estimate.relative_sd < single
