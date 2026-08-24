"""The whole pipeline, end to end, against a face whose true geometry is known.

There is no photograph here and no model. The subject is a table of
three-dimensional landmark coordinates in millimetres; a pinhole camera projects
them; the projection is drawn into an image; and fake backends stand in for the
detector, the landmarker, the head-pose network and the iris measurer. That
buys three things a test against a real photograph cannot.

**Ground truth exists.** Every expected value below is computed from the same
coordinate table by evaluating the catalogue's own formula on it, so the test
compares the pipeline against the geometry rather than against a constant
somebody typed once and never revisited.

**The experiment can be controlled.** Rotating the head by exactly ten degrees
and re-running is a two-line change, which makes the pose-sensitivity prediction
in ``core/sensitivity`` directly falsifiable: a transverse width should shrink by
``1 - cos(10 deg)`` and a ratio of two transverse widths should barely move.

**It runs anywhere.** No weights, no network, no GPU, a couple of seconds. The
acceptance criterion for this project is that the measurement layer is testable
without any of the machinery, and this file is where that claim is cashed.

The residual error the tolerances allow is perspective. The camera sits two
metres away, so a landmark twenty-five millimetres in front of the eye plane is
magnified by about one percent relative to one in it, exactly as
``scale.magnification_distortion`` says. That is a real property of photography,
not a defect of the pipeline, and the tolerances are sized to it.
"""

from __future__ import annotations

import json
import math
import socket
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest
from PIL import ExifTags
from PIL import Image as PILImage

from vitruve.core.geometry import rotation_matrix
from vitruve.core.landmarks import Landmark as L
from vitruve.core.landmarks import PointSet
from vitruve.core.scale import ScaleSource
from vitruve.core.sensitivity import POSE_ESTIMATOR_SD_DEG
from vitruve.core.spec import Reportability, Unit, View
from vitruve.measure.evaluate import Unavailable
from vitruve.measure.registry import BY_ID, for_view
from vitruve.models import licensing
from vitruve.models.licensing import Tier
from vitruve.pipeline.ingest import Ruler, pixel_sha256, save_stripped
from vitruve.pipeline.manifest import RunManifest
from vitruve.pipeline.ports import Backends
from vitruve.pipeline.run import analyze

# ---------------------------------------------------------------------------
# The subject: a face, in millimetres, in the canonical frame.
# +x is the subject's right, +y is up, +z is anterior.
# ---------------------------------------------------------------------------

#: The chin is deliberately given a normal degree of retrusion rather than a
#: straight profile. An angle computed as ``arctan2(|cross|, dot)`` is folded at
#: 180 degrees, so a nearly straight profile has a median biased downward by the
#: landmark noise -- the same mechanism that gives a symmetric face a positive
#: asymmetry. Keeping the convexity angle near 169 degrees puts the reference
#: values away from that boundary, where the tolerances below mean what they say.
FACE_MM: dict[L, tuple[float, float, float]] = {
    L.GLABELLA: (0.0, 25.0, 18.0),
    L.NASION: (0.0, 14.0, 10.0),
    L.SELLION: (0.0, 10.0, 6.0),
    L.PUPIL_R: (31.0, 0.0, 0.0),
    L.PUPIL_L: (-31.0, 0.0, 0.0),
    L.ENDOCANTHION_R: (16.0, 1.0, -4.0),
    L.ENDOCANTHION_L: (-16.0, 1.0, -4.0),
    L.EXOCANTHION_R: (46.0, 3.6, -10.0),
    L.EXOCANTHION_L: (-46.0, 3.6, -10.0),
    L.PALPEBRALE_SUP_R: (31.0, 6.0, -2.0),
    L.PALPEBRALE_SUP_L: (-31.0, 6.0, -2.0),
    L.PALPEBRALE_INF_R: (31.0, -4.0, -2.0),
    L.PALPEBRALE_INF_L: (-31.0, -4.0, -2.0),
    L.ZYGION_R: (68.0, -2.0, -35.0),
    L.ZYGION_L: (-68.0, -2.0, -35.0),
    L.TRAGION_R: (72.0, 2.0, -75.0),
    L.TRAGION_L: (-72.0, 2.0, -75.0),
    L.ALARE_R: (17.0, -32.0, 12.0),
    L.ALARE_L: (-17.0, -32.0, 12.0),
    L.SUBNASALE: (0.0, -40.0, 15.0),
    L.PRONASALE: (0.0, -30.0, 30.0),
    L.COLUMELLA: (0.0, -34.0, 24.0),
    L.LABIALE_SUPERIUS: (0.0, -53.0, 14.0),
    L.STOMION: (0.0, -57.0, 13.0),
    L.LABIALE_INFERIUS: (0.0, -62.0, 13.0),
    L.CHEILION_R: (25.0, -57.0, 5.0),
    L.CHEILION_L: (-25.0, -57.0, 5.0),
    L.SUBLABIALE: (0.0, -70.0, 2.0),
    L.POGONION: (0.0, -85.0, 4.0),
    L.GNATHION: (0.0, -93.0, 0.0),
    L.MENTON: (0.0, -95.0, -4.0),
    L.GONION_R: (55.0, -70.0, -35.0),
    L.GONION_L: (-55.0, -70.0, -35.0),
    L.CERVICALE: (0.0, -100.0, -50.0),
}

#: The landmark model in this fixture emits no philtral crests, which is true of
#: several real 68-point backends. It is here so that "unavailable" has
#: something to be.
OMITTED = (L.CRISTA_PHILTRI_L, L.CRISTA_PHILTRI_R)

#: At ninety degrees of yaw the subject's right side is turned away from the
#: camera, so a real landmarker sees only the left one. Supplying both would let
#: a bilateral measurement succeed on a landmark that is behind the head.
PROFILE_HIDDEN = (
    L.PUPIL_R,
    L.ENDOCANTHION_R,
    L.EXOCANTHION_R,
    L.PALPEBRALE_SUP_R,
    L.PALPEBRALE_INF_R,
    L.ZYGION_R,
    L.TRAGION_R,
    L.ALARE_R,
    L.CHEILION_R,
    L.GONION_R,
)

IMAGE_W, IMAGE_H = 1600, 2000
IMAGE_DIAGONAL_PX = math.hypot(IMAGE_W, IMAGE_H)
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)
FOCAL_35MM = 85.0
FOCAL_PX = FOCAL_35MM / (FULL_FRAME_DIAGONAL_MM / IMAGE_DIAGONAL_PX)
SUBJECT_DISTANCE_MM = 2000.0

#: A 100 mm reference held in the eye plane. Using an external ruler rather than
#: the subject's own interpupillary distance matters for the pose arm: a scale
#: derived from the pupils absorbs the very cosine that yaw imposes on every
#: transverse width, and the effect under test would cancel itself out.
RULER = Ruler(known_mm=100.0, pixel_span=100.0 * FOCAL_PX / SUBJECT_DISTANCE_MM)

LANDMARK_SD_PX = 1.2

#: Perspective at two metres, plus Monte-Carlo jitter on the median. Sized from
#: the geometry rather than tuned until green.
LENGTH_TOLERANCE = 0.02
RATIO_TOLERANCE = 0.02
ANGLE_TOLERANCE_DEG = 0.6


def project(
    landmarks: dict[L, tuple[float, float, float]],
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> tuple[list[L], np.ndarray]:
    """Pinhole projection of the rotated head into image pixel coordinates.

    The camera sits on the ``+z`` axis looking back at the origin, so the
    viewer's right is the subject's left and image ``y`` runs downward. Both are
    the conventions the canonical stage has to undo, which is the point of
    projecting here rather than fabricating image coordinates directly.
    """
    names = list(landmarks)
    world = np.array([landmarks[n] for n in names], dtype=float)
    world = world @ rotation_matrix(yaw, pitch, roll).T
    x_cam = -world[:, 0]
    y_cam = world[:, 1]
    z_cam = SUBJECT_DISTANCE_MM - world[:, 2]
    u = IMAGE_W / 2 + FOCAL_PX * x_cam / z_cam
    v = IMAGE_H / 2 - FOCAL_PX * y_cam / z_cam
    return names, np.stack([u, v], axis=-1)


def render(uv: np.ndarray, *, seed: int = 0, blur_radius: int = 0) -> np.ndarray:
    """A face-shaped image with enough high-frequency detail to look in focus.

    The landmark model does not read this image -- it is handed the true
    projection -- but the quality gate does, and a gate tested only against
    contrived arrays is a gate nobody has run.
    """
    rng = np.random.default_rng(seed)
    img = np.full((IMAGE_H, IMAGE_W, 3), 128.0)
    yy, xx = np.mgrid[0:IMAGE_H, 0:IMAGE_W]
    cx, cy = float(uv[:, 0].mean()), float(uv[:, 1].mean())
    img[((xx - cx) / 300.0) ** 2 + ((yy - cy) / 380.0) ** 2 <= 1.0] = 165.0
    for u, v in uv:
        img[(xx - u) ** 2 + (yy - v) ** 2 <= 25.0] = 60.0
    img += rng.normal(0.0, 8.0, img.shape)
    img = np.clip(img, 0, 255)
    for _ in range(blur_radius):
        img = (
            img
            + np.roll(img, 1, 0)
            + np.roll(img, -1, 0)
            + np.roll(img, 1, 1)
            + np.roll(img, -1, 1)
        ) / 5.0
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Fake backends. Each is the narrowest thing that satisfies its protocol.
# ---------------------------------------------------------------------------


@dataclass
class FakeDetection:
    bbox: tuple[float, float, float, float]
    score: float
    keypoints: np.ndarray


@dataclass
class FakeLandmarkSet:
    points: PointSet
    covariances: np.ndarray
    yaw_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None


@dataclass
class FakeHeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class FakeIris:
    iris_diameter_px_l: float | None
    iris_diameter_px_r: float | None


@dataclass
class Registered:
    detection: FakeDetection
    landmarks: FakeLandmarkSet
    pose: FakeHeadPose
    iris_px: float | None


class _Table:
    """Per-image fixture lookup, keyed by the pipeline's own pixel hash.

    A single backend bundle serves both views of a run, so the fakes have to
    tell the photographs apart the same way the manifest does. The single-entry
    fallback covers the one arm that re-encodes as JPEG, where the pixels
    legitimately change after the fixture was registered.
    """

    def __init__(self) -> None:
        self._entries: dict[str, Registered] = {}

    def add(self, pixels: np.ndarray, entry: Registered) -> None:
        self._entries[pixel_sha256(pixels)] = entry

    def get(self, image: np.ndarray) -> Registered:
        key = pixel_sha256(image)
        if key in self._entries:
            return self._entries[key]
        if len(self._entries) == 1:
            return next(iter(self._entries.values()))
        raise KeyError("no fixture registered for this image")


class FakeDetector:
    provenance = licensing.YUNET
    weights_sha256 = "0" * 64

    def __init__(self, table: _Table) -> None:
        self._table = table

    def detect(self, image):
        return [self._table.get(image).detection]


class FakeLandmarker:
    provenance = licensing.SPIGA
    weights_sha256 = "1" * 64

    def __init__(self, table: _Table) -> None:
        self._table = table

    def locate(self, image, detection):
        return self._table.get(image).landmarks


class FakePoseEstimator:
    provenance = licensing.SIXDREPNET
    weights_sha256 = "2" * 64

    def __init__(self, table: _Table, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self._table = table
        self._offset = offset

    def estimate(self, image, detection):
        p = self._table.get(image).pose
        return FakeHeadPose(
            p.yaw_deg + self._offset[0],
            p.pitch_deg + self._offset[1],
            p.roll_deg + self._offset[2],
        )


class FakeIrisMeasurer:
    provenance = licensing.MEDIAPIPE_FACE_LANDMARKER
    weights_sha256 = "3" * 64

    def __init__(self, table: _Table) -> None:
        self._table = table

    def measure_iris(self, image, detection):
        px = self._table.get(image).iris_px
        return None if px is None else FakeIris(px, px)


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def make_view(
    tmp_path,
    name: str,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    omit: tuple[L, ...] = OMITTED,
    with_iris: bool = True,
    blur_radius: int = 0,
    exif: bool = False,
    render_seed: int = 0,
):
    """Write one synthetic photograph and build the fixture entry that goes with it."""
    visible = {k: v for k, v in FACE_MM.items() if k not in omit}
    names, uv = project(visible, yaw=yaw, pitch=pitch, roll=roll)
    pixels = render(uv, seed=render_seed, blur_radius=blur_radius)

    path = tmp_path / f"{name}.{'jpg' if exif else 'png'}"
    image = PILImage.fromarray(pixels)
    if exif:
        tags = image.getexif()
        tags[ExifTags.Base.Make.value] = "Vitruve"
        tags[ExifTags.Base.Model.value] = "Synthetic"
        sub = tags.get_ifd(ExifTags.IFD.Exif.value)
        sub[ExifTags.Base.FocalLengthIn35mmFilm.value] = int(FOCAL_35MM)
        gps = tags.get_ifd(ExifTags.IFD.GPSInfo.value)
        gps[ExifTags.GPS.GPSLatitudeRef.value] = "N"
        gps[ExifTags.GPS.GPSLatitude.value] = (
            Fraction(41),
            Fraction(52),
            Fraction(0),
        )  # a real coordinate, which must not survive into anything written out
        image.save(path, quality=97, exif=tags)
        pixels = np.asarray(PILImage.open(path).convert("RGB"), dtype=np.uint8)
    else:
        image.save(path)

    index = {n: i for i, n in enumerate(names)}
    landmarks = FakeLandmarkSet(
        points=PointSet(index=index, coords=uv),
        covariances=np.broadcast_to(
            np.eye(2) * LANDMARK_SD_PX**2, (len(names), 2, 2)
        ).copy(),
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
    )
    keypoint_names = (L.PUPIL_R, L.PUPIL_L, L.PRONASALE, L.CHEILION_R, L.CHEILION_L)
    available = [n for n in keypoint_names if n in index]
    if len(available) < 5:
        # Whatever is visible, padded out to five with the nose. A detector's
        # keypoints only seed the alignment, so approximate ones are fine.
        available = (available + [L.PRONASALE] * 5)[:5]
    detection = FakeDetection(
        bbox=(
            float(uv[:, 0].min()),
            float(uv[:, 1].min()),
            float(np.ptp(uv[:, 0])),
            float(np.ptp(uv[:, 1])),
        ),
        score=0.99,
        keypoints=np.stack([uv[index[n]] for n in available]),
    )
    entry = Registered(
        detection=detection,
        landmarks=landmarks,
        pose=FakeHeadPose(yaw, pitch, roll),
        iris_px=(11.84 * FOCAL_PX / SUBJECT_DISTANCE_MM) if with_iris else None,
    )
    return path, pixels, entry


def make_stack(*views, pose_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Backends:
    table = _Table()
    for _path, pixels, entry in views:
        table.add(pixels, entry)
    return Backends(
        detector=FakeDetector(table),
        landmarker=FakeLandmarker(table),
        pose_estimator=FakePoseEstimator(table, pose_offset),
        iris=FakeIrisMeasurer(table),
        device="cpu",
    )


def run(tmp_path, **kwargs):
    """Build a single frontal fixture and analyse it. Returns the result."""
    analyze_kwargs = {
        k: kwargs.pop(k)
        for k in ("ruler_mm", "seed", "n_samples", "correct_roll", "declared_sex")
        if k in kwargs
    }
    pose_offset = kwargs.pop("pose_offset", (0.0, 0.0, 0.0))
    view = make_view(tmp_path, kwargs.pop("name", "frontal"), **kwargs)
    stack = make_stack(view, pose_offset=pose_offset)
    analyze_kwargs.setdefault("ruler_mm", RULER)
    analyze_kwargs.setdefault("seed", 11)
    analyze_kwargs.setdefault("n_samples", 512)
    return analyze(view[0], tier=Tier.PERMISSIVE, backends=stack, **analyze_kwargs)


# ---------------------------------------------------------------------------
# Ground truth, computed from the coordinate table by the catalogue's own formula
# ---------------------------------------------------------------------------


def frontal_truth(spec_id: str) -> float:
    """The measurement an ideal orthographic frontal camera would return."""
    points = PointSet.from_mapping(
        {n: np.array([v[0], v[1]]) for n, v in FACE_MM.items()}
    )
    return float(BY_ID[spec_id].formula.eval(points))


def profile_truth(spec_id: str) -> float:
    """The measurement an ideal profile camera would return.

    The subject's right-left coordinate is zeroed, because a profile photograph
    projects everything into the sagittal plane whether it belongs there or not.
    That is a property of the view, so the reference has to share it.
    """
    points = PointSet.from_mapping(
        {n: np.array([0.0, v[1], v[2]]) for n, v in FACE_MM.items()}
    )
    return float(BY_ID[spec_id].formula.eval(points))


def assert_close(result, spec_id: str, expected: float, *, why: str = "") -> None:
    measured = result.by_id(spec_id)
    assert measured is not None and not isinstance(measured, Unavailable), (
        f"{spec_id} was not measured: {measured}"
    )
    unit = BY_ID[spec_id].unit
    if unit is Unit.DEGREES:
        assert measured.value == pytest.approx(expected, abs=ANGLE_TOLERANCE_DEG), why
    else:
        tolerance = RATIO_TOLERANCE if unit is Unit.RATIO else LENGTH_TOLERANCE
        assert measured.value == pytest.approx(expected, rel=tolerance), why
    assert measured.ci_low < measured.value < measured.ci_high


# ---------------------------------------------------------------------------
# Recovery of the known geometry
# ---------------------------------------------------------------------------


def test_end_to_end_recovers_the_synthetic_geometry(tmp_path):
    result = run(tmp_path)
    assert not result.failed, result.failure_reasons
    assert result.scale is not None and result.scale.source is ScaleSource.RULER

    for spec_id in (
        "interpupillary_distance",
        "intercanthal_width",
        "biocular_width",
        "nose_breadth",
        "mouth_width",
        "nose_height",
        "face_height_sellion_menton",
        "middle_third_height",
        "lower_third_height",
        "palpebral_fissure_width_r",
        "palpebral_fissure_height_r",
    ):
        assert_close(result, spec_id, frontal_truth(spec_id))

    for spec_id in (
        "intercanthal_biocular_ratio",
        "nose_mouth_width_ratio",
        "eye_spacing_ratio",
        "facial_thirds_ratio",
        "eye_aspect_ratio_r",
    ):
        assert_close(result, spec_id, frontal_truth(spec_id))

    for spec_id in ("canthal_tilt_r", "canthal_tilt_l"):
        assert_close(result, spec_id, frontal_truth(spec_id))


def test_a_perfectly_symmetric_face_still_reports_a_positive_asymmetry(tmp_path):
    """A folded measurement cannot be unbiased, and the size of the bias is known.

    The asymmetry measurements are absolute differences. Under landmark noise the
    difference is a zero-mean normal, so its absolute value is half-normal, whose
    median is ``0.6745 * sigma`` and whose standard deviation is
    ``0.6028 * sigma``. The median therefore sits at ``1.119`` standard
    deviations above zero *for a face with no asymmetry at all*.

    This is asserted rather than tolerated because it is a property a reader of
    the report needs: a small positive asymmetry is what a symmetric face looks
    like through this instrument, and the number to compare it against is the
    measurement's own spread, not zero.
    """
    result = run(tmp_path)
    half_normal_median_over_sd = 0.6745 / math.sqrt(1.0 - 2.0 / math.pi)

    for spec_id in ("canthal_tilt_asymmetry", "ocular_height_asymmetry", "mouth_corner_asymmetry"):
        assert frontal_truth(spec_id) == pytest.approx(0.0, abs=1e-9)
        measured = result.by_id(spec_id)
        assert measured.value > 0.0
        assert measured.ci_low > 0.0, "a folded quantity cannot bracket zero"
        assert measured.value == pytest.approx(
            half_normal_median_over_sd * measured.sd, rel=0.25
        ), f"{spec_id} should read as pure folding noise on a symmetric face"


def test_a_ruler_collapses_the_scale_assumption(tmp_path):
    with_ruler = run(tmp_path, name="a", ruler_mm=RULER)
    from_prior = run(tmp_path, name="b", ruler_mm=None)

    assert from_prior.scale.source is ScaleSource.FUSED
    # The prior carries a population spread; the ruler carries a reading error.
    assert with_ruler.scale.relative_sd < from_prior.scale.relative_sd / 3
    ruler_value = with_ruler.by_id("intercanthal_width")
    prior_value = from_prior.by_id("intercanthal_width")
    assert ruler_value.relative_ci_width < prior_value.relative_ci_width


def test_ratios_do_not_need_a_scale_at_all(tmp_path):
    """Scale cancels in a ratio, which is why the project prefers them."""
    result = run(tmp_path, ruler_mm=None)
    for spec_id in ("intercanthal_biocular_ratio", "nose_mouth_width_ratio"):
        assert_close(result, spec_id, frontal_truth(spec_id))


def test_without_any_scale_cue_millimetre_values_are_unavailable(tmp_path):
    view = make_view(
        tmp_path,
        "noscale",
        omit=(*OMITTED, L.PUPIL_L, L.PUPIL_R),
        with_iris=False,
    )
    stack = make_stack(view)
    result = analyze(
        view[0], tier=Tier.PERMISSIVE, backends=stack, seed=11, n_samples=512
    )
    assert result.scale is None

    unavailable = {u.spec_id: u for u in result.unavailable}
    assert "intercanthal_width" in unavailable
    assert "no scale reference" in unavailable["intercanthal_width"].reason
    # The dimensionless outputs are untouched, which is the point.
    assert_close(result, "intercanthal_biocular_ratio", frontal_truth("intercanthal_biocular_ratio"))


# ---------------------------------------------------------------------------
# Withheld and unavailable are different outcomes, and both are results
# ---------------------------------------------------------------------------


def test_measurements_that_need_3d_are_withheld_with_the_published_reason(tmp_path):
    result = run(tmp_path)
    for spec_id in ("bizygomatic_width", "bigonial_width", "facial_width_height_ratio"):
        measured = result.by_id(spec_id)
        assert measured.verdict.reportability is Reportability.WITHHOLD
        assert any("self-occluding" in r for r in measured.verdict.reasons), (
            f"{spec_id} was withheld without naming why"
        )
        assert any("Lim et al. 2022" in r for r in measured.verdict.reasons)
    assert set(result.withheld) & set(result.measured)
    assert not set(result.withheld) & set(result.reportable)


def test_a_missing_landmark_is_unavailable_and_names_the_landmark(tmp_path):
    result = run(tmp_path)
    unavailable = {u.spec_id: u for u in result.unavailable}
    assert "philtrum_width" in unavailable
    assert set(unavailable["philtrum_width"].missing_landmarks) == {
        "crista_philtri_l",
        "crista_philtri_r",
    }
    # And it is not in the measured list at all, which is what distinguishes
    # "the model cannot see it" from "the number would not mean anything".
    assert result.by_id("philtrum_width") is unavailable["philtrum_width"]


def test_profile_measurements_are_unavailable_without_a_profile_photograph(tmp_path):
    result = run(tmp_path)
    profile_ids = {s.id for s in for_view(View.PROFILE)}
    unavailable = {u.spec_id: u for u in result.unavailable}
    assert profile_ids <= set(unavailable)
    assert "no profile photograph was supplied" in unavailable["nasofrontal_angle"].reason


# ---------------------------------------------------------------------------
# Two views
# ---------------------------------------------------------------------------


def test_a_profile_photograph_produces_the_profile_measurements(tmp_path):
    frontal = make_view(tmp_path, "front")
    profile = make_view(
        tmp_path, "side", yaw=90.0, omit=(*OMITTED, *PROFILE_HIDDEN), render_seed=3
    )
    stack = make_stack(frontal, profile)
    result = analyze(
        frontal[0],
        profile[0],
        tier=Tier.PERMISSIVE,
        backends=stack,
        ruler_mm=RULER,
        seed=11,
        n_samples=512,
    )
    assert not result.failed, result.failure_reasons
    assert result.profile is not None and result.profile.frame.points.dim == 3
    assert result.profile.frame.have_3d is False

    for spec_id in ("nasofrontal_angle", "nasolabial_angle", "facial_convexity_angle"):
        assert_close(result, spec_id, profile_truth(spec_id))
    assert_close(result, "submental_length", profile_truth("submental_length"))

    # The far side of the head is not visible in a profile, so the measurement
    # that needs it is unavailable rather than guessed.
    unavailable = {u.spec_id for u in result.unavailable}
    assert "gonial_angle_r" in unavailable
    assert result.by_id("gonial_angle_l") not in (None,)
    assert not isinstance(result.by_id("gonial_angle_l"), Unavailable)

    # The frontal measurements are unaffected by the profile being present.
    assert_close(result, "intercanthal_width", frontal_truth("intercanthal_width"))


# ---------------------------------------------------------------------------
# Pose: the predictions in core/sensitivity, made falsifiable
# ---------------------------------------------------------------------------


def test_yaw_moves_measurements_the_way_the_sensitivity_model_predicts(tmp_path):
    """Ten degrees of yaw, against the first-order orthographic prediction.

    A transverse width carries the full ``cos(yaw)``, so it must *shrink* by
    ``1 - cos(10 deg)``, about 1.5 percent. A ratio of two transverse widths
    carries the same cosine twice and it cancels, so it must barely move. Both
    the size and the direction are asserted: a change that got the magnitude
    right and the sign wrong would mean the pipeline was measuring the mirror
    image.
    """
    level = run(tmp_path, name="level")
    turned = run(tmp_path, name="turned", yaw=10.0)
    predicted = 1.0 - math.cos(math.radians(10.0))

    for spec_id in ("biocular_width", "intercanthal_width", "mouth_width"):
        before = level.by_id(spec_id).value
        after = turned.by_id(spec_id).value
        relative = (after - before) / before
        assert relative < 0, f"{spec_id} should shrink under yaw, not grow"
        assert abs(relative) == pytest.approx(predicted, rel=0.25)

    for spec_id in ("intercanthal_biocular_ratio", "nose_mouth_width_ratio"):
        before = level.by_id(spec_id).value
        after = turned.by_id(spec_id).value
        relative = abs((after - before) / before)
        assert relative < predicted / 10, (
            f"{spec_id} is a same-plane ratio and should be nearly yaw-invariant"
        )

    # Verticals are untouched by a rotation about the vertical axis, which is
    # the control that says the widths moved because of yaw and not because
    # something global changed between the two runs.
    for spec_id in ("nose_height", "face_height_sellion_menton"):
        before = level.by_id(spec_id).value
        after = turned.by_id(spec_id).value
        assert abs((after - before) / before) < predicted / 10


def test_roll_is_corrected_in_the_value_and_retained_in_the_uncertainty(tmp_path):
    level = run(tmp_path, name="level")
    rolled = run(tmp_path, name="rolled", roll=8.0)
    uncorrected = run(tmp_path, name="raw", roll=8.0, correct_roll=False)

    truth = frontal_truth("canthal_tilt_r")
    assert level.by_id("canthal_tilt_r").value == pytest.approx(truth, abs=ANGLE_TOLERANCE_DEG)
    assert rolled.by_id("canthal_tilt_r").value == pytest.approx(truth, abs=ANGLE_TOLERANCE_DEG)
    # Uncorrected, eight degrees of camera tilt lands almost entirely in a
    # measurement whose whole normal range spans about eight degrees.
    # Canthal tilt is now measured against the interpupillary line rather than
    # the image horizon, so it is roll-invariant in the formula itself and does
    # not depend on the pipeline's correction having run. This assertion used
    # to read `truth + 8.0`, which was the symptom of the older definition: a
    # tilted camera was reported as a tilted face.
    assert uncorrected.by_id("canthal_tilt_r").value == pytest.approx(
        truth, abs=ANGLE_TOLERANCE_DEG
    )

    frame = rolled.frontal.frame
    assert frame.applied_roll_deg == pytest.approx(8.0, abs=0.05)
    assert frame.roll_deg == pytest.approx(
        POSE_ESTIMATOR_SD_DEG * math.sqrt(2.0 / math.pi), rel=1e-6
    )
    # The correction is not free: the gate still sees a residual and still says
    # so on the pose-critical measurement.
    assert any("roll" in r for r in rolled.by_id("canthal_tilt_r").verdict.reasons)


def test_pose_source_disagreement_reaches_the_report(tmp_path):
    agreeing = run(tmp_path, name="agree")
    diverging = run(tmp_path, name="diverge", pose_offset=(12.0, 0.0, 0.0))

    assert agreeing.frontal.pose.agreement.max_delta_deg == pytest.approx(0.0)
    assert diverging.frontal.pose.agreement.max_delta_deg == pytest.approx(12.0)
    assert diverging.frontal.pose.yaw_sd_deg > agreeing.frontal.pose.yaw_sd_deg
    codes = {i.code for i in diverging.frontal.quality.issues}
    assert "pose_sources_diverge" in codes


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_the_manifest_round_trips_and_explains_every_measurement(tmp_path):
    result = run(tmp_path)
    manifest = result.manifest

    assert RunManifest.from_json(manifest.to_json()) == manifest
    written = manifest.write(tmp_path / "out")
    assert RunManifest.from_json(written.read_text()) == manifest

    recorded = {m.spec_id for m in manifest.measurements}
    assert recorded == {m.spec_id for m in result.measured} | {
        u.spec_id for u in result.unavailable
    }
    assert len(recorded) == len(BY_ID)

    for record in manifest.measurements:
        assert record.formula_fingerprint == BY_ID[record.spec_id].fingerprint
        if record.outcome == "withheld":
            assert record.value is None
            assert record.reasons, f"{record.spec_id} was withheld without a reason"
        if record.outcome == "unavailable":
            assert record.reasons
        if record.outcome == "measured":
            assert record.value is not None
            if record.reportability != Reportability.REPORT.value:
                # A clean measurement owes no explanation; a qualified one does.
                assert record.reasons, f"{record.spec_id} is caveated silently"

    roles = {m.role for m in manifest.models}
    assert roles == {"detector", "landmarker", "pose_estimator", "iris"}
    assert all(m.weights_sha256 for m in manifest.models)
    assert manifest.images[0].sha256 == result.frontal.source.sha256
    assert manifest.scale["source"] == "ruler"
    assert manifest.seed == 11


def test_the_manifest_times_every_stage(tmp_path):
    result = run(tmp_path)
    names = [s.name for s in result.manifest.stages]
    for expected in (
        "frontal.ingest",
        "frontal.detect",
        "frontal.align",
        "frontal.landmark",
        "frontal.pose",
        "frontal.quality",
        "frontal.canonical",
        "scale",
        "measure",
    ):
        assert expected in names
    assert all(s.seconds >= 0 for s in result.manifest.stages)


def test_no_aggregate_score_appears_anywhere_in_the_result(tmp_path):
    """Rule one of the core contract, checked against the real output."""
    result = run(tmp_path)
    banned = ("score", "overall", "harmony", "rating", "rank", "attractiveness", "beauty")
    payload = json.dumps(json.loads(result.manifest.to_json())).lower()
    for word in banned:
        assert f'"{word}"' not in payload
    assert not [a for a in dir(result) if any(b in a.lower() for b in banned)]


def test_the_run_is_deterministic_under_a_fixed_seed(tmp_path):
    first = run(tmp_path, name="det", seed=5)
    second = run(tmp_path, name="det", seed=5)
    assert [m.value for m in first.measured] == [m.value for m in second.measured]
    assert [m.ci_high for m in first.measured] == [m.ci_high for m in second.measured]

    different = run(tmp_path, name="det", seed=6)
    assert [m.value for m in first.measured] != [m.value for m in different.measured]


# ---------------------------------------------------------------------------
# Privacy, failure, and the offline guarantee
# ---------------------------------------------------------------------------


def test_exif_is_read_for_optics_and_never_written_back(tmp_path):
    view = make_view(tmp_path, "withexif", exif=True)
    stack = make_stack(view)
    result = analyze(
        view[0], tier=Tier.PERMISSIVE, backends=stack, ruler_mm=RULER, seed=11, n_samples=256
    )

    exif = result.frontal.source.exif
    assert exif.focal_length_35mm == pytest.approx(FOCAL_35MM)
    assert exif.had_gps
    # The optics are used: a distance follows from the focal length and the
    # measured interpupillary span, and it lands near where the camera was put.
    assert result.frontal.subject_distance is not None
    assert result.frontal.subject_distance.metres == pytest.approx(
        SUBJECT_DISTANCE_MM / 1000.0, rel=0.1
    )

    payload = json.loads(result.manifest.to_json())
    tags_seen = payload["images"][0]["exif_tags_present"]
    assert any(t.startswith("GPSLatitude") for t in tags_seen)
    # The names travel, the coordinates do not.
    assert "41" not in json.dumps(tags_seen)

    stripped = save_stripped(result.frontal.source.pixels, tmp_path / "stripped.jpg")
    with PILImage.open(stripped) as reloaded:
        assert not dict(reloaded.getexif())


def test_a_photograph_that_cannot_be_measured_fails_with_an_actionable_message(tmp_path):
    result = run(tmp_path, name="blurred", blur_radius=30)
    assert result.failed
    assert not result.measured and not result.unavailable
    assert result.failure_reasons
    joined = " ".join(result.failure_reasons)
    assert "Remedy:" in joined
    # The manifest is still written: a run that produced nothing is exactly when
    # a reader most needs to know what was attempted, and on which pixels.
    assert result.manifest.failure_reasons == result.failure_reasons
    assert result.manifest.images and result.manifest.models


def test_analysis_opens_no_sockets(tmp_path, monkeypatch):
    """The offline guarantee, asserted rather than documented."""

    class Blocked(socket.socket):
        def __init__(self, *a, **k):
            raise AssertionError("analysis attempted a network connection")

    monkeypatch.setattr(socket, "socket", Blocked)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("analysis attempted a network connection")
        ),
    )
    result = run(tmp_path, name="offline", n_samples=256)
    assert not result.failed
