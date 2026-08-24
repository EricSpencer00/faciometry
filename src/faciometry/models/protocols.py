"""The backend boundary: what a model must return, and in what frame.

Every stage of the pipeline talks to models through these protocols, so a
backend swap is a registry entry and not a rewrite. Three properties are
enforced by the types rather than by convention, because each of them is a
failure that would otherwise be silent:

* **Provenance travels with the result.** A number in the report has to be
  traceable to the weights that produced it, and a `Provenance` attached to the
  result survives caching and serialisation in a way that a global "which model
  did we load" variable does not.
* **Points are named, never indexed.** A backend emits a `Landmark -> index`
  map and a `PointSet`; integer indices do not cross this boundary. What a
  backend cannot see is *absent* from the map, and `evaluate()` turns that into
  an `Unavailable` naming the landmark. Substituting a nearby point instead is
  the single easiest way to make this whole system dishonest.
* **Uncertainty is anisotropic and comes from the model.** `heatmap_moments`
  is here rather than in a backend because every heatmap landmarker needs the
  same second-moment computation, and because the argmax shortcut is exactly
  what the rest of Faciometry is built to avoid.

Coordinate frames, stated once. Backends return **image coordinates**: +x
right across the pixel grid, +y down, origin at the top-left pixel. The
canonical frame that measurements are evaluated in is **+x the subject's
right, +y up**. In a mirror-free frontal photograph the subject's right is the
viewer's left, so the conversion negates both axes; :func:`to_canonical` is
the only place that negation is written down.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..core.landmarks import Landmark, PointSet
from ..measure.evaluate import LandmarkUncertainty
from .licensing import Provenance

Image = NDArray[np.uint8]
"""An **RGB** image, shape ``(h, w, 3)``, uint8, EXIF already discarded.

RGB rather than OpenCV's BGR because that is what `pipeline.ingest` produces
and what `pipeline.ports` declares. Two of the four backends were trained on
BGR and one wants RGB, so the swap happens inside each backend, once, next to
the model that cares. A convention that lives in a docstring and is applied
four different ways is the kind of thing that produces a report with a plausible
blue cast in every skin measurement.
"""

#: Row order of the five detector keypoints, matching `pipeline.ports`. The
#: sides are the subject's, so ``eye_r`` is the eye at the *smaller* image x in
#: an unmirrored frontal photograph.
KEYPOINT_ORDER: tuple[str, ...] = ("eye_r", "eye_l", "nose", "mouth_r", "mouth_l")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceBox:
    """One detected face: a box, a score, and five coarse keypoints.

    The five keypoints are the detector's own, at detector accuracy, and they
    exist to seed alignment and to cross-check the landmark model's left/right
    assignment. They are deliberately *not* promoted into the `PointSet`: a
    detector eye centre is several pixels from a pupil centre, and a pipeline
    that quietly substituted one for the other would report a shifted
    interpupillary distance with a landmark-model-sized error bar.

    Naming is subject-relative throughout, so ``right_eye`` is the eye on the
    subject's right, which appears at the smaller image x in a frontal
    photograph.
    """

    x: float
    y: float
    w: float
    h: float
    score: float
    right_eye: NDArray[np.float64]
    left_eye: NDArray[np.float64]
    nose: NDArray[np.float64]
    right_mouth: NDArray[np.float64]
    left_mouth: NDArray[np.float64]
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"face box must have positive extent, got {self.w}x{self.h}")
        for name in ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth"):
            kp = np.asarray(getattr(self, name), dtype=float)
            if kp.shape != (2,):
                raise ValueError(f"{name} must be a 2-vector, got shape {kp.shape}")

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def centre(self) -> NDArray[np.float64]:
        return np.array([self.x + self.w / 2.0, self.y + self.h / 2.0])

    @property
    def area(self) -> float:
        return float(self.w * self.h)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Alias for :attr:`xywh`, under the name `pipeline.ports` uses."""
        return self.xywh

    @property
    def keypoints(self) -> NDArray[np.float64]:
        """The five keypoints as ``(5, 2)`` in :data:`KEYPOINT_ORDER`.

        An array rather than a mapping because the alignment stage does a
        least-squares fit against a template and wants rows, and because this is
        the shape `pipeline.ports.Detection` declares. The named attributes
        remain the readable way to reach one of them.
        """
        return np.stack(
            [self.right_eye, self.left_eye, self.nose, self.right_mouth, self.left_mouth],
            axis=0,
        )

    @property
    def named_keypoints(self) -> Mapping[str, NDArray[np.float64]]:
        return dict(zip(KEYPOINT_ORDER, self.keypoints, strict=True))

    @classmethod
    def from_detection(cls, detection: object, provenance: Provenance) -> FaceBox:
        """Coerce anything satisfying `pipeline.ports.Detection` into a FaceBox.

        The pipeline is written against a structural protocol and its tests
        supply their own minimal detections, so a landmarker that assumed it
        would always be handed one of *these* objects would work in production
        and fail in the test that exists to catch exactly that.
        """
        if isinstance(detection, cls):
            return detection
        x, y, w, h = (float(v) for v in detection.bbox)  # type: ignore[attr-defined]
        kp = np.asarray(detection.keypoints, dtype=float)  # type: ignore[attr-defined]
        if kp.shape != (5, 2):
            raise ValueError(f"expected (5, 2) keypoints in KEYPOINT_ORDER, got {kp.shape}")
        return cls(
            x=x,
            y=y,
            w=w,
            h=h,
            score=float(getattr(detection, "score", 1.0)),
            right_eye=kp[0],
            left_eye=kp[1],
            nose=kp[2],
            right_mouth=kp[3],
            left_mouth=kp[4],
            provenance=getattr(detection, "provenance", provenance),
        )

    def expanded(self, factor: float) -> FaceBox:
        """Grow the box about its centre.

        Landmark models are trained on a particular crop convention and a box
        that clips the chin or the hairline moves every contour point. The
        expansion factor belongs to the landmarker, not to the detector, which
        is why this returns a new box rather than mutating the detection.
        """
        if factor <= 0:
            raise ValueError("expansion factor must be positive")
        cx, cy = self.centre
        w, h = self.w * factor, self.h * factor
        return FaceBox(
            x=cx - w / 2.0,
            y=cy - h / 2.0,
            w=w,
            h=h,
            score=self.score,
            right_eye=self.right_eye,
            left_eye=self.left_eye,
            nose=self.nose,
            right_mouth=self.right_mouth,
            left_mouth=self.left_mouth,
            provenance=self.provenance,
        )

    @property
    def looks_mirrored(self) -> bool:
        """True when the eye keypoints contradict the subject-relative naming.

        A mirrored selfie, or a backend whose keypoint order was misread, puts
        the subject's right eye at the larger image x. Nothing downstream can
        detect that from the landmarks alone, because a face is close enough to
        symmetric that a transposed map still looks like a face.
        """
        return bool(self.right_eye[0] > self.left_eye[0])


@runtime_checkable
class FaceDetector(Protocol):
    """Finds faces and their five coarse keypoints."""

    provenance: Provenance

    def detect(self, image: Image) -> tuple[FaceBox, ...]:
        """Detections in descending score order, possibly empty."""
        ...


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LandmarkResult:
    """Named 2D points in image coordinates, with their covariances.

    ``yaw_deg``/``pitch_deg``/``roll_deg`` are optional because not every
    landmarker estimates pose, and ``None`` here means "this backend does not
    say", which the pipeline must resolve from a `PoseEstimator` rather than
    silently treat as zero.
    """

    points: PointSet
    uncertainty: LandmarkUncertainty
    provenance: Provenance
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    #: Landmarks in the `Landmark` enum that this backend structurally cannot
    #: supply. Recorded so the report can distinguish "the model cannot see
    #: this" from "the model saw it badly".
    unsupported: tuple[Landmark, ...] = ()

    def __post_init__(self) -> None:
        if self.points.dim != 2:
            raise ValueError("a 2D landmarker must return 2D points")
        missing_cov = set(self.points.index) - set(self.uncertainty.index)
        if missing_cov:
            raise ValueError(
                "every named point needs a covariance; missing "
                + ", ".join(sorted(m.value for m in missing_cov))
            )

    @property
    def has_pose(self) -> bool:
        return self.yaw_deg is not None

    @property
    def covariances(self) -> NDArray[np.float64]:
        """``(n_landmarks, 2, 2)`` in squared pixels, in `points.index` order.

        The same array `uncertainty` holds. It is surfaced under this name
        because `pipeline.ports.LandmarkSet` reads it directly, and because a
        consumer that only wants the matrices should not have to know that they
        travel inside an object whose other method draws Monte-Carlo samples.
        """
        return self.uncertainty.covariances


@runtime_checkable
class Landmarker(Protocol):
    """Locates named landmarks inside a detected box, with covariances."""

    provenance: Provenance

    def locate(self, image: Image, box: FaceBox) -> LandmarkResult:
        ...


# ---------------------------------------------------------------------------
# Dense mesh
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseResult:
    """A dense mesh, its iris ring, and the fitted rigid transform.

    The iris points are separated out rather than left inside the mesh because
    they are the only permissively-licensed metric cue Faciometry has:
    :func:`faciometry.core.scale.from_iris` turns a horizontal iris diameter in
    pixels into millimetres per pixel against a pooled corneal-diameter prior.
    Keeping them in a named field means a backend that lacks iris refinement
    fails loudly at construction instead of yielding a plausible wrong scale.

    ``transform`` is the backend's own 4x4 object-to-camera matrix. It is
    carried for the run manifest and for pose cross-checks; it is *not* a
    metric reconstruction, and MediaPipe's own millimetre output rests on a
    constant-scale assumption that Faciometry does not use.
    """

    points: PointSet
    all_points_px: NDArray[np.float64]
    iris_right_px: NDArray[np.float64]
    iris_left_px: NDArray[np.float64]
    transform: NDArray[np.float64]
    provenance: Provenance
    unsupported: tuple[Landmark, ...] = ()

    def __post_init__(self) -> None:
        if self.all_points_px.shape != (478, 2):
            raise ValueError(
                f"expected 478 dense points, got shape {self.all_points_px.shape}"
            )
        for name in ("iris_right_px", "iris_left_px"):
            ring = np.asarray(getattr(self, name))
            if ring.shape != (5, 2):
                raise ValueError(f"{name} must be a centre plus four ring points, got {ring.shape}")
        if self.transform.shape != (4, 4):
            raise ValueError(f"transform must be 4x4, got {self.transform.shape}")

    @staticmethod
    def _horizontal_diameter(ring: NDArray[np.float64]) -> float:
        """Horizontal extent of an iris ring, in pixels.

        Horizontal rather than mean-radius because the visible iris is occluded
        top and bottom by the eyelids in most photographs, and the pooled
        11.84 mm prior in `core.scale` is a *horizontal* corneal diameter. A
        vertical or averaged diameter would be measuring the eyelid aperture
        and calling it an iris.
        """
        return float(np.ptp(ring[1:, 0]))

    @property
    def iris_diameter_px(self) -> tuple[float, float]:
        """(right, left) horizontal iris diameters in pixels."""
        return (
            self._horizontal_diameter(self.iris_right_px),
            self._horizontal_diameter(self.iris_left_px),
        )

    @property
    def iris_diameter_px_r(self) -> float | None:
        """Subject's right iris, under the name `pipeline.ports` reads."""
        value = self._horizontal_diameter(self.iris_right_px)
        return value if value > 0 else None

    @property
    def iris_diameter_px_l(self) -> float | None:
        value = self._horizontal_diameter(self.iris_left_px)
        return value if value > 0 else None


@runtime_checkable
class DenseLandmarker(Protocol):
    """Produces a dense mesh with a separated iris ring."""

    provenance: Provenance

    def locate(self, image: Image, box: FaceBox) -> DenseResult:
        ...


# ---------------------------------------------------------------------------
# Head pose
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadPose:
    """Head rotation in degrees, with the estimator's own scatter.

    Sign convention, as far as it has been verified: pitch is invariant under a
    horizontal mirror while yaw and roll negate, which is what fixes *which*
    channel is which. The absolute direction (whether a positive yaw is a turn
    toward the subject's own left or right) has **not** been verified against a
    labelled benchmark here, and it is not load-bearing today because
    `core.sensitivity.gated_pose` takes an absolute value before gating. A
    measurement that ever consumes a *signed* pose must verify the direction
    first, and the docstring of the backend it comes from says how.

    ``sd_deg`` is the estimator's standard deviation, not a per-image
    confidence. `core.sensitivity` gates on ``|pose| + k * sd``, so a pose that
    arrived without its scatter would let a badly posed face through a
    five-degree gate.
    """

    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    sd_deg: float
    provenance: Provenance
    source: str = "model"

    def __post_init__(self) -> None:
        if self.sd_deg < 0:
            raise ValueError("pose standard deviation cannot be negative")

    @property
    def max_abs_deg(self) -> float:
        """Largest rotation magnitude, which is what the pose gate reads."""
        return max(abs(self.yaw_deg), abs(self.pitch_deg), abs(self.roll_deg))

    def disagreement(self, other: HeadPose) -> float:
        """Largest per-axis disagreement with an independent estimate, degrees.

        Two estimators that disagree by more than their combined scatter mean
        the pose number is not to be trusted, which is the entire reason
        Faciometry carries a second pose model at all.
        """
        return max(
            abs(self.yaw_deg - other.yaw_deg),
            abs(self.pitch_deg - other.pitch_deg),
            abs(self.roll_deg - other.roll_deg),
        )


@runtime_checkable
class PoseEstimator(Protocol):
    """Estimates head rotation from a face crop."""

    provenance: Provenance

    def estimate(self, image: Image, box: FaceBox) -> HeadPose:
        ...


# ---------------------------------------------------------------------------
# Frames and heatmaps
# ---------------------------------------------------------------------------


def to_canonical(
    points_px: NDArray[np.float64],
    *,
    origin_px: NDArray[np.float64],
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Image coordinates to the canonical frame: +x subject's right, +y up.

    Both axes negate. The y negation is the familiar one (image rows run
    downward); the x negation is the one that gets forgotten, because in a
    mirror-free frontal photograph the subject's right side is at the *smaller*
    image x. Getting it wrong swaps every lateralised finding while leaving
    every symmetric measurement correct, so nothing downstream would notice.
    """
    pts = np.asarray(points_px, dtype=float)
    if pts.shape[-1] != 2:
        raise ValueError(f"expected 2D points, got last axis {pts.shape[-1]}")
    origin = np.asarray(origin_px, dtype=float)
    return np.stack(
        [
            -(pts[..., 0] - origin[0]) * scale,
            -(pts[..., 1] - origin[1]) * scale,
        ],
        axis=-1,
    )


def canonicalise(ps: PointSet, *, origin_px: NDArray[np.float64], scale: float = 1.0) -> PointSet:
    """:func:`to_canonical` applied to a whole `PointSet`, names preserved."""
    return ps.with_coords(to_canonical(ps.coords, origin_px=origin_px, scale=scale))


#: Variance contributed by rounding a continuous position onto a pixel grid.
#: A uniform distribution over one pixel has variance 1/12, and adding it to
#: the heatmap covariance keeps a single-pixel-sharp channel from claiming zero
#: positional uncertainty, which would make its Monte-Carlo draws degenerate.
PIXEL_QUANTISATION_VAR = 1.0 / 12.0


def heatmap_moments(
    heatmaps: NDArray[np.float64],
    *,
    window: int = 11,
    floor_var: float = PIXEL_QUANTISATION_VAR,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-channel centroid and 2x2 covariance from landmark heatmaps.

    ``heatmaps`` is ``(n_landmarks, h, w)``; the returned centroid is
    ``(n_landmarks, 2)`` in (x, y) heatmap pixels and the covariance is
    ``(n_landmarks, 2, 2)`` in the same units.

    Two choices here are not obvious and both matter.

    **Second moment, not argmax.** The argmax is one number and the whole point
    of a heatmap backend is that the field around it says how well the point is
    pinned down. A landmark on the jaw contour is sharply localised across the
    contour and badly localised along it, and that anisotropy is what makes a
    contour-derived measurement's interval honest rather than decorative.

    **A window around the peak, with the window minimum subtracted.** The
    attention maps these networks emit are sigmoid-activated and have a small
    but non-zero response over the entire face. Taking the second moment over
    the full map would measure the width of the face rather than the width of
    the peak, and would report the *same* covariance for every landmark. The
    window truncates the tails, so the result is a lower bound on the true
    positional variance; that bound is stated rather than hidden, and it is the
    right direction to err in only because the alternative measures nothing at
    all.
    """
    hm = np.asarray(heatmaps, dtype=float)
    if hm.ndim != 3:
        raise ValueError(f"expected (n_landmarks, h, w), got shape {hm.shape}")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer of at least 3")
    n, h, w = hm.shape
    half = window // 2

    centres = np.empty((n, 2))
    covs = np.empty((n, 2, 2))
    for i in range(n):
        chan = hm[i]
        py, px = np.unravel_index(int(np.argmax(chan)), chan.shape)
        y0, y1 = max(0, py - half), min(h, py + half + 1)
        x0, x1 = max(0, px - half), min(w, px + half + 1)
        patch = chan[y0:y1, x0:x1]
        # Subtract the window's own floor so a uniform background contributes
        # no mass. Clipping at zero keeps the weights a valid distribution.
        weights = np.clip(patch - patch.min(), 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            # A dead channel: the peak carries no information, so report the
            # argmax with a covariance the size of the window rather than
            # pretending to a sharp localisation.
            centres[i] = (px, py)
            covs[i] = np.eye(2) * (half**2)
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        mx = float((weights * xs).sum() / total)
        my = float((weights * ys).sum() / total)
        dx, dy = xs - mx, ys - my
        vxx = float((weights * dx * dx).sum() / total)
        vyy = float((weights * dy * dy).sum() / total)
        vxy = float((weights * dx * dy).sum() / total)
        centres[i] = (mx, my)
        covs[i] = [[vxx + floor_var, vxy], [vxy, vyy + floor_var]]
    return centres, covs


def scale_covariances(
    covs: NDArray[np.float64], *, sx: float, sy: float
) -> NDArray[np.float64]:
    """Push covariances through an axis-aligned scaling.

    ``cov' = J cov J^T`` with ``J = diag(sx, sy)``. Written out rather than
    inlined because the cross term scales as ``sx * sy`` and not as either one
    alone, which is easy to get wrong when the crop is not square.
    """
    j = np.diag([float(sx), float(sy)])
    return np.einsum("ab,nbc,dc->nad", j, np.asarray(covs, dtype=float), j)


def uncertainty_from_covariances(
    index: Mapping[Landmark, int],
    covs_by_backend_index: NDArray[np.float64],
    backend_index: Mapping[Landmark, int],
) -> LandmarkUncertainty:
    """Re-key a backend's per-row covariances onto the named point set.

    The backend produces covariances in *its* row order; the `PointSet` uses a
    compact order over only the landmarks that survived the map. Doing this by
    hand at each backend is how a covariance ends up attached to the wrong
    landmark, which would be invisible in every output.
    """
    dim = 2
    n = len(index)
    out = np.zeros((n, dim, dim))
    for name, row in index.items():
        out[row] = covs_by_backend_index[backend_index[name]]
    return LandmarkUncertainty(index=dict(index), covariances=out)
