"""Mapping a detected face onto a fixed canonical crop, reversibly.

Landmark backends want a face at a predictable size and orientation, so the
five detector keypoints are fitted to a fixed template by a similarity
transform. Similarity and not affine: an affine fit has six free parameters and
will happily shear a face to match the template, which changes every angle in
the catalogue. A similarity has four, it can only rotate, uniformly scale and
translate, and those are exactly the degrees of freedom that a measurement in
the canonical frame is invariant to.

The transform is returned in both directions, and that is the point of this
module rather than a detail of it. A measurement is taken in one frame and has
to be drawn in another: the report overlays intercanthal width on the
photograph the user recognises, not on a 512-pixel crop of it. Keeping the
inverse as a first-class object means the drawing code never re-derives it and
never gets it subtly wrong.

The canonical template is the ArcFace five-point layout, scaled from its native
112 pixels. It is used here only as a stable reference geometry; no ArcFace
weights are involved and none of the licence obligations attach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .ports import Detection, Image

#: Side of the canonical crop, in pixels.
CANONICAL_SIZE = 512

#: ArcFace five-point template at 112 pixels, in :data:`~vitruve.pipeline.ports.KEYPOINT_ORDER`.
#: Row 0 is the eye at the smaller x coordinate, which in an unmirrored frontal
#: photograph is the subject's right eye.
_ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=float,
)


def canonical_keypoints(size: int = CANONICAL_SIZE) -> NDArray[np.float64]:
    """The five-point template at the requested crop size."""
    return _ARCFACE_112 * (float(size) / 112.0)


@dataclass(frozen=True)
class Similarity:
    """A rotation, a uniform scale and a translation, stored as a 2x3 matrix.

    Row-vector convention throughout, matching every other array in Vitruve:
    ``apply`` computes ``p @ A.T + t`` for points shaped ``(..., 2)``.
    """

    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        if m.shape != (2, 3):
            raise ValueError(f"a similarity is a 2x3 matrix, got {m.shape}")
        object.__setattr__(self, "matrix", m)

    @property
    def linear(self) -> NDArray[np.float64]:
        return self.matrix[:, :2]

    @property
    def translation(self) -> NDArray[np.float64]:
        return self.matrix[:, 2]

    @property
    def scale(self) -> float:
        """Uniform scale factor, in destination pixels per source pixel."""
        return float(math.sqrt(abs(np.linalg.det(self.linear))))

    @property
    def rotation_deg(self) -> float:
        return float(math.degrees(math.atan2(self.matrix[1, 0], self.matrix[0, 0])))

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform points shaped ``(..., 2)``, broadcasting over leading axes."""
        p = np.asarray(points, dtype=float)
        if p.shape[-1] != 2:
            raise ValueError(f"expected 2D points, got last axis {p.shape[-1]}")
        return p @ self.linear.T + self.translation

    def apply_covariance(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """Push a ``(..., 2, 2)`` covariance through, as ``A C A^T``.

        Transforming the points and leaving the covariances behind is the
        classic way to end up with an interval that belongs to a different
        coordinate system than the value it brackets.
        """
        a = self.linear
        return np.einsum("ij,...jk,lk->...il", a, np.asarray(cov, dtype=float), a)

    @property
    def inverse(self) -> Similarity:
        a_inv = np.linalg.inv(self.linear)
        return Similarity(np.concatenate([a_inv, (-a_inv @ self.translation)[:, None]], axis=1))

    def as_matrix3(self) -> NDArray[np.float64]:
        """Homogeneous 3x3 form, for composing with other transforms."""
        return np.vstack([self.matrix, np.array([0.0, 0.0, 1.0])])

    @classmethod
    def identity(cls) -> Similarity:
        return cls(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def estimate_similarity(
    src: NDArray[np.float64], dst: NDArray[np.float64]
) -> Similarity:
    """Least-squares similarity taking ``src`` onto ``dst`` (Umeyama 1991).

    The reflection guard matters. With five near-coplanar keypoints and a
    left/right transposition somewhere upstream, the unconstrained optimum is a
    *mirrored* fit that lands the points beautifully and flips the face. Forcing
    a positive determinant means such a bug surfaces as a visibly bad alignment
    instead of as a silently mirrored report.
    """
    s = np.asarray(src, dtype=float)
    d = np.asarray(dst, dtype=float)
    if s.shape != d.shape or s.ndim != 2 or s.shape[1] != 2:
        raise ValueError(f"need matching (n, 2) arrays, got {s.shape} and {d.shape}")
    n = s.shape[0]
    if n < 2:
        raise ValueError("a similarity needs at least two point correspondences")

    mu_s, mu_d = s.mean(axis=0), d.mean(axis=0)
    sc, dc = s - mu_s, d - mu_d
    var_s = float((sc**2).sum() / n)
    if var_s <= 0:
        raise ValueError("source points are coincident; no similarity is defined")

    cov = (dc.T @ sc) / n
    u, sing, vt = np.linalg.svd(cov)
    corr = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        corr[1, 1] = -1.0
    rot = u @ corr @ vt
    scale = float((sing * np.diag(corr)).sum() / var_s)
    trans = mu_d - scale * (rot @ mu_s)
    return Similarity(np.concatenate([scale * rot, trans[:, None]], axis=1))


@dataclass(frozen=True)
class AlignedFace:
    """A canonical crop and the two transforms that relate it to the original.

    ``forward`` maps original-image coordinates into the crop; ``inverse`` maps
    them back. Both are kept rather than one plus a note, because "invert this
    when you draw" is an instruction that gets forgotten.
    """

    pixels: NDArray[np.uint8]
    forward: Similarity
    inverse: Similarity
    size: int
    detection_score: float

    @property
    def pixels_per_source_pixel(self) -> float:
        return self.forward.scale


def _bilinear_sample(image: Image, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Sample ``image`` at fractional coordinates, clamping at the border.

    Written out rather than delegated to OpenCV so that the pipeline's
    alignment stage has no dependency beyond numpy. It runs over a quarter of a
    million output pixels and takes a few milliseconds, which is far below the
    model inference it precedes.
    """
    h, w = image.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    img = image.astype(np.float64)
    top = img[y0c, x0c] * (1.0 - fx) + img[y0c, x1c] * fx
    bot = img[y1c, x0c] * (1.0 - fx) + img[y1c, x1c] * fx
    out = top * (1.0 - fy) + bot * fy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def align(
    image: Image, detection: Detection, *, size: int = CANONICAL_SIZE
) -> AlignedFace:
    """Warp the detected face onto a ``size`` x ``size`` canonical crop."""
    kps = np.asarray(detection.keypoints, dtype=float)
    if kps.shape != (5, 2):
        raise ValueError(f"expected five detector keypoints shaped (5, 2), got {kps.shape}")

    forward = estimate_similarity(kps, canonical_keypoints(size))
    inverse = forward.inverse

    grid_y, grid_x = np.mgrid[0:size, 0:size]
    # Sample at pixel centres. Half-pixel offsets are the difference between a
    # crop that lines up with its landmarks and one that is consistently half a
    # pixel out, which at 512 pixels across a face is a tenth of a millimetre
    # of systematic bias on every measurement.
    dst = np.stack([grid_x + 0.5, grid_y + 0.5], axis=-1)
    src = inverse.apply(dst)
    pixels = _bilinear_sample(image, src[..., 0] - 0.5, src[..., 1] - 0.5)

    return AlignedFace(
        pixels=pixels,
        forward=forward,
        inverse=inverse,
        size=size,
        detection_score=float(detection.score),
    )
