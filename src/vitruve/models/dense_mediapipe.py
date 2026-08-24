"""MediaPipe Face Landmarker: 478 points, and the ten that recover scale.

The reason this backend exists is the iris. No permissively-licensed model
returns metric geometry, so every millimetre Vitruve prints descends from an
assumption about the true size of something in the frame, and the pooled
corneal diameter (11.84 mm, SD 0.79, over 296,887 eyes) is the best such
assumption available without a ruler: it is adult-equivalent from age four, so
it needs no age or sex conditioning, unlike the interpupillary prior which
needs both and is still shifted up to 12% by ancestry. MediaPipe's iris
refinement is the only permissively-licensed way to measure that diameter, and
the `.task` bundle is Apache-2.0 like the code.

What this backend deliberately does **not** provide:

* **Covariances.** The mesh comes from a coordinate-regression head, so there
  is no field around each vertex and nothing honest to compute a second moment
  from. `DenseResult` therefore carries no `LandmarkUncertainty`, and a caller
  that wants to measure from these points has to state an isotropic assumption
  itself. Measurements come from the heatmap landmarker; this one supplies
  scale and a cross-check.
* **MediaPipe's own metric output.** The 4x4 transformation matrix is carried
  for the run manifest and for pose comparison, but its translation rests on a
  fixed canonical head size. That is a constant, not a measurement, and using
  it would mean reporting one manufacturer's average skull as this subject's.

**Landmark mapping.** The 478-point mesh has no published anatomical index
table worth trusting, so the assignments below were derived rather than
recalled: SPIGA's 98 points were mapped to landmarks first (from that model's
own canonical 3D shape), both models were run on the same frontal portrait, and
each MediaPipe index was chosen as the vertex nearest the corresponding SPIGA
landmark. Twenty of the twenty-four shared landmarks agreed to under three
pixels on a 270-pixel-tall face. Every bilateral pair was then checked for
mirror symmetry about the interpupillary midline, which is what rejected 359 as
a partner for 33 (asymmetric by five pixels) in favour of 263, and rejected
129/358 as alare (nine pixels too high, on the nasal sidewall rather than the
alar base) in favour of 98/327.

Left and right are the subject's. That was confirmed rather than assumed: iris
centre 468 and YuNet's first keypoint, which its documentation calls the right
eye, land within two pixels of each other and both at the smaller image x.

**Glabella is included here and absent from SPIGA.** That is not an
inconsistency. WFLW has no vertex above the inner brow at all, whereas the
MediaPipe mesh has one on the midline at brow-ridge height. Only its (x, y) is
used; the "most prominent" part of the anatomical definition is a depth
criterion that no frontal 2D model can evaluate. Trichion stays absent from
both: MediaPipe vertex 10 is the top of the mesh, which tracks the mesh
boundary rather than a hairline, and calling it trichion would put a
measurement's endpoint wherever the mesh happened to end.

**Version.** MediaPipe 1.0.1 aborts on macOS arm64 inside
`TensorsToDetectionsCalculator::Open` with ``Check failed: service_ Service is
unavailable`` -- a Metal helper is initialised for a graph running on CPU, and
the process dies rather than raising. It is a hard crash, so it cannot be
caught. The 0.10.x line works; this backend requires it and says so.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..core.landmarks import Landmark, PointSet
from ..core.landmarks import Landmark as L
from ..core.scale import ScaleEstimate, from_iris, fuse
from . import licensing, weights
from .licensing import Provenance, Tier
from .protocols import DenseResult, FaceBox, Image

WEIGHT_KEY = "mediapipe_face_landmarker"

PROVENANCE: Provenance = licensing.MEDIAPIPE_FACE_LANDMARKER

N_POINTS = 478

#: Iris vertices: a centre followed by four ring points. Right eye first, and
#: right means the subject's.
IRIS_RIGHT: tuple[int, ...] = (468, 469, 470, 471, 472)
IRIS_LEFT: tuple[int, ...] = (473, 474, 475, 476, 477)

#: Mesh vertex index for each landmark, derived as described in the module
#: docstring.
MESH_INDEX: dict[Landmark, int] = {
    L.GLABELLA: 9,
    L.SELLION: 168,
    L.SUPERCILIARE_R: 105,
    L.SUPERCILIARE_L: 334,
    L.EXOCANTHION_R: 33,
    L.ENDOCANTHION_R: 133,
    L.ENDOCANTHION_L: 362,
    L.EXOCANTHION_L: 263,
    L.PALPEBRALE_SUP_R: 159,
    L.PALPEBRALE_INF_R: 145,
    L.PALPEBRALE_SUP_L: 386,
    L.PALPEBRALE_INF_L: 374,
    L.PUPIL_R: 468,
    L.PUPIL_L: 473,
    L.PRONASALE: 1,
    L.ALARE_R: 98,
    L.ALARE_L: 327,
    L.SUBNASALE: 2,
    L.CRISTA_PHILTRI_R: 37,
    L.CRISTA_PHILTRI_L: 267,
    L.LABIALE_SUPERIUS: 0,
    L.LABIALE_INFERIUS: 17,
    L.CHEILION_R: 61,
    L.CHEILION_L: 291,
    L.MENTON: 152,
}

UNSUPPORTED: tuple[Landmark, ...] = (
    L.TRICHION,
    L.NASION,
    L.ORBITALE_L,
    L.ORBITALE_R,
    L.SUBALARE_L,
    L.SUBALARE_R,
    L.COLUMELLA,
    L.STOMION,
    L.SUBLABIALE,
    L.POGONION,
    L.GNATHION,
    L.GONION_L,
    L.GONION_R,
    L.ZYGION_L,
    L.ZYGION_R,
    L.TRAGION_L,
    L.TRAGION_R,
    L.PORION_L,
    L.PORION_R,
    L.CERVICALE,
)


class _MissingDependency(RuntimeError):
    pass


def _import_mediapipe() -> tuple[Any, Any, Any]:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise _MissingDependency(
            "mediapipe is not installed. Install it with "
            "`uv pip install 'mediapipe>=0.10.31,<1'`. Version 1.0.x aborts the "
            "process on macOS arm64 inside the face detector subgraph, so the "
            "upper bound is load-bearing rather than caution."
        ) from exc
    version = getattr(mp, "__version__", "unknown")
    if version.startswith("1."):
        raise _MissingDependency(
            f"mediapipe {version} aborts the process on macOS arm64 when the face "
            "landmarker graph opens (Check failed: service_ Service is unavailable). "
            "It is a hard abort and cannot be caught at the call site, so it is "
            "refused here. Install 0.10.31 or later in the 0.10 line."
        )
    return mp, mp_python, vision


class MediaPipeDenseLandmarker:
    """478-vertex mesh with iris refinement.

    The task runner holds native state and is not thread-safe, so one instance
    per thread. It is closed explicitly rather than left to the garbage
    collector because the underlying graph owns a thread pool.
    """

    provenance = PROVENANCE

    def __init__(
        self,
        *,
        allowed_tier: Tier = Tier.PERMISSIVE,
        weights_path: Path | None = None,
        min_detection_confidence: float = 0.5,
        num_faces: int = 1,
    ) -> None:
        licensing.require(PROVENANCE, allowed_tier)

        mp, mp_python, vision = _import_mediapipe()
        self._mp = mp
        self._path = weights_path or weights.resolve(WEIGHT_KEY)
        self._sha256 = weights.spec_for(WEIGHT_KEY).sha256
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self._path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=int(num_faces),
            min_face_detection_confidence=float(min_detection_confidence),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._runner = vision.FaceLandmarker.create_from_options(options)

    @property
    def weights_path(self) -> Path:
        return self._path

    @property
    def weights_sha256(self) -> str:
        return self._sha256

    @property
    def index_map(self) -> Mapping[Landmark, int]:
        return dict(MESH_INDEX)

    def close(self) -> None:
        self._runner.close()

    def __enter__(self) -> MediaPipeDenseLandmarker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def measure_iris(self, image: Image, detection: object) -> DenseResult:
        """The `pipeline.ports.IrisMeasurer` entry point.

        `DenseResult` already exposes ``iris_diameter_px_l`` and
        ``iris_diameter_px_r``, so the whole mesh is what comes back rather than
        a two-field wrapper. The pipeline reads the two fields it declared and
        anything richer that wants the mesh has it without a second inference.
        """
        return self.locate(image, detection)

    def locate(self, image: Image, box: object | None = None) -> DenseResult:
        """Mesh for the face nearest ``box``, or the only face if ``box`` is None.

        MediaPipe runs its own detector, so the box is used to *select* among
        its detections rather than to crop. Cropping first would change the
        scale the mesh model sees and the iris refinement is the part that
        would suffer, which is the one measurement this backend exists for.
        """
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected an (h, w, 3) RGB image, got shape {img.shape}")
        h, w = img.shape[:2]

        # MediaPipe wants RGB, which is what the pipeline already hands over.
        frame = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)
        )
        result = self._runner.detect(frame)
        if not result.face_landmarks:
            raise RuntimeError(
                "MediaPipe found no face. The detector that produced the box and "
                "this one disagree, which is worth surfacing rather than papering "
                "over: scale recovery has no fallback that does not involve a "
                "population prior."
            )

        meshes = [
            np.array([[p.x * w, p.y * h] for p in face], dtype=np.float64)
            for face in result.face_landmarks
        ]
        which = 0
        if box is not None and len(meshes) > 1:
            centre = FaceBox.from_detection(box, PROVENANCE).centre
            which = int(np.argmin([np.linalg.norm(m.mean(axis=0) - centre) for m in meshes]))
        mesh = meshes[which]
        if mesh.shape[0] != N_POINTS:
            raise RuntimeError(
                f"expected {N_POINTS} vertices with iris refinement, got {mesh.shape[0]}. "
                "The iris points are the reason this backend is loaded at all, so a "
                "mesh without them is an error rather than a degraded result."
            )

        transforms = getattr(result, "facial_transformation_matrixes", None)
        transform = (
            np.asarray(transforms[which], dtype=np.float64)
            if transforms
            else np.eye(4, dtype=np.float64)
        )

        index = {name: i for i, name in enumerate(MESH_INDEX)}
        coords = np.stack([mesh[MESH_INDEX[name]] for name in index], axis=0)

        return DenseResult(
            points=PointSet(index=index, coords=coords),
            all_points_px=mesh,
            iris_right_px=mesh[list(IRIS_RIGHT)],
            iris_left_px=mesh[list(IRIS_LEFT)],
            transform=transform,
            provenance=PROVENANCE,
            unsupported=UNSUPPORTED,
        )


def scale_from_dense(result: DenseResult) -> ScaleEstimate:
    """Millimetres per pixel from both irides.

    Both eyes are measured and fused rather than averaged in pixel space. They
    are two observations of the same prior read by the same model on the same
    image, so `core.scale.fuse` combines them with an explicit correlation
    term; averaging the diameters first would produce a narrower interval than
    the evidence supports, which is the exact error this pipeline is built to
    avoid making.
    """
    right, left = result.iris_diameter_px
    if right <= 0 or left <= 0:
        raise ValueError("iris ring has zero horizontal extent, so no scale can be read")
    return fuse(from_iris(right), from_iris(left))


def build(
    *, allowed_tier: Tier = Tier.PERMISSIVE, **kwargs: object
) -> MediaPipeDenseLandmarker:
    return MediaPipeDenseLandmarker(allowed_tier=allowed_tier, **kwargs)  # type: ignore[arg-type]
