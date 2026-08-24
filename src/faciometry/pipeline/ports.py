"""What the pipeline needs from a model backend, stated as narrowly as possible.

The model layer and the pipeline layer are written independently, so the
dependency between them is expressed structurally rather than nominally: the
pipeline declares the *shape* of what it consumes and never imports a concrete
backend class. Three things fall out of that choice.

* The whole orchestration layer is testable with a dozen lines of fake
  implementation and no weights, no torch, and no network. That is not a
  convenience, it is the only way the end-to-end test can be part of the
  ordinary test run.
* A backend that grows an extra field, or that returns its own richer result
  type, keeps working. Only removing something the pipeline names can break it.
* The seam is documented in one file. When a member below turns out to already
  exist under a different name in ``faciometry.models.protocols``, deleting the
  local ``Protocol`` and importing the real one is a one-line change that the
  type checker will verify.

The five detector keypoints deserve their own paragraph because getting them
backwards silently mirrors every lateralised finding in the report. They follow
the RetinaFace/ArcFace convention, and the left/right labels are the
**subject's**, matching :mod:`faciometry.core.landmarks`. In an unmirrored frontal
photograph the subject's right eye appears at the *smaller* image x coordinate,
so ``keypoints[0]`` is the leftmost eye on screen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..core.landmarks import PointSet
from ..models.licensing import Provenance, Tier, require

#: An image as the pipeline passes it around: HxWx3, uint8, RGB, EXIF already
#: discarded by the ingest stage.
Image = NDArray[np.uint8]

#: Row order of the five detector keypoints. ``_l`` and ``_r`` are the
#: subject's left and right, so in an unmirrored frontal photograph ``eye_r``
#: is the eye nearer the left edge of the frame.
KEYPOINT_ORDER: tuple[str, ...] = ("eye_r", "eye_l", "nose", "mouth_r", "mouth_l")


@runtime_checkable
class Detection(Protocol):
    """One detected face.

    ``bbox`` is ``(x, y, width, height)`` in pixels of the *original* image,
    not of any crop. ``keypoints`` is ``(5, 2)`` in the same coordinates and in
    :data:`KEYPOINT_ORDER`.
    """

    bbox: tuple[float, float, float, float]
    score: float
    keypoints: NDArray[np.float64]


@runtime_checkable
class LandmarkSet(Protocol):
    """Dense landmarks in original-image pixel coordinates, with covariances.

    ``points`` carries the name-to-index map; ``covariances`` is
    ``(n_landmarks, 2, 2)`` indexed the same way, in squared pixels. A backend
    that only regresses coordinates has no honest covariance to offer and
    should say so by supplying an isotropic fallback explicitly rather than by
    returning zeros, because a zero covariance reads downstream as a perfectly
    known landmark.

    The pose fields are optional. When a landmarker reports its own pose it
    becomes one of the two independent sources the quality gate compares; when
    it does not, the gate falls back to the interocular axis for roll and says
    so.
    """

    points: PointSet
    covariances: NDArray[np.float64]
    yaw_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None


@runtime_checkable
class HeadPose(Protocol):
    """Head pose in degrees, in the canonical frame.

    Sign conventions, stated operationally so that no handedness argument is
    needed to check them:

    * ``roll_deg`` is positive when the subject's right eye sits *above* the
      subject's left eye in the photograph.
    * ``yaw_deg`` is positive when the subject's face turns toward the
      subject's own left, so that the subject's right cheek becomes more
      visible.
    * ``pitch_deg`` is positive when the chin rises.
    """

    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@runtime_checkable
class IrisMeasurement(Protocol):
    """Horizontal visible iris diameter per eye, in pixels of the original image.

    ``None`` on a side means the backend could not measure it, which is a
    different statement from zero and is the difference between a scale cue
    being fused and being skipped.
    """

    iris_diameter_px_l: float | None
    iris_diameter_px_r: float | None


@runtime_checkable
class Backend(Protocol):
    """Anything the manifest has to account for."""

    provenance: Provenance
    weights_sha256: str | None


@runtime_checkable
class Detector(Backend, Protocol):
    def detect(self, image: Image) -> Sequence[Detection]: ...


@runtime_checkable
class Landmarker(Backend, Protocol):
    def locate(self, image: Image, detection: Detection) -> LandmarkSet: ...


@runtime_checkable
class PoseEstimator(Backend, Protocol):
    def estimate(self, image: Image, detection: Detection) -> HeadPose: ...


@runtime_checkable
class IrisMeasurer(Backend, Protocol):
    def measure_iris(self, image: Image, detection: Detection) -> IrisMeasurement | None: ...


@dataclass(frozen=True)
class Backends:
    """The set of backends one run uses, bundled so a run is reproducible.

    ``pose_estimator`` and ``iris`` are optional and their absence degrades the
    report rather than breaking it: without the independent pose estimator
    there is nothing to cross-check the landmarker against, and the quality
    report records that the cross-check was not available instead of quietly
    reporting agreement.
    """

    detector: Detector
    landmarker: Landmarker
    pose_estimator: PoseEstimator | None = None
    iris: IrisMeasurer | None = None
    device: str = "cpu"

    def all(self) -> tuple[tuple[str, Backend], ...]:
        """Every backend with the role it plays, for the manifest."""
        roles = (
            ("detector", self.detector),
            ("landmarker", self.landmarker),
            ("pose_estimator", self.pose_estimator),
            ("iris", self.iris),
        )
        return tuple((role, b) for role, b in roles if b is not None)

    def check_licences(self, allowed: Tier) -> None:
        """Re-assert the licence tier over an already-constructed bundle.

        The real enforcement point is inside the model layer, before weights
        are read off disk, because a check that runs after loading has already
        let the obligation attach. This is a second, cheap assertion that the
        bundle handed to :func:`faciometry.pipeline.run.analyze` is the bundle the
        caller's tier permits, and it exists so that a caller assembling
        backends by hand cannot bypass the tier by accident.
        """
        for _role, backend in self.all():
            require(backend.provenance, allowed)


def load_backends(*, tier: Tier, device: str | None = None) -> Backends:
    """Build the default backend bundle from :mod:`faciometry.models`.

    Imported lazily and inside the function on purpose. ``core``, ``measure``
    and ``norms`` carry no torch dependency, and the pipeline is only as heavy
    as the backends actually asked for; importing the model registry at module
    scope would drag torch into every test that only wanted to check a formula.
    """
    try:
        from ..models import registry as model_registry
    except ImportError as exc:  # pragma: no cover - depends on the model layer
        raise RuntimeError(
            "no model backends are installed. Faciometry's measurement layer runs "
            "without them, but an analysis needs a detector and a landmarker; "
            "install the model extras or pass backends= explicitly."
        ) from exc
    return model_registry.default_backends(tier=tier, device=device)  # type: ignore[no-any-return]
