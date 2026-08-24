"""YuNet face detection, through OpenCV's own ONNX runtime.

YuNet is here rather than a YOLO face detector for one reason and it is a
licensing reason: every third-party "yolov8-face" checkpoint descends from the
Ultralytics trainer, which asserts AGPL-3.0 over the models it produces, and
the permissive tag those checkpoints carry does not launder that. YuNet is MIT
end to end, under a megabyte, and reports WIDER Face val AP 0.834 / 0.824 /
0.708 -- easy on the easy set and respectable on the hard one, which is the
regime a standardised portrait sits in anyway.

It is also the only backend Faciometry loads that needs no PyTorch. `cv2.dnn`
reads the ONNX directly, so face detection stays available on a machine where
the torch install is broken, which matters because the detector is what tells
the user "no face in this photograph" rather than a stack trace.

**Keypoint order and the left/right trap.** `cv2.FaceDetectorYN` returns fifteen
numbers per face: ``x, y, w, h``, then five keypoints as ``(x, y)`` pairs in the
order right eye, left eye, nose tip, right mouth corner, left mouth corner, then
the score. Those names are *subject-relative*, so in a mirror-free frontal
photograph the "right eye" keypoint carries the smaller image x. That was
checked rather than assumed: on a frontal portrait, YuNet's first keypoint,
SPIGA's landmark 96 and MediaPipe's iris centre 468 all land within two pixels
of each other on the same physical eye. Three independently trained models
agreeing on which eye is which is the only evidence worth having here, because
a transposition produces a perfectly plausible face either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import licensing, weights
from .licensing import Provenance, Tier
from .protocols import FaceBox, Image

#: Key in ``assets/weights.lock.json``.
WEIGHT_KEY = "yunet"

PROVENANCE: Provenance = licensing.YUNET

#: Column layout of one row of `cv2.FaceDetectorYN.detect`'s output.
_BOX = slice(0, 4)
_KEYPOINTS = slice(4, 14)
_SCORE = 14


class YuNetDetector:
    """Detects faces and their five coarse keypoints.

    The input size handed to OpenCV must match the image, and OpenCV silently
    produces nonsense boxes if it does not, so it is reset per image rather
    than fixed at construction.
    """

    provenance = PROVENANCE

    def __init__(
        self,
        *,
        allowed_tier: Tier = Tier.PERMISSIVE,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        weights_path: Path | None = None,
    ) -> None:
        # Before anything is read from disk: a tier refusal must not be
        # reachable only after a 200 MB load has already happened.
        licensing.require(PROVENANCE, allowed_tier)

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "YuNet needs opencv-python, which is not installed. "
                "Install it with `uv pip install opencv-python`."
            ) from exc

        self._cv2 = cv2
        self._score_threshold = float(score_threshold)
        self._path = weights_path or weights.resolve(WEIGHT_KEY)
        self._sha256 = weights.spec_for(WEIGHT_KEY).sha256
        self._detector = cv2.FaceDetectorYN.create(
            str(self._path),
            "",
            (320, 320),
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
        )

    @property
    def weights_path(self) -> Path:
        return self._path

    @property
    def weights_sha256(self) -> str:
        """The pin the manifest records, so a report names its own weights."""
        return self._sha256

    def detect(self, image: Image) -> tuple[FaceBox, ...]:
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected an (h, w, 3) RGB image, got shape {img.shape}")
        h, w = img.shape[:2]
        # YuNet was trained through OpenCV, so it expects BGR. The pipeline
        # hands out RGB. One reversal here beats a convention everyone remembers
        # differently.
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        self._detector.setInputSize((int(w), int(h)))
        _, raw = self._detector.detect(bgr)
        if raw is None:
            return ()

        boxes: list[FaceBox] = []
        for row in np.asarray(raw, dtype=np.float64):
            x, y, bw, bh = row[_BOX]
            if bw <= 0 or bh <= 0:
                # OpenCV occasionally emits a degenerate row at the edge of the
                # frame. Dropping it here keeps FaceBox's invariant meaningful.
                continue
            kp = row[_KEYPOINTS].reshape(5, 2)
            boxes.append(
                FaceBox(
                    x=float(x),
                    y=float(y),
                    w=float(bw),
                    h=float(bh),
                    score=float(row[_SCORE]),
                    right_eye=kp[0].copy(),
                    left_eye=kp[1].copy(),
                    nose=kp[2].copy(),
                    right_mouth=kp[3].copy(),
                    left_mouth=kp[4].copy(),
                    provenance=PROVENANCE,
                )
            )
        boxes.sort(key=lambda b: b.score, reverse=True)
        return tuple(boxes)

    def detect_largest(self, image: Image) -> FaceBox | None:
        """The biggest face, which is the subject in a portrait.

        Largest rather than highest-scoring: a sharp bystander in the
        background can outscore a slightly soft foreground subject, and a
        morphometric report on the wrong person is worse than no report.
        """
        found = self.detect(image)
        if not found:
            return None
        return max(found, key=lambda b: b.area)


def build(*, allowed_tier: Tier = Tier.PERMISSIVE, **kwargs: object) -> YuNetDetector:
    return YuNetDetector(allowed_tier=allowed_tier, **kwargs)  # type: ignore[arg-type]
