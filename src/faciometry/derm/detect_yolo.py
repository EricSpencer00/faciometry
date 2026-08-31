"""Acne lesion detection with YOLO, over region crops rather than whole faces.

This is the one place in Faciometry where YOLO is the right tool rather than the
familiar one. Acne is genuinely multi-instance: a face carries anywhere from
zero to a hundred separate lesions, each a few pixels across, each needing its
own box and its own confidence. That is the problem class one-stage detectors
were designed for, and it is a poor fit for anything else in this pipeline.
Dense landmarking is not detection and is served better by heatmaps, which is
why YOLO does not appear there.

Two structural decisions:

**Crop, then detect.** A 4000 px selfie fed to a 640 px network is downsampled
by six, and a 12 px papule becomes two pixels. Running the detector over region
crops at native resolution keeps the lesion inside the range of scales the model
was trained on, and it also means every detection arrives already attributed to
a named region, which is what the findings layer needs. Overlapping tiles are
merged afterwards by IoU, because a lesion on a tile boundary would otherwise be
counted twice.

**The licence is checked before the import.** Ultralytics asserts AGPL-3.0 over
models produced by its training code, not only over the code, so every acne
checkpoint trained with the Ultralytics trainer carries that obligation whatever
its model card says. The detector therefore sits at :attr:`Tier.COPYLEFT` and
:func:`licensing.require` runs before ``import ultralytics`` and before any
weight is opened. A check that runs afterwards has not prevented anything.

The severity mapping is Hayashi's. Hayashi et al. (2008, *Journal of
Dermatology*) graded acne by counting **inflammatory lesions on one half of the
face**: 0 to 5 mild, 6 to 20 moderate, 21 to 50 severe, more than 50 very
severe. ACNE04 grades to that convention, so a detector fine-tuned on ACNE04 or
on a set labelled to match it inherits the convention along with the weights,
including its definition of what counts as a countable lesion. That is a
property of the training data and not of this code, so it is recorded in the
detector's model card and reprinted with every severity output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.spec import Reportability, Verdict
from ..models.licensing import YOLO_DERM_SEG, Provenance, Tier, require
from .regions import Region, RegionSet

#: Regions a lesion detector is run over. The periorbital regions are excluded:
#: eyelashes and lid margins generate false positives at the rate that matters,
#: and acne does not occur on the eyelid in any quantity worth the noise.
DETECTION_REGIONS: tuple[Region, ...] = (
    Region.FOREHEAD,
    Region.GLABELLA,
    Region.NASAL,
    Region.MALAR_L,
    Region.MALAR_R,
    Region.LATERAL_CHEEK_L,
    Region.LATERAL_CHEEK_R,
    Region.PERIORAL,
)

#: Side of the square the detector sees, in pixels.
TILE_PX = 640

#: Fraction of a tile shared with its neighbour, so a lesion straddling a seam
#: is fully visible in at least one tile.
TILE_OVERLAP = 0.25

#: IoU above which two detections in full-image coordinates are one lesion.
MERGE_IOU = 0.4


class Severity(str, Enum):
    """Hayashi grade. Ordered, and defined by a count, not by an opinion."""

    CLEAR = "clear"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    VERY_SEVERE = "very_severe"


#: Upper bounds on the inflammatory lesion count per half face, Hayashi 2008.
HAYASHI_BOUNDS: tuple[tuple[int, Severity], ...] = (
    (0, Severity.CLEAR),
    (5, Severity.MILD),
    (20, Severity.MODERATE),
    (50, Severity.SEVERE),
)

HAYASHI_NOTE = (
    "severity follows Hayashi et al. (2008): inflammatory lesions counted on one "
    "half of the face, 0-5 mild, 6-20 moderate, 21-50 severe, above 50 very "
    "severe. The count comes from a detector whose idea of a countable lesion is "
    "inherited from its training set's labelling convention, so the grade is "
    "calibrated on that convention and not on a clinician's examination"
)

#: Why the graded count is the quantity Hayashi defined, when a midline exists.
MEASURED_HALF_NOTE = "the larger of the two half-face counts, split at the facial midline"

#: Why it is not, when no midline exists. Half the whole-face count rounded up
#: is the least misleading estimator available, but it estimates the quantity
#: the grade is defined on rather than measuring it, so a grade built on it is
#: a caveated grade.
ESTIMATED_HALF_NOTE = (
    "half the whole-face count rounded up, because no facial midline was "
    "supplied; Hayashi grades one half face, so this is an estimate of the "
    "quantity the grade is defined on rather than the quantity itself"
)


def hayashi_severity(count_per_half_face: int) -> Severity:
    """Map an inflammatory lesion count on one half face to a Hayashi grade."""
    if count_per_half_face < 0:
        raise ValueError("lesion count cannot be negative")
    for bound, severity in HAYASHI_BOUNDS:
        if count_per_half_face <= bound:
            return severity
    return Severity.VERY_SEVERE


# ---------------------------------------------------------------------------
# Geometry helpers, kept free of any model dependency so they are testable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """A square crop of the source image, in full-image pixel coordinates."""

    region: Region
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, image: NDArray) -> NDArray:
        return image[self.y0 : self.y1, self.x0 : self.x1]


def tiles_for_region(
    region: Region,
    bounds: tuple[float, float, float, float],
    height: int,
    width: int,
    *,
    tile_px: int = TILE_PX,
    overlap: float = TILE_OVERLAP,
) -> tuple[Tile, ...]:
    """Cover a region's bounding box with overlapping square tiles.

    ``bounds`` is in *array* coordinates (row-major, y down), which is what the
    caller gets after flipping a canonical-frame polygon. A region smaller than
    one tile yields a single tile centred on it, padded out to the full tile
    size when the image allows, so the detector always sees its trained input
    scale rather than a letterboxed sliver.
    """
    x_lo, y_lo, x_hi, y_hi = bounds
    x_lo, y_lo = max(int(math.floor(x_lo)), 0), max(int(math.floor(y_lo)), 0)
    x_hi = min(int(math.ceil(x_hi)), width)
    y_hi = min(int(math.ceil(y_hi)), height)
    if x_hi <= x_lo or y_hi <= y_lo:
        return ()

    step = max(int(tile_px * (1.0 - overlap)), 1)
    xs = list(range(x_lo, max(x_hi - tile_px, x_lo) + 1, step))
    ys = list(range(y_lo, max(y_hi - tile_px, y_lo) + 1, step))
    out: list[Tile] = []
    for y in ys:
        for x in xs:
            x0 = min(x, max(width - tile_px, 0))
            y0 = min(y, max(height - tile_px, 0))
            out.append(
                Tile(region, x0, y0, min(x0 + tile_px, width), min(y0 + tile_px, height))
            )
    # Dedupe: small regions produce the same clamped tile from several offsets.
    seen: dict[tuple[int, int, int, int], Tile] = {}
    for t in out:
        seen.setdefault((t.x0, t.y0, t.x1, t.y1), t)
    return tuple(seen.values())


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lesion:
    """One detected lesion instance, in full-image pixel coordinates."""

    region: Region
    box: tuple[float, float, float, float]
    confidence: float
    lesion_class: str = "inflammatory"
    #: Instance mask contour, when the model is a segmentation model. Area from
    #: a mask is a far better size estimate than area from a box, and box area
    #: overstates a round lesion by about a quarter.
    contour: NDArray[np.float64] | None = None

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def box_area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)

    @property
    def equivalent_diameter(self) -> float:
        return 2.0 * math.sqrt(max(self.box_area, 0.0) / math.pi)


def merge_duplicates(
    lesions: Iterable[Lesion], *, threshold: float = MERGE_IOU
) -> tuple[Lesion, ...]:
    """Greedy IoU suppression over the union of all tiles.

    Overlapping tiles mean the same papule is detected twice, and a count is the
    output here, so a double count is not a cosmetic problem. Suppression runs
    in full-image coordinates across tiles and across regions, keeping the most
    confident instance and its region attribution.
    """
    ordered = sorted(lesions, key=lambda d: d.confidence, reverse=True)
    kept: list[Lesion] = []
    for cand in ordered:
        if any(iou(cand.box, k.box) >= threshold for k in kept):
            continue
        kept.append(cand)
    return tuple(kept)


@dataclass(frozen=True)
class RegionLesions:
    """Lesions attributed to one region, with the count that grades it."""

    region: Region
    lesions: tuple[Lesion, ...]
    tiles: int
    area_px: float

    @property
    def count(self) -> int:
        return len(self.lesions)

    @property
    def density_per_kpx(self) -> float:
        """Lesions per thousand square pixels, so regions are comparable."""
        return 1000.0 * self.count / self.area_px if self.area_px > 0 else 0.0


@dataclass(frozen=True)
class Grade:
    """A Hayashi grade with the verdict that gates it.

    The grade and the count it came from travel together with the reasons, so
    no caller can print a severity without the statement of what it was counted
    on. ``estimated`` is true when no facial midline was available and the
    count is half the whole-face total rather than one half face.
    """

    severity: Severity
    half_face_count: int
    estimated: bool
    verdict: Verdict

    @property
    def reasons(self) -> tuple[str, ...]:
        return self.verdict.reasons


@dataclass(frozen=True)
class AcneDetection:
    """Everything one detector pass produced, plus what it cost in obligations."""

    per_region: dict[Region, RegionLesions]
    provenance: Provenance
    confidence_threshold: float
    model_card: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def lesions(self) -> tuple[Lesion, ...]:
        return tuple(l for r in self.per_region.values() for l in r.lesions)

    @property
    def total_count(self) -> int:
        return sum(r.count for r in self.per_region.values())

    def count_on(self, side: str, midline_x: float) -> int:
        """Lesions on one half of the face, which is what Hayashi actually grades.

        Every lesion is assigned by its own centre against the facial midline,
        including those in midline regions: a forehead spans both halves, so
        attributing the whole forehead to one side would double the count that
        the grade is defined on. ``midline_x`` is ``regions.frame.origin[0]``,
        and since +x is the subject's right, a centre to the right of it is a
        right-side lesion.
        """
        if side not in ("l", "r"):
            raise ValueError("side must be 'l' or 'r'")
        want_right = side == "r"
        return sum(
            1
            for lesion in self.lesions
            if (lesion.centre[0] >= midline_x) == want_right
        )

    def hayashi_count(self, midline_x: float) -> int:
        """The larger of the two half-face counts, which is the graded one."""
        return max(self.count_on("l", midline_x), self.count_on("r", midline_x))

    def severity(self, *, half_face_count: int | None = None) -> Grade:
        """Hayashi grade, carrying the verdict that says what it rests on.

        Without a midline to split on, half the total is the least misleading
        estimator, and rounding up keeps the grade from being flattered by a
        face whose lesions all sit on one side. That substitution is an
        estimate and not the graded quantity, so it arrives as a caveat on the
        verdict. A caller cannot otherwise tell it apart from a count taken on
        a real half face.
        """
        estimated = half_face_count is None
        if half_face_count is None:
            half_face_count = int(math.ceil(self.total_count / 2.0))
        basis = ESTIMATED_HALF_NOTE if estimated else MEASURED_HALF_NOTE
        return Grade(
            severity=hayashi_severity(half_face_count),
            half_face_count=half_face_count,
            estimated=estimated,
            verdict=Verdict(
                Reportability.CAVEAT,
                (
                    (Reportability.CAVEAT, HAYASHI_NOTE),
                    (Reportability.CAVEAT, basis),
                ),
            ),
        )


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


class DetectorUnavailable(RuntimeError):
    """Raised when weights are absent. Distinct from a licence refusal."""


@dataclass
class AcneDetector:
    """YOLO instance segmentation over region crops.

    The constructor does not load anything. :meth:`load` checks the licence
    first, then imports ultralytics, then opens the weights, in that order.
    """

    weights: str
    allowed_tier: Tier = Tier.PERMISSIVE
    provenance: Provenance = YOLO_DERM_SEG
    confidence: float = 0.25
    tile_px: int = TILE_PX
    overlap: float = TILE_OVERLAP
    device: str = "cpu"
    model_card: dict[str, Any] = field(default_factory=dict)
    _model: Any = field(default=None, repr=False, compare=False)

    def load(self) -> "AcneDetector":
        """Check the tier, then import, then load. The order is the point."""
        require(self.provenance, self.allowed_tier)
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional extra
            raise DetectorUnavailable(
                "acne detection needs `ultralytics`, which is AGPL-3.0; install "
                "the copyleft extra and re-run with --license-tier copyleft"
            ) from exc
        import os

        if not os.path.exists(self.weights):
            raise DetectorUnavailable(
                f"no weights at {self.weights}; train them with "
                "`python -m faciometry.derm.train` against a CC BY 4.0 acne set"
            )
        self._model = YOLO(self.weights)
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _predict_tile(self, crop: NDArray) -> list[tuple[tuple[float, float, float, float], float, str, NDArray | None]]:
        """Run the model on one crop. Isolated so it can be stubbed in tests."""
        if self._model is None:  # pragma: no cover - guarded by detect()
            raise DetectorUnavailable("detector is not loaded")
        results = self._model.predict(
            crop, conf=self.confidence, device=self.device, verbose=False
        )
        out = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            if boxes is None:
                continue
            xyxy = np.asarray(boxes.xyxy)
            conf = np.asarray(boxes.conf).reshape(-1)
            cls = np.asarray(boxes.cls).reshape(-1).astype(int)
            contours = list(getattr(masks, "xy", []) or []) if masks is not None else []
            for i in range(xyxy.shape[0]):
                contour = np.asarray(contours[i], dtype=float) if i < len(contours) else None
                out.append(
                    (
                        (float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3])),
                        float(conf[i]),
                        str(names.get(int(cls[i]), "inflammatory")),
                        contour,
                    )
                )
        return out

    def detect(
        self,
        image: NDArray,
        regions: RegionSet,
        *,
        only: Sequence[Region] = DETECTION_REGIONS,
        y_up: bool = True,
    ) -> AcneDetection:
        """Detect lesions inside every requested region.

        A detection is kept only when its centre falls inside the region polygon
        it came from. The tile is a rectangle and the region is not, so without
        that test a forehead tile would claim lesions found in the hair.
        """
        if not self.loaded:
            raise DetectorUnavailable("call load() before detect()")
        arr = np.asarray(image)
        height, width = arr.shape[:2]
        per_region: dict[Region, RegionLesions] = {}

        for region in only:
            if region not in regions:
                continue
            poly = regions.get(region)
            x0, y_lo, x1, y_hi = poly.bounds
            if y_up:
                row_lo, row_hi = height - 1 - y_hi, height - 1 - y_lo
            else:
                row_lo, row_hi = y_lo, y_hi
            tiles = tiles_for_region(
                region,
                (x0, row_lo, x1, row_hi),
                height,
                width,
                tile_px=self.tile_px,
                overlap=self.overlap,
            )
            found: list[Lesion] = []
            for tile in tiles:
                for box, conf, name, contour in self._predict_tile(tile.crop(arr)):
                    full = (
                        box[0] + tile.x0,
                        box[1] + tile.y0,
                        box[2] + tile.x0,
                        box[3] + tile.y0,
                    )
                    cx = 0.5 * (full[0] + full[2])
                    cy_row = 0.5 * (full[1] + full[3])
                    cy = (height - 1 - cy_row) if y_up else cy_row
                    if not bool(poly.contains(np.array([cx, cy]))):
                        continue
                    if contour is not None:
                        contour = contour + np.array([tile.x0, tile.y0], dtype=float)
                    found.append(
                        Lesion(
                            region=region,
                            box=full,
                            confidence=conf,
                            lesion_class=name,
                            contour=contour,
                        )
                    )
            per_region[region] = RegionLesions(
                region=region,
                lesions=merge_duplicates(found),
                tiles=len(tiles),
                area_px=poly.area,
            )

        return AcneDetection(
            per_region=per_region,
            provenance=self.provenance,
            confidence_threshold=self.confidence,
            model_card=dict(self.model_card),
            notes=(
                self.provenance.describe(),
                f"detection ran on {self.tile_px} px crops at native resolution; "
                "whole-face inference at this model's input size would downsample "
                "a typical lesion below the scale it was trained on",
                "recall and precision on this subject's skin tone are unmeasured "
                "unless the model card records a stratified evaluation",
            ),
        )


__all__ = [
    "DETECTION_REGIONS",
    "TILE_PX",
    "TILE_OVERLAP",
    "MERGE_IOU",
    "HAYASHI_BOUNDS",
    "HAYASHI_NOTE",
    "MEASURED_HALF_NOTE",
    "ESTIMATED_HALF_NOTE",
    "Grade",
    "Severity",
    "Tile",
    "Lesion",
    "RegionLesions",
    "AcneDetection",
    "AcneDetector",
    "DetectorUnavailable",
    "hayashi_severity",
    "tiles_for_region",
    "iou",
    "merge_duplicates",
]
