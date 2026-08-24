"""Getting pixels off disk, and getting the metadata off the pixels.

Two jobs that pull in opposite directions. EXIF is the only place a photograph
volunteers its own optics -- focal length, and occasionally the focused subject
distance -- and those are exactly what the perspective-distortion warning needs.
EXIF is also where the GPS coordinates live. So this module reads the tags it
needs, keeps four numbers, and drops the block; every image the pipeline writes
back out is re-encoded from a bare pixel array, so there is no path by which an
input's metadata reaches an output.

The image identity recorded in the manifest is a sha256 of the *decoded,
orientation-normalised pixel buffer*, not of the file. Hashing the file would
mean that stripping EXIF changed the identity of the thing that was measured,
which is backwards: the measurement is a function of the pixels.

Subject distance is estimated rather than assumed. The pinhole relation needs a
focal length in the same units as the sensor, and the only EXIF tag that gives
that without a sensor-size database is ``FocalLengthIn35mmFilm``. When it is
absent the honest answer is ``None``. A guessed distance would feed straight
into the ICAO magnification warning and into the near-fixation correction on the
interpupillary prior, so a wrong guess is worse than no guess.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import ExifTags, ImageOps
from PIL import Image as PILImage

from ..core.scale import IPD_PRIORS
from ..core.spec import View

#: Diagonal of a 36x24 mm frame, the reference the ``FocalLengthIn35mmFilm``
#: tag is defined against.
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

#: Range of subject distances the EXIF tag is believed within. Phone cameras
#: write 0 for "unknown" and occasionally 65535 for "infinity", and a
#: focus-derived distance beyond a few metres is not a portrait anyway.
_PLAUSIBLE_DISTANCE_M = (0.1, 20.0)

_EXIF_IFD = ExifTags.IFD.Exif.value
_GPS_IFD = ExifTags.IFD.GPSInfo.value


def _rational(value: object) -> float | None:
    """Coerce an EXIF rational to a float, tolerating every form Pillow returns.

    Pillow hands back an ``IFDRational`` normally, a bare ``int`` for the
    short-typed tags, and a raw ``(numerator, denominator)`` tuple for images
    whose IFD it could not fully parse. All three appear in real files.
    """
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 2 or not value[1]:
            return None
        return float(value[0]) / float(value[1])
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@dataclass(frozen=True)
class ExifFacts:
    """The handful of EXIF values the pipeline is allowed to keep.

    ``tag_names_seen`` records tag *names* and never their values. It exists so
    that the manifest can say what the file contained without becoming a copy
    of it, which matters most for the tags this class deliberately does not
    read: a run manifest that listed GPS coordinates would defeat the point of
    stripping them.
    """

    focal_length_mm: float | None = None
    focal_length_35mm: float | None = None
    subject_distance_m: float | None = None
    orientation: int | None = None
    camera: str | None = None
    had_gps: bool = False
    tag_names_seen: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.tag_names_seen


def read_exif(img: PILImage.Image) -> ExifFacts:
    """Extract the optics tags, note what else was there, keep none of it."""
    try:
        exif = img.getexif()
    except Exception:  # pragma: no cover - corrupt metadata is not a read error
        return ExifFacts()
    if not exif:
        return ExifFacts()

    try:
        sub = dict(exif.get_ifd(_EXIF_IFD))
    except Exception:  # pragma: no cover
        sub = {}
    try:
        gps = dict(exif.get_ifd(_GPS_IFD))
    except Exception:  # pragma: no cover
        gps = {}

    names: list[str] = []
    for tag_id in list(exif.keys()) + list(sub.keys()):
        names.append(ExifTags.TAGS.get(tag_id, f"unknown_{tag_id}"))
    for tag_id in gps:
        # GPSTAGS names already carry the "GPS" prefix; adding another produces
        # "GPSGPSLatitude", which no reader will match against.
        names.append(ExifTags.GPSTAGS.get(tag_id, f"GPS_unknown_{tag_id}"))

    distance = _rational(sub.get(ExifTags.Base.SubjectDistance.value))
    lo, hi = _PLAUSIBLE_DISTANCE_M
    if distance is not None and not lo <= distance <= hi:
        # 0 means "unknown" and the saturated value means "infinity"; both are
        # written by real cameras and neither is a portrait distance.
        distance = None

    make = exif.get(ExifTags.Base.Make.value)
    model = exif.get(ExifTags.Base.Model.value)
    camera = " ".join(str(x).strip() for x in (make, model) if x) or None

    return ExifFacts(
        focal_length_mm=_rational(sub.get(ExifTags.Base.FocalLength.value)),
        focal_length_35mm=_rational(sub.get(ExifTags.Base.FocalLengthIn35mmFilm.value)),
        subject_distance_m=distance,
        orientation=(
            int(exif[ExifTags.Base.Orientation.value])
            if ExifTags.Base.Orientation.value in exif
            else None
        ),
        camera=camera,
        had_gps=bool(gps),
        tag_names_seen=tuple(sorted(set(names))),
    )


@dataclass(frozen=True)
class SourceImage:
    """Decoded pixels plus the little that is known about how they were made."""

    path: Path | None
    view: View
    pixels: NDArray[np.uint8]
    sha256: str
    exif: ExifFacts

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def diagonal_px(self) -> float:
        return math.hypot(self.width, self.height)


def pixel_sha256(pixels: NDArray[np.uint8]) -> str:
    """Content hash of a pixel buffer, shape and dtype included.

    Shape goes into the hash because two different images can share a byte
    sequence under different shapes, and the manifest is meant to identify what
    was measured rather than what was stored.
    """
    arr = np.ascontiguousarray(pixels)
    h = hashlib.sha256()
    h.update(f"{arr.shape}:{arr.dtype}:".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def load_image(path: str | Path, *, view: View = View.FRONTAL) -> SourceImage:
    """Read an image, normalise its orientation, and drop its metadata.

    ``ImageOps.exif_transpose`` is applied before anything else so that every
    later stage can assume image ``+y`` points down the subject's face. A
    portrait photograph stored as landscape-plus-orientation-tag is extremely
    common on phones, and a detector handed the unrotated buffer finds nothing
    at all.
    """
    p = Path(path)
    with PILImage.open(p) as raw:
        exif = read_exif(raw)
        upright = ImageOps.exif_transpose(raw)
        rgb = upright.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.uint8).copy()
    return SourceImage(
        path=p,
        view=view,
        pixels=pixels,
        sha256=pixel_sha256(pixels),
        exif=exif,
    )


def from_array(
    pixels: NDArray[np.uint8], *, view: View = View.FRONTAL, exif: ExifFacts | None = None
) -> SourceImage:
    """Wrap an in-memory array, for callers that never touched a file."""
    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got shape {arr.shape}")
    return SourceImage(
        path=None, view=view, pixels=arr, sha256=pixel_sha256(arr), exif=exif or ExifFacts()
    )


def strip_exif(pixels: NDArray[np.uint8]) -> PILImage.Image:
    """A PIL image built from bare pixels, carrying no metadata at all.

    Everything Vitruve writes -- report overlays, cached crops, debug dumps --
    goes through here. Constructing from the array rather than copying and
    editing an existing image is deliberate: it is not possible to forget to
    remove a tag from a buffer that never had one.
    """
    return PILImage.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8), mode="RGB")


def save_stripped(pixels: NDArray[np.uint8], path: str | Path) -> Path:
    """Write pixels to disk with no metadata. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    strip_exif(pixels).save(out)
    return out


@dataclass(frozen=True)
class SubjectDistance:
    """Camera-to-subject distance, with the uncertainty it was derived under."""

    metres: float
    relative_sd: float
    source: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.metres > 0:
            raise ValueError(f"subject distance must be positive, got {self.metres}")


def estimate_subject_distance(
    exif: ExifFacts,
    *,
    feature_px: float | None = None,
    image_diagonal_px: float | None = None,
    feature_mm: float | None = None,
    feature_sd_mm: float | None = None,
    landmark_sd_px: float = 1.5,
) -> SubjectDistance | None:
    """Camera-to-subject distance, or ``None`` when it cannot be determined.

    Two routes, in order of trust.

    The camera's own ``SubjectDistance`` tag is a focus reading and is used
    directly when present and plausible. It is coarse -- most phones quantise
    it hard -- so it carries a flat 20% interval rather than something derived.

    Otherwise the pinhole relation over a feature of known physical size. With
    a 35 mm-equivalent focal length ``f``, an image whose diagonal is
    ``image_diagonal_px``, and a feature spanning ``feature_px``, the feature's
    image on a 36x24 mm frame is ``feature_px * 43.27 / image_diagonal_px`` mm
    across, and the distance follows. The default feature is the interpupillary
    distance, whose population spread is the dominant error term.

    Returning ``None`` is the common case and it is the correct one. Every
    downstream consumer -- the ICAO magnification warning, the near-fixation
    correction on the interpupillary prior, the ``subject_distance_m`` argument
    to :func:`~vitruve.measure.evaluate.evaluate` -- treats ``None`` as "not
    known" and says so, whereas a guessed 0.5 m would silently withhold
    measurements and a guessed 2 m would silently pass them.
    """
    if exif.subject_distance_m is not None:
        return SubjectDistance(
            metres=exif.subject_distance_m,
            relative_sd=0.20,
            source="exif_subject_distance",
            notes=(
                "read from the camera's own focus distance tag, which most "
                "devices quantise coarsely; the interval is a flat 20% rather "
                "than a derived one",
            ),
        )

    f35 = exif.focal_length_35mm
    if f35 is None or not f35 > 0:
        return None
    if not feature_px or feature_px <= 0 or not image_diagonal_px or image_diagonal_px <= 0:
        return None

    if feature_mm is None:
        mean, sd = IPD_PRIORS[None]
        feature_mm, feature_sd_mm = mean, sd
    if feature_sd_mm is None:
        feature_sd_mm = 0.0

    mm_per_px_on_sensor = FULL_FRAME_DIAGONAL_MM / image_diagonal_px
    feature_on_sensor_mm = feature_px * mm_per_px_on_sensor
    if feature_on_sensor_mm <= 0:
        return None
    distance_mm = f35 * feature_mm / feature_on_sensor_mm

    # Error terms, in quadrature and all relative. The feature's population
    # spread dominates at any sane face size; the landmark term only matters
    # for a face a few dozen pixels across, which the quality gate rejects on
    # other grounds anyway. The 35 mm-equivalent tag is an integer, so it
    # carries a half-unit quantisation of its own.
    rel_feature = feature_sd_mm / feature_mm if feature_mm else 0.0
    rel_landmark = landmark_sd_px / feature_px
    rel_focal = 0.5 / f35
    relative_sd = math.sqrt(rel_feature**2 + rel_landmark**2 + rel_focal**2)

    return SubjectDistance(
        metres=distance_mm / 1000.0,
        relative_sd=relative_sd,
        source="pinhole_from_exif_focal_length",
        notes=(
            f"from a {f35:.0f} mm 35mm-equivalent focal length and a "
            f"{feature_px:.0f} px feature assumed to be {feature_mm:.2f} mm across",
            "assumes the feature lies in a plane normal to the optical axis; "
            "yaw shortens it and biases the distance upward",
        ),
    )


@dataclass(frozen=True)
class Ruler:
    """A physical reference of known length, measured in the photograph.

    This is the only scale cue that is an observation rather than a population
    prior, and it is what collapses the roughly 5.5% interpupillary-prior error
    that otherwise sits under every millimetre value in the report.
    """

    known_mm: float
    pixel_span: float
    reading_error_mm: float = 1.0

    def __post_init__(self) -> None:
        if self.known_mm <= 0 or self.pixel_span <= 0:
            raise ValueError("a ruler needs a positive length and a positive pixel span")
