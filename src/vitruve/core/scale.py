"""Recovering millimetres from pixels, and being honest about the cost.

No permissively-licensed 3D face model returns metric geometry -- MICA is the
only genuinely metric option and it is research-licence only -- so every
millimetre Vitruve prints descends from an assumption about the true size of
something visible in the frame. This module makes that assumption explicit,
attaches an uncertainty to it, and lets the user collapse it by putting a ruler
in the photograph.

Three cues, in descending order of trust:

* **Ruler.** A graduated scale held in the facial plane. This is the gold
  standard and it is what Qoves itself asks its customers for. Residual error
  is the user's reading error plus out-of-plane tilt, not a population prior.
* **Iris.** Horizontal visible iris diameter. Pooled corneal diameter is
  11.84 mm with SD 0.79 (Healy & Stephan 2026, n = 296,887 eyes), and it is
  adult-equivalent from age four, so it needs no age or sex conditioning. Its
  weakness is pixels: a 12 mm span carries roughly a fifth of the pixels of an
  interpupillary span, so landmark noise hurts it five times as much.
* **Interpupillary distance.** ANSUR gives 63.36 mm with SD 3.83 over 3,976
  adults, and sex conditioning tightens it a little (male 64.67 ± 3.71, female
  62.31 ± 3.60). Two corrections matter: ancestry shifts the mean by up to 12%
  across populations, and near fixation at arm's length reduces IPD by about
  3 mm relative to distance IPD -- a 4.7% systematic bias on exactly the
  selfies people submit.

When more than one cue is available they are fused by inverse-variance
weighting with an explicit correlation term, because both are measured by the
same landmark model on the same image and are therefore not independent.
Treating them as independent would understate the fused uncertainty, which is
the failure mode this whole module exists to avoid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ScaleSource(str, Enum):
    RULER = "ruler"
    IRIS = "iris"
    INTERPUPILLARY = "interpupillary"
    FUSED = "fused"
    NONE = "none"


#: Pooled corneal (visible iris) diameter, mm. Healy & Stephan 2026.
IRIS_DIAMETER_MM = 11.84
IRIS_DIAMETER_SD = 0.79

#: Interpupillary distance priors, mm. Dodgson 2004 over ANSUR 1988.
IPD_PRIORS: dict[str | None, tuple[float, float]] = {
    None: (63.36, 3.832),
    "male": (64.67, 3.708),
    "female": (62.31, 3.599),
}

#: Multiplicative correction applied when the subject is fixating at close
#: range: near IPD runs about 3 mm below distance IPD (Evereklioglu et al.).
NEAR_FIXATION_FACTOR = 60.36 / 63.36
NEAR_FIXATION_THRESHOLD_M = 0.7

#: Assumed correlation between two image-derived scale cues. They share a
#: landmark model and an image, so they are far from independent.
CUE_CORRELATION = 0.5


@dataclass(frozen=True)
class ScaleEstimate:
    """Millimetres per pixel, with the uncertainty that comes with it."""

    mm_per_px: float
    relative_sd: float
    source: ScaleSource
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.mm_per_px > 0:
            raise ValueError(f"mm_per_px must be positive, got {self.mm_per_px}")
        if self.relative_sd < 0:
            raise ValueError("relative_sd must not be negative")

    @property
    def relative_ci95(self) -> float:
        """Half-width of the 95% interval as a fraction of the value."""
        return 1.96 * self.relative_sd

    def to_mm(self, pixels: float) -> float:
        return pixels * self.mm_per_px


def from_ruler(px_per_known_mm: float, known_mm: float, reading_error_mm: float = 1.0) -> ScaleEstimate:
    """Scale from a graduated ruler held in the facial plane."""
    if known_mm <= 0 or px_per_known_mm <= 0:
        raise ValueError("ruler length and its pixel span must both be positive")
    return ScaleEstimate(
        mm_per_px=known_mm / px_per_known_mm,
        relative_sd=reading_error_mm / known_mm,
        source=ScaleSource.RULER,
        notes=(
            f"scale from a {known_mm:.0f} mm reference in frame, assuming "
            f"{reading_error_mm:.1f} mm reading error; assumes the ruler lies in "
            "the facial plane, and out-of-plane tilt shortens it",
        ),
    )


def from_iris(iris_diameter_px: float) -> ScaleEstimate:
    """Scale from the horizontal visible iris diameter."""
    if iris_diameter_px <= 0:
        raise ValueError("iris diameter in pixels must be positive")
    return ScaleEstimate(
        mm_per_px=IRIS_DIAMETER_MM / iris_diameter_px,
        relative_sd=IRIS_DIAMETER_SD / IRIS_DIAMETER_MM,
        source=ScaleSource.IRIS,
        notes=(
            f"assumes an {IRIS_DIAMETER_MM:.2f} mm iris (SD {IRIS_DIAMETER_SD:.2f}); "
            "age-invariant above four years, but measured across only a few dozen "
            "pixels, so landmark noise weighs heavily",
        ),
    )


def from_interpupillary(
    ipd_px: float,
    *,
    declared_sex: str | None = None,
    subject_distance_m: float | None = None,
) -> ScaleEstimate:
    """Scale from interpupillary distance against a population prior.

    ``declared_sex`` is used only if the subject stated it. Vitruve does not
    infer it, so an undeclared subject gets the pooled prior and a wider
    interval -- which is the correct trade, not a degradation.
    """
    if ipd_px <= 0:
        raise ValueError("interpupillary distance in pixels must be positive")
    key = declared_sex if declared_sex in IPD_PRIORS else None
    mean, sd = IPD_PRIORS[key]
    notes = [
        f"assumes a {mean:.2f} mm interpupillary distance (SD {sd:.2f}) from ANSUR"
        + (f", conditioned on declared sex '{key}'" if key else ", pooled over sex")
    ]
    if key is None:
        notes.append(
            "sex was not declared, so the prior is pooled; declaring it narrows "
            "the scale interval by roughly a tenth"
        )
    notes.append(
        "ancestry shifts the population mean by up to 12%, and Vitruve does not "
        "infer ancestry, so this interval does not cover that shift"
    )
    if subject_distance_m is not None and subject_distance_m < NEAR_FIXATION_THRESHOLD_M:
        mean *= NEAR_FIXATION_FACTOR
        notes.append(
            f"subject appears to be about {subject_distance_m:.2f} m away, so the "
            "prior is corrected downward for near fixation (about 3 mm)"
        )
    return ScaleEstimate(
        mm_per_px=mean / ipd_px,
        relative_sd=sd / mean,
        source=ScaleSource.INTERPUPILLARY,
        notes=tuple(notes),
    )


def fuse(*estimates: ScaleEstimate) -> ScaleEstimate:
    """Combine scale cues by inverse-variance weighting.

    A ruler, when present, wins outright: it is a direct observation rather
    than a population prior, and averaging it against a prior would only add
    the prior's error back in.

    Otherwise the estimates are combined in log space (scale error is
    multiplicative) with a shared correlation of :data:`CUE_CORRELATION`. With
    two equally precise, half-correlated cues the fused standard deviation is
    about 87% of either one -- a real but modest gain, and much less than the
    71% an independence assumption would claim.
    """
    ests = [e for e in estimates if e is not None]
    if not ests:
        raise ValueError("fuse() needs at least one estimate")
    rulers = [e for e in ests if e.source is ScaleSource.RULER]
    if rulers:
        best = min(rulers, key=lambda e: e.relative_sd)
        if len(ests) > 1:
            return ScaleEstimate(
                best.mm_per_px,
                best.relative_sd,
                best.source,
                best.notes + ("a physical reference was present, so population priors were not used",),
            )
        return best
    if len(ests) == 1:
        return ests[0]

    logs = [math.log(e.mm_per_px) for e in ests]
    sds = [max(e.relative_sd, 1e-9) for e in ests]
    weights = [1.0 / s**2 for s in sds]
    total = sum(weights)
    fused_log = sum(w * x for w, x in zip(weights, logs)) / total

    # Variance of a weighted mean of correlated terms:
    #   var = sum_i sum_j w_i w_j cov_ij / (sum w)^2
    var = 0.0
    for i, si in enumerate(sds):
        for j, sj in enumerate(sds):
            rho = 1.0 if i == j else CUE_CORRELATION
            var += weights[i] * weights[j] * rho * si * sj
    var /= total**2

    return ScaleEstimate(
        mm_per_px=math.exp(fused_log),
        relative_sd=math.sqrt(var),
        source=ScaleSource.FUSED,
        notes=tuple(
            [f"fused {len(ests)} cues ({', '.join(e.source.value for e in ests)}) with an "
             f"assumed correlation of {CUE_CORRELATION}, since both are read by the same "
             "landmark model from the same image"]
        )
        + tuple(n for e in ests for n in e.notes),
    )


def magnification_distortion(subject_distance_m: float, plane_depth_mm: float = 50.0) -> float:
    """Fractional magnification between the eye plane and a plane behind it.

    The ICAO portrait-quality formula: a feature ``plane_depth_mm`` in front of
    the reference plane is enlarged by roughly that depth over the
    camera-to-subject distance. At 0.3 m the nose plane is about 17% larger
    relative to the eye plane; at 2 m it is 2.5%. Any ratio whose numerator and
    denominator straddle those planes carries this as bias.
    """
    if subject_distance_m <= 0:
        raise ValueError("subject distance must be positive")
    return (plane_depth_mm / 1000.0) / subject_distance_m
