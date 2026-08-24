"""Colour measurement, for the findings no public dataset can support.

Dark circles, erythema and pore visibility are the three things a consumer skin
report always claims to detect, and they are exactly the three for which no
usable annotated public data exists. Not "small datasets": none. Acne has real
boxes (ACNE04, and the Roboflow sets under CC BY 4.0). Wrinkles have real pixel
masks (FFHQ-Wrinkle, non-commercial). Periorbital hyperpigmentation, erythema
and pores have neither a dataset nor a credible model, so any tool that reports
them from a learned detector has trained on something it is not showing you.

The honest alternative is not to detect them but to *measure* them, and the
measurement is a within-face paired contrast in CIELAB.

Why paired, and why that is the load-bearing decision:

* An absolute threshold on redness or darkness is a statement about the
  illuminant, the camera's white balance, the JPEG pipeline and the subject's
  constitutive pigmentation, in roughly that order, before it is a statement
  about the skin. It fails hardest on dark skin, where the same absolute a*
  elevation corresponds to a much larger perceptual and clinical change.
* A difference between two regions of the *same face in the same photograph*
  cancels the illuminant, the camera response to first order, and the person's
  own baseline tone. What survives is the thing being asked about.

So erythema is a* in a region minus a* in adjacent cheek skin, and periorbital
hyperpigmentation is the infraorbital-minus-malar difference in L* and b*. The
periorbital pairing in particular is the standard clinical construct: the
infraorbital region is darker and yellower than the cheek directly below it, and
how much darker is the finding.

Three disciplines carry over from the measurement layer:

1. **Every value carries an interval.** The interval comes from within-region
   pixel variance, computed over spatial blocks rather than over pixels, because
   neighbouring skin pixels are strongly correlated and a pixel-count standard
   error would be optimistic by an order of magnitude.
2. **A contrast smaller than its own uncertainty is withheld**, which is the
   discriminability rule of ``core.sensitivity`` applied to a colour difference:
   a number that cannot be distinguished from the noise in its own measurement
   is not a finding.
3. **Uncalibrated colour is comparable within one photograph and not between
   photographs.** A consumer photograph has no colour reference in it, so its
   absolute L*a*b* is unanchored. Paired contrasts survive this; absolute
   values (skin tone, ITA) do not, and they say so every time they are printed.
   Put a grey card in the frame and :func:`grey_card_balance` turns the absolute
   numbers into real ones.

Skin tone is reported on the **Monk Skin Tone scale**, not Fitzpatrick.
Fitzpatrick was constructed in 1975 to predict erythemal response to UVA dosing
in light skin; its two darkest categories absorb the whole range of dark skin,
and it was never a scale of appearance. Monk's ten-point scale was built for
appearance, and in a 2,214-person US survey (Heldreth et al. 2024, ACM Journal
on Responsible Computing) it was rated more representative than the alternatives
by participants across skin tones. Tone here selects a stratum for reporting
error, and is never itself a finding about a person.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .regions import Region, RegionSet

# ---------------------------------------------------------------------------
# sRGB to CIELAB, written out rather than imported
# ---------------------------------------------------------------------------

#: sRGB (IEC 61966-2-1) linear-RGB to CIE XYZ under D65.
SRGB_TO_XYZ: NDArray[np.float64] = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=float,
)

#: D65 white point, normalised to Y = 1.
D65_WHITE: NDArray[np.float64] = np.array([0.95047, 1.00000, 1.08883], dtype=float)

_DELTA = 6.0 / 29.0
_DELTA3 = _DELTA**3


def srgb_to_linear(channel: NDArray[np.float64]) -> NDArray[np.float64]:
    """Undo the sRGB transfer function. Input and output in [0, 1]."""
    c = np.asarray(channel, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(channel: NDArray[np.float64]) -> NDArray[np.float64]:
    c = np.clip(np.asarray(channel, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _lab_f(t: NDArray[np.float64]) -> NDArray[np.float64]:
    t = np.asarray(t, dtype=float)
    return np.where(t > _DELTA3, np.cbrt(t), t / (3.0 * _DELTA**2) + 4.0 / 29.0)


def xyz_to_lab(xyz: NDArray[np.float64], white: NDArray[np.float64] = D65_WHITE) -> NDArray[np.float64]:
    """CIE XYZ (Y normalised to 1 for the white point) to L*a*b*."""
    ratio = np.asarray(xyz, dtype=float) / np.asarray(white, dtype=float)
    fx, fy, fz = _lab_f(ratio[..., 0]), _lab_f(ratio[..., 1]), _lab_f(ratio[..., 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def rgb_to_lab(
    image: NDArray,
    *,
    balance: "WhiteBalance | None" = None,
) -> NDArray[np.float64]:
    """Convert an sRGB image to CIELAB, optionally through a white balance.

    ``image`` is ``(..., 3)``, either uint8 in [0, 255] or float in [0, 1]. The
    balance, when supplied, is applied in *linear* light, which is where a von
    Kries channel gain belongs; applying it to gamma-encoded values is a common
    and quietly wrong shortcut that leaves a residual hue cast.
    """
    arr = np.asarray(image)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected an RGB image with 3 channels, got {arr.shape}")
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(float) / 255.0
    else:
        arr = arr.astype(float)
        if arr.max(initial=0.0) > 1.0 + 1e-6:
            raise ValueError("float images must be scaled to [0, 1]")
    lin = srgb_to_linear(arr)
    if balance is not None:
        lin = lin * np.asarray(balance.gains, dtype=float)
    xyz = lin @ SRGB_TO_XYZ.T
    return xyz_to_lab(xyz)


def delta_e76(lab_a: Sequence[float], lab_b: Sequence[float]) -> float:
    """CIE76 colour difference. Crude next to CIEDE2000, and adequate here.

    Everything this module compares is a small difference between two skin
    colours, which sits in the part of the space where CIE76 and CIEDE2000 do
    not disagree in a way that would change a verdict.
    """
    a = np.asarray(lab_a, dtype=float)
    b = np.asarray(lab_b, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


# ---------------------------------------------------------------------------
# White balance
# ---------------------------------------------------------------------------

UNCALIBRATED_NOTE = (
    "no colour reference was present in this photograph, so these values are "
    "comparable within this image and not between images; a different camera, "
    "white balance or light source shifts them without the skin changing"
)

ABSOLUTE_UNCALIBRATED_NOTE = (
    "this is an absolute colour value, so an uncalibrated white balance moves it "
    "directly; include a grey card in frame to make it a measurement rather than "
    "an indication"
)


@dataclass(frozen=True)
class WhiteBalance:
    """Per-channel gains applied in linear light, and whether they are real."""

    gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    calibrated: bool = False
    source: str = "none"
    notes: tuple[str, ...] = (UNCALIBRATED_NOTE,)

    @classmethod
    def uncalibrated(cls) -> "WhiteBalance":
        return cls()


def grey_card_balance(
    patch_rgb: NDArray,
    *,
    reflectance: float | None = None,
) -> WhiteBalance:
    """Channel gains that neutralise a grey card in frame.

    ``patch_rgb`` is any array of sRGB pixels from a neutral card. The card
    defines what "no colour" looks like under this illuminant and this camera,
    so dividing each linear channel by its own mean and rescaling to the patch's
    mean level maps it to neutral without changing exposure.

    ``reflectance`` (0.18 for a standard grey card, 0.90 for a white patch) also
    anchors the exposure, which is what makes absolute L* meaningful rather than
    merely hue-correct. Without it the hue is calibrated and the lightness is
    not, and the returned notes say so.
    """
    arr = np.asarray(patch_rgb)
    if arr.shape[-1] != 3:
        raise ValueError("grey card patch must have 3 channels")
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(float) / 255.0
    lin = srgb_to_linear(arr).reshape(-1, 3)
    if lin.shape[0] < 4:
        raise ValueError("grey card patch needs at least 4 pixels")
    means = lin.mean(axis=0)
    if np.any(means <= 0):
        raise ValueError("grey card patch has a fully black channel; it is not usable")
    if np.any(means > 0.98):
        raise ValueError("grey card patch is clipped; it carries no colour information")
    gains = float(means.mean()) / means
    notes = [f"white balance from a neutral reference patch ({lin.shape[0]} px)"]
    if reflectance is not None:
        if not 0.0 < reflectance <= 1.0:
            raise ValueError("reflectance must be in (0, 1]")
        gains = gains * (reflectance / float(means.mean()))
        notes.append(
            f"exposure anchored to a {reflectance * 100:.0f}% reflectance reference, "
            "so absolute L* is meaningful and comparable between photographs"
        )
    else:
        notes.append(
            "no reflectance was given for the reference, so hue is calibrated but "
            "absolute lightness still depends on exposure"
        )
    return WhiteBalance(
        gains=(float(gains[0]), float(gains[1]), float(gains[2])),
        calibrated=True,
        source="grey card",
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Region colour
# ---------------------------------------------------------------------------

#: Floor on any channel standard error, in L*a*b* units.
#:
#: 8-bit quantisation, JPEG chroma subsampling and demosaicing all put a hard
#: bottom under how finely a colour can be read out of a consumer photograph,
#: and a region of ten thousand pixels would otherwise report a standard error
#: near zero and make every contrast look decisive. One CIELAB unit is roughly
#: the classical just-noticeable difference, so half a unit is the smallest
#: standard error this module will claim.
SE_FLOOR = 0.5

#: Side of the spatial block used for the cluster-robust standard error.
BLOCK_PX = 8

#: Minimum masked fraction for a block to contribute, so that edge blocks
#: holding three pixels do not get the same weight as full ones.
BLOCK_MIN_FILL = 0.5


@dataclass(frozen=True)
class RegionColour:
    """Mean L*a*b* over a region, with a standard error that respects correlation."""

    region: Region | None
    n_pixels: int
    n_blocks: int
    mean: tuple[float, float, float]
    sd: tuple[float, float, float]
    se: tuple[float, float, float]
    calibrated: bool = False
    notes: tuple[str, ...] = ()

    @property
    def L(self) -> float:
        return self.mean[0]

    @property
    def a(self) -> float:
        return self.mean[1]

    @property
    def b(self) -> float:
        return self.mean[2]

    @property
    def ita(self) -> float:
        return individual_typology_angle(self.L, self.b)

    def format(self) -> str:
        return (
            f"L* {self.mean[0]:.1f} +/- {1.96 * self.se[0]:.1f}, "
            f"a* {self.mean[1]:+.1f} +/- {1.96 * self.se[1]:.1f}, "
            f"b* {self.mean[2]:+.1f} +/- {1.96 * self.se[2]:.1f} "
            f"({self.n_pixels} px, {self.n_blocks} blocks)"
        )


def _block_means(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    block: int,
) -> NDArray[np.float64]:
    """Per-block channel means over the masked pixels. Shape ``(n_blocks, 3)``."""
    h, w = mask.shape
    rows = range(0, h, block)
    cols = range(0, w, block)
    out: list[NDArray[np.float64]] = []
    area = float(block * block)
    for r in rows:
        for c in cols:
            sub = mask[r : r + block, c : c + block]
            n = int(sub.sum())
            if n == 0 or n < BLOCK_MIN_FILL * area:
                continue
            out.append(values[r : r + block, c : c + block][sub].mean(axis=0))
    return np.asarray(out, dtype=float) if out else np.zeros((0, 3), dtype=float)


def region_colour(
    lab: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    region: Region | None = None,
    calibrated: bool = False,
    block: int = BLOCK_PX,
    trim: float = 0.02,
) -> RegionColour:
    """Mean colour over a masked region, with a cluster-robust standard error.

    The standard error is computed from the spread of *block* means, not pixel
    means. Skin pixels a few micrometres apart are not independent samples: the
    same shading gradient, the same lens blur and the same JPEG macroblock run
    through all of them. Dividing the pixel standard deviation by the square
    root of the pixel count would understate the uncertainty by roughly the
    block side, and that inflation is exactly what turns a shading gradient into
    a "statistically significant" finding.

    ``trim`` drops the darkest and lightest tails before averaging, which is a
    cheap defence against specular highlights and stray hair inside the polygon.
    """
    lab = np.asarray(lab, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError(f"lab image must be (h, w, 3), got {lab.shape}")
    if mask.shape != lab.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match image {lab.shape[:2]}")

    notes: list[str] = []
    work = mask
    n = int(work.sum())
    if n == 0:
        raise ValueError(
            f"region {region.value if region else '?'} has no pixels inside the image"
        )

    if trim > 0 and n >= 50:
        lightness = lab[..., 0][work]
        lo, hi = np.quantile(lightness, [trim, 1.0 - trim])
        keep = np.zeros_like(work)
        keep[work] = (lightness >= lo) & (lightness <= hi)
        if keep.sum() >= 0.5 * n:
            work = keep
            notes.append(
                f"trimmed the darkest and lightest {trim * 100:.0f}% of pixels to "
                "suppress specular highlights and stray hair"
            )

    px = lab[work]
    mean = px.mean(axis=0)
    sd = px.std(axis=0, ddof=1) if px.shape[0] > 1 else np.zeros(3)

    blocks = _block_means(lab, work, block)
    if blocks.shape[0] >= 4:
        se = blocks.std(axis=0, ddof=1) / math.sqrt(blocks.shape[0])
        n_blocks = int(blocks.shape[0])
    else:
        se = sd / math.sqrt(max(px.shape[0], 1))
        n_blocks = int(blocks.shape[0])
        notes.append(
            f"fewer than four {block}px blocks fit inside this region, so the "
            "standard error falls back to a pixel count and is optimistic, "
            "because neighbouring skin pixels are not independent samples"
        )
    se = np.maximum(se, SE_FLOOR)

    if not calibrated:
        notes.append(UNCALIBRATED_NOTE)

    return RegionColour(
        region=region,
        n_pixels=int(work.sum()),
        n_blocks=n_blocks,
        mean=(float(mean[0]), float(mean[1]), float(mean[2])),
        sd=(float(sd[0]), float(sd[1]), float(sd[2])),
        se=(float(se[0]), float(se[1]), float(se[2])),
        calibrated=calibrated,
        notes=tuple(notes),
    )


def sample_regions(
    image: NDArray,
    regions: RegionSet,
    *,
    balance: WhiteBalance | None = None,
    erode: float = 0.12,
    y_up: bool = True,
    only: Sequence[Region] | None = None,
) -> dict[Region, RegionColour]:
    """Convert once, then read every available region out of the same LAB array."""
    lab = rgb_to_lab(image, balance=balance)
    h, w = lab.shape[:2]
    calibrated = bool(balance and balance.calibrated)
    wanted = list(only) if only is not None else sorted(regions.available, key=lambda r: r.value)
    out: dict[Region, RegionColour] = {}
    for region in wanted:
        if region not in regions:
            continue
        mask = regions.mask_for(region, h, w, erode=erode, y_up=y_up)
        if not mask.any():
            continue
        out[region] = region_colour(lab, mask, region=region, calibrated=calibrated)
    return out


# ---------------------------------------------------------------------------
# Paired contrasts
# ---------------------------------------------------------------------------

CHANNELS = ("L*", "a*", "b*")
_CHANNEL_INDEX = {"L*": 0, "a*": 1, "b*": 2}


@dataclass(frozen=True)
class PairedContrast:
    """A within-face colour difference, with the uncertainty that decides it.

    ``delta`` is target minus reference. ``uncertainty`` is the standard error
    of that difference, propagated from the two regions' cluster-robust standard
    errors. :attr:`discriminable` applies the rule from ``core.sensitivity``:
    the signal has to exceed its own measurement error before it is a finding.
    The numerator there is a between-person spread and here it is a within-face
    contrast, so this is the same principle rather than the same statistic, and
    it is implemented separately for that reason.
    """

    label: str
    channel: str
    target: Region | None
    reference: Region | None
    delta: float
    uncertainty: float
    calibrated: bool = False
    notes: tuple[str, ...] = ()

    @property
    def ratio(self) -> float:
        return abs(self.delta) / self.uncertainty if self.uncertainty > 0 else math.inf

    @property
    def discriminable(self) -> bool:
        return self.ratio > 1.0

    @property
    def ci_low(self) -> float:
        return self.delta - 1.96 * self.uncertainty

    @property
    def ci_high(self) -> float:
        return self.delta + 1.96 * self.uncertainty

    @property
    def verdict(self) -> str:
        if self.ratio > 3.0:
            return "clearly separated from the reference region"
        if self.ratio > 1.5:
            return "separated from the reference region"
        if self.ratio > 1.0:
            return "marginal: the contrast barely exceeds its own uncertainty"
        return (
            "withheld: the contrast is smaller than the uncertainty of the "
            "measurement that produced it"
        )

    def format(self) -> str:
        return (
            f"{self.label}: {self.delta:+.2f} {self.channel} "
            f"(95% CI {self.ci_low:+.2f} to {self.ci_high:+.2f}) -- {self.verdict}"
        )


def paired_contrast(
    target: RegionColour,
    reference: RegionColour,
    channel: str,
    *,
    label: str | None = None,
) -> PairedContrast:
    """Target minus reference on one CIELAB channel.

    The two standard errors add in quadrature. They are not independent -- both
    regions are lit by the same lamps and read by the same sensor -- but the
    shared component is what the subtraction removes, so treating the residuals
    as independent is the conservative direction here rather than the optimistic
    one.
    """
    if channel not in _CHANNEL_INDEX:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    i = _CHANNEL_INDEX[channel]
    delta = target.mean[i] - reference.mean[i]
    unc = math.sqrt(target.se[i] ** 2 + reference.se[i] ** 2)
    calibrated = target.calibrated and reference.calibrated
    notes: list[str] = [
        "a within-face paired contrast, so the illuminant, the camera white "
        "balance and the subject's own baseline tone cancel to first order"
    ]
    if not calibrated:
        notes.append(
            "the contrast survives an uncalibrated white balance; the absolute "
            "colours it was computed from do not"
        )
    return PairedContrast(
        label=label or f"{channel} contrast",
        channel=channel,
        target=target.region,
        reference=reference.region,
        delta=float(delta),
        uncertainty=float(max(unc, SE_FLOOR)),
        calibrated=calibrated,
        notes=tuple(notes),
    )


def erythema(target: RegionColour, reference: RegionColour) -> PairedContrast:
    """a* elevation of a region against paired reference skin.

    a* is the red-green opponent axis, and cutaneous erythema is haemoglobin
    absorption in the green band, so a* is the channel that moves. The reference
    should be adjacent skin at a similar illumination angle; the default pairing
    in ``regions.REFERENCE_PAIRS`` is malar against lateral cheek.

    The residual confound is shading. The lateral cheek curves away from a
    frontal light source, so it is darker, and CIELAB chroma is not perfectly
    lightness-independent, which puts a small illumination-geometry term into
    every malar-minus-lateral contrast. It is small next to a real erythema and
    it is not zero, so it is stated rather than corrected.
    """
    c = paired_contrast(target, reference, "a*", label="erythema (a* elevation)")
    return PairedContrast(
        label=c.label,
        channel=c.channel,
        target=c.target,
        reference=c.reference,
        delta=c.delta,
        uncertainty=c.uncertainty,
        calibrated=c.calibrated,
        notes=c.notes
        + (
            "a* tracks haemoglobin absorption, which is why redness is read on "
            "this axis rather than from an RGB ratio",
            "the reference region sits further from a frontal light source, so a "
            "small part of any contrast is illumination geometry rather than skin",
        ),
    )


@dataclass(frozen=True)
class PigmentationContrast:
    """Infraorbital-minus-malar difference, the dark-circle construct.

    Periorbital hyperpigmentation shows up as a region that is both **darker**
    (lower L*) and **yellower or browner** (higher b*) than the cheek below it.
    Reporting the two channels separately is deliberate: a purely darker
    infraorbital band with no b* shift is usually shadow from the orbital rim or
    from tear-trough volume loss, and a b* shift with little darkening is
    usually pigment. The combined magnitude is offered for ranking, never in
    place of the two components.
    """

    lightness: PairedContrast
    yellowness: PairedContrast
    target: Region | None
    reference: Region | None

    @property
    def magnitude(self) -> float:
        """Distance in the (L*, b*) plane, in CIELAB units."""
        return math.hypot(self.lightness.delta, self.yellowness.delta)

    @property
    def uncertainty(self) -> float:
        m = self.magnitude
        if m <= 0:
            return math.hypot(self.lightness.uncertainty, self.yellowness.uncertainty)
        return (
            math.hypot(
                self.lightness.delta * self.lightness.uncertainty,
                self.yellowness.delta * self.yellowness.uncertainty,
            )
            / m
        )

    @property
    def ratio(self) -> float:
        u = self.uncertainty
        return self.magnitude / u if u > 0 else math.inf

    @property
    def discriminable(self) -> bool:
        return self.ratio > 1.0

    @property
    def darker_and_yellower(self) -> bool:
        return self.lightness.delta < 0 and self.yellowness.delta > 0

    @property
    def interpretation(self) -> str:
        if not self.discriminable:
            return "no infraorbital contrast above the measurement uncertainty"
        if self.darker_and_yellower:
            return (
                "infraorbital region is both darker and yellower than the cheek "
                "below it, the pattern associated with pigmentation"
            )
        if self.lightness.delta < 0:
            return (
                "infraorbital region is darker than the cheek with no shift along "
                "b*, which is the pattern shadowing produces"
            )
        return "infraorbital region differs from the cheek in a direction pigmentation does not predict"


def periorbital_pigmentation(
    infraorbital: RegionColour, malar: RegionColour
) -> PigmentationContrast:
    """Infraorbital minus ipsilateral malar, on L* and b*.

    Ipsilateral rather than contralateral, and malar rather than any other
    patch, because it is the nearest large area of the same person's skin under
    the same light. Comparing to the other eye would measure asymmetry; comparing
    to a population mean would measure the population.
    """
    return PigmentationContrast(
        lightness=paired_contrast(
            infraorbital, malar, "L*", label="infraorbital darkening (L* deficit)"
        ),
        yellowness=paired_contrast(
            infraorbital, malar, "b*", label="infraorbital yellowing (b* elevation)"
        ),
        target=infraorbital.region,
        reference=malar.region,
    )


# ---------------------------------------------------------------------------
# Skin tone
# ---------------------------------------------------------------------------


def individual_typology_angle(lightness: float, yellowness: float) -> float:
    """ITA in degrees: ``atan((L* - 50) / b*)``.

    The classical colorimetric summary of constitutive pigmentation. High
    positive angles are light skin, negative angles are dark skin. It is a
    two-channel projection, so it discards a* entirely, which is why an
    erythematous cheek and a pigmented one can share an ITA.
    """
    if yellowness == 0:
        return 90.0 if lightness >= 50 else -90.0
    return math.degrees(math.atan((lightness - 50.0) / yellowness))


#: Conventional ITA bands (Chardon et al.), kept because they are the vocabulary
#: the dermatology literature uses. They are *not* the reporting scale here.
ITA_CLASSES: tuple[tuple[float, str], ...] = (
    (55.0, "very light"),
    (41.0, "light"),
    (28.0, "intermediate"),
    (10.0, "tan"),
    (-30.0, "brown"),
    (-math.inf, "dark"),
)


def ita_class(ita: float) -> str:
    for threshold, name in ITA_CLASSES:
        if ita > threshold:
            return name
    return ITA_CLASSES[-1][1]


#: The ten Monk Skin Tone reference swatches, as published at skintone.google.
#: The scale is defined by these swatches, so the mapping below is a nearest
#: swatch in CIELAB rather than a set of invented numeric thresholds.
MONK_SWATCHES_HEX: tuple[str, ...] = (
    "#f6ede4",
    "#f3e7db",
    "#f7ead0",
    "#eadaba",
    "#d7bd96",
    "#a07e56",
    "#825c43",
    "#604134",
    "#3a312a",
    "#292420",
)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


MONK_SWATCH_LAB: NDArray[np.float64] = rgb_to_lab(
    np.array([_hex_to_rgb(h) for h in MONK_SWATCHES_HEX], dtype=np.uint8)
)


@dataclass(frozen=True)
class SkinTone:
    """A Monk tone, the ITA it came with, and how far the match actually was."""

    monk: int
    swatch_hex: str
    delta_e: float
    ita: float
    ita_class: str
    calibrated: bool
    region: Region | None = None
    notes: tuple[str, ...] = ()

    def format(self) -> str:
        return (
            f"Monk tone {self.monk} (nearest swatch {self.swatch_hex}, "
            f"dE76 {self.delta_e:.1f}); ITA {self.ita:+.1f} deg, {self.ita_class}"
        )


def monk_tone(
    colour: RegionColour,
    *,
    region: Region | None = None,
) -> SkinTone:
    """Nearest Monk swatch to a measured region colour.

    Monk rather than Fitzpatrick. Fitzpatrick is a 1975 instrument for
    predicting how skin burns under UVA, built by asking light-skinned patients
    how they reacted; types V and VI were appended later and between them absorb
    the entire range of dark skin. It has been used as an appearance scale for
    decades because it was the scale that existed, and computer-vision fairness
    audits inherit its compression. Monk's ten swatches were built for
    appearance and were rated more representative in Heldreth et al. (2024).

    The nearest-swatch match reports its own distance, and the caller can see
    when a face's colour fell between two swatches or nowhere near any of them.
    """
    lab = np.array(colour.mean, dtype=float)
    d = np.sqrt(((MONK_SWATCH_LAB - lab) ** 2).sum(axis=1))
    i = int(np.argmin(d))
    ita = individual_typology_angle(colour.L, colour.b)
    notes = [
        "skin tone selects a stratum for reporting measurement error; it is not "
        "itself a finding about the person, and it is never inferred as a "
        "demographic attribute",
        "the Monk scale is defined by ten reference swatches, so this is a "
        "nearest-swatch match in CIELAB and not a conversion formula",
    ]
    if not colour.calibrated:
        notes.append(ABSOLUTE_UNCALIBRATED_NOTE)
    if d[i] > 12.0:
        notes.append(
            f"the nearest swatch is {d[i]:.0f} CIELAB units away, which is further "
            "than any swatch spacing; the reading is more likely a lighting or "
            "white-balance artifact than a tone"
        )
    return SkinTone(
        monk=i + 1,
        swatch_hex=MONK_SWATCHES_HEX[i],
        delta_e=float(d[i]),
        ita=ita,
        ita_class=ita_class(ita),
        calibrated=colour.calibrated,
        region=region if region is not None else colour.region,
        notes=tuple(notes),
    )


def tone_from_regions(
    colours: dict[Region, RegionColour],
    *,
    prefer: Sequence[Region] = (
        Region.MALAR_R,
        Region.MALAR_L,
        Region.LATERAL_CHEEK_R,
        Region.LATERAL_CHEEK_L,
        Region.FOREHEAD,
    ),
) -> SkinTone | None:
    """Read tone from the largest available unremarkable patch of cheek.

    The cheek is preferred to the forehead because the forehead is more often
    specular and more often shaded by hair, and to the nose because the nose is
    the most erythematous part of most faces.
    """
    for region in prefer:
        if region in colours:
            return monk_tone(colours[region], region=region)
    return None


__all__ = [
    "SRGB_TO_XYZ",
    "D65_WHITE",
    "SE_FLOOR",
    "BLOCK_PX",
    "CHANNELS",
    "UNCALIBRATED_NOTE",
    "ABSOLUTE_UNCALIBRATED_NOTE",
    "MONK_SWATCHES_HEX",
    "MONK_SWATCH_LAB",
    "ITA_CLASSES",
    "WhiteBalance",
    "RegionColour",
    "PairedContrast",
    "PigmentationContrast",
    "SkinTone",
    "srgb_to_linear",
    "linear_to_srgb",
    "xyz_to_lab",
    "rgb_to_lab",
    "delta_e76",
    "grey_card_balance",
    "region_colour",
    "sample_regions",
    "paired_contrast",
    "erythema",
    "periorbital_pigmentation",
    "individual_typology_angle",
    "ita_class",
    "monk_tone",
    "tone_from_regions",
]
