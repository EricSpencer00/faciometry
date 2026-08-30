"""One finding type over two very different kinds of evidence.

The dermatological layer produces two things that look nothing alike: a count of
discrete objects a detector found, and a continuous colour difference between
two patches of skin. They reach a report through the same type anyway, because
the report's obligations do not depend on where a number came from. Every
finding carries an interval rather than a point estimate. Every finding that is
withheld says why it was withheld instead of vanishing. Every finding names the
provenance of whatever produced it, which for the detector means naming an
AGPL-3.0 lineage and a CC BY 4.0 training set.

The withholding rule is the same one ``core.sensitivity`` applies to
measurements: a quantity smaller than the uncertainty of the measurement that
produced it is not a result. For a colour contrast that is the cluster-robust
standard error of the difference. For a lesion count it is the Poisson spread of
the count itself, which is why a report of "3 lesions" on a region is a claim
about roughly zero to seven.

Two things this module deliberately does not contain:

**No recommendations.** No products, no routines, no "consider seeing a
dermatologist". This is a measurement layer. Advice is the part of a consumer
skin report with no measurement basis and the part most likely to cause harm,
and the boundary is enforced here in :func:`contains_advice` rather than left to
whoever writes the renderer.

**No overall skin score.** The same rule the measurement layer follows, for the
same reason: a scalar summary is the component with the least defensible basis
and the clearest path to harm.

And one thing it always contains: :data:`DISCLAIMER`. Nothing here is a
diagnosis, and the distance between "the infraorbital region is 4 L* units
darker than the cheek" and any clinical statement is not a distance the software
gets to cross.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from ..core.spec import Reportability, Verdict
from ..models.licensing import Provenance
from .colorimetry import (
    ABSOLUTE_UNCALIBRATED_NOTE,
    PairedContrast,
    PigmentationContrast,
    RegionColour,
    SkinTone,
)
from .detect_yolo import (
    ESTIMATED_HALF_FACE_NOTE,
    HAYASHI_NOTE,
    AcneDetection,
    RegionLesions,
    Severity,
    estimated_half_face_count,
    hayashi_severity,
)
from .regions import Region

DISCLAIMER = (
    "These are measurements of a photograph, not a medical assessment. Faciometry "
    "is not a medical device, performs no diagnosis, and its outputs have not "
    "been validated against clinical examination. A colour difference between "
    "two regions of skin is a property of the image; what causes it is outside "
    "what this software can determine."
)

#: Phrases that turn a measurement layer into an advice layer. Checked at
#: construction, so a renderer cannot smuggle one in through a note.
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "recommend",
    "you should",
    "we suggest",
    "consult a",
    "consult your",
    "see a doctor",
    "see a dermatologist",
    "seek medical",
    "treatment",
    "regimen",
    "skincare",
    "prescri",
    "diagnos",
    "improve your",
    "reduce your",
    "get rid of",
)


def contains_advice(text: str) -> str | None:
    """Return the offending phrase, or ``None``.

    Substring matching is blunt and that is intended: the cost of a false
    positive is rewording a note, and the cost of a false negative is a
    measurement tool telling someone what to do about their face.
    """
    lowered = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            return phrase
    return None


class FindingKind(str, Enum):
    """What was measured. Not what it means."""

    ACNE_LESION_COUNT = "acne_lesion_count"
    ACNE_SEVERITY = "acne_severity"
    ERYTHEMA = "erythema"
    PERIORBITAL_PIGMENTATION = "periorbital_pigmentation"
    SKIN_TONE = "skin_tone"

    @property
    def is_colorimetric(self) -> bool:
        return self in (
            FindingKind.ERYTHEMA,
            FindingKind.PERIORBITAL_PIGMENTATION,
            FindingKind.SKIN_TONE,
        )


@dataclass(frozen=True)
class Finding:
    """One dermatological observation, with everything needed to judge it."""

    kind: FindingKind
    label: str
    region: Region | None
    #: The measured quantity. A count for detections, a CIELAB difference for
    #: contrasts, a Monk index for tone.
    magnitude: float
    unit: str
    #: Standard error of :attr:`magnitude`, in the same unit.
    uncertainty: float
    ci_low: float
    ci_high: float
    verdict: Verdict
    method: str
    provenance: Provenance | None = None
    reference_region: Region | None = None
    count: int | None = None
    severity: Severity | None = None
    calibrated: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.ci_low > self.ci_high:
            raise ValueError(f"{self.label}: interval is inverted")
        if self.uncertainty < 0:
            raise ValueError(f"{self.label}: uncertainty cannot be negative")
        for text in (self.label, *self.notes, *self.verdict.reasons):
            offender = contains_advice(text)
            if offender is not None:
                raise ValueError(
                    f"{self.label}: findings carry measurements, not advice; "
                    f"the phrase {offender!r} appeared in {text!r}"
                )

    @property
    def reportable(self) -> bool:
        return self.verdict.shown

    @property
    def ratio(self) -> float:
        """Magnitude over its own uncertainty, the quantity the gate reads."""
        return abs(self.magnitude) / self.uncertainty if self.uncertainty > 0 else math.inf

    def format(self) -> str:
        """One line. Never the value without the interval, never a bare refusal."""
        where = f" [{self.region.value}]" if self.region else ""
        if not self.reportable:
            blocking = self.verdict.blocking or self.verdict.reasons
            reason = blocking[0] if blocking else "no reason recorded"
            return f"{self.label}{where}: withheld -- {reason}"
        if self.count is not None:
            body = f"{self.count} {self.unit} (95% CI {self.ci_low:.0f} to {self.ci_high:.0f})"
        else:
            body = (
                f"{self.magnitude:+.2f} {self.unit} "
                f"(95% CI {self.ci_low:+.2f} to {self.ci_high:+.2f})"
            )
        if self.severity is not None:
            body += f", Hayashi grade {self.severity.value}"
        caveat = ""
        if self.verdict.caveats:
            caveat = f" -- {self.verdict.caveats[0]}"
        return f"{self.label}{where}: {body}{caveat}"


@dataclass(frozen=True)
class FindingSet:
    """Findings plus the statement that has to travel with them."""

    findings: tuple[Finding, ...] = ()
    disclaimer: str = DISCLAIMER
    notes: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    @property
    def reportable(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.reportable)

    @property
    def withheld(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if not f.reportable)

    def of_kind(self, kind: FindingKind) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.kind is kind)

    def format(self) -> str:
        lines = [f.format() for f in self.findings]
        if self.withheld:
            lines.append(
                f"{len(self.withheld)} of {len(self.findings)} findings were "
                "withheld; a withheld finding is a result, and the reason is above"
            )
        lines.append(self.disclaimer)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _contrast_verdict(contrast: PairedContrast, extra: Sequence[str] = ()) -> Verdict:
    """Attribute each reason to its own severity, not to the verdict's.

    ``core.spec.Verdict`` keeps a severity per finding because a withheld
    measurement usually accumulates several, only one of which caused the
    withholding. An uncalibrated white balance is a caveat whether or not the
    contrast cleared its uncertainty, and flattening the two would make the
    renderer guess which one did the blocking.
    """
    findings: list[tuple[Reportability, str]] = [(Reportability.CAVEAT, e) for e in extra]
    if not contrast.discriminable:
        # The reason bounds the contrast without printing it. A withheld value
        # that arrives in the reason string has not been withheld.
        findings.insert(
            0,
            (
                Reportability.WITHHOLD,
                f"{contrast.verdict}: the {contrast.channel} contrast is below its own "
                f"standard error of {contrast.uncertainty:.2f}",
            ),
        )
        return Verdict(Reportability.WITHHOLD, tuple(findings))
    if contrast.ratio < 1.5:
        findings.insert(0, (Reportability.CAVEAT, contrast.verdict))
        return Verdict(Reportability.CAVEAT, tuple(findings))
    return Verdict(
        Reportability.CAVEAT if findings else Reportability.REPORT, tuple(findings)
    )


def from_erythema(contrast: PairedContrast) -> Finding:
    """Erythema as an a* contrast against paired reference skin."""
    extra: list[str] = []
    if not contrast.calibrated:
        extra.append(
            "white balance is uncalibrated; the paired contrast tolerates that, "
            "the absolute colours behind it do not"
        )
    verdict = _contrast_verdict(contrast, extra)
    return Finding(
        kind=FindingKind.ERYTHEMA,
        label="erythema, a* elevation over paired reference skin",
        region=contrast.target,
        reference_region=contrast.reference,
        magnitude=contrast.delta,
        unit="a*",
        uncertainty=contrast.uncertainty,
        ci_low=contrast.ci_low,
        ci_high=contrast.ci_high,
        verdict=verdict,
        method="CIELAB paired within-face contrast, no learned model",
        calibrated=contrast.calibrated,
        notes=contrast.notes,
    )


def from_pigmentation(contrast: PigmentationContrast) -> Finding:
    """Periorbital hyperpigmentation as an infraorbital-minus-malar contrast."""
    findings: list[tuple[Reportability, str]] = []
    if not contrast.discriminable:
        findings.append(
            (
                Reportability.WITHHOLD,
                "the infraorbital contrast is smaller than the uncertainty of the "
                f"measurement that produced it (standard error {contrast.uncertainty:.2f} "
                "CIELAB units), so the value is withheld rather than printed",
            )
        )
        verdict = Verdict(Reportability.WITHHOLD, tuple(findings))
    else:
        findings.append((Reportability.CAVEAT, contrast.interpretation))
        if not contrast.lightness.calibrated:
            findings.append(
                (
                    Reportability.CAVEAT,
                    "white balance is uncalibrated; the paired contrast tolerates that, "
                    "the absolute colours behind it do not",
                )
            )
        if contrast.ratio < 1.5:
            findings.insert(
                0,
                (
                    Reportability.CAVEAT,
                    "marginal: the contrast barely exceeds its own uncertainty",
                ),
            )
        verdict = Verdict(Reportability.CAVEAT, tuple(findings))
    return Finding(
        kind=FindingKind.PERIORBITAL_PIGMENTATION,
        label="periorbital pigmentation, infraorbital against ipsilateral malar",
        region=contrast.target,
        reference_region=contrast.reference,
        magnitude=contrast.magnitude,
        unit="CIELAB units in the (L*, b*) plane",
        uncertainty=contrast.uncertainty,
        ci_low=contrast.magnitude - 1.96 * contrast.uncertainty,
        ci_high=contrast.magnitude + 1.96 * contrast.uncertainty,
        verdict=verdict,
        method="CIELAB paired within-face contrast on L* and b*, no learned model",
        calibrated=contrast.lightness.calibrated,
        notes=(
            contrast.lightness.format(),
            contrast.yellowness.format(),
            "no annotated public dataset for periorbital hyperpigmentation exists, "
            "so this is measured as a paired contrast rather than detected",
            "shadow cast by the orbital rim reads as an L* deficit with no b* shift; "
            "the two components are reported separately so that case is visible",
        ),
    )


def from_tone(tone: SkinTone, colour: RegionColour) -> Finding:
    """Skin tone on the Monk scale, always caveated when uncalibrated."""
    findings: list[tuple[Reportability, str]] = []
    if not tone.calibrated:
        findings.append((Reportability.CAVEAT, ABSOLUTE_UNCALIBRATED_NOTE))
    if tone.delta_e > 12.0:
        findings.append(
            (
                Reportability.CAVEAT,
                f"the nearest Monk swatch is {tone.delta_e:.0f} CIELAB units away, which "
                "is further than the spacing between swatches",
            )
        )
    reportability = Reportability.REPORT if not findings else Reportability.CAVEAT
    return Finding(
        kind=FindingKind.SKIN_TONE,
        label="skin tone, nearest Monk Skin Tone swatch",
        region=tone.region,
        magnitude=float(tone.monk),
        unit="Monk tone (1-10)",
        uncertainty=0.0,
        ci_low=float(tone.monk),
        ci_high=float(tone.monk),
        verdict=Verdict(reportability, tuple(findings)),
        method="nearest CIELAB swatch on the Monk Skin Tone scale",
        calibrated=tone.calibrated,
        notes=tone.notes
        + (
            colour.format(),
            f"ITA {tone.ita:+.1f} degrees, {tone.ita_class} under the Chardon bands",
            "reported on the Monk scale rather than Fitzpatrick, which was built in "
            "1975 to predict UV erythemal response in light skin and compresses the "
            "whole range of dark skin into two categories",
        ),
    )


def from_region_lesions(
    group: RegionLesions,
    provenance: Provenance,
    *,
    detector_validated: bool = False,
) -> Finding:
    """A lesion count over one region, with a Poisson interval.

    A count is a Poisson draw, so ``n`` carries a standard error of ``sqrt(n)``
    before any detector error is considered, and the interval printed here is
    that alone. Detector recall is a second, larger and unmeasured term unless
    the model card records a stratified evaluation, which is why the finding is
    caveated until it does.
    """
    n = group.count
    se = math.sqrt(n) if n > 0 else 1.0
    findings: list[tuple[Reportability, str]] = []
    if not detector_validated:
        findings.append(
            (
                Reportability.CAVEAT,
                "detector recall and precision on this subject's skin tone are "
                "unmeasured; the interval covers counting noise only",
            )
        )
    if n == 0:
        findings.append(
            (
                Reportability.CAVEAT,
                "no lesions were detected, which is a statement about the detector at "
                "this confidence threshold and not about the skin",
            )
        )
    return Finding(
        kind=FindingKind.ACNE_LESION_COUNT,
        label="acne lesions detected",
        region=group.region,
        magnitude=float(n),
        unit="lesions",
        uncertainty=se,
        ci_low=max(n - 1.96 * se, 0.0),
        ci_high=n + 1.96 * se,
        verdict=Verdict(
            Reportability.CAVEAT if findings else Reportability.REPORT, tuple(findings)
        ),
        method="YOLO instance segmentation over region crops at native resolution",
        provenance=provenance,
        count=n,
        calibrated=True,
        notes=(
            f"{group.count} instances over {group.tiles} crops, "
            f"{group.density_per_kpx:.2f} per thousand square pixels of region area",
        ),
    )


def from_acne_severity(
    detection: AcneDetection,
    *,
    midline_x: float | None = None,
    detector_validated: bool = False,
) -> Finding:
    """The Hayashi grade, with the convention it was calibrated on."""
    if midline_x is not None:
        half = detection.hayashi_count(midline_x)
        basis = "the larger of the two half-face counts, split at the facial midline"
    else:
        half = estimated_half_face_count(detection.total_count)
        basis = ESTIMATED_HALF_FACE_NOTE
    severity = hayashi_severity(half)
    se = math.sqrt(half) if half > 0 else 1.0
    findings: list[tuple[Reportability, str]] = [
        (Reportability.CAVEAT, HAYASHI_NOTE),
        (Reportability.CAVEAT, basis),
    ]
    if not detector_validated:
        findings.append(
            (
                Reportability.CAVEAT,
                "the detector behind this count has no stratified evaluation recorded "
                "in its model card",
            )
        )
    return Finding(
        kind=FindingKind.ACNE_SEVERITY,
        label="acne severity, Hayashi grade from inflammatory lesion count",
        region=None,
        magnitude=float(half),
        unit="inflammatory lesions per half face",
        uncertainty=se,
        ci_low=max(half - 1.96 * se, 0.0),
        ci_high=half + 1.96 * se,
        verdict=Verdict(Reportability.CAVEAT, tuple(findings)),
        method="YOLO instance segmentation over region crops, graded by count",
        provenance=detection.provenance,
        count=half,
        severity=severity,
        calibrated=True,
        notes=detection.notes,
    )


def collect(
    *,
    erythema_contrasts: Iterable[PairedContrast] = (),
    pigmentation: Iterable[PigmentationContrast] = (),
    tone: tuple[SkinTone, RegionColour] | None = None,
    detection: AcneDetection | None = None,
    midline_x: float | None = None,
    detector_validated: bool = False,
    unavailable: dict[Region, tuple[str, ...]] | None = None,
) -> FindingSet:
    """Assemble every finding from one analysis pass.

    ``unavailable`` comes straight from ``RegionSet.unavailable`` and is turned
    into notes rather than dropped, so a report can distinguish "no erythema was
    found" from "the malar region could not be built because the landmark model
    did not supply cheilion_l".
    """
    findings: list[Finding] = []
    for contrast in erythema_contrasts:
        findings.append(from_erythema(contrast))
    for contrast in pigmentation:
        findings.append(from_pigmentation(contrast))
    if tone is not None:
        findings.append(from_tone(tone[0], tone[1]))
    if detection is not None:
        for group in detection.per_region.values():
            findings.append(
                from_region_lesions(group, detection.provenance, detector_validated=detector_validated)
            )
        findings.append(
            from_acne_severity(
                detection, midline_x=midline_x, detector_validated=detector_validated
            )
        )
    notes: list[str] = []
    for region, missing in (unavailable or {}).items():
        notes.append(
            f"{region.value} was not built, so nothing was measured there: "
            f"missing {', '.join(missing)}"
        )
    return FindingSet(findings=tuple(findings), notes=tuple(notes))


__all__ = [
    "DISCLAIMER",
    "FORBIDDEN_PHRASES",
    "FindingKind",
    "Finding",
    "FindingSet",
    "contains_advice",
    "from_erythema",
    "from_pigmentation",
    "from_tone",
    "from_region_lesions",
    "from_acne_severity",
    "collect",
]
