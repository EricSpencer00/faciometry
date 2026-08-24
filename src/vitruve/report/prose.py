"""The written half of the report.

Every sentence in this module is generated from a `Measured`, an `Unavailable`
or a `MeasurementSpec`. None of it is a template with a number dropped into a
slot, because the sentence that has to be written about a withheld measurement
is a different sentence from the one written about a reported measurement, and
collapsing the two is how a refusal turns into a footnote.

Four rules constrain everything here, and each one is enforced by a test rather
than by good intentions:

1. A value never appears without its interval. There is no function in this
   module that returns a formatted value on its own, so a caller cannot print
   one by accident.
2. A withheld measurement gets a sentence naming its cause in plain language.
   The causes are a closed set: pose, scale, a self-occluding landmark, an
   interval wider than the value, a camera too close, poor repeatability, an
   unknown between-person spread, or the finding that the quantity varies less
   between people than between photographs of one person.
3. A normative comparison names the stratum and its n. "Compared against 589
   Black female respirator users measured with calipers" is a claim a reader
   can check. "Compared against the population" is not.
4. No prescriptions. A reference range says where a cited sample fell. It is
   not a target, this module never writes that a value is ideal or ought to be
   anything else, and it recommends nothing at all. Vitruve reports
   measurements.
"""

from __future__ import annotations

from collections.abc import Sequence

from dataclasses import dataclass

from ..core.spec import Evidence, MeasurementSpec, Reportability, Unit
from ..measure.evaluate import Measured, Unavailable
from ..measure.registry import BY_ID
from .model import NormativeStratum, ReportInput

#: Vocabulary that would turn a measurement into a target or a prescription.
#: The test suite scans every generated sentence against this list. Adding a
#: word here is cheap; removing one is a decision about what the project is.
PRESCRIPTIVE_TERMS: tuple[str, ...] = (
    "ideal",
    "should be",
    "ought to",
    "deviates",
    "deviation from",
    "improve",
    "correct the",
    "consider ",
    "recommend",
    "we suggest",
    "optimal",
    "flaw",
    "imperfect",
    "too wide",
    "too narrow",
    "better than",
    "worse than",
    "attractive",
    "harmony",
    "overall score",
)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _places(unit: Unit) -> str:
    return {
        Unit.MILLIMETRES: "{:.1f}",
        Unit.DEGREES: "{:.1f}",
        Unit.PIXELS: "{:.0f}",
        Unit.RATIO: "{:.3g}",
    }[unit]


def _unit_suffix(unit: Unit) -> str:
    return "" if unit is Unit.RATIO else f" {unit.value}"


def value_phrase(m: Measured) -> str:
    """The value with its interval, which is the only form a value takes here.

    Deliberately the whole phrase and not the number: a helper that returned
    the point estimate alone would eventually get called on its own.
    """
    fmt = _places(m.unit)
    return (
        f"{fmt.format(m.value)}{_unit_suffix(m.unit)} "
        f"(95% interval {fmt.format(m.ci_low)} to {fmt.format(m.ci_high)})"
    )


def interval_width_phrase(m: Measured) -> str:
    """How wide the interval is relative to the value."""
    span = m.relative_ci_width
    if span == float("inf"):
        return "the interval cannot be expressed relative to a value this close to zero"
    return f"the interval spans {span * 100:.0f}% of the value"


# ---------------------------------------------------------------------------
# Causes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cause:
    """A plain-language reason a measurement is not printed as a number."""

    key: str
    #: Two or three words, for a heading or a count.
    name: str
    #: A sentence a reader who has never read a photogrammetry paper can use.
    plain: str


CAUSE_DISCRIMINABILITY = Cause(
    "discriminability",
    "varies more between photographs than between people",
    "This quantity moves more from one photograph of a single person to the next "
    "than it moves from one person to another, so a number taken from this "
    "photograph would describe the photograph.",
)
CAUSE_UNKNOWN_SPREAD = Cause(
    "unknown_spread",
    "unknown between-person spread",
    "Nobody has published how much this quantity varies between people, so "
    "whether it distinguishes one face from another cannot be established.",
)
CAUSE_MARGINAL = Cause(
    "marginal",
    "measurement error close to the between-person spread",
    "The spread between people is only a little larger than the error on this "
    "photograph, so small differences in the number carry little information.",
)
CAUSE_SELF_OCCLUDING = Cause(
    "self_occluding",
    "self-occluding landmark",
    "The endpoints sit on a part of the face that curves away from the camera. "
    "A photograph shows where the silhouette falls, which is not where the "
    "anatomical point is.",
)
CAUSE_ROLL = Cause(
    "roll",
    "camera roll",
    "This measurement is defined against the horizon, so tilting the camera "
    "enters the number almost degree for degree, against a normal range only a "
    "few degrees wide.",
)
CAUSE_POSE = Cause(
    "pose",
    "head pose",
    "The head was turned further from the camera than this measurement "
    "tolerates, and turning the head shortens what the photograph records.",
)
CAUSE_SCALE = Cause(
    "scale",
    "assumed scale",
    "The millimetre value rests on an assumed interpupillary distance rather "
    "than a measured one, and that assumption carries its own spread.",
)
CAUSE_LANDMARK = Cause(
    "landmark_uncertainty",
    "landmark uncertainty",
    "The landmark model could not place the points precisely enough, so the "
    "interval grew to a large fraction of the value.",
)
CAUSE_PERSPECTIVE = Cause(
    "perspective",
    "camera distance",
    "The camera was close enough that features nearer the lens are magnified "
    "relative to features behind them, which changes apparent proportions.",
)
CAUSE_REPEATABILITY = Cause(
    "repeatability",
    "poor repeatability",
    "Repeat photographs of the same face did not reproduce this number closely "
    "enough for it to be worth printing.",
)
CAUSE_NO_AGREEMENT = Cause(
    "no_agreement_study",
    "no agreement study",
    "The measurement is standard in the facial-aesthetics literature, but no "
    "study has checked whether a photograph recovers it in agreement with a "
    "caliper.",
)

#: Matched in order, so the specific patterns come before the general ones.
#: "no published between-person spread" and "between-person spread ... against
#: measurement error" share most of their words and mean opposite things.
_MATCHERS: tuple[tuple[str, Cause], ...] = (
    ("no published between-person spread", CAUSE_UNKNOWN_SPREAD),
    ("uninformative:", CAUSE_DISCRIMINABILITY),
    ("this photograph contributes more variance", CAUSE_DISCRIMINABILITY),
    ("marginal:", CAUSE_MARGINAL),
    ("self-occluding", CAUSE_SELF_OCCLUDING),
    ("image roll", CAUSE_ROLL),
    ("head pose", CAUSE_POSE),
    ("scale assumption", CAUSE_SCALE),
    ("scale prior", CAUSE_SCALE),
    ("95% interval spans", CAUSE_LANDMARK),
    ("perspective magnification", CAUSE_PERSPECTIVE),
    ("test-retest", CAUSE_REPEATABILITY),
    ("no published agreement study", CAUSE_NO_AGREEMENT),
)


def cause_of(reason: str) -> Cause:
    """Classify one verdict reason into a plain-language cause.

    An unrecognised reason is passed through verbatim rather than dropped. A
    silent default here would mean a new rule in the core gate produced a
    withheld measurement whose explanation vanished on the way to the reader.
    """
    low = reason.lower()
    for needle, cause in _MATCHERS:
        if needle in low:
            return cause
    return Cause("other", "see the recorded reason", reason)


def causes(m: Measured) -> tuple[Cause, ...]:
    """The distinct causes behind a verdict, in the order the gate found them."""
    out: list[Cause] = []
    for reason in m.verdict.reasons:
        cause = cause_of(reason)
        if cause.key not in {c.key for c in out}:
            out.append(cause)
    return tuple(out)


# ---------------------------------------------------------------------------
# Sentences about one measurement
# ---------------------------------------------------------------------------


def _spec_for(m: Measured | Unavailable) -> MeasurementSpec | None:
    return BY_ID.get(m.spec_id)


EVIDENCE_PROSE: dict[Evidence, str] = {
    Evidence.VALIDATED_2D: (
        "A published study compared this measurement on photographs against the "
        "same measurement taken with calipers on the same people, and the two "
        "agreed to about a millimetre."
    ),
    Evidence.POSE_INVARIANT_RATIO: (
        "Both terms of this ratio lie in the same plane, so the pixel scale "
        "cancels and, to first order, so does head rotation. This is the "
        "sturdiest kind of number a photograph can give."
    ),
    Evidence.REQUIRES_3D: (
        "The endpoints lie on a laterally curved surface that hides itself from "
        "the camera, so this measurement is emitted only from a three-dimensional "
        "fit and is flagged wherever it appears."
    ),
    Evidence.POSE_CRITICAL: (
        "The measurement is defined relative to the horizon, so camera tilt "
        "transfers into it almost degree for degree."
    ),
    Evidence.CONVENTIONAL: (
        "The facial-aesthetics literature uses this measurement and publishes a "
        "range for it, but no study has checked whether a photograph recovers it "
        "in agreement with a caliper."
    ),
}


def evidence_sentence(spec: MeasurementSpec) -> str:
    return EVIDENCE_PROSE[spec.evidence]


def statement(m: Measured) -> str:
    """The value sentence for a measurement that may be shown."""
    if not m.shown:
        return withheld_paragraph(m)
    return f"{m.label} measures {value_phrase(m)}."


def discriminability_sentence(m: Measured) -> str:
    d = m.discriminability
    if d is None:
        return CAUSE_UNKNOWN_SPREAD.plain
    return (
        f"Between-person spread is {d.ratio:.1f} times the measurement error on "
        f"this photograph. Of that error, head pose contributes "
        f"{d.pose_component:.3g} and landmark placement {d.landmark_component:.3g}, "
        f"against a between-person spread of {d.between_subject_sd:.3g}."
    )


def reference_range_sentence(spec: MeasurementSpec) -> str | None:
    """A published range, framed as context.

    The framing sentence is not decoration. A range printed next to a value
    reads as a target unless it is told not to.
    """
    if spec.reference_range is None:
        return None
    lo, hi, source = spec.reference_range
    fmt = _places(spec.unit)
    return (
        f"{source} reports {fmt.format(lo)} to {fmt.format(hi)}"
        f"{_unit_suffix(spec.unit)} for this measurement. That range describes "
        "where a cited sample fell. It is context for reading the number and "
        "not a target."
    )


def stratum_sentence(m: Measured, stratum: NormativeStratum) -> str:
    """Name the reference sample, its size, and how it was measured."""
    fmt = _places(m.unit)
    method = stratum.method or "the method the source describes"
    return (
        f"Compared against {stratum.label} measured by {method}, whose "
        f"{m.label.lower()} averaged {fmt.format(stratum.mean)}"
        f"{_unit_suffix(m.unit)} with a standard deviation of "
        f"{fmt.format(stratum.sd)}{_unit_suffix(m.unit)} "
        f"({stratum.source}). That sample is not a general population, and the "
        "comparison carries no further than the sample does."
    )


def stratum_caveat_sentence(stratum: NormativeStratum) -> str | None:
    return stratum.caveat or None


def _dedupe(found: Sequence[Cause]) -> list[Cause]:
    seen: set[str] = set()
    out: list[Cause] = []
    for c in found:
        if c.key not in seen:
            seen.add(c.key)
            out.append(c)
    return out


def primary_cause(m: Measured) -> Cause | None:
    """The cause that actually suppressed the measurement.

    A withheld measurement usually accumulates several findings, only one of
    which did the suppressing; the rest would have been caveats anyway. Reading
    them as an undifferentiated list tells someone their jaw width was withheld
    because of an assumed scale, when the real answer is that the landmark is
    on a surface the camera cannot see.

    The core verdict carries a severity per finding, so this reads it rather
    than inferring it from the wording.
    """
    blocking = _dedupe([cause_of(r) for r in m.verdict.blocking])
    if blocking:
        return blocking[0]
    found = _dedupe(causes(m))
    return found[0] if found else None


def secondary_causes(m: Measured) -> list[Cause]:
    """Findings recorded against the measurement that did not suppress it."""
    primary = primary_cause(m)
    return [
        c
        for c in _dedupe([cause_of(r) for r in m.verdict.caveats])
        if primary is None or c.key != primary.key
    ]


def withheld_paragraph(m: Measured) -> str:
    """Why a measurement is not printed, naming the cause in plain language."""
    lead = f"{m.label} is withheld."
    primary = primary_cause(m)
    if primary is None:
        return f"{lead} The gate recorded no reason, which is itself a defect."
    text = f"{lead} Cause: {primary.name}. {primary.plain}"
    others = secondary_causes(m)
    if others:
        text += " Also recorded against this measurement: " + "; ".join(
            c.name for c in others
        ) + "."
    return text


def caveat_paragraph(m: Measured) -> str | None:
    """What to hold against a measurement that is shown with reservations."""
    if m.verdict.reportability is not Reportability.CAVEAT:
        return None
    found = _dedupe(causes(m))
    if not found:
        return None
    named = "; ".join(c.name for c in found)
    body = " ".join(c.plain for c in found)
    return f"Read with caution. {named.capitalize()}. {body}"


def unavailable_sentence(u: Unavailable) -> str:
    """Why a measurement was never attempted.

    Distinct from a withheld measurement: this one says the model could not see
    what the formula asks for, not that the number would mean nothing.
    """
    missing = list(u.missing_landmarks)
    if len(missing) == 1 and " " in missing[0]:
        return f"{u.label} was not attempted: {missing[0]}."
    named = ", ".join(missing)
    plural = "landmarks" if len(missing) > 1 else "the landmark"
    return (
        f"{u.label} was not attempted. The landmark model does not supply "
        f"{plural} {named}, and Vitruve does not guess a point it cannot see."
    )


def scale_sentence(m: Measured) -> str | None:
    if m.scale_source is None:
        return None
    notes = " ".join(m.notes) if m.notes else ""
    lead = f"Millimetres come from the {m.scale_source} scale cue."
    return f"{lead} {notes}".strip()


def provenance_sentence(m: Measured) -> str:
    """Formula identity and the points it read.

    Two reports of the same face can be compared only if both say which
    definition they used, so the fingerprint travels with the number.
    """
    return (
        f"Formula fingerprint {m.formula_fingerprint}, evaluated over "
        f"{m.n_samples} draws from the landmark covariances, "
        f"{m.n_valid} of which were finite. Landmarks read: "
        f"{', '.join(m.landmarks_used)}."
    )


@dataclass(frozen=True)
class MeasurementProse:
    """Every sentence the renderer has about one measurement."""

    spec_id: str
    label: str
    headline: str
    evidence: str
    discriminability: str
    caveat: str | None = None
    reference_range: str | None = None
    stratum: str | None = None
    stratum_caveat: str | None = None
    scale: str | None = None
    provenance: str = ""

    def sentences(self) -> tuple[str, ...]:
        return tuple(
            s
            for s in (
                self.headline,
                self.evidence,
                self.discriminability,
                self.caveat,
                self.reference_range,
                self.stratum,
                self.stratum_caveat,
                self.scale,
                self.provenance,
            )
            if s
        )


def describe(m: Measured, stratum: NormativeStratum | None = None) -> MeasurementProse:
    """All the prose for one measurement, shown or withheld."""
    spec = _spec_for(m)
    return MeasurementProse(
        spec_id=m.spec_id,
        label=m.label,
        headline=statement(m),
        evidence=evidence_sentence(spec) if spec else "",
        discriminability=discriminability_sentence(m),
        caveat=caveat_paragraph(m),
        reference_range=reference_range_sentence(spec) if spec else None,
        stratum=stratum_sentence(m, stratum) if (stratum and m.shown) else None,
        stratum_caveat=stratum_caveat_sentence(stratum) if stratum else None,
        scale=scale_sentence(m) if m.shown else None,
        provenance=provenance_sentence(m),
    )


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def cause_counts(report: ReportInput) -> tuple[tuple[Cause, int], ...]:
    """How many measurements each cause accounts for, most common first.

    A count of refusals, not an average of values. The distinction is the
    subject of ``tests/unit/test_no_aggregate_score.py``.
    """
    tally: dict[str, tuple[Cause, int]] = {}
    for m in report.withheld:
        for c in causes(m):
            cause, n = tally.get(c.key, (c, 0))
            tally[c.key] = (cause, n + 1)
    return tuple(sorted(tally.values(), key=lambda t: (-t[1], t[0].name)))


def summary(report: ReportInput) -> tuple[str, ...]:
    """The paragraphs that open the report.

    They lead with the count of what is not here. A report that opens with its
    successes and buries its refusals in an appendix has told the reader that
    the refusals are an embarrassment, and they are the most informative part
    of the document.
    """
    paras: list[str] = []
    attempted = report.n_attempted
    shown = report.n_shown
    paras.append(
        f"{shown} of {attempted} attempted measurements are reported here. "
        f"{report.n_reported} carry no reservation and {report.n_caveated} are "
        f"printed with one. {report.n_withheld} were computed and then withheld, "
        f"and {report.n_unavailable} could not be attempted at all."
    )

    counts = cause_counts(report)
    if counts:
        listed = "; ".join(f"{c.name} ({n})" for c, n in counts)
        paras.append(f"The withheld measurements break down by cause as: {listed}.")

    if report.unavailable:
        missing: dict[str, int] = {}
        for u in report.unavailable:
            for name in u.missing_landmarks:
                missing[name] = missing.get(name, 0) + 1
        ranked = sorted(missing.items(), key=lambda t: (-t[1], t[0]))[:5]
        listed = ", ".join(f"{name} ({n})" for name, n in ranked)
        paras.append(
            "Measurements that were not attempted are blocked by points the "
            f"landmark model does not supply: {listed}."
        )

    paras.append(
        "A withheld measurement is a result. Vitruve judges every quantity "
        "against one question, which is whether it varies more between people "
        "than it varies between photographs of the same person. Kleinberg and "
        "Vanezis (2007) photographed subjects in ten-degree steps and watched "
        "facial indices move 8 to 19 percent at ten degrees of yaw, against a "
        "spread between subjects of 1.2 percent. Where a quantity cannot beat "
        "its own photograph, the reason is printed instead of the number."
    )

    if report.declared_sex or report.declared_ancestry:
        declared = ", ".join(
            v for v in (report.declared_sex, report.declared_ancestry) if v
        )
        paras.append(
            f"The subject declared {declared}. Those attributes select which "
            "reference sample a measurement is placed against. Vitruve never "
            "infers them and never reports them as a finding."
        )
    else:
        paras.append(
            "No sex or ancestry was declared, so any normative comparison uses "
            "a cell pooled over both. Vitruve does not infer either attribute "
            "from the photograph."
        )

    paras.append(
        "This report contains no single number standing for the face as a "
        "whole. There is no ground truth for such a number, only panels of "
        "judges, and roughly half the stable variance in those panels is "
        "private taste rather than anything shared (Hönekopp 2006)."
    )
    return tuple(paras)


def quality_sentences(report: ReportInput) -> tuple[str, ...]:
    out = []
    for issue in report.quality:
        reading = f" ({issue.reading})" if issue.reading else ""
        out.append(f"{issue.detail}{reading}")
    return tuple(out)


def report_text(report: ReportInput) -> str:
    """The whole report as plain text.

    Used by the tests and by anything that wants the prose without the markup.
    The HTML renderer builds from the same sentences, so the two cannot drift.
    """
    lines: list[str] = [report.subject_label, "=" * len(report.subject_label), ""]
    lines.extend(summary(report))
    lines.append("")
    for group in report.groups():
        lines.append(group.region.title)
        lines.append("-" * len(group.region.title))
        lines.append(group.region.blurb)
        lines.append("")
        for m in group.measurements:
            lines.extend(describe(m, report.strata.get(m.spec_id)).sentences())
            lines.append("")
        for u in group.unavailable:
            lines.append(unavailable_sentence(u))
            lines.append("")
    if report.quality:
        lines.append("Photograph quality")
        lines.append("------------------")
        lines.extend(quality_sentences(report))
        lines.append("")
    if report.obligations:
        lines.append("License obligations of the backends that ran")
        lines.append("--------------------------------------------")
        lines.extend(report.obligations)
        lines.append("")
    if report.references:
        lines.append("References")
        lines.append("----------")
        lines.extend(report.references)
    return "\n".join(lines)
