"""One HTML file, no network, nothing hidden.

The report is a single document with the stylesheet inlined and every image
carried as a data URI. That constraint is not tidiness. An analysis that ran
with the network blocked must produce an artifact that renders with the network
blocked, otherwise the offline guarantee lasts exactly until somebody opens the
result.

The ordering of the document is the argument it makes. It opens with how many
measurements survived out of how many were attempted and why the rest did not,
because a report that opens with its successes and appendixes its refusals has
told the reader which of the two it thinks is embarrassing. Then the
measurements by region, each carrying its interval, its evidence tier, its
discriminability ratio and the fingerprint of the formula that produced it.
Then the photograph's own quality, then the licence obligations the run
incurred, then the literature.

No dependency on ``vitruve.models``. The obligations arrive as strings from
whoever loaded the weights, which keeps the renderer free of torch and keeps
the module boundary in ``docs/CORE_API.md`` true.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError as exc:  # pragma: no cover - a packaging problem, not a bug
    raise ImportError(
        "vitruve.report.html renders through a Jinja template; add "
        "jinja2>=3.1 to the project dependencies"
    ) from exc

from ..core.spec import Evidence, MeasurementSpec, Reportability, Unit
from ..measure.evaluate import Measured
from ..measure.registry import BY_ID
from . import prose
from .model import MeasurementGroup, NormativeStratum, ReportInput

TEMPLATES = Path(__file__).parent / "templates"

#: Short codes for the evidence tier, expanded in the legend and in the prose
#: attached to every measurement. An instrument marks its channels.
TIER_CODE: dict[Evidence, str] = {
    Evidence.VALIDATED_2D: "AGREED",
    Evidence.POSE_INVARIANT_RATIO: "RATIO",
    Evidence.REQUIRES_3D: "3D",
    Evidence.POSE_CRITICAL: "TILT",
    Evidence.CONVENTIONAL: "CONV",
}

TIER_GLOSS: dict[Evidence, str] = {
    Evidence.VALIDATED_2D: "agreement with calipers published",
    Evidence.POSE_INVARIANT_RATIO: "same-plane ratio, scale and pose cancel",
    Evidence.REQUIRES_3D: "needs a 3D fit, endpoints self-occlude in a photograph",
    Evidence.POSE_CRITICAL: "defined against the horizon, absorbs camera roll",
    Evidence.CONVENTIONAL: "conventional, no agreement study",
}


@dataclass(frozen=True)
class IntervalBar:
    """Positions along a shared track, in percent.

    Everything is precomputed here rather than in the template, because a
    number that decides how wide a bar is drawn deserves to be reviewable in
    Python and not in a Jinja expression.
    """

    lo_pct: float
    width_pct: float
    mid_pct: float
    axis_lo: str
    axis_hi: str
    band_lo_pct: float = 0.0
    band_width_pct: float = 0.0
    band_label: str = ""

    @property
    def has_band(self) -> bool:
        return self.band_width_pct > 0.0


def _band(
    spec: MeasurementSpec | None, stratum: NormativeStratum | None, fmt: str
) -> tuple[float, float, str] | None:
    """The hatched strip behind the interval, and what it is.

    A published reference range wins where one exists. Otherwise the reference
    sample stands in, drawn as the middle 95% of that sample and labelled with
    who the sample was. Neither is a target, and the key line under the bar
    says so in as many words.
    """
    if spec is not None and spec.reference_range is not None:
        lo, hi, source = spec.reference_range
        return (
            float(lo),
            float(hi),
            f"published range for context, {fmt.format(lo)} to {fmt.format(hi)} "
            f"({source})",
        )
    if stratum is not None and stratum.sd > 0:
        lo, hi = stratum.mean - 2 * stratum.sd, stratum.mean + 2 * stratum.sd
        return (
            lo,
            hi,
            f"where the middle 95% of {stratum.label} fell, "
            f"{fmt.format(lo)} to {fmt.format(hi)}",
        )
    return None


def _interval_bar(
    m: Measured,
    spec: MeasurementSpec | None,
    stratum: NormativeStratum | None = None,
) -> IntervalBar | None:
    fmt = prose._places(m.unit)
    lo, hi = float(m.ci_low), float(m.ci_high)
    if not all(map(_finite, (lo, hi, m.value))):
        return None
    band = _band(spec, stratum, fmt)
    points = [lo, hi, float(m.value)]
    if band is not None:
        points += [float(band[0]), float(band[1])]
    dom_lo, dom_hi = min(points), max(points)
    span = dom_hi - dom_lo
    pad = span * 0.12 if span > 0 else max(abs(float(m.value)) * 0.1, 1e-6)
    dom_lo, dom_hi = dom_lo - pad, dom_hi + pad
    span = dom_hi - dom_lo

    def pct(x: float) -> float:
        return round((x - dom_lo) / span * 100.0, 3)

    bar = IntervalBar(
        lo_pct=pct(lo),
        width_pct=round(pct(hi) - pct(lo), 3),
        mid_pct=pct(float(m.value)),
        axis_lo=fmt.format(dom_lo),
        axis_hi=fmt.format(dom_hi),
    )
    if band is None:
        return bar
    b_lo, b_hi, label = band
    return IntervalBar(
        lo_pct=bar.lo_pct,
        width_pct=bar.width_pct,
        mid_pct=bar.mid_pct,
        axis_lo=bar.axis_lo,
        axis_hi=bar.axis_hi,
        band_lo_pct=pct(float(b_lo)),
        band_width_pct=round(pct(float(b_hi)) - pct(float(b_lo)), 3),
        band_label=label,
    )


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


@dataclass(frozen=True)
class MeasurementRow:
    """One measurement as the template needs it."""

    spec_id: str
    label: str
    shown: bool
    status: str
    tier_code: str
    tier_gloss: str
    value_text: str
    interval_text: str
    unit_text: str
    discriminability: str
    discriminability_title: str
    prose: prose.MeasurementProse
    reasons: tuple[str, ...]
    bar: IntervalBar | None = None


def _row(m: Measured, report: ReportInput) -> MeasurementRow:
    spec = BY_ID.get(m.spec_id)
    stratum = report.strata.get(m.spec_id)
    fmt = prose._places(m.unit)
    d = m.discriminability
    shown = m.shown
    return MeasurementRow(
        spec_id=m.spec_id,
        label=m.label,
        shown=shown,
        status={
            Reportability.REPORT: "reported",
            Reportability.CAVEAT: "reported with a caveat",
            Reportability.WITHHOLD: "withheld",
        }[m.verdict.reportability],
        tier_code=TIER_CODE[spec.evidence] if spec else "",
        tier_gloss=TIER_GLOSS[spec.evidence] if spec else "",
        value_text=fmt.format(m.value) if shown else "withheld",
        interval_text=(
            f"{fmt.format(m.ci_low)} to {fmt.format(m.ci_high)}" if shown else ""
        ),
        unit_text="" if (m.unit is Unit.RATIO or not shown) else m.unit.value,
        discriminability=f"{d.ratio:.2f}" if d else "unknown",
        discriminability_title=prose.discriminability_sentence(m),
        prose=prose.describe(m, stratum),
        reasons=m.verdict.reasons,
        bar=_interval_bar(m, spec, stratum) if shown else None,
    )


@dataclass(frozen=True)
class GroupView:
    group: MeasurementGroup
    rows: tuple[MeasurementRow, ...]
    unavailable: tuple[str, ...]

    @property
    def region(self):
        return self.group.region

    @property
    def counted(self) -> str:
        shown = sum(1 for r in self.rows if r.shown)
        return f"{shown} of {len(self.rows) + len(self.unavailable)}"


def _manifest_rows(manifest: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    out = []
    for key in sorted(manifest):
        value = manifest[key]
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, sort_keys=True, default=str)
        else:
            text = str(value)
        out.append((key, text))
    return tuple(out)


def build_context(report: ReportInput) -> dict[str, Any]:
    """Everything the template may read, and nothing it must compute.

    Kept as a plain dictionary so a test can assert on what the template is
    allowed to see, which is how the no-aggregate rule is checked at the
    template boundary rather than only in the rendered string.
    """
    groups = report.groups()
    views = tuple(
        GroupView(
            group=g,
            rows=tuple(_row(m, report) for m in g.measurements),
            unavailable=tuple(prose.unavailable_sentence(u) for u in g.unavailable),
        )
        for g in groups
    )
    generated = report.generated_at or datetime.now(UTC).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    return {
        "report": report,
        "css": (TEMPLATES / "report.css").read_text(),
        "subject": report.subject_label,
        "generated_at": generated,
        "summary": prose.summary(report),
        "cause_counts": prose.cause_counts(report),
        "groups": views,
        "quality": report.quality,
        "obligations": report.obligations,
        "references": report.references,
        "manifest": _manifest_rows(report.manifest),
        "tier_legend": tuple(
            (TIER_CODE[e], TIER_GLOSS[e]) for e in Evidence
        ),
        "counts": {
            "attempted": report.n_attempted,
            "shown": report.n_shown,
            "reported": report.n_reported,
            "caveated": report.n_caveated,
            "withheld": report.n_withheld,
            "unavailable": report.n_unavailable,
        },
    }


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        # Unconditionally, not by file extension: the template is named
        # ``report.html.j2`` and an extension-driven policy would look at the
        # ``.j2`` and quietly decide this was not HTML.
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(report: ReportInput) -> str:
    """The whole report as one self-contained HTML string."""
    env = _environment()
    return env.get_template("report.html.j2").render(**build_context(report))


def write(report: ReportInput, path: str | Path) -> Path:
    out = Path(path)
    out.write_text(render(report), encoding="utf-8")
    return out
