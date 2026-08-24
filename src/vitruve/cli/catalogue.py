"""Printing the measurement catalogue as a table you can decide from.

This is the command to run first. The catalogue is 45 measurements and the
useful thing about it is not the list of names, it is the last three columns:
how far each measurement moves when the head turns ten degrees, how far it
spreads between different people, and the ratio of the two. Kleinberg and
Vanezis (2007) found facial indices moving 8 to 19 percent at ten degrees of
yaw against a between-subject spread of 1.2 percent, so that ratio is the
number that decides whether a measurement carries information about a person
or about a photograph.

The rows are the same data the API serves at ``GET /catalogue`` and the same
data the report gates on. There is one catalogue and it is
``vitruve.measure.registry.CATALOGUE``.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from dataclasses import dataclass

from ..core.spec import Evidence, MeasurementSpec, Unit, View
from ..measure.registry import BY_ID, CATALOGUE

#: The pose Vitruve quotes sensitivity at: ten degrees on each of yaw, pitch
#: and roll, combined in quadrature. Ten degrees because that is the step
#: Kleinberg and Vanezis photographed at, and all three axes because a
#: hand-held photograph rarely misses on only one.
QUOTED_POSE_DEG = 10.0

EVIDENCE_SHORT: dict[Evidence, str] = {
    Evidence.VALIDATED_2D: "validated",
    Evidence.POSE_INVARIANT_RATIO: "ratio-inv",
    Evidence.REQUIRES_3D: "needs-3d",
    Evidence.POSE_CRITICAL: "pose-crit",
    Evidence.CONVENTIONAL: "convention",
}

#: Sort order: most trustworthy tier first.
EVIDENCE_RANK: dict[Evidence, int] = {
    Evidence.VALIDATED_2D: 0,
    Evidence.POSE_INVARIANT_RATIO: 1,
    Evidence.REQUIRES_3D: 2,
    Evidence.POSE_CRITICAL: 3,
    Evidence.CONVENTIONAL: 4,
}

EVIDENCE_MEANING: dict[Evidence, str] = {
    Evidence.VALIDATED_2D: (
        "published Bland-Altman agreement with direct anthropometry under about a "
        "millimetre. Report freely."
    ),
    Evidence.POSE_INVARIANT_RATIO: (
        "a ratio of two quantities in the same plane, so scale cancels and to first "
        "order so does pose. The most robust output this pipeline has."
    ),
    Evidence.REQUIRES_3D: (
        "the endpoints sit on a laterally curved, self-occluding surface, so a 2D "
        "photograph gives a silhouette artifact instead of the anatomical point. "
        "Lim et al. 2022 measured bigonial breadth against direct anthropometry at a "
        "mean difference of 9.3 mm with limits of agreement from -0.9 to 19.6 mm. "
        "Emitted only from a 3D fit."
    ),
    Evidence.POSE_CRITICAL: (
        "absorbs pose error close to one for one against a normal range only a few "
        "degrees wide. Canthal tilt is defined against the horizon, so image roll "
        "transfers straight into it."
    ),
    Evidence.CONVENTIONAL: (
        "standard vocabulary in the facial-aesthetics literature with a cited "
        "reference range, and no published agreement study against a caliper."
    ),
}


@dataclass(frozen=True)
class Row:
    """One catalogue row, flattened for a table or for JSON."""

    id: str
    label: str
    view: str
    unit: str
    evidence: str
    description: str
    pose_tolerance_deg: float
    #: How far this measurement moves at :data:`QUOTED_POSE_DEG` on all three
    #: axes: a fraction of its own value for lengths and ratios, degrees for
    #: angles.
    move_at_quoted_pose: float
    #: Between-subject spread in the same units as ``move_at_quoted_pose``.
    #: ``None`` means nobody has published one, which is reported as unknown
    #: rather than guessed.
    between_subject_spread: float | None
    #: Spread over movement. Above one, the measurement separates people
    #: better than it separates photographs of the same person.
    discriminability_at_quoted_pose: float | None
    measured_within_person_spread: float | None
    landmarks: tuple[str, ...]
    references: tuple[str, ...]
    reference_range: tuple[float, float, str] | None
    formula_fingerprint: str
    sensitivity_source: str


def _row(spec: MeasurementSpec) -> Row:
    move = spec.sensitivity.error_at(QUOTED_POSE_DEG, QUOTED_POSE_DEG, QUOTED_POSE_DEG)
    if spec.measured_within_person_rsd is not None:
        # A measured within-person spread contains the pose term plus
        # everything the projection model cannot see, so it replaces the
        # derived figure exactly as `assess_discriminability` does.
        move = max(move, spec.measured_within_person_rsd)
    spread = spec.between_subject_rsd
    ratio = (spread / move) if (spread is not None and move > 0) else None
    return Row(
        id=spec.id,
        label=spec.label,
        view=spec.view.value,
        unit=spec.unit.value,
        evidence=spec.evidence.value,
        description=spec.description,
        pose_tolerance_deg=spec.pose_tolerance_deg,
        move_at_quoted_pose=move,
        between_subject_spread=spread,
        discriminability_at_quoted_pose=ratio,
        measured_within_person_spread=spec.measured_within_person_rsd,
        landmarks=tuple(sorted(m.value for m in spec.landmarks)),
        references=spec.references,
        reference_range=spec.reference_range,
        formula_fingerprint=spec.fingerprint,
        sensitivity_source=spec.sensitivity.source,
    )


def rows(
    *, view: View | None = None, evidence: Evidence | None = None
) -> tuple[Row, ...]:
    """Catalogue rows, filtered, worst-behaved measurements last."""
    specs = [
        s
        for s in CATALOGUE
        if (view is None or s.view is view or s.view is View.EITHER)
        and (evidence is None or s.evidence is evidence)
    ]
    specs.sort(key=lambda s: (EVIDENCE_RANK[s.evidence], -(_row(s).discriminability_at_quoted_pose or 0.0), s.id))
    return tuple(_row(s) for s in specs)


def _fmt_amount(value: float | None, unit: str) -> str:
    """Angles in degrees, everything else as a percentage of its own value."""
    if value is None:
        return "unknown"
    if unit == Unit.DEGREES.value:
        return f"{value:.2f} deg"
    return f"{value * 100:.2f}%"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 100:
        return f"{value:.0f}x"
    return f"{value:.2f}x"


def render_table(selected: tuple[Row, ...]) -> str:
    """The full table, grouped by view."""
    width = min(shutil.get_terminal_size((100, 24)).columns, 104)
    out: list[str] = []
    out.append(f"Vitruve measurement catalogue: {len(selected)} of {len(CATALOGUE)} measurements")
    out.append("")

    header = (
        f"{'measurement':<32} {'unit':<5} {'evidence':<10} {'tol':>4} "
        f"{'moves @10deg':>13} {'between people':>15} {'ratio':>8}"
    )
    for view_name in ("frontal", "profile"):
        group = [r for r in selected if r.view == view_name]
        if not group:
            continue
        out.append(f"{view_name.upper()} VIEW ({len(group)})")
        out.append(header)
        out.append("-" * min(len(header), width))
        for r in group:
            out.append(
                f"{r.id:<32.32} {r.unit:<5} {EVIDENCE_SHORT[Evidence(r.evidence)]:<10} "
                f"{r.pose_tolerance_deg:>3.0f}d "
                f"{_fmt_amount(r.move_at_quoted_pose, r.unit):>13} "
                f"{_fmt_amount(r.between_subject_spread, r.unit):>15} "
                f"{_fmt_ratio(r.discriminability_at_quoted_pose):>8}"
            )
        out.append("")

    out.append("COLUMNS")
    out.extend(
        textwrap.fill(line, width=width, initial_indent="  ", subsequent_indent="      ")
        for line in (
            "tol   the head-pose tolerance for this measurement, in degrees. Past it the "
            "value is caveated, and past twice it the value is withheld.",
            "moves @10deg   how far the value shifts at 10 degrees of yaw, pitch and roll "
            "together, in quadrature. Degrees for an angle, a percentage of its own value "
            "otherwise. Where a study has measured the real within-person spread, that "
            "number is used instead, because it also contains expression and landmark "
            "drift, which no projection model predicts.",
            "between people   the spread of this measurement across different people, in "
            "the same units. 'unknown' means nobody has published one, which is reported "
            "rather than guessed.",
            "ratio   between people divided by moves @10deg. Above 1 the measurement "
            "carries more person than photograph. Below 1 Vitruve withholds it.",
        )
    )
    out.append("")
    out.append("EVIDENCE TIERS")
    for ev in sorted(EVIDENCE_MEANING, key=lambda e: EVIDENCE_RANK[e]):
        n = sum(1 for r in selected if r.evidence == ev.value)
        out.append(
            textwrap.fill(
                f"{EVIDENCE_SHORT[ev]} ({n}) {EVIDENCE_MEANING[ev]}",
                width=width,
                initial_indent="  ",
                subsequent_indent="    ",
            )
        )
    out.append("")
    ratios = [
        r.discriminability_at_quoted_pose
        for r in selected
        if r.discriminability_at_quoted_pose is not None
    ]
    known = ratios
    beats = [x for x in ratios if x > 1.0]
    out.append(
        textwrap.fill(
            f"{len(beats)} of the {len(known)} measurements with a published between-person "
            f"spread beat their own pose error at 10 degrees. The other "
            f"{len(selected) - len(known)} have no published spread, so whether they "
            "distinguish anyone is unknown. Kleinberg and Vanezis (2007) is where the "
            "10-degree figure comes from; FISWG 2026 (V2.1, 6.4.1) prohibits "
            "photo-anthropometry for identification on the same evidence.",
            width=width,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )
    return "\n".join(out)


def render_detail(spec_id: str) -> str:
    """Everything known about one measurement."""
    spec = BY_ID[spec_id]
    r = _row(spec)
    out = [f"{spec.label}", "=" * len(spec.label), ""]
    if spec.description:
        out.append(textwrap.fill(spec.description, width=88))
        out.append("")
    pairs: list[tuple[str, str]] = [
        ("id", r.id),
        ("view", r.view),
        ("unit", r.unit),
        ("evidence", f"{r.evidence} -- {EVIDENCE_MEANING[spec.evidence]}"),
        ("formula", f"{r.formula_fingerprint} over {', '.join(r.landmarks)}"),
        ("pose tolerance", f"{r.pose_tolerance_deg:.0f} degrees"),
        (
            "pose sensitivity",
            f"yaw {spec.sensitivity.yaw:.4g}, pitch {spec.sensitivity.pitch:.4g}, "
            f"roll {spec.sensitivity.roll:.4g} per degree ({r.sensitivity_source})",
        ),
        ("moves at 10 deg", _fmt_amount(r.move_at_quoted_pose, r.unit)),
        ("between people", _fmt_amount(r.between_subject_spread, r.unit)),
        ("ratio", _fmt_ratio(r.discriminability_at_quoted_pose)),
    ]
    if spec.measured_within_person_rsd is not None:
        pairs.append(
            ("measured within person", _fmt_amount(spec.measured_within_person_rsd, r.unit))
        )
    if spec.reference_range is not None:
        lo, hi, src = spec.reference_range
        pairs.append(("reference range", f"{lo:g} to {hi:g} {r.unit} ({src})"))
    for key, value in pairs:
        out.append(
            textwrap.fill(
                f"{key:<22} {value}", width=88, subsequent_indent=" " * 23
            )
        )
    if spec.within_person_source:
        out.append("")
        out.append(textwrap.fill(spec.within_person_source, width=88, initial_indent="  ", subsequent_indent="  "))
    if spec.references:
        out.append("")
        out.append("references")
        for ref in spec.references:
            out.append(textwrap.fill(ref, width=88, initial_indent="  - ", subsequent_indent="    "))
    return "\n".join(out)


def to_json(selected: tuple[Row, ...]) -> str:
    return json.dumps(
        {
            "quoted_pose_deg": QUOTED_POSE_DEG,
            "n_total": len(CATALOGUE),
            "measurements": [r.__dict__ for r in selected],
        },
        indent=2,
        default=list,
    )


def run(
    *,
    view: View | None,
    evidence: Evidence | None,
    spec_id: str | None,
    as_json: bool,
) -> int:
    if spec_id is not None:
        if spec_id not in BY_ID:
            print(f"no measurement with id {spec_id!r}. Run `vitruve catalogue` for the list.")
            return 2
        if as_json:
            print(json.dumps(_row(BY_ID[spec_id]).__dict__, indent=2, default=list))
        else:
            print(render_detail(spec_id))
        return 0
    selected = rows(view=view, evidence=evidence)
    print(to_json(selected) if as_json else render_table(selected))
    return 0
