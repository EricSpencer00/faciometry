"""The analyze command.

Its job is narrow: turn paths and flags into an :class:`AnalysisRequest`, hand
that to the pipeline, and turn what comes back into an exit code and a
rendering. The measurement decisions all happen further in; nothing here
decides whether a number may be printed.

One thing it does decide is that selecting a tier above permissive prints what
that tier costs before the run rather than after. The obligation attaches when
the weights load, and a user who finds out afterwards has already accepted it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ..models.licensing import Tier
from .exits import Exit
from .runner import (
    AnalysisRequest,
    BadImage,
    Status,
    load_image_file,
    render_html,
    render_text,
    run_analysis,
    to_dict,
)

_STATUS_EXIT = {
    Status.OK: Exit.OK,
    Status.BAD_INPUT: Exit.BAD_INPUT,
    Status.QUALITY_GATE: Exit.QUALITY_GATE,
    Status.LICENSE: Exit.LICENSE,
    Status.UNAVAILABLE: Exit.ERROR,
}


def run(
    *,
    frontal: str | Sequence[str],
    profile: str | None = None,
    license_tier: Tier = Tier.PERMISSIVE,
    declared_sex: str | None = None,
    declared_ancestry: str | None = None,
    ruler_mm: float | None = None,
    out: str | None = None,
    as_json: bool = False,
    seed: int = 0,
    pdf: bool = False,
) -> int:
    frontal_paths = [frontal] if isinstance(frontal, str) else list(frontal)
    try:
        frontal_images = [load_image_file(p) for p in frontal_paths]
        profile_image = load_image_file(profile) if profile else None
    except BadImage as exc:
        print(f"vitruve analyze: {exc}", file=sys.stderr)
        return Exit.BAD_INPUT

    if license_tier > Tier.PERMISSIVE:
        print(
            f"vitruve analyze: license tier {license_tier.name.lower()} selected. "
            f"Run `vitruve licenses --tier {license_tier.name.lower()}` for what that "
            "obliges you to.",
            file=sys.stderr,
        )

    request = AnalysisRequest(
        frontal=frontal_images[0],
        extra_frontals=tuple(frontal_images[1:]),
        profile=profile_image,
        license_tier=license_tier,
        declared_sex=declared_sex,
        declared_ancestry=declared_ancestry,
        ruler_mm=ruler_mm,
        seed=seed,
    )
    outcome = run_analysis(request)

    if outcome.status is Status.OK:
        payload = to_dict(outcome)
        text = render_text(outcome)
        print(json.dumps(payload, indent=2) if as_json else text)
        if out:
            _write(Path(out), payload, text, render_html(outcome))
            if pdf and (code := _write_pdf(Path(out), outcome)) != Exit.OK:
                return code
        elif pdf:
            print(
                "vitruve analyze: --pdf needs --out, because a PDF is a file "
                "rather than something to print to a terminal",
                file=sys.stderr,
            )
            return Exit.BAD_INPUT
        return Exit.OK

    print(f"vitruve analyze: {outcome.message}", file=sys.stderr)
    for reason in outcome.reasons:
        print(f"  {reason}", file=sys.stderr)
    return _STATUS_EXIT[outcome.status]


def _write_pdf(directory: Path, outcome) -> int:
    """Write the typeset report, or explain why it could not be.

    Kept out of `_write` because the PDF is the one output with a dependency
    the core install does not carry, and a missing optional extra should name
    its install command rather than produce a stack trace.
    """
    try:
        from ..report.pdf import PDFUnavailable, render_pdf
        from .runner import build_report_input
    except ImportError as exc:  # pragma: no cover - defensive
        print(f"vitruve analyze: the PDF report is unavailable ({exc})", file=sys.stderr)
        return Exit.ERROR
    try:
        written = render_pdf(build_report_input(outcome), directory / "report.pdf")
    except PDFUnavailable as exc:
        print(f"vitruve analyze: {exc}", file=sys.stderr)
        return Exit.ERROR
    print(f"wrote {written}", file=sys.stderr)
    return Exit.OK


def _write(directory: Path, payload: dict, text: str, html: str | None) -> None:
    """Write the report, and nothing else.

    The source photographs are not copied. The HTML report does embed the
    annotated overlays, which are crops of the face with the landmarks and
    their uncertainty ellipses drawn on, so it is a picture of the subject and
    `docs/PRIVACY.md` says so. The JSON and text reports carry measurements and
    provenance only.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = [directory / "report.json", directory / "report.txt"]
    (directory / "report.json").write_text(json.dumps(payload, indent=2))
    (directory / "report.txt").write_text(text + "\n")
    if html is not None:
        (directory / "report.html").write_text(html)
        written.append(directory / "report.html")
    print("wrote " + ", ".join(str(p) for p in written), file=sys.stderr)
