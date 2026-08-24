"""No scalar aggregate over measurements, anywhere, in any output.

This is the load-bearing test of the project. Deleting it is a decision about
what Vitruve is, so the reasoning lives here next to the assertions rather than
only in the design document.

**A single number has no construct validity.** There is no ground truth for
facial attractiveness to validate a predictor against. There is only a panel of
raters, and a model trained on panel ratings learns that panel: its
demographics, its era, and its instructions. Calling the output a measurement
does not make a measurement of it. Every quantity Vitruve does print can be
checked against a caliper on the same face, and a rating cannot.

**Roughly half of the stable variance in those panels is private taste.**
Hoenekopp (2006) decomposed attractiveness ratings into shared and private
components and found the private component about as large as the shared one.
An aggregate built from such ratings averages away the half that belongs to the
individual rater and then presents the remainder as a property of the face.

**It is also the documented harm vector.** The measurements are what people
come for; the ranking is what hurts them. Vitruve reports the measurements and
does not compute the ranking. A withheld measurement is a result, a
discriminability ratio is a result, and neither of them is a verdict on a face.

What is *not* forbidden, and what these tests take care to permit:

* Counts of outcomes. How many measurements were reportable, how many were
  withheld and under which cause, are facts about the photograph and the state
  of the literature. They say nothing about the face.
* A per-measurement number: a value, its interval, its discriminability ratio.
  One measurement is not an aggregate over measurements.
* The mean and standard deviation of a *reference sample*, which describes the
  cited population and not this subject.

The line these tests draw is that no object holding more than one measurement
may expose a number derived from the values of those measurements. That is
checked structurally rather than by name: the whole report is rendered twice,
with every measured value scaled by a constant in the second pass, and every
number the multi-measurement objects expose has to come back unchanged.
"""

from __future__ import annotations

import dataclasses
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_report_prose import synthetic_report

from vitruve.report import html, model, overlay, prose
from vitruve.report.model import MeasurementGroup, ReportInput

#: Phrases that would mean Vitruve had started rating faces. Matched against
#: the rendered document with the stylesheet and the base64 images removed.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "overall score",
    "overall rating",
    "harmony",
    "harmony index",
    "attractiveness",
    "attractive",
    "beauty score",
    "facial score",
    "composite score",
    "total score",
    "aggregate score",
    "rating",
    "out of 10",
    "out of ten",
    "percentile rank",
    "your score",
    "final score",
    "overall grade",
)

#: Fragments no public name in the report package may contain.
FORBIDDEN_NAMES: tuple[str, ...] = (
    "score",
    "overall",
    "harmony",
    "attractiv",
    "beauty",
    "rating",
    "aggregate",
    "composite",
    "ranking",
    "percentile_rank",
)

TEMPLATE = Path(html.TEMPLATES) / "report.html.j2"


@pytest.fixture(scope="module")
def report() -> ReportInput:
    return synthetic_report()


def _text_only(rendered: str) -> str:
    body = re.sub(r"<style>.*?</style>", "", rendered, flags=re.S)
    return re.sub(r'data:image/png;base64,[^"]*', "", body)


# ---------------------------------------------------------------------------
# The rendered document
# ---------------------------------------------------------------------------


def test_rendered_report_contains_no_rating_vocabulary(report: ReportInput):
    body = _text_only(html.render(report)).lower()
    hits = [phrase for phrase in FORBIDDEN_TEXT if phrase in body]
    assert hits == [], f"the rendered report reads like a rating: {hits}"


def test_plain_text_report_contains_no_rating_vocabulary(report: ReportInput):
    text = prose.report_text(report).lower()
    hits = [phrase for phrase in FORBIDDEN_TEXT if phrase in text]
    assert hits == [], hits


def test_the_document_says_it_does_not_do_this(report: ReportInput):
    """The refusal is stated to the reader, not only to the test suite."""
    body = _text_only(html.render(report))
    assert "single number standing for the face as a whole" in body
    assert "does not combine them, rank them" in body


# ---------------------------------------------------------------------------
# The public surface
# ---------------------------------------------------------------------------


def _public_names(module) -> set[str]:
    out = set()
    for name, value in vars(module).items():
        if name.startswith("_") or isinstance(value, types.ModuleType):
            continue
        origin = getattr(value, "__module__", "vitruve.report")
        if origin is not None and not str(origin).startswith("vitruve.report"):
            continue
        out.add(name)
        if dataclasses.is_dataclass(value):
            out.update(f.name for f in dataclasses.fields(value))
    return out


def test_no_public_name_suggests_an_aggregate():
    names: set[str] = set()
    for module in (model, prose, html, overlay):
        names |= _public_names(module)
    assert names, "the scan found nothing, so it is not scanning anything"
    hits = sorted(n for n in names if any(f in n.lower() for f in FORBIDDEN_NAMES))
    assert hits == [], hits


def test_template_variables_suggest_no_aggregate(report: ReportInput):
    source = TEMPLATE.read_text()
    used = set(re.findall(r"\{\{\s*([a-zA-Z_][\w.]*)", source))
    hits = sorted(v for v in used if any(f in v.lower() for f in FORBIDDEN_NAMES))
    assert hits == [], hits
    context = set(html.build_context(report))
    hits = sorted(k for k in context if any(f in k.lower() for f in FORBIDDEN_NAMES))
    assert hits == [], hits


def test_template_does_not_aggregate_measurements():
    """No Jinja filter in the template folds the rows into one number."""
    source = TEMPLATE.read_text()
    for construct in ("|sum", "| sum", "|average", "| average", "sum(", "mean("):
        assert construct not in source, construct


# ---------------------------------------------------------------------------
# The structural argument
# ---------------------------------------------------------------------------


def _numbers_on(obj) -> dict[str, object]:
    """Every number an object exposes under a public name."""
    out: dict[str, object] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # pragma: no cover - a property that needs arguments
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = value
    return out


def _multi_measurement_objects(report: ReportInput) -> list[object]:
    """Everything in the report layer that holds more than one measurement.

    An aggregate over measurements has nowhere else to live. A number hanging
    off a single row is a property of that measurement.
    """
    objects: list[object] = [report]
    objects.extend(report.groups())
    objects.extend(html.build_context(report)["groups"])
    return objects


def _scaled(report: ReportInput, factor: float) -> ReportInput:
    """The same report with every measured value multiplied by a constant.

    Verdicts, intervals relative to their values, discriminability ratios and
    every count are untouched, so anything that moves under this transform is a
    function of the measured values themselves.
    """
    return dataclasses.replace(
        report,
        measurements=tuple(
            dataclasses.replace(
                m,
                value=m.value * factor,
                ci_low=m.ci_low * factor,
                ci_high=m.ci_high * factor,
                sd=m.sd * factor,
            )
            for m in report.measurements
        ),
    )


def test_multi_measurement_objects_expose_only_counts(report: ReportInput):
    for obj in _multi_measurement_objects(report):
        for name, value in _numbers_on(obj).items():
            assert isinstance(value, int), (
                f"{type(obj).__name__}.{name} is a float spanning several "
                "measurements, which is what an aggregate score looks like"
            )


def test_no_number_moves_when_every_value_is_scaled(report: ReportInput):
    """The property that makes the previous test more than a naming rule."""
    other = _scaled(report, 7.3)
    before = [_numbers_on(o) for o in _multi_measurement_objects(report)]
    after = [_numbers_on(o) for o in _multi_measurement_objects(other)]
    assert len(before) == len(after)
    for a, b in zip(before, after, strict=True):
        assert a == b, "a number spanning several measurements moved with them"


def test_the_scaling_control_actually_changes_the_report(report: ReportInput):
    """Without this, the test above would pass on a transform that did nothing."""
    other = _scaled(report, 7.3)
    assert report.measurements != other.measurements
    assert html.render(report) != html.render(other)
    shown = report.shown[0]
    assert prose.value_phrase(shown) != prose.value_phrase(
        next(m for m in other.shown if m.spec_id == shown.spec_id)
    )


def test_counts_add_up_to_the_number_attempted(report: ReportInput):
    """The one figure the report leads with has to be the whole of it."""
    assert (
        report.n_reported
        + report.n_caveated
        + report.n_withheld
        + report.n_unavailable
        == report.n_attempted
    )
    assert report.n_shown == report.n_reported + report.n_caveated


def test_the_report_is_not_itself_a_number(report: ReportInput):
    for dunder in ("__float__", "__int__", "__index__"):
        assert not hasattr(report, dunder)
    for group in report.groups():
        for dunder in ("__float__", "__int__", "__index__"):
            assert not hasattr(group, dunder)


def test_groups_hold_measurements_rather_than_summarising_them(report: ReportInput):
    groups = report.groups()
    assert groups
    assert sum(len(g.measurements) for g in groups) == len(report.measurements)
    assert all(isinstance(g, MeasurementGroup) for g in groups)
