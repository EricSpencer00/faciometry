"""The HTML renderer: one file, no network, and nothing buried.

The document has to survive being opened on the machine that produced it, with
the network switched off, which is the machine the analysis was supposed to run
on. Every assertion about self-containment here is really an assertion about
that.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from markupsafe import escape
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from test_report_prose import FACE, point_set, synthetic_report, uncertainty

from vitruve.report import html, overlay, prose
from vitruve.report.model import ReportInput


def esc(text: str) -> str:
    """The same text as the template would have written it."""
    return str(escape(text))


def text_only(rendered: str) -> str:
    """The document with the stylesheet and the base64 images taken out.

    A base64 payload is a few hundred thousand arbitrary characters, so any
    substring search over the raw document finds whatever it looks for.
    """
    body = re.sub(r"<style>.*?</style>", "", rendered, flags=re.S)
    return re.sub(r"data:image/png;base64,[^\"]*", "", body)


IMAGE_W, IMAGE_H = 760, 900
_ORIGIN = (IMAGE_W / 2, 430.0)
_ZOOM = 0.82


def _to_image(x: float, y: float) -> tuple[float, float]:
    """Canonical frame to image pixels: +y is up in one and down in the other."""
    return _ORIGIN[0] + x * _ZOOM, _ORIGIN[1] - y * _ZOOM


def synthetic_photograph() -> Image.Image:
    """A stand-in for a face photograph, built from numpy so nothing is fetched."""
    yy, xx = np.mgrid[0:IMAGE_H, 0:IMAGE_W]
    r = np.hypot(xx - IMAGE_W / 2, (yy - IMAGE_H / 2) * 1.25) / (IMAGE_W / 2)
    shade = np.clip(1.05 - 0.55 * r, 0.0, 1.0)
    rgb = np.stack(
        [shade * 196, shade * 168, shade * 148], axis=-1
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def plotted_landmarks() -> tuple[overlay.PlottedLandmark, ...]:
    ps = point_set()
    unc = uncertainty(ps)
    flip = np.diag([_ZOOM, -_ZOOM])
    out = []
    for name, i in ps.index.items():
        x, y = _to_image(*FACE[name][:2])
        cov = flip @ np.asarray(unc.covariances[i])[:2, :2] @ flip.T
        out.append(overlay.PlottedLandmark(name=name.value, x=x, y=y, cov=cov))
    return tuple(out)


@pytest.fixture(scope="module")
def report() -> ReportInput:
    base = synthetic_report()
    overlays = overlay.overlays_for_groups(
        synthetic_photograph(), plotted_landmarks(), base.groups(), roll_deg=1.5
    )
    return dataclasses.replace(base, overlays=overlays)


@pytest.fixture(scope="module")
def rendered(report: ReportInput) -> str:
    return html.render(report)


# ---------------------------------------------------------------------------
# One file, no network
# ---------------------------------------------------------------------------


def test_renders_a_whole_document(rendered: str):
    assert rendered.lstrip().startswith("<!doctype html>")
    assert rendered.rstrip().endswith("</html>")
    assert "<style>" in rendered


def test_nothing_is_fetched_when_the_report_is_opened(rendered: str):
    for forbidden in ('src="http', "src='http", 'href="http', "href='http",
                      "<link", "<script", "@import", "url(http", "//cdn"):
        assert forbidden not in rendered, forbidden


def test_every_image_is_inlined(rendered: str):
    sources = re.findall(r'<img[^>]*src="([^"]{0,40})', rendered)
    assert sources, "the report carries no overlay at all"
    assert all(s.startswith("data:image/png;base64,") for s in sources)


def test_overlays_appear_with_a_caption_explaining_the_ellipses(
    report: ReportInput, rendered: str
):
    assert report.overlays
    for o in report.overlays:
        assert o.title in rendered
        assert "uncertainty ellipses" in rendered
        assert "95% of where the model believes the point lies" in rendered


# ---------------------------------------------------------------------------
# The honest parts are not buried
# ---------------------------------------------------------------------------


def test_the_document_opens_with_the_count(report: ReportInput, rendered: str):
    tally = f"{report.n_shown}<span>/{report.n_attempted}</span>"
    assert tally in rendered
    assert "measurements reportable, of those attempted" in rendered
    first_region = min(
        rendered.index(g.region.title) for g in report.groups()
    )
    assert rendered.index(tally) < first_region


def test_the_ledger_accounts_for_every_attempted_measurement(report: ReportInput, rendered: str):
    total = (
        report.n_reported
        + report.n_caveated
        + report.n_withheld
        + report.n_unavailable
    )
    assert total == report.n_attempted
    for label in ("reported without reservation", "reported with a caveat",
                  "computed, then withheld", "not attempted, landmark missing"):
        assert label in rendered


def test_every_shown_value_is_printed_with_its_interval(report: ReportInput, rendered: str):
    body = text_only(rendered)
    for m in report.shown:
        fmt = prose._places(m.unit)
        interval = f"95% interval {fmt.format(m.ci_low)} to {fmt.format(m.ci_high)}"
        assert interval in body, m.spec_id


def test_no_bare_value_element_exists(rendered: str):
    """Every value block in the markup carries an interval block beside it."""
    values = re.findall(r'<div class="m__value">(.*?)</div>', rendered, re.S)
    assert values
    for block in values:
        assert ("m__interval" in block) or ("withheld" in block)


def test_withheld_measurements_show_the_reason_and_not_the_number(report: ReportInput):
    victim = next(m for m in report.withheld)
    # The sentinel needs a digit run long enough that it cannot occur by
    # coincidence. A three-digit one collided with another measurement's own
    # interval ("0.891 to 0.987"), which failed the test while nothing had
    # leaked. The same trap caught the base64 payloads earlier in this file.
    loud = dataclasses.replace(victim, value=913579.246, ci_low=902468.0, ci_high=924680.0)
    others = tuple(m for m in report.measurements if m.spec_id != victim.spec_id)
    out = text_only(html.render(dataclasses.replace(report, measurements=(loud, *others))))
    assert "913579" not in out
    assert "902468" not in out
    assert "924680" not in out
    assert "withheld" in out
    assert esc(prose.withheld_paragraph(loud)) in out


def test_each_measurement_carries_its_provenance(report: ReportInput, rendered: str):
    body = text_only(rendered)
    for m in report.measurements:
        assert m.formula_fingerprint in body
        for name in m.landmarks_used:
            assert name in body


def test_each_measurement_carries_its_tier_and_discriminability(rendered: str):
    for code in ("AGREED", "RATIO", "CONV"):
        assert code in rendered
    assert "discriminability" in rendered


def test_unavailable_measurements_are_named_in_their_region(report: ReportInput, rendered: str):
    assert report.unavailable
    for u in report.unavailable:
        assert esc(prose.unavailable_sentence(u)) in rendered


# ---------------------------------------------------------------------------
# Quality, obligations, references, manifest
# ---------------------------------------------------------------------------


def test_quality_issues_are_rendered_with_their_readings(report: ReportInput, rendered: str):
    for issue in report.quality:
        assert esc(issue.detail) in rendered
        assert issue.reading in rendered
        assert issue.severity in rendered


def test_licence_obligations_appear_in_full(report: ReportInput, rendered: str):
    assert report.obligations
    for line in report.obligations:
        assert esc(line) in rendered
    assert "obligations of the backends that actually loaded" in rendered


def test_an_empty_obligation_list_says_so_rather_than_disappearing(report: ReportInput):
    out = html.render(dataclasses.replace(report, obligations=()))
    assert "No backend obligations were recorded" in out
    assert "did not record them" in out


def test_references_and_manifest_are_present(report: ReportInput, rendered: str):
    for ref in report.references:
        assert esc(ref) in rendered
    assert "landmark_backend" in rendered
    assert "synthetic fixture" in rendered


def test_manifest_values_are_escaped(report: ReportInput):
    out = html.render(
        dataclasses.replace(report, manifest={"note": "<script>alert(1)</script>"})
    )
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def test_no_typographic_tells_in_the_rendered_prose(rendered: str):
    body = text_only(rendered)
    assert "—" not in body
    assert "–" not in body  # noqa: RUF001


def test_no_prescriptive_language_survives_into_the_html(rendered: str):
    body = text_only(rendered).lower()
    hits = [t for t in prose.PRESCRIPTIVE_TERMS if t in body]
    assert hits == [], hits


def test_the_stylesheet_asks_for_no_webfont(rendered: str):
    style = re.search(r"<style>(.*?)</style>", rendered, re.S).group(1)
    assert "@font-face" not in style
    assert "fonts.googleapis" not in style
    for generic in ("Inter", "Roboto"):
        assert generic not in style


# ---------------------------------------------------------------------------
# Writing it out
# ---------------------------------------------------------------------------


def test_write_produces_a_single_file(report: ReportInput, tmp_path: Path):
    out = html.write(report, tmp_path / "report.html")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert list(tmp_path.iterdir()) == [out]


def test_context_is_a_plain_dictionary(report: ReportInput):
    ctx = html.build_context(report)
    assert set(ctx) >= {"summary", "groups", "counts", "manifest", "obligations"}
    assert ctx["counts"]["attempted"] == report.n_attempted


def test_a_report_with_nothing_in_it_still_renders():
    """An analysis that measured nothing must still say so on a page."""
    out = html.render(ReportInput(subject_label="Empty"))
    assert "0<span>/0</span>" in out
    assert "No quality issues were recorded" in out
    assert "No manifest was supplied" in out
