"""The PDF report, checked against the artifact rather than against the code.

Every assertion here reads a real PDF back. The renderer is exercised twice --
once over the whole synthetic face the other report tests share, and once over
a ``Measured`` built by hand in this file -- and then the bytes are parsed, the
text is pulled out of the content streams, and the project's three rules are
applied to what a reader would actually see:

1. no scalar standing for the face as a whole,
2. no value without its interval,
3. no prescription.

Checking the rendered text rather than the rendering code is the point. A rule
that holds in ``prose.py`` and breaks in the page layout is a rule the reader
does not get.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_no_aggregate_score import FORBIDDEN_NAMES, FORBIDDEN_TEXT
from test_report_prose import synthetic_report

from vitruve.core.sensitivity import Discriminability
from vitruve.core.spec import Reportability, Unit, Verdict
from vitruve.measure.evaluate import Measured, Unavailability, Unavailable
from vitruve.report import pdf as pdf_module
from vitruve.report import prose
from vitruve.report.model import QualityIssue, ReportInput

pytest.importorskip(
    "reportlab",
    reason=(
        "the PDF report is the optional [pdf] extra; "
        "install it with `pip install 'vitruve[pdf]'`"
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report() -> ReportInput:
    return synthetic_report()


@pytest.fixture(scope="module")
def document(report: ReportInput) -> bytes:
    return pdf_module.render_pdf_bytes(report)


@pytest.fixture(scope="module")
def text(document: bytes) -> str:
    return pdf_module.extract_text(document)


@pytest.fixture(scope="module")
def flowed(text: str) -> str:
    """The text with the line breaks the layout introduced taken back out.

    A sentence broken across three printed lines is still that sentence, and a
    rule about what the document says has to be applied to the sentence.
    """
    return re.sub(r"\s+", " ", text)


def hand_built_report() -> ReportInput:
    """A whole report from two ``Measured`` objects written out by hand.

    Deliberately not routed through ``evaluate``: this asserts the renderer can
    be driven from the dataclasses alone, which is what lets somebody with
    measurements from another source print them here.
    """
    shown = Measured(
        spec_id="intercanthal_width",
        label="Intercanthal width",
        unit=Unit.MILLIMETRES,
        value=31.4,
        ci_low=29.2,
        ci_high=33.7,
        sd=1.1,
        verdict=Verdict(
            reportability=Reportability.CAVEAT,
            findings=((Reportability.CAVEAT, "scale assumption is a prior, not a ruler"),),
        ),
        discriminability=Discriminability(
            ratio=1.8,
            between_subject_sd=3.2,
            total_error_sd=1.8,
            pose_component=0.6,
            landmark_component=1.6,
            scale_component=0.4,
        ),
        formula_fingerprint="ab" * 8,
        landmarks_used=("endocanthion_l", "endocanthion_r"),
        n_samples=512,
        n_valid=512,
        scale_source="interpupillary prior",
        notes=("The prior carries its own spread.",),
    )
    held = Measured(
        spec_id="bigonial_width",
        label="Bigonial width",
        unit=Unit.MILLIMETRES,
        value=123.456,
        ci_low=100.0,
        ci_high=150.0,
        sd=12.0,
        verdict=Verdict(
            reportability=Reportability.WITHHOLD,
            findings=(
                (Reportability.WITHHOLD, "gonion is self-occluding in a photograph"),
            ),
        ),
        discriminability=None,
        formula_fingerprint="cd" * 8,
        landmarks_used=("gonion_l", "gonion_r"),
        n_samples=512,
        n_valid=498,
    )
    missing = Unavailable(
        spec_id="mentocervical_angle",
        label="Mentocervical angle",
        missing_landmarks=("cervicale",),
        kind=Unavailability.MISSING_LANDMARKS,
    )
    return ReportInput(
        measurements=(shown, held),
        unavailable=(missing,),
        quality=(
            QualityIssue(
                code="pose_yaw",
                detail="The head is turned away from the camera. Retake square on.",
                severity="caveat",
                reading="yaw 9.1 deg",
            ),
        ),
        manifest={"seed": 3, "vitruve_version": "0.0.0+test"},
        obligations=("SPIGA landmark model: BSD-3-Clause",),
        references=("Farkas 1994, Anthropometry of the Head and Face, 2nd ed.",),
        subject_label="Unidentified subject",
        generated_at="2026-08-23 00:00 UTC",
    )


def page_count(data: bytes) -> int:
    """Pages in the file, from the page objects rather than from the renderer."""
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


# ---------------------------------------------------------------------------
# It is a PDF, and it is several pages long
# ---------------------------------------------------------------------------


def test_it_is_a_pdf(document: bytes):
    assert document.startswith(b"%PDF-")
    assert document.rstrip().endswith(b"%%EOF")


def test_it_runs_to_several_pages(document: bytes):
    assert page_count(document) >= 8, "a whole catalogue does not fit on eight pages"


def test_render_pdf_writes_the_file(report: ReportInput, tmp_path: Path):
    out = pdf_module.render_pdf(report, tmp_path / "nested" / "report.pdf")
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_a_hand_built_report_renders():
    """No pipeline, no photograph, no model: two dataclasses and a page."""
    data = pdf_module.render_pdf_bytes(hand_built_report())
    text = pdf_module.extract_text(data)
    assert "Intercanthal width" in text
    assert "Bigonial width" in text
    assert "Unidentified subject" in text


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_two_runs_of_the_same_input_are_byte_identical(report: ReportInput):
    assert pdf_module.render_pdf_bytes(report) == pdf_module.render_pdf_bytes(report)


def test_the_document_date_comes_from_the_report_and_not_the_clock(report: ReportInput):
    data = pdf_module.render_pdf_bytes(report)
    assert b"D:20260823000000+00'00'" in data
    assert data.count(b"D:20260823000000+00'00'") >= 2, "creation and modification"


def test_a_report_with_no_date_falls_back_to_a_fixed_epoch():
    import dataclasses

    undated = dataclasses.replace(hand_built_report(), generated_at="", manifest={})
    data = pdf_module.render_pdf_bytes(undated)
    assert pdf_module.FALLBACK_STAMP.encode("ascii") in data
    assert data == pdf_module.render_pdf_bytes(undated)


def test_the_date_is_read_out_of_the_manifest_when_the_field_is_empty():
    import dataclasses

    r = dataclasses.replace(
        hand_built_report(),
        generated_at="",
        manifest={"generated_at": "2024-01-02T03:04:05+00:00"},
    )
    assert pdf_module.pdf_datestamp(r) == "D:20240102030405+00'00'"


def test_changing_one_measurement_changes_the_file(report: ReportInput):
    """Without this, byte-identity would be satisfied by rendering nothing."""
    import dataclasses

    other = dataclasses.replace(
        report,
        measurements=tuple(
            dataclasses.replace(
                m, value=m.value * 1.5, ci_low=m.ci_low * 1.5, ci_high=m.ci_high * 1.5
            )
            for m in report.measurements
        ),
    )
    assert pdf_module.render_pdf_bytes(other) != pdf_module.render_pdf_bytes(report)


# ---------------------------------------------------------------------------
# Rule 1: no scalar aggregate, anywhere in the artifact
# ---------------------------------------------------------------------------


def test_the_extracted_text_reads_like_no_rating(flowed: str):
    """The same scan ``test_no_aggregate_score`` runs over the HTML."""
    body = flowed.lower()
    hits = [phrase for phrase in FORBIDDEN_TEXT if phrase in body]
    assert hits == [], f"the PDF reads like a rating: {hits}"


def test_the_hand_built_report_passes_the_same_scan():
    body = re.sub(
        r"\s+", " ", pdf_module.extract_text(pdf_module.render_pdf_bytes(hand_built_report()))
    ).lower()
    hits = [phrase for phrase in FORBIDDEN_TEXT if phrase in body]
    assert hits == [], hits


def test_no_public_name_in_the_pdf_module_suggests_an_aggregate():
    import dataclasses

    names: set[str] = set()
    for name, value in vars(pdf_module).items():
        if name.startswith("_") or isinstance(value, types.ModuleType):
            continue
        names.add(name)
        if dataclasses.is_dataclass(value):
            names.update(f.name for f in dataclasses.fields(value))
    assert names
    hits = sorted(n for n in names if any(f in n.lower() for f in FORBIDDEN_NAMES))
    assert hits == [], hits


def test_the_document_tells_the_reader_it_does_not_do_this(flowed: str):
    assert "single number standing for the face as a whole" in flowed
    assert "does not combine them, rank them" in flowed


def test_the_cover_leads_with_the_count_of_what_is_not_here(report: ReportInput, text: str):
    cover = pdf_module.cover_of(report)
    assert cover.ratio == f"{report.n_shown}/{report.n_attempted}"
    lines = text.splitlines()
    assert str(cover.shown) in lines[: len(lines) // 2]
    assert f"/{cover.attempted}" in lines


# ---------------------------------------------------------------------------
# Rule 2: a value never appears without its interval
# ---------------------------------------------------------------------------


def _value_head(m: Measured) -> str:
    fmt = prose._places(m.unit)
    unit = "" if m.unit is Unit.RATIO else f" {m.unit.value}"
    return f"{fmt.format(m.value)}{unit}"


def test_every_printed_value_is_followed_by_its_interval(report: ReportInput, text: str):
    lines = text.splitlines()
    assert report.shown, "the fixture reports nothing, so this proves nothing"
    for m in report.shown:
        head = _value_head(m)
        positions = [i for i, line in enumerate(lines) if line == head]
        assert positions, f"{m.spec_id}: the value never appears in the document"
        assert any(
            lines[i + 1].startswith("95% interval") for i in positions if i + 1 < len(lines)
        ), f"{m.spec_id}: a value is printed without its interval"


def test_the_interval_printed_is_the_one_that_was_measured(report: ReportInput, text: str):
    for m in report.shown:
        fmt = prose._places(m.unit)
        wanted = f"95% interval {fmt.format(m.ci_low)} to {fmt.format(m.ci_high)}"
        assert wanted in text, f"{m.spec_id}: {wanted!r} is not in the document"


def test_withheld_measurements_print_a_cause_instead_of_a_number(
    report: ReportInput, flowed: str
):
    assert report.withheld, "the fixture withholds nothing, so this proves nothing"
    for m in report.withheld:
        assert f"{m.label} is withheld." in flowed, m.spec_id
        cause = prose.primary_cause(m)
        assert cause is not None and cause.name in flowed, m.spec_id


def test_a_withheld_value_is_not_smuggled_in_anywhere():
    """The gate computed 123.456 mm. Printing it beside the refusal would undo it."""
    data = pdf_module.render_pdf_bytes(hand_built_report())
    body = pdf_module.extract_text(data)
    assert "123.5" not in body
    assert "123.456" not in body
    assert "Bigonial width is withheld." in re.sub(r"\s+", " ", body)


def test_measurements_that_were_never_attempted_say_which_point_was_missing(
    report: ReportInput, flowed: str
):
    assert report.unavailable
    for u in report.unavailable:
        assert u.label in flowed, u.spec_id
    assert "does not guess a point it cannot see" in flowed


# ---------------------------------------------------------------------------
# Rule 3: no prescription
# ---------------------------------------------------------------------------


def test_the_document_prescribes_nothing(report: ReportInput, flowed: str):
    """Scanned over what this renderer writes, not over what it quotes.

    A catalogue label is the anatomical name of a quantity and its wording is
    the registry's business -- "nasal dorsal deviation from the midline" is
    what the measurement is called, not an instruction to the reader -- so the
    labels are lifted out before the scan. ``test_report_prose`` polices the
    label vocabulary itself.
    """
    body = flowed.lower()
    for m in list(report.measurements) + list(report.unavailable):
        body = body.replace(m.label.lower(), " ")
    hits = [term for term in prose.PRESCRIPTIVE_TERMS if term in body]
    assert hits == [], f"the PDF tells the reader what to be: {hits}"


def test_a_reference_range_is_framed_as_context(flowed: str):
    if "reports" in flowed and "for this measurement" in flowed:
        assert "It is context for reading the number and not a target." in flowed


# ---------------------------------------------------------------------------
# The structure the reader was promised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading",
    [
        "HOW TO READ THIS",
        "CAPTURE QUALITY",
        "MEASUREMENTS BY REGION",
        "WITHHELD, AND WHY",
        "PLATES",
        "METHOD",
        "REFERENCES",
        "LICENCE OBLIGATIONS INCURRED BY THIS RUN",
        "RUN MANIFEST",
    ],
)
def test_every_promised_section_is_present(text: str, heading: str):
    assert heading in text


def test_the_evidence_tier_of_every_measurement_is_legible(text: str):
    from vitruve.core.spec import Evidence
    from vitruve.report.html import TIER_CODE, TIER_GLOSS

    for evidence in Evidence:
        assert TIER_CODE[evidence] in text
        assert TIER_GLOSS[evidence][:24] in re.sub(r"\s+", " ", text)


def test_the_quality_findings_carry_their_remedy(report: ReportInput, flowed: str):
    assert report.quality
    for issue in report.quality:
        assert issue.detail.split(".")[0] in flowed, issue.code


def test_the_licence_obligations_are_printed_in_full(report: ReportInput, flowed: str):
    for line in report.obligations:
        assert line.split(":")[0] in flowed


def test_the_manifest_travels_with_the_document(report: ReportInput, flowed: str):
    for key in report.manifest:
        assert key[:22] in flowed


def test_a_manifest_value_the_size_of_the_report_is_cut_and_says_so():
    """A real run's manifest carries a serialised copy of every measurement.

    Printed whole it is three pages of JSON running off the bottom of the page.
    Printed short and silently it is a manifest that lies about the run. So it
    is cut, counted, and the count is on the page.
    """
    import dataclasses
    import json

    blob = json.dumps([{"spec_id": f"m{i}", "value": i} for i in range(400)])
    bloated = dataclasses.replace(
        hand_built_report(), manifest={"measurements": blob, "seed": 3}
    )
    data = pdf_module.render_pdf_bytes(bloated)
    lines = pdf_module.extract_text(data).splitlines()
    flowed = re.sub(r"\s+", " ", "\n".join(lines))
    assert "further lines are not printed here" in flowed
    assert "the whole value is in the JSON manifest" in flowed
    # The cut held: the blob did not turn into pages of JSON.
    printed = [line for line in lines if '"spec_id"' in line]
    assert len(printed) <= pdf_module.MANIFEST_MAX_LINES
    assert page_count(data) <= page_count(
        pdf_module.render_pdf_bytes(hand_built_report())
    ) + 1


# ---------------------------------------------------------------------------
# Offline, self-contained, no CDN
# ---------------------------------------------------------------------------


def test_the_file_asks_the_network_for_nothing(document: bytes):
    for construct in (b"/URI", b"/Launch", b"/JavaScript", b"/JS", b"/EmbeddedFile"):
        assert construct not in document, construct


def test_the_plates_are_carried_as_bytes_and_not_as_a_link():
    import dataclasses

    from vitruve.report.model import OverlayImage

    with_plate = dataclasses.replace(
        hand_built_report(),
        overlays=(
            OverlayImage(
                region="jaw",
                title="Jaw and chin",
                caption="Landmarks with their 95% ellipses.",
                png=_tiny_png(),
            ),
        ),
    )
    data = pdf_module.render_pdf_bytes(with_plate)
    assert b"/Image" in data, "the plate did not become an image object"
    assert "Jaw and chin" in re.sub(r"\s+", " ", pdf_module.extract_text(data))
    assert b"/URI" not in data


def test_a_photographic_plate_is_downsampled_and_still_deterministic():
    """The plate path a real run takes: large, photographic, JPEG-encoded.

    A real overlay stored losslessly puts ten megabytes into the file, so
    plates above a threshold are resampled and JPEG-encoded. That is the one
    step in the renderer that could have been non-deterministic, so it is
    rendered twice and compared.
    """
    import dataclasses

    from vitruve.report.model import OverlayImage

    plate = OverlayImage("jaw", "Jaw and chin", "A photograph.", _photo_png(1400))
    big = dataclasses.replace(hand_built_report(), overlays=(plate,))
    first = pdf_module.render_pdf_bytes(big)
    assert first == pdf_module.render_pdf_bytes(big)
    assert b"/DCTDecode" in first, "the photograph was not stored as a JPEG"
    # Resampled on the way in, not carried at capture size.
    assert f"/Width {pdf_module.PLATE_MAX_PX}".encode() in first
    assert b"/Width 1400" not in first


def test_an_undecodable_plate_costs_the_plate_and_not_the_report():
    import dataclasses

    from vitruve.report.model import OverlayImage

    broken = dataclasses.replace(
        hand_built_report(),
        overlays=(OverlayImage("jaw", "Jaw", "A plate.", b"not a png"),),
    )
    body = pdf_module.extract_text(pdf_module.render_pdf_bytes(broken))
    assert "could not be decoded" in re.sub(r"\s+", " ", body)
    assert "Intercanthal width" in body


# ---------------------------------------------------------------------------
# The optional neighbours
# ---------------------------------------------------------------------------


def _fake_composite_module(monkeypatch: pytest.MonkeyPatch, **attrs) -> None:
    """Stand a module in for ``vitruve.report.composite``.

    Both the package attribute and ``sys.modules`` are replaced. Patching only
    ``sys.modules`` is not enough once the real module has been imported: the
    ``from . import composite`` machinery finds the attribute on the package
    first and never looks at ``sys.modules`` at all.
    """
    import vitruve.report as package

    module = types.ModuleType("vitruve.report.composite")
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "vitruve.report.composite", module)
    monkeypatch.setattr(package, "composite", module, raising=False)


def test_plates_from_the_report_alone_are_optional(report: ReportInput):
    """`report/composite.py` may not exist, and must not be required."""
    assert isinstance(pdf_module.extra_plates(report), tuple)


def test_a_mirror_composite_can_be_handed_straight_to_the_renderer():
    """The contract with the composite module: title, caption, png.

    Asserted against the real dataclass where it exists, so a rename there
    fails here rather than silently dropping the plates out of the PDF.
    """
    import dataclasses

    composite = pytest.importorskip("vitruve.report.composite")
    fields = {f.name for f in dataclasses.fields(composite.MirrorComposite)}
    assert {"title", "caption", "png"} <= fields


def test_a_duck_typed_plate_is_printed():
    plate = types.SimpleNamespace(
        title="Subject's left half, mirrored",
        caption="A picture and not a measurement.",
        png=_tiny_png(),
    )
    body = pdf_module.extract_text(
        pdf_module.render_pdf_bytes(hand_built_report(), plates=[plate])
    )
    flowed = re.sub(r"\s+", " ", body)
    assert "Subject's left half, mirrored" in flowed
    assert "A picture and not a measurement." in flowed


def test_a_plate_module_that_offers_the_agreed_callable_is_used(
    report: ReportInput, monkeypatch: pytest.MonkeyPatch
):
    png = _tiny_png()
    _fake_composite_module(
        monkeypatch,
        plates_for_report=lambda _report: [
            types.SimpleNamespace(title="Mirror", caption="A plate.", png=png)
        ],
    )
    assert [p.title for p in pdf_module.extra_plates(report)] == ["Mirror"]


def test_a_plate_module_that_raises_costs_the_plates_and_not_the_report(
    report: ReportInput, monkeypatch: pytest.MonkeyPatch
):
    def explode(_report):
        raise RuntimeError("half written")

    _fake_composite_module(monkeypatch, plates_for_report=explode)
    assert pdf_module.extra_plates(report) == ()
    assert pdf_module.render_pdf_bytes(report).startswith(b"%PDF-")


def test_a_plate_module_with_the_wrong_signature_is_ignored(
    report: ReportInput, monkeypatch: pytest.MonkeyPatch
):
    _fake_composite_module(
        monkeypatch, plates_for_report=lambda image, points: [("nope", None)]
    )
    assert pdf_module.extra_plates(report) == ()


def test_skin_findings_are_printed_when_the_input_carries_them():
    findings = types.SimpleNamespace(
        findings=(
            types.SimpleNamespace(
                format=lambda: (
                    "Periorbital pigmentation [periorbital_l]: +2.10 dL "
                    "(95% CI +1.20 to +3.00)"
                ),
                method="CIELAB difference against the malar reference region",
            ),
        ),
        disclaimer="These are observations of a photograph, not a diagnosis.",
    )
    body = pdf_module.extract_text(
        pdf_module.render_pdf_bytes(hand_built_report(), skin=findings)
    )
    flowed = re.sub(r"\s+", " ", body)
    assert "SKIN FINDINGS" in body
    assert "Periorbital pigmentation" in flowed
    assert "95% CI +1.20 to +3.00" in flowed
    assert "not a diagnosis" in flowed


def test_no_skin_section_when_there_are_no_findings():
    assert "SKIN FINDINGS" not in pdf_module.extract_text(
        pdf_module.render_pdf_bytes(hand_built_report())
    )


# ---------------------------------------------------------------------------
# The degraded path
# ---------------------------------------------------------------------------


def test_the_missing_dependency_names_the_install_command():
    message = str(pdf_module.PDFUnavailable())
    assert "reportlab" in message
    assert "pip install" in message


def test_the_print_html_fallback_carries_a_cover_and_print_rules(report: ReportInput):
    document = pdf_module.render_print_html(report)
    assert "pcover__ratio" in document
    assert "@media print" in document
    assert f"{report.n_shown}<span>/{report.n_attempted}</span>" in document
    # Same offline guarantee as the screen report: nothing is fetched. An XML
    # namespace in an inline SVG is a string, not a request, so the check is
    # for the constructs that would actually go to the network.
    for construct in ('src="http', "url(http", '<link', "@import"):
        assert construct not in document, construct


def test_the_print_stylesheet_never_hides_a_measurement():
    css = (Path(pdf_module.TEMPLATES) / "print.css").read_text()
    assert ".m {" not in css.replace(" ", "") or "display:none" not in css.replace(" ", "")
    assert "break-inside: avoid" in css


# ---------------------------------------------------------------------------
# The reader used by the tests above
# ---------------------------------------------------------------------------


def test_the_extractor_actually_reads_the_document(document: bytes, text: str):
    """A silent extractor would make every text assertion above vacuous."""
    assert len(text) > 4000
    assert "VITRUVE" in text
    assert "MORPHOMETRIC REPORT" in text


def test_the_extractor_unescapes_what_the_writer_escaped():
    import dataclasses

    r = dataclasses.replace(
        hand_built_report(), subject_label="Subject (left) \\ right"
    )
    body = pdf_module.extract_text(pdf_module.render_pdf_bytes(r))
    assert "Subject (left) \\ right" in body


def _tiny_png() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (200, 92, 42)).save(buffer, format="PNG")
    return buffer.getvalue()


def _photo_png(side: int) -> bytes:
    """A plate the size and busyness of a real overlay.

    Flat colour would compress to nothing and prove nothing about the size of a
    real plate, so this is noise with a drawn line through it.
    """
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (side, side))
    pixels = img.load()
    assert pixels is not None
    value = 7
    for y in range(side):
        for x in range(0, side, 4):
            value = (value * 1103515245 + 12345) % 2147483648
            shade = value >> 23
            for dx in range(4):
                if x + dx < side:
                    pixels[x + dx, y] = (shade, (shade + 40) % 256, (shade + 90) % 256)
    ImageDraw.Draw(img).line([(0, 0), (side, side)], fill=(200, 92, 42), width=3)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_the_budget_section_reaches_the_pdf(text: str, report):
    from vitruve.report import prose

    assert prose.BUDGET_TITLE.upper() in text
    for line in prose.lever_lines(report):
        assert line.action in text.replace("\n", " ")
