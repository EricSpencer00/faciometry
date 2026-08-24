"""The report as a PDF: the artifact a reader keeps.

Why reportlab
-------------

Four engines were considered, and the deciding constraint was that a core
install of Vitruve is ``numpy + pillow`` and has to stay that way.

* **WeasyPrint** renders the existing HTML almost exactly, and would have been
  the shortest path. It links cairo, pango, harfbuzz and gobject through
  ``ctypes``. On macOS that is a Homebrew prerequisite the user has to install
  by hand before ``pip install`` will produce a working program, and a report
  generator that fails at import on a clean laptop is not a feature.
* **Headless Chromium through Playwright** is perfect fidelity and a
  three-hundred-megabyte browser download. It is also a network fetch on first
  use, in a project whose whole claim is that it runs with the network off.
* **Print the HTML from a browser** costs nothing and is nobody's idea of a
  downloadable report, because what comes out depends on which browser printed
  it. It is kept, deliberately, as the degraded path: :func:`render_print_html`
  emits the same document with ``templates/print.css`` and a cover page
  attached, so a user with no extra packages can still get a paginated PDF out
  of ctrl-P.
* **reportlab** is a pure-Python wheel with no system libraries, it is the only
  one of the four that installs the same way on every platform, and the price
  is that the layout in this module is written by hand. That price is paid once
  here.

So: reportlab, as the optional extra ``[pdf]``. It is imported lazily and
:class:`PDFUnavailable` names the install command, so a core install still
imports ``vitruve.report`` and only fails when a PDF is actually asked for.

Type is the PDF base-14 set, Times and Courier. Not a fashionable choice, but
the base-14 faces need no font file, embed nothing, and render identically in
every viewer, which is what makes the output byte-identical across machines.
The screen report's Iowan/Palatino stack cannot be used without shipping or
scraping a font, and a report that looks different depending on which
typefaces a laptop happens to have is not a reproducible artifact.

Reproducibility
---------------

Two runs over the same :class:`~vitruve.report.model.ReportInput` produce
byte-identical files. Nothing in this module reads the clock or a random
source: the document dates come from ``report.generated_at`` (or from the run
manifest, or from a fixed epoch when neither is present), reportlab is
constructed with ``invariant=1`` so its own timestamps and document identifier
are pinned, and every mapping is walked in sorted order. ``tests/unit/
test_pdf.py`` asserts the byte-identity rather than trusting this paragraph.

What it will not print
----------------------

The same three rules the rest of the report layer is held to, checked against
this module's own extracted text in the tests: no scalar standing for the face
as a whole, no value without its interval, no prescription.
"""

from __future__ import annotations

import base64
import io
import re
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.spec import Evidence, Unit
from ..measure.evaluate import Measured
from ..measure.registry import BY_ID
from . import prose
from .html import TIER_CODE, TIER_GLOSS, IntervalBar, _interval_bar
from .model import ReportInput

TEMPLATES = Path(__file__).parent / "templates"

INSTALL_HINT = "pip install 'vitruve[pdf]'   (or: pip install reportlab)"


class PDFUnavailable(RuntimeError):
    """reportlab is not installed, and the PDF path needs it."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "the PDF report needs reportlab, which is an optional extra. "
            f"Install it with:\n    {INSTALL_HINT}"
            + (f"\n({detail})" if detail else "")
        )


def available() -> bool:
    """Whether a PDF can be rendered in this environment."""
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# The sheet
# ---------------------------------------------------------------------------

#: US Letter. The alternative was A4; Letter loses the bottom of an A4 page on
#: a US printer, and A4 loses nothing on either, but the project's own capture
#: guidance is written against ICAO portrait sizing and the author prints on
#: Letter. The constant is here so changing it is one edit.
PAGE_W, PAGE_H = 612.0, 792.0

MARGIN = 60.0
GUTTER = 66.0
COL_X = MARGIN + GUTTER
COL_W = PAGE_W - COL_X - 72.0
PROSE_W = 396.0
TOP_Y = PAGE_H - 54.0
FOOT_Y = 42.0

#: Lines of one manifest value that reach the page. The run manifest can carry
#: a serialised copy of every measurement, which is three pages of JSON nobody
#: reads; what is cut is counted and said, not silently dropped.
MANIFEST_MAX_LINES = 12

SERIF = "Times-Roman"
SERIF_I = "Times-Italic"
MONO = "Courier"
MONO_B = "Courier-Bold"

#: Lifted from ``templates/report.css`` so the two renderings are the same
#: document in two media rather than two documents.
PAPER = "#f3f0e8"
INK = "#1b1815"
INK_2 = "#5d564b"
INK_3 = "#8b8274"
RULE = "#cbc4b4"
RULE_SOFT = "#ded8c9"
ACCENT = "#b4501f"

#: Characters above Latin-1 that a report might carry, and what they become.
#: Written as escapes rather than as themselves: a folding table whose keys
#: are an em dash and an en dash sitting one line apart is a table nobody can
#: proofread, and the linter is right to say so.
_SUBS = {
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # ellipsis
    "\u00d7": "x",    # multiplication sign
    "\u00b7": "-",    # middle dot
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2212": "-",    # minus sign
    "\u00a0": " ",    # non-breaking space
}


def _flat(text: object) -> str:
    """Text the base-14 faces can set.

    WinAnsi covers Latin-1, which covers every author name in the reference
    list. The handful of typographic characters above it are folded rather than
    dropped, because a reference that loses its dash is still a reference and a
    reference that loses a letter is not.
    """
    s = str(text)
    for bad, good in _SUBS.items():
        s = s.replace(bad, good)
    return s.encode("latin-1", "replace").decode("latin-1")


class Sheet:
    """A cursor down a ruled page, with the page breaks handled.

    Deliberately not reportlab's Platypus. Platypus wants a document made of
    flowables and gives back a frame it has decided how to fill; the interval
    bars, the gutter marks and the hairlines here are drawn against absolute
    coordinates, and mixing the two models produces a layout nobody can
    predict from reading it.
    """

    def __init__(self, canvas: Any, *, subject: str, generated_at: str) -> None:
        self.c = canvas
        self.subject = _flat(subject)
        self.generated_at = _flat(generated_at)
        self.page = 0
        self.y = TOP_Y
        self.running_head = False
        self._begin_page()

    # -- page furniture --------------------------------------------------

    def _begin_page(self) -> None:
        from reportlab.lib.colors import HexColor

        if self.page:
            self.c.showPage()
        self.page += 1
        c = self.c
        c.setFillColor(HexColor(PAPER))
        c.rect(-1, -1, PAGE_W + 2, PAGE_H + 2, stroke=0, fill=1)
        self.y = TOP_Y
        if self.running_head:
            draw_tracked(
                c,
                MARGIN,
                PAGE_H - 40.0,
                "VITRUVE",
                font=MONO,
                size=6.4,
                tracking=1.5,
                color=INK_3,
            )
            c.setFillColor(HexColor(INK_3))
            c.setFont(MONO, 6.4)
            c.drawRightString(PAGE_W - 72.0, PAGE_H - 40.0, self.subject)
            c.setStrokeColor(HexColor(RULE))
            c.setLineWidth(0.4)
            c.line(MARGIN, PAGE_H - 48.0, PAGE_W - 72.0, PAGE_H - 48.0)
            self.y = PAGE_H - 72.0
        self._footer()

    def _footer(self) -> None:
        from reportlab.lib.colors import HexColor

        c = self.c
        c.setStrokeColor(HexColor(RULE_SOFT))
        c.setLineWidth(0.4)
        c.line(MARGIN, FOOT_Y + 12.0, PAGE_W - 72.0, FOOT_Y + 12.0)
        c.setFillColor(HexColor(INK_3))
        c.setFont(MONO, 6.4)
        c.drawString(MARGIN, FOOT_Y, _flat(self.generated_at))
        c.drawRightString(PAGE_W - 72.0, FOOT_Y, f"{self.page:02d}")

    def new_page(self) -> None:
        self._begin_page()

    def need(self, height: float) -> None:
        """Break the page if ``height`` will not fit below the cursor."""
        if self.y - height < FOOT_Y + 26.0:
            self._begin_page()

    def space(self, height: float) -> None:
        self.y -= height

    # -- marks -----------------------------------------------------------

    def rule(
        self,
        *,
        color: str = RULE,
        width: float | None = None,
        thickness: float = 0.5,
        gap_before: float = 6.0,
        gap_after: float = 6.0,
        x: float = COL_X,
    ) -> None:
        from reportlab.lib.colors import HexColor

        self.need(gap_before + gap_after + thickness)
        self.y -= gap_before
        self.c.setStrokeColor(HexColor(color))
        self.c.setLineWidth(thickness)
        self.c.line(x, self.y, x + (width if width is not None else COL_W), self.y)
        self.y -= gap_after

    def tracked(
        self,
        text: str,
        *,
        x: float = COL_X,
        font: str = MONO,
        size: float = 7.0,
        tracking: float = 1.6,
        color: str = INK_2,
        leading: float = 12.0,
    ) -> None:
        self.need(leading)
        self.y -= leading
        draw_tracked(
            self.c,
            x,
            self.y,
            text,
            font=font,
            size=size,
            tracking=tracking,
            color=color,
        )

    def line(
        self,
        text: str,
        *,
        x: float = COL_X,
        font: str = SERIF,
        size: float = 10.0,
        color: str = INK,
        leading: float = 13.0,
    ) -> None:
        from reportlab.lib.colors import HexColor

        self.need(leading)
        self.y -= leading
        self.c.setFillColor(HexColor(color))
        self.c.setFont(font, size)
        self.c.drawString(x, self.y, _flat(text))

    def para(
        self,
        text: str,
        *,
        x: float = COL_X,
        font: str = SERIF,
        size: float = 9.8,
        leading: float = 13.4,
        color: str = INK,
        width: float = PROSE_W,
        gap_after: float = 7.0,
    ) -> None:
        from reportlab.lib.colors import HexColor

        for row in wrap(_flat(text), font, size, width):
            self.need(leading)
            self.y -= leading
            self.c.setFillColor(HexColor(color))
            self.c.setFont(font, size)
            self.c.drawString(x, self.y, row)
        self.y -= gap_after

    def section(self, number: str, title: str, *, new_page: bool = False) -> None:
        """A numbered section: the figure in the gutter, the title ruled."""
        from reportlab.lib.colors import HexColor

        if new_page or self.y < FOOT_Y + 180.0:
            self._begin_page()
        else:
            self.y -= 26.0
        self.need(46.0)
        c = self.c
        y = self.y - 11.0
        draw_tracked(
            c, MARGIN, y, number, font=MONO, size=7.0, tracking=1.2, color=ACCENT
        )
        draw_tracked(
            c,
            COL_X,
            y,
            _flat(title).upper(),
            font=SERIF,
            size=10.5,
            tracking=2.0,
            color=INK,
        )
        self.y = y - 7.0
        c.setStrokeColor(HexColor(INK))
        c.setLineWidth(0.7)
        c.line(COL_X, self.y, COL_X + COL_W, self.y)
        self.y -= 14.0

    def chip(self, text: str, *, y: float, color: str = INK) -> None:
        """The small ruled box that carries an evidence tier."""
        from reportlab.lib.colors import HexColor

        label = _flat(text)
        c = self.c
        w = stringw(label, MONO, 6.0) + 7.0
        c.setStrokeColor(HexColor(RULE))
        c.setLineWidth(0.4)
        c.rect(MARGIN, y - 2.6, w, 10.0, stroke=1, fill=0)
        c.setFillColor(HexColor(color))
        c.setFont(MONO, 6.0)
        c.drawString(MARGIN + 3.5, y, label)

    def key_value(
        self,
        rows: Sequence[tuple[str, str]],
        *,
        key_w: float = 96.0,
        max_lines: int = MANIFEST_MAX_LINES,
    ) -> None:
        """A mono two-column table, the run manifest's natural shape.

        Two things a manifest does that a page does not forgive. A value can be
        a serialised copy of the whole measurement set, tens of thousands of
        characters long, which is three pages of unreadable JSON in the middle
        of a keepsake; it is cut at ``max_lines`` and the cut is stated, with
        the count, so the reader knows exactly what is not on the page. And a
        value can outrun the paper, so the break is taken per line rather than
        per row: reserving the height of the whole row and then drawing it
        anyway is how text ends up printed over the footer.
        """
        from reportlab.lib.colors import HexColor

        for key, value in rows:
            wrapped = wrap(_flat(value), MONO, 6.6, COL_W - key_w) or [""]
            dropped = 0
            if max_lines and len(wrapped) > max_lines:
                dropped = len(wrapped) - max_lines
                wrapped = wrapped[:max_lines]
            self.need(20.0)
            first = True
            for row in wrapped:
                self.need(9.4)
                self.y -= 9.4
                self.c.setFont(MONO, 6.6)
                if first:
                    self.c.setFillColor(HexColor(INK_3))
                    self.c.drawString(COL_X, self.y, _flat(key)[:22])
                    first = False
                self.c.setFillColor(HexColor(INK))
                self.c.drawString(COL_X + key_w, self.y, row)
            if dropped:
                self.need(9.4)
                self.y -= 9.4
                self.c.setFont(MONO, 6.6)
                self.c.setFillColor(HexColor(ACCENT))
                self.c.drawString(
                    COL_X + key_w,
                    self.y,
                    _flat(
                        f"[{dropped} further lines are not printed here; "
                        "the whole value is in the JSON manifest]"
                    ),
                )
            self.y -= 1.6


def stringw(text: str, font: str, size: float) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    return float(stringWidth(text, font, size))


def tracked_width(text: str, font: str, size: float, tracking: float) -> float:
    return stringw(text, font, size) + tracking * len(text)


def draw_tracked(
    canvas: Any,
    x: float,
    y: float,
    text: str,
    *,
    font: str,
    size: float,
    tracking: float,
    color: str,
    align: str = "left",
) -> None:
    """Letterspaced text.

    Character spacing lives on reportlab's text object rather than on the
    canvas, so every tracked line here is set through one. Right alignment is
    done by measuring, because the text object has no right-aligned draw and
    the tracking has to be counted into the width or the line hangs into the
    margin.

    The save/restore is not decoration. Character spacing is a graphics-state
    parameter and outlives the text object that set it, so without it the first
    tracked heading on a page silently tracks out every paragraph after it.
    """
    from reportlab.lib.colors import HexColor

    flat = _flat(text)
    if align == "right":
        x -= tracked_width(flat, font, size, tracking)
    canvas.saveState()
    canvas.setFillColor(HexColor(color))
    obj = canvas.beginText(x, y)
    obj.setFont(font, size)
    obj.setCharSpace(tracking)
    obj.textOut(flat)
    canvas.drawText(obj)
    canvas.restoreState()


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    """Greedy line breaking. No hyphenation, no justification.

    A long unbroken token -- a formula fingerprint, a URL -- is cut at the
    measure rather than allowed to run off the page, because a fingerprint that
    ends at the paper's edge is a fingerprint nobody can check.
    """
    out: list[str] = []
    for source in text.split("\n"):
        row = ""
        for word in source.split(" "):
            trial = f"{row} {word}".strip()
            if row and stringw(trial, font, size) > width:
                out.append(row)
                row = word
            else:
                row = trial
            while stringw(row, font, size) > width and len(row) > 1:
                cut = len(row)
                while cut > 1 and stringw(row[:cut], font, size) > width:
                    cut -= 1
                out.append(row[:cut])
                row = row[cut:]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# The cover, shared with the print stylesheet's cover
# ---------------------------------------------------------------------------

STANDFIRST = (
    "This document records what one photograph of one face supports, and what "
    "it does not. Every number in it is a measurement carrying the 95% "
    "interval that the landmark uncertainty puts around it. Nothing in it is "
    "combined into a single number for the face, nothing in it compares this "
    "face with another person's, and nothing in it is a diagnosis or advice. "
    "Measurements the gate refused are printed as refusals, with the cause in "
    "plain language where the number would have been, because on a single "
    "photograph the refusals are the larger half of the finding."
)


@dataclass(frozen=True)
class Cover:
    """Everything on page one, computed once for both renderings."""

    subject: str
    generated_at: str
    shown: int
    attempted: int
    withheld: int
    unavailable: int
    standfirst: str = STANDFIRST

    @property
    def ratio(self) -> str:
        return f"{self.shown}/{self.attempted}"

    @property
    def caption(self) -> str:
        return "measurements reportable, of those attempted"

    @property
    def rows(self) -> tuple[tuple[str, str], ...]:
        return (
            ("subject", self.subject),
            ("generated", self.generated_at),
            ("reportable", str(self.shown)),
            ("computed, then withheld", str(self.withheld)),
            ("not attempted", str(self.unavailable)),
        )


def cover_of(report: ReportInput) -> Cover:
    return Cover(
        subject=report.subject_label or "Unidentified subject",
        generated_at=report.generated_at or "date not recorded",
        shown=report.n_shown,
        attempted=report.n_attempted,
        withheld=report.n_withheld,
        unavailable=report.n_unavailable,
    )


# ---------------------------------------------------------------------------
# Static prose that belongs to the printed document
# ---------------------------------------------------------------------------

HOW_TO_READ: tuple[tuple[str, str], ...] = (
    (
        "An interval",
        "Every printed value is followed by a 95% interval. The landmark model "
        "returns a position and a covariance for each point it places; the "
        "measurement is recomputed over draws from those covariances, the value "
        "printed is the median of the draws and the interval is where the middle "
        "95% of them fell. A wide interval is not a property of the face. It is "
        "what this photograph, at this resolution and this pose, could support.",
    ),
    (
        "Withheld",
        "A withheld measurement was computed and then not printed. One question "
        "decides it: does the quantity vary more between people than it varies "
        "between photographs of the same person? Where the answer is no, the "
        "cause takes the place of the number. The causes are a closed set -- head "
        "pose, camera roll, an assumed scale, a self-occluding landmark, camera "
        "distance, landmark uncertainty, poor repeatability, an unpublished "
        "between-person spread -- and each one is named in plain language on the "
        "measurement it suppressed.",
    ),
    (
        "Not attempted",
        "A measurement that was never attempted is a different thing again: the "
        "landmark model does not supply a point the formula reads. Vitruve does "
        "not guess a point it cannot see, so the formula is not evaluated and the "
        "missing points are named.",
    ),
    (
        "Discriminability",
        "The ratio printed with each measurement is the between-person spread "
        "divided by the measurement error on this photograph. At 1.0 the two are "
        "the same size and the number describes the photograph as much as the "
        "face. The figure is a property of one measurement and is never combined "
        "across measurements.",
    ),
    (
        "The band behind a bar",
        "Where a bar carries a hatched strip, that strip is a published range or "
        "the middle 95% of a named reference sample, drawn on the same axis. It "
        "records where a cited sample fell. It is context for reading the number "
        "and it is not a target; Vitruve does not hold a face against it.",
    ),
    (
        "Provenance",
        "Each measurement carries the fingerprint of the formula that produced it "
        "and the landmarks that formula read. Two reports of the same face can "
        "only be compared if both say which definition they used, so the "
        "fingerprint travels with the number.",
    ),
)

METHOD: tuple[str, ...] = (
    "Landmarks are placed by a neural model that returns a position and a "
    "positional covariance for every point. Head pose is estimated twice, once "
    "from the landmark geometry and once by an independent estimator, and where "
    "the two disagree the pose interval is widened to cover the disagreement "
    "rather than averaged into a number that hides it.",
    "Each measurement in the catalogue is a formula over named anatomical "
    "points. It is evaluated over draws from the landmark covariances, giving a "
    "distribution rather than a value; the median of that distribution is "
    "printed with the interval that contains the middle 95% of it. Millimetres, "
    "where they appear, rest on a scale cue, and where that cue is an assumed "
    "interpupillary distance rather than a ruler in the facial plane, the "
    "assumption carries its own spread and enters the interval.",
    "The gate then compares the total error on each measurement against the "
    "between-person spread published for it. Kleinberg and Vanezis (2007) "
    "photographed subjects in ten-degree steps and watched facial indices move "
    "8 to 19 percent at ten degrees of yaw, against a spread between subjects of "
    "1.2 percent; FISWG's guidance withdraws photo-anthropometry from "
    "identification on the same grounds. Where a quantity cannot beat its own "
    "photograph, the cause is printed instead of the number.",
    "This document contains no single number standing for the face as a whole. "
    "There is no ground truth for such a number, only panels of judges, and "
    "roughly half the stable variance in those panels is private taste rather "
    "than anything shared (Hoenekopp 2006). Vitruve reports measurements. It "
    "does not combine them, rank them, or compare this face to another.",
)

COLOPHON = (
    "Vitruve reports measurements. It does not combine them, rank them, or "
    "compare this face to another. Every value here is a median over draws from "
    "the landmark covariances, shown with the interval those draws produced, and "
    "every refusal here is a result."
)


# ---------------------------------------------------------------------------
# Optional neighbours, imported defensively
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plate:
    """One printed image with its caption. The PDF's own plate record."""

    title: str
    caption: str
    png: bytes


#: Names ``report/composite.py`` might expose that take the report alone.
#:
#: The mirror composites in that module are built from the photograph and the
#: landmark positions, neither of which ``ReportInput`` carries, so the normal
#: way in is for the caller to build them and hand them over::
#:
#:     from vitruve.report.composite import mirror_composites
#:     render_pdf(report, path, plates=mirror_composites(image, points,
#:                                                       measurements=measured))
#:
#: A ``MirrorComposite`` already has ``title``, ``caption`` and ``png``, which
#: is the whole of what this renderer reads off a plate, so it goes straight in
#: with no adapter. This list is the second way in, for a report-only helper
#: appearing later, and it is probed rather than required.
_PLATE_ENTRY_POINTS = (
    "plates_for_report",
    "plates_for",
    "build_plates",
)


def extra_plates(report: ReportInput) -> tuple[Plate, ...]:
    """Plates a neighbouring module can build from the report alone.

    Written to fail into an empty tuple, on a missing module, a missing
    function, a wrong signature or a raising one. A missing plate costs the
    reader an illustration; an exception here would cost them the report.
    """
    try:
        from . import composite
    except Exception:
        return ()
    for name in _PLATE_ENTRY_POINTS:
        fn = getattr(composite, name, None)
        if not callable(fn):
            continue
        try:
            produced = fn(report)
        except Exception:
            continue
        plates = tuple(_as_plates(produced))
        if plates:
            return plates
    return ()


def _as_plates(produced: Any) -> Iterable[Plate]:
    """Anything with a title, a caption and PNG bytes, as a plate.

    Duck-typed on purpose. It accepts this module's :class:`Plate`, the report
    layer's ``OverlayImage`` and ``composite.MirrorComposite`` without any of
    them having to know about the others.
    """
    for item in produced or ():
        png = getattr(item, "png", None)
        if not isinstance(png, (bytes, bytearray)):
            continue
        yield Plate(
            title=str(getattr(item, "title", "") or "Plate"),
            caption=str(getattr(item, "caption", "") or ""),
            png=bytes(png),
        )


def skin_findings(report: ReportInput, given: Any = None) -> tuple[Any, ...]:
    """Dermatological findings carried on the input, if there are any.

    ``vitruve.derm`` is optional and ``ReportInput`` has no field for it yet,
    so the findings are looked for under any of the names a caller might have
    attached them under, and passed explicitly otherwise.
    """
    source = given
    if source is None:
        for name in ("skin", "derm", "derm_findings", "skin_findings", "findings"):
            source = getattr(report, name, None)
            if source is not None:
                break
    if source is None:
        return ()
    items = getattr(source, "findings", source)
    try:
        return tuple(items)
    except TypeError:
        return ()


def skin_disclaimer(given: Any) -> str:
    text = getattr(given, "disclaimer", "")
    if text:
        return str(text)
    try:
        from ..derm.findings import DISCLAIMER

        return str(DISCLAIMER)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

#: Longest side a plate is stored at, in pixels. At a printed width of 414pt
#: this is about 190 dpi, which is past what the eye resolves on paper and well
#: short of carrying a 1350px portrait into a file somebody has to email.
PLATE_MAX_PX = 1100

#: JPEG quality for the plates. They are photographs of a face with an
#: annotation drawn over them, and Flate on a photograph is barely compression
#: at all: the eight overlays of a real run come to 10 MB stored losslessly and
#: to well under 2 MB here, at a quality where the uncertainty ellipses and the
#: caption bar are still crisp. Small plates stay lossless, so a synthetic
#: fixture is not quietly resampled.
PLATE_JPEG_QUALITY = 86
PLATE_JPEG_ABOVE_PX = 240


def _prepared(png: bytes) -> tuple[bytes, float]:
    """A plate flattened onto the paper colour and cut to a printable size.

    Flattening rather than carrying an alpha channel: an overlay drawn with
    translucent ellipses over a soft mask renders differently in every viewer,
    and the ivory ground is part of the document.

    Returns encoded bytes rather than an image, because reportlab embeds a JPEG
    stream verbatim when it is handed one and re-encodes anything else.
    """
    from PIL import Image

    opened = Image.open(io.BytesIO(png))
    opened.load()
    img: Image.Image = opened
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        ground = Image.new("RGBA", img.size, (*_rgb(PAPER), 255))
        img = Image.alpha_composite(ground, img)
    img = img.convert("RGB")
    longest = max(img.size)
    if longest > PLATE_MAX_PX:
        k = PLATE_MAX_PX / longest
        img = img.resize(
            (max(1, round(img.width * k)), max(1, round(img.height * k))),
            Image.Resampling.LANCZOS,
        )
    aspect = img.width / img.height
    if max(img.size) > PLATE_JPEG_ABOVE_PX:
        buffer = io.BytesIO()
        try:
            img.save(
                buffer,
                format="JPEG",
                quality=PLATE_JPEG_QUALITY,
                # No chroma subsampling: the annotation is drawn in one
                # saturated colour over skin, which is exactly what subsampling
                # smears. No `optimize`, either: it makes the encoder buffer a
                # whole scan against ``ImageFile.MAXBLOCK``, and a busy plate
                # then dies with "broken data stream" on the way to a BytesIO.
                # It bought about six percent.
                subsampling=0,
            )
            return buffer.getvalue(), aspect
        except OSError:
            # An encoder that refuses is not a reason to lose the plate.
            pass
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), aspect


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def draw_plate(sheet: Sheet, title: str, caption: str, png: bytes) -> None:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import ImageReader

    try:
        encoded, aspect = _prepared(png)
    except Exception:
        sheet.para(
            f"{title}: the plate could not be decoded, so it is not printed here.",
            font=SERIF_I,
            color=INK_2,
        )
        return
    width = COL_W
    height = width / aspect if aspect > 0 else width
    if height > 420.0:
        height = 420.0
        width = height * aspect
    caption_rows = wrap(_flat(caption), SERIF, 8.6, PROSE_W) if caption else []
    sheet.need(height + 24.0 + 11.4 * len(caption_rows))
    sheet.tracked(title, size=6.6, color=INK_2)
    sheet.y -= 4.0
    sheet.y -= height
    sheet.c.drawImage(
        ImageReader(io.BytesIO(encoded)),
        COL_X,
        sheet.y,
        width=width,
        height=height,
        preserveAspectRatio=True,
        anchor="sw",
        mask=None,
    )
    sheet.c.setStrokeColor(HexColor(RULE))
    sheet.c.setLineWidth(0.4)
    sheet.c.rect(COL_X, sheet.y, width, height, stroke=1, fill=0)
    sheet.y -= 4.0
    for row in caption_rows:
        sheet.y -= 11.4
        sheet.c.setFillColor(HexColor(INK_2))
        sheet.c.setFont(SERIF, 8.6)
        sheet.c.drawString(COL_X, sheet.y, row)
    sheet.y -= 12.0


# ---------------------------------------------------------------------------
# The interval, drawn to scale
# ---------------------------------------------------------------------------

BAR_W = 340.0
BAR_H = 13.0


def draw_bar(sheet: Sheet, bar: IntervalBar) -> None:
    """The same geometry the HTML draws, in points instead of percent."""
    from reportlab.lib.colors import HexColor

    sheet.need(BAR_H + 26.0)
    sheet.y -= 6.0
    top = sheet.y
    bottom = top - BAR_H
    c = sheet.c

    def at(pct: float) -> float:
        return COL_X + BAR_W * (float(pct) / 100.0)

    if bar.has_band:
        left, right = at(bar.band_lo_pct), at(bar.band_lo_pct + bar.band_width_pct)
        c.saveState()
        path = c.beginPath()
        path.rect(left, bottom, max(right - left, 0.6), BAR_H)
        c.clipPath(path, stroke=0, fill=0)
        c.setStrokeColor(HexColor(RULE))
        c.setLineWidth(0.4)
        step = 4.0
        x = left - BAR_H
        while x < right + BAR_H:
            c.line(x, bottom, x + BAR_H, top)
            x += step
        c.restoreState()
        c.setStrokeColor(HexColor(RULE))
        c.setLineWidth(0.4)
        c.line(left, bottom, left, top)
        c.line(right, bottom, right, top)

    c.setStrokeColor(HexColor(RULE))
    c.setLineWidth(0.4)
    mid = (top + bottom) / 2.0
    c.line(COL_X, mid, COL_X + BAR_W, mid)
    c.line(COL_X, bottom, COL_X, top)
    c.line(COL_X + BAR_W, bottom, COL_X + BAR_W, top)

    lo, hi = at(bar.lo_pct), at(bar.lo_pct + bar.width_pct)
    c.setFillColor(HexColor(INK))
    c.rect(lo, mid - 2.2, max(hi - lo, 1.2), 4.4, stroke=0, fill=1)
    c.setFillColor(HexColor(ACCENT))
    c.rect(at(bar.mid_pct) - 0.7, bottom + 1.0, 1.4, BAR_H - 2.0, stroke=0, fill=1)

    sheet.y = bottom - 8.0
    c.setFillColor(HexColor(INK_3))
    c.setFont(MONO, 6.2)
    c.drawString(COL_X, sheet.y, _flat(bar.axis_lo))
    c.drawRightString(COL_X + BAR_W, sheet.y, _flat(bar.axis_hi))
    key = "solid bar, 95% interval; rule, the value"
    if bar.has_band:
        key += f"; hatched, {bar.band_label}"
    sheet.y -= 1.0
    for row in wrap(_flat(key), MONO, 6.2, COL_W):
        sheet.need(8.4)
        sheet.y -= 8.4
        c.setFont(MONO, 6.2)
        c.setFillColor(HexColor(INK_3))
        c.drawString(COL_X, sheet.y, row)
    sheet.y -= 4.0


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def draw_measurement(sheet: Sheet, m: Measured, report: ReportInput) -> None:
    """One reported measurement: the value, its interval, and its provenance."""
    from reportlab.lib.colors import HexColor

    spec = BY_ID.get(m.spec_id)
    stratum = report.strata.get(m.spec_id)
    text = prose.describe(m, stratum)
    fmt = prose._places(m.unit)
    unit = "" if m.unit is Unit.RATIO else f" {m.unit.value}"

    sheet.rule(color=RULE, gap_before=9.0, gap_after=2.0)
    # Enough for the label, the value and the first line of the bar, so a
    # measurement never leaves its label alone at the foot of a page.
    sheet.need(76.0)
    if spec is not None:
        sheet.chip(TIER_CODE[spec.evidence], y=sheet.y - 11.0)
    sheet.line(m.label, font=SERIF, size=11.4, leading=14.0, color=INK)

    sheet.need(20.0)
    sheet.y -= 17.0
    c = sheet.c
    c.setFillColor(HexColor(INK))
    c.setFont(MONO_B, 14.0)
    head = _flat(f"{fmt.format(m.value)}{unit}")
    c.drawString(COL_X, sheet.y, head)
    c.setFillColor(HexColor(INK_2))
    c.setFont(MONO, 7.4)
    c.drawString(
        COL_X + stringw(head, MONO_B, 14.0) + 10.0,
        sheet.y + 1.0,
        _flat(
            f"95% interval {fmt.format(m.ci_low)} to {fmt.format(m.ci_high)}{unit}"
        ),
    )
    sheet.y -= 4.0

    bar = _interval_bar(m, spec, stratum)
    if bar is not None:
        draw_bar(sheet, bar)

    for note in (
        text.caveat,
        text.evidence,
        text.reference_range,
        text.stratum,
        text.stratum_caveat,
        text.scale,
    ):
        if note:
            sheet.para(note, size=9.2, leading=12.6, color=INK_2, gap_after=3.0)

    sheet.y -= 3.0
    d = m.discriminability
    meta = [
        ("discriminability", f"{d.ratio:.2f}" if d else "unknown"),
        ("provenance", text.provenance),
    ]
    if m.verdict.reasons:
        meta.append(("recorded reasons", " | ".join(m.verdict.reasons)))
    for key, value in meta:
        rows = wrap(_flat(f"{key}  {value}"), MONO, 6.4, COL_W)
        for i, row in enumerate(rows):
            sheet.need(8.6)
            sheet.y -= 8.6
            c.setFont(MONO, 6.4)
            c.setFillColor(HexColor(INK_2 if i == 0 else INK_3))
            c.drawString(COL_X, sheet.y, row)
    sheet.y -= 4.0


def draw_withheld_line(sheet: Sheet, m: Measured) -> None:
    """A refusal in its own region, compactly. The full account is section 05."""
    from reportlab.lib.colors import HexColor

    cause = prose.primary_cause(m)
    sheet.rule(color=RULE_SOFT, gap_before=7.0, gap_after=2.0)
    sheet.need(26.0)
    sheet.chip("HELD", y=sheet.y - 11.0, color=ACCENT)
    sheet.line(m.label, font=SERIF, size=10.4, leading=13.0, color=INK_2)
    sheet.y -= 10.6
    c = sheet.c
    c.setFont(MONO, 6.8)
    c.setFillColor(HexColor(ACCENT))
    c.drawString(COL_X, sheet.y, _flat("WITHHELD"))
    c.setFillColor(HexColor(INK_2))
    c.drawString(
        COL_X + 58.0,
        sheet.y,
        _flat(cause.name if cause else "no reason recorded"),
    )
    sheet.y -= 5.0


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def _manifest_rows(manifest: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    import json

    out = []
    for key in sorted(manifest):
        value = manifest[key]
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, sort_keys=True, default=str)
        else:
            text = str(value)
        out.append((str(key), text))
    return tuple(out)


def draw_cover(sheet: Sheet, cover: Cover) -> None:
    from reportlab.lib.colors import HexColor

    c = sheet.c
    y = PAGE_H - 96.0
    draw_tracked(
        c, MARGIN, y, "VITRUVE", font=SERIF, size=30.0, tracking=9.0, color=INK
    )
    c.setStrokeColor(HexColor(INK))
    c.setLineWidth(1.4)
    c.line(MARGIN, y - 14.0, PAGE_W - 72.0, y - 14.0)
    draw_tracked(
        c,
        MARGIN,
        y - 28.0,
        "MORPHOMETRIC REPORT",
        font=MONO,
        size=7.0,
        tracking=2.0,
        color=INK_2,
    )
    draw_tracked(
        c,
        PAGE_W - 72.0,
        y - 28.0,
        _flat(cover.generated_at).upper(),
        font=MONO,
        size=7.0,
        tracking=2.0,
        color=INK_2,
        align="right",
    )

    y -= 150.0
    c.setFillColor(HexColor(INK))
    c.setFont(SERIF, 96.0)
    shown = f"{cover.shown}"
    c.drawString(MARGIN, y, shown)
    w = stringw(shown, SERIF, 96.0)
    c.setFillColor(HexColor(INK_3))
    c.drawString(MARGIN + w, y, f"/{cover.attempted}")
    caption_x = MARGIN + w + stringw(f"/{cover.attempted}", SERIF, 96.0) + 22.0
    for i, row in enumerate(wrap(cover.caption.upper(), MONO, 7.0, 132.0)):
        draw_tracked(
            c,
            caption_x,
            y + 26.0 - i * 11.0,
            row,
            font=MONO,
            size=7.0,
            tracking=1.5,
            color=INK_2,
        )

    y -= 74.0
    c.setStrokeColor(HexColor(RULE))
    c.setLineWidth(0.5)
    c.line(MARGIN, y, PAGE_W - 72.0, y)

    y -= 26.0
    c.setFillColor(HexColor(INK))
    c.setFont(SERIF, 20.0)
    for row in wrap(_flat(cover.subject), SERIF, 20.0, PAGE_W - MARGIN - 72.0):
        c.drawString(MARGIN, y, row)
        y -= 24.0

    y -= 12.0
    for row in wrap(_flat(cover.standfirst), SERIF, 10.2, 400.0):
        c.setFillColor(HexColor(INK))
        c.setFont(SERIF, 10.2)
        c.drawString(MARGIN, y, row)
        y -= 14.2

    y -= 16.0
    c.setStrokeColor(HexColor(RULE))
    c.setLineWidth(0.5)
    c.line(MARGIN, y, PAGE_W - 72.0, y)
    y -= 13.0
    for key, value in cover.rows:
        c.setFont(MONO, 6.8)
        c.setFillColor(HexColor(INK_3))
        c.drawString(MARGIN, y, _flat(key))
        c.setFillColor(HexColor(INK))
        for row in wrap(_flat(value), MONO, 6.8, 300.0)[:2]:
            c.drawString(MARGIN + 150.0, y, row)
            y -= 9.6
        y -= 1.4


def build(sheet: Sheet, report: ReportInput, *, plates: Sequence[Plate], skin: Any) -> None:
    """Draw the whole document. One function, in the order it is read."""
    cover = cover_of(report)
    draw_cover(sheet, cover)
    sheet.running_head = True

    # -- 01 how to read this ---------------------------------------------
    sheet.new_page()
    sheet.section("01", "How to read this", new_page=False)
    for heading, body in HOW_TO_READ:
        sheet.tracked(heading, size=6.8, color=ACCENT, tracking=1.4)
        sheet.y -= 2.0
        sheet.para(body, size=9.4, leading=12.8, gap_after=8.0)
    sheet.tracked("Evidence tiers", size=6.8, color=ACCENT, tracking=1.4)
    sheet.y -= 2.0
    for evidence in Evidence:
        sheet.need(10.0)
        sheet.y -= 10.4
        _mono_pair(sheet, TIER_CODE[evidence], TIER_GLOSS[evidence])

    # -- 02 capture quality ----------------------------------------------
    sheet.section("02", "Capture quality")
    if report.quality:
        sheet.para(
            "What the gate found in the photograph itself, with the remedy for "
            "each. These findings are the reason many of the measurements "
            "further on carry the intervals they carry.",
            color=INK_2,
        )
        for issue in report.quality:
            sheet.rule(color=RULE_SOFT, gap_before=7.0, gap_after=3.0)
            sheet.need(28.0)
            sheet.tracked(
                issue.severity.upper(),
                size=6.4,
                color=ACCENT if issue.severity != "note" else INK_3,
                tracking=1.4,
                leading=10.0,
            )
            sheet.para(issue.detail, size=9.4, leading=12.8, gap_after=2.0)
            if issue.reading:
                sheet.line(
                    issue.reading, font=MONO, size=6.6, color=INK_3, leading=9.0
                )
                sheet.y -= 3.0
    else:
        sheet.para(
            "No quality issues were recorded for this photograph. That is the "
            "gate reporting silence, not a guarantee: it checks pose, blur, "
            "exposure, occlusion and camera distance, and says nothing about "
            "anything else."
        )

    # -- 03 measurements by region ---------------------------------------
    groups = report.groups()
    sheet.section("03", "Measurements by region")
    sheet.para(
        "Grouped by the part of the face they describe rather than by how well "
        "they survive a photograph, because the reader is looking at a face and "
        "not at a taxonomy. The evidence tier travels in the margin of every "
        "measurement instead.",
        color=INK_2,
    )
    for i, view in enumerate(groups, start=1):
        sheet.section(f"03.{i}", view.region.title)
        sheet.para(
            f"{view.region.blurb} {view.n_shown} of "
            f"{len(view.measurements) + len(view.unavailable)} reportable.",
            color=INK_2,
            size=9.4,
        )
        if view.overlay is not None:
            draw_plate(
                sheet, view.overlay.title, view.overlay.caption, view.overlay.png
            )
        for m in view.measurements:
            if m.shown:
                draw_measurement(sheet, m, report)
            else:
                draw_withheld_line(sheet, m)
        if view.unavailable:
            sheet.rule(color=RULE_SOFT, gap_before=8.0, gap_after=4.0)
            for u in view.unavailable:
                sheet.para(
                    prose.unavailable_sentence(u),
                    size=9.0,
                    leading=12.2,
                    color=INK_2,
                    gap_after=3.0,
                )

    # -- 04 withheld ------------------------------------------------------
    sheet.section("04", "Withheld, and why")
    counts = prose.cause_counts(report)
    sheet.para(
        "Each of these was computed and then not printed. A withheld "
        "measurement is a result: it says this photograph cannot carry that "
        "quantity, and it names what stopped it.",
        color=INK_2,
    )
    if counts:
        sheet.tracked("By cause", size=6.8, color=ACCENT, tracking=1.4)
        sheet.y -= 2.0
        widest = counts[0][1]
        for cause, n in counts:
            sheet.need(15.0)
            sheet.y -= 10.0
            _mono_pair(sheet, f"{n:>3}", cause.name)
            _tally_rule(sheet, n / widest if widest else 0.0)
    for m in report.withheld:
        sheet.rule(color=RULE_SOFT, gap_before=8.0, gap_after=3.0)
        sheet.need(30.0)
        sheet.chip("HELD", y=sheet.y - 11.0, color=ACCENT)
        sheet.line(m.label, font=SERIF, size=10.6, leading=13.4)
        sheet.y -= 2.0
        sheet.para(
            prose.withheld_paragraph(m), size=9.2, leading=12.6, gap_after=2.0
        )
        sheet.para(
            prose.discriminability_sentence(m),
            font=MONO,
            size=6.4,
            leading=8.8,
            color=INK_3,
            width=COL_W,
            gap_after=2.0,
        )
    if report.unavailable:
        sheet.rule(color=RULE, gap_before=10.0, gap_after=5.0)
        sheet.tracked("Not attempted at all", size=6.8, color=ACCENT, tracking=1.4)
        sheet.y -= 3.0
        for u in report.unavailable:
            sheet.para(
                prose.unavailable_sentence(u),
                size=9.0,
                leading=12.2,
                color=INK_2,
                gap_after=3.0,
            )

    # -- 05 plates --------------------------------------------------------
    all_plates = [
        Plate(o.title, o.caption, o.png) for o in report.overlays
    ] + list(plates)
    sheet.section("05", "Plates")
    if all_plates:
        sheet.para(
            "Every landmark is drawn as its 95% ellipse rather than as a dot. A "
            "dot claims a certainty the model did not report; the ellipse shows "
            "the shape of the error, which is why a point on the jaw contour "
            "smears along the jawline and a pupil does not.",
            color=INK_2,
        )
        for plate in all_plates:
            draw_plate(sheet, plate.title, plate.caption, plate.png)
    else:
        sheet.para(
            "No plates were produced for this run. The overlays are drawn from "
            "the landmark positions and their covariances, and one of those was "
            "not carried through to the renderer.",
            color=INK_2,
        )

    # -- 06 skin ----------------------------------------------------------
    findings = skin_findings(report, skin)
    if findings:
        sheet.section("06", "Skin findings")
        sheet.para(
            "Measured from the photograph in the same way as everything above: "
            "each finding carries its own interval, and a finding the gate "
            "refused is printed as a refusal. These are observations of an "
            "image. They are not a diagnosis.",
            color=INK_2,
        )
        for finding in findings:
            try:
                line = finding.format()
            except Exception:
                continue
            sheet.rule(color=RULE_SOFT, gap_before=6.0, gap_after=3.0)
            sheet.para(line, font=MONO, size=6.8, leading=9.4, width=COL_W, gap_after=2.0)
            method = getattr(finding, "method", "")
            if method:
                sheet.para(
                    str(method), size=8.8, leading=11.8, color=INK_3, gap_after=1.0
                )
        disclaimer = skin_disclaimer(skin)
        if disclaimer:
            sheet.rule(color=RULE, gap_before=10.0, gap_after=5.0)
            sheet.para(disclaimer, font=SERIF_I, size=9.0, leading=12.2, color=INK_2)

    # -- 07 method --------------------------------------------------------
    sheet.section("07", "Method")
    for para in METHOD:
        sheet.para(para, size=9.4, leading=12.8)

    # -- 08 references ----------------------------------------------------
    sheet.section("08", "References")
    if report.references:
        for ref in report.references:
            sheet.para(
                ref, font=MONO, size=6.6, leading=9.2, color=INK, width=COL_W,
                gap_after=3.0,
            )
    else:
        sheet.para("No references were supplied with this run.", color=INK_2)

    # -- 09 licence obligations -------------------------------------------
    sheet.section("09", "Licence obligations incurred by this run")
    if report.obligations:
        sheet.para(
            "The obligations of the backends that actually loaded, printed in "
            "full. A weight file's licence is a property of the run, not a "
            "footnote to the project.",
            color=INK_2,
        )
        for line in report.obligations:
            sheet.para(
                line, font=MONO, size=6.6, leading=9.2, width=COL_W, gap_after=4.0
            )
    else:
        sheet.para(
            "No backend obligations were recorded for this run. Either every "
            "model that ran was permissively licensed with nothing inherited "
            "from its training data, or the run did not record them, and those "
            "two are not the same thing.",
            color=INK_2,
        )

    # -- 10 run manifest ---------------------------------------------------
    sheet.section("10", "Run manifest")
    rows = _manifest_rows(report.manifest)
    if rows:
        sheet.key_value(rows)
    else:
        sheet.para(
            "No manifest was supplied with this run, so nothing here can be "
            "reproduced from the report alone.",
            color=INK_2,
        )

    sheet.rule(color=INK, thickness=1.2, gap_before=18.0, gap_after=6.0)
    sheet.para(COLOPHON, font=MONO, size=6.4, leading=9.0, color=INK_3, width=COL_W)


def _mono_pair(sheet: Sheet, left: str, right: str, *, left_w: float = 62.0) -> None:
    from reportlab.lib.colors import HexColor

    c = sheet.c
    c.setFont(MONO, 6.6)
    c.setFillColor(HexColor(ACCENT))
    c.drawString(COL_X, sheet.y, _flat(left))
    c.setFillColor(HexColor(INK_2))
    rows = wrap(_flat(right), MONO, 6.6, COL_W - left_w)
    c.drawString(COL_X + left_w, sheet.y, rows[0] if rows else "")
    for row in rows[1:]:
        sheet.need(8.8)
        sheet.y -= 8.8
        c.setFont(MONO, 6.6)
        c.setFillColor(HexColor(INK_2))
        c.drawString(COL_X + left_w, sheet.y, row)


def _tally_rule(sheet: Sheet, fraction: float) -> None:
    from reportlab.lib.colors import HexColor

    sheet.y -= 3.4
    sheet.c.setFillColor(HexColor(ACCENT))
    sheet.c.rect(
        COL_X + 62.0, sheet.y, max(1.0, 200.0 * float(fraction)), 1.6, stroke=0, fill=1
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

#: What a PDF viewer shows in its Document Properties panel when the run
#: recorded no date. A fixed epoch rather than the clock, because the clock is
#: the one thing that would make two runs of the same input differ.
FALLBACK_STAMP = "D:19700101000000+00'00'"

_STAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?")


def pdf_datestamp(report: ReportInput) -> str:
    """The document date, from the report rather than from the clock."""
    for candidate in (
        report.generated_at,
        str(report.manifest.get("generated_at", "")),
        str(report.manifest.get("started_at", "")),
    ):
        match = _STAMP.search(candidate or "")
        if match:
            y, mo, d, h, mi, s = match.groups()
            return f"D:{y}{mo}{d}{h}{mi}{s or '00'}+00'00'"
    return FALLBACK_STAMP


def render_pdf_bytes(
    report: ReportInput,
    *,
    plates: Sequence[Any] = (),
    skin: Any = None,
) -> bytes:
    """The whole report as PDF bytes. Deterministic for a given input.

    ``plates`` is anything with ``title``, ``caption`` and ``png``: this
    module's :class:`Plate`, or the ``MirrorComposite`` objects
    ``report.composite.mirror_composites`` returns. ``skin`` is a
    ``vitruve.derm`` ``FindingSet``, or anything iterable of findings that can
    format themselves.
    """
    try:
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise PDFUnavailable(str(exc)) from exc

    plates = tuple(_as_plates(plates)) or extra_plates(report)

    buffer = io.BytesIO()
    stamp = pdf_datestamp(report)
    c = rl_canvas.Canvas(
        buffer,
        pagesize=(PAGE_W, PAGE_H),
        invariant=1,
        pageCompression=1,
    )
    c.setDateFormatter(lambda *_: stamp)
    c.setTitle(_flat(f"Vitruve morphometric report, {report.subject_label}"))
    c.setSubject(
        _flat(
            f"{report.n_shown} of {report.n_attempted} attempted measurements "
            "reportable, each with its 95% interval"
        )
    )
    c.setAuthor("Vitruve")
    c.setCreator("Vitruve")
    c.setProducer("Vitruve (reportlab)")
    sheet = Sheet(
        c,
        subject=report.subject_label or "Unidentified subject",
        generated_at=report.generated_at or "date not recorded",
    )
    build(sheet, report, plates=plates, skin=skin)
    c.showPage()
    c.save()
    return buffer.getvalue()


def render_pdf(
    report: ReportInput,
    path: str | Path,
    *,
    plates: Sequence[Any] = (),
    skin: Any = None,
) -> Path:
    """Write the report to ``path``. The one call a caller needs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_pdf_bytes(report, plates=plates, skin=skin))
    return out


# ---------------------------------------------------------------------------
# The degraded path: print the HTML
# ---------------------------------------------------------------------------


def render_print_html(report: ReportInput) -> str:
    """The screen report with a cover page and print rules attached.

    For an install without the ``pdf`` extra. The browser's own print-to-PDF
    then produces something paginated with a cover, rather than a screenshot of
    a web page.
    """
    from .html import render as render_html

    document = render_html(report)
    cover = _render_cover_fragment(report)
    css = (TEMPLATES / "print.css").read_text(encoding="utf-8")
    document = document.replace("</head>", f"<style>{css}</style>\n</head>", 1)
    return document.replace("<body>", f"<body>\n{cover}", 1)


def _render_cover_fragment(report: ReportInput) -> str:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("cover.html.j2").render(cover=cover_of(report))


# ---------------------------------------------------------------------------
# Reading the document back
# ---------------------------------------------------------------------------

_TEXT_OP = re.compile(rb"\((?:\\.|[^\\()])*\)\s*Tj|\[(?:\\.|[^\\\[\]])*\]\s*TJ")
_STRING = re.compile(rb"\((?:\\.|[^\\()])*\)")
_ESCAPES = {
    b"n": b"\n",
    b"r": b"\r",
    b"t": b"\t",
    b"b": b"\b",
    b"f": b"\f",
    b"(": b"(",
    b")": b")",
    b"\\": b"\\",
}


def extract_text(data: bytes) -> str:
    """Every string the document draws, in the order it draws it.

    Enough of a PDF reader to let the project's own rules be checked against
    the artifact rather than against the code that wrote it, and small enough
    not to add a dependency for the privilege. Content streams only: an image
    stream decompresses to samples, not to text, and is skipped by the test for
    the text operators.
    """
    out: list[str] = []
    for stream in _streams(data):
        if b"BT" not in stream:
            continue
        for match in _TEXT_OP.finditer(stream):
            chunk = match.group(0)
            pieces = [_unescape(s[1:-1]) for s in _STRING.findall(chunk)]
            out.append("".join(pieces))
    return "\n".join(out)


def _streams(data: bytes) -> list[bytes]:
    """Every stream in the file, decoded as far as it decodes.

    reportlab writes content through ASCII85 and then Flate, so both are undone
    here, in that order, and anything that refuses either is handed back raw
    for the caller to reject.
    """
    out: list[bytes] = []
    start = 0
    marker = b"stream"
    while True:
        i = data.find(marker, start)
        if i < 0:
            break
        if data[max(i - 3, 0) : i] == b"end":
            start = i + len(marker)
            continue
        j = i + len(marker)
        if data[j : j + 2] == b"\r\n":
            j += 2
        elif data[j : j + 1] in (b"\n", b"\r"):
            j += 1
        k = data.find(b"endstream", j)
        if k < 0:
            break
        out.append(_decoded(data[j:k]))
        start = k + len(b"endstream")
    return out


def _decoded(raw: bytes) -> bytes:
    body = raw.strip()
    if body.endswith(b"~>"):
        try:
            body = base64.a85decode(body, adobe=True)
        except Exception:
            return raw
    try:
        return zlib.decompress(body)
    except zlib.error:
        return body


def _unescape(raw: bytes) -> str:
    out = bytearray()
    i = 0
    while i < len(raw):
        ch = raw[i : i + 1]
        if ch != b"\\":
            out += ch
            i += 1
            continue
        nxt = raw[i + 1 : i + 2]
        if nxt in _ESCAPES:
            out += _ESCAPES[nxt]
            i += 2
        elif nxt.isdigit():
            digits = raw[i + 1 : i + 4]
            keep = bytes(d for d in digits if 0x30 <= d <= 0x37)
            out.append(int(keep, 8) & 0xFF)
            i += 1 + len(keep)
        else:
            out += nxt
            i += 2
    return out.decode("latin-1")


__all__ = [
    "Cover",
    "PDFUnavailable",
    "Plate",
    "Sheet",
    "available",
    "cover_of",
    "extra_plates",
    "extract_text",
    "pdf_datestamp",
    "render_pdf",
    "render_pdf_bytes",
    "render_print_html",
]
