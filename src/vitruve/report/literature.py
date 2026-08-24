"""What the clinical literature associates with a measurement, and nothing else.

This layer exists because "nasofrontal angle: 128 degrees" tells a reader
nothing about why anyone measures a nasofrontal angle, and the honest answer --
that rhinoplasty surgeons measure it when planning a dorsal profile -- is a fact
about a body of published work, not a fact about the face in the photograph.
The distinction is the whole module:

    allowed      "The rhinoplasty literature measures the nasofrontal angle at
                 the radix when planning the dorsal profile (Metgudmath et al.
                 2023)."
    forbidden    Anything second-person, anything imperative, anything implying
                 the photographed face has a defect or a goal state.

Four things enforce that rather than leaving it to whoever writes the renderer.

**It is off by default.** :func:`literature_for` takes ``enabled`` and defaults
it to ``False``. Body dysmorphic disorder runs at 18.6 percent in aesthetic
plastic surgery pools, and Joseph et al. (2017) found surgeons identified 2 of
43 screen-positive patients, so the population most likely to read a list of
surgical topics next to their own face is the population least well served by
one and the least likely to be spotted. A named procedure attached to a number
is not neutral to that reader, whatever the grammar of the sentence. So the
default is silence, and the caller has to ask.

**It attaches only to a measurement that was actually reported.** A withheld
measurement has no number. Attaching literature to it would put a surgical
topic next to a quantity the gate has already refused to print, which is a
finding smuggled past the gate in prose. :func:`literature_for` reads
``Measured.shown`` and ``Finding.reportable`` and skips everything else.

**It attaches only where the value falls outside a published range.** A value
inside the range the literature reports has nothing to be said about it that
the range sentence in ``prose.py`` does not already say. Where a range exists
and the value is inside it, no note is emitted at all. Where no range exists,
the note carries no range sentence, because there is nothing to compare
against.

**Every string is checked at import.** The check is
``derm.findings.contains_advice`` -- the same function that guards every
dermatological finding -- plus ``prose.PRESCRIPTIVE_TERMS`` and a second-person
pronoun scan. A phrase that fails any of the three raises on import rather than
reaching a reader. That includes :data:`DISCLAIMER`, which is why it says "not
a medical assessment" and "no clinical determination" rather than using the
word the shared gate forbids.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..derm.findings import Finding, contains_advice
from ..measure.evaluate import Measured
from ..measure.registry import BY_ID
from .prose import PRESCRIPTIVE_TERMS

_DATA = Path(__file__).parent / "data" / "associations.json"

#: Travels with every note, and is exposed on its own so a renderer that shows
#: one association shows this too. It says "not a medical assessment" and "no
#: clinical determination" rather than the shorter word, because
#: ``contains_advice`` forbids that word outright -- it cannot tell a claim of
#: one from a denial of one, and a blunt check that costs a rewording is the
#: trade the derm layer already made.
DISCLAIMER = (
    "This note records what a body of published literature associates with a "
    "measurement. It is not a medical assessment, it makes no clinical "
    "determination about the person in the photograph, and it is not a "
    "statement that anything about that person needs changing. Vitruve is not "
    "a medical device, and its outputs have not been validated against "
    "clinical examination."
)

#: Why the layer is off unless a caller asks for it, in one sentence a reader
#: of the report can check.
WHY_OPT_IN = (
    "This section is off unless it is asked for. Body dysmorphic disorder runs "
    "at 18.6 percent in aesthetic plastic surgery pools, and Joseph et al. "
    "(2017) found that surgeons identified 2 of 43 screen-positive patients, "
    "so a list of surgical topics printed beside a measurement of one's own "
    "face reaches its worst-served reader unrecognised."
)

_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself|yourselves)\b", re.IGNORECASE)


def contains_second_person(text: str) -> str | None:
    """Return the offending pronoun, or ``None``.

    Vitruve writes about the measurement, never about the person holding the
    report. The moment a sentence addresses a reader it has stopped describing
    a photograph and started advising someone.
    """
    match = _SECOND_PERSON.search(text)
    return match.group(0) if match else None


def contains_prescription(text: str) -> str | None:
    """Return the offending term from :data:`prose.PRESCRIPTIVE_TERMS`.

    Shared with the written half of the report rather than redefined here. A
    second list would drift, and the drift would show up as a word this module
    is allowed to use and the rest of the report is not.
    """
    lowered = text.lower()
    for term in PRESCRIPTIVE_TERMS:
        if term in lowered:
            return term
    return None


def audit(text: str) -> str | None:
    """The offending phrase under any of the three checks, or ``None``."""
    return (
        contains_advice(text) or contains_prescription(text) or contains_second_person(text)
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """One source, with an identifier a reader can resolve.

    A citation without a resolvable identifier is an assertion that a paper
    exists. In a health-adjacent tool that is worse than no entry at all, so
    the constructor refuses one.
    """

    authors: str
    year: int
    journal: str
    doi: str | None = None
    pmcid: str | None = None
    isbn: str | None = None
    access: str = ""

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError(f"{self.authors} {self.year}: citation has no resolvable identifier")

    @property
    def identifier(self) -> str:
        if self.doi:
            return f"doi:{self.doi}"
        if self.pmcid:
            return self.pmcid
        if self.isbn:
            return f"ISBN {self.isbn}"
        return ""

    @property
    def url(self) -> str | None:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmcid:
            return f"https://pmc.ncbi.nlm.nih.gov/articles/{self.pmcid}/"
        return None

    def format(self) -> str:
        access = f", {self.access}" if self.access else ""
        return f"{self.authors} {self.year}, {self.journal}, {self.identifier}{access}"


@dataclass(frozen=True)
class PublishedRange:
    """A range a named source reports, used only to decide whether to speak."""

    low: float
    high: float
    unit: str
    source: str

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"{self.source}: published range must be increasing")

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class Association:
    """What the literature associates with one measurement or finding."""

    key: str
    #: ``"measurement"`` for a catalogue id, ``"derm_finding"`` for a
    #: :class:`~vitruve.derm.findings.FindingKind` value.
    target: str
    topic: str
    summary: str
    citations: tuple[Citation, ...]
    caveat: str | None = None
    published_range: PublishedRange | None = None

    def __post_init__(self) -> None:
        if self.target not in ("measurement", "derm_finding"):
            raise ValueError(f"{self.key}: unknown target {self.target!r}")
        if not self.citations:
            raise ValueError(f"{self.key}: an association without a citation is an opinion")
        for text in self.strings():
            offender = audit(text)
            if offender is not None:
                raise ValueError(
                    f"{self.key}: the literature layer states what a literature "
                    f"contains, and nothing else; the phrase {offender!r} appeared "
                    f"in {text!r}"
                )

    def strings(self) -> tuple[str, ...]:
        """Every string this association can put in front of a reader."""
        out = [self.key, self.target, self.topic, self.summary]
        if self.caveat:
            out.append(self.caveat)
        if self.published_range is not None:
            out.append(self.published_range.source)
            out.append(self.published_range.unit)
        for c in self.citations:
            out.extend([c.authors, c.journal, c.access, c.format()])
        return tuple(out)


def _citation(raw: dict) -> Citation:
    return Citation(
        authors=raw["authors"],
        year=int(raw["year"]),
        journal=raw["journal"],
        doi=raw.get("doi"),
        pmcid=raw.get("pmcid"),
        isbn=raw.get("isbn"),
        access=raw.get("access", ""),
    )


def _association(raw: dict) -> Association:
    pr = raw.get("published_range")
    return Association(
        key=raw["key"],
        target=raw["target"],
        topic=raw["topic"],
        summary=raw["summary"],
        citations=tuple(_citation(c) for c in raw["citations"]),
        caveat=raw.get("caveat"),
        published_range=(
            PublishedRange(
                low=float(pr["low"]),
                high=float(pr["high"]),
                unit=pr["unit"],
                source=pr["source"],
            )
            if pr
            else None
        ),
    )


def _load() -> tuple[Association, ...]:
    raw = json.loads(_DATA.read_text(encoding="utf-8"))
    entries = tuple(_association(a) for a in raw["associations"])
    seen: set[tuple[str, str]] = set()
    for a in entries:
        if (a.target, a.key) in seen:
            raise RuntimeError(f"duplicate association for {a.target} {a.key}")
        seen.add((a.target, a.key))
    return entries


ASSOCIATIONS: tuple[Association, ...] = _load()

BY_KEY: dict[tuple[str, str], Association] = {(a.target, a.key): a for a in ASSOCIATIONS}


def association_for(key: str, target: str = "measurement") -> Association | None:
    return BY_KEY.get((target, key))


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

_PLACES: dict[str, str] = {
    "mm": "{:.1f}",
    "deg": "{:.1f}",
    "px": "{:.0f}",
    "ratio": "{:.3g}",
}


def _fmt(value: float, unit: str) -> str:
    body = _PLACES.get(unit, "{:.3g}").format(value)
    return body if unit == "ratio" else f"{body} {unit}"


@dataclass(frozen=True)
class LiteratureNote:
    """One association, bound to one reported value."""

    key: str
    target: str
    label: str
    topic: str
    summary: str
    citations: tuple[Citation, ...]
    caveat: str | None = None
    #: Only present when a published range exists and the value fell outside
    #: it. Phrased as a position relative to a cited sample, never as a
    #: departure from a norm.
    range_note: str | None = None
    disclaimer: str = DISCLAIMER

    @property
    def subject(self) -> str:
        return "measurement" if self.target == "measurement" else "finding"

    def sentences(self) -> tuple[str, ...]:
        cited = "; ".join(c.format() for c in self.citations)
        return tuple(
            s
            for s in (
                f"{self.label}: the literature associates this {self.subject} with "
                f"{self.topic}.",
                self.summary,
                self.range_note,
                self.caveat,
                f"Sources: {cited}.",
                self.disclaimer,
            )
            if s
        )

    def format(self) -> str:
        return " ".join(self.sentences())


def _range_note(pr: PublishedRange | None, value: float, subject: str) -> str | None:
    """The neutral sentence about where a value sits, or ``None``.

    "Outside the range this source reports" is a statement about a published
    sample. "Abnormal" and "deviates" are statements about a person, and both
    are refused by the import-time audit anyway.

    The source name is audited here rather than trusted, because a range
    falling back to ``registry.py`` brings that file's source string with it
    and this module does not own that file. A source name that would fail the
    audit is replaced by a pointer to the catalogue instead of being dropped,
    so the numbers stay attributable and nothing unaudited reaches a reader.
    """
    if pr is None or pr.contains(value):
        return None
    source = pr.source
    if audit(source) is not None:
        source = "the source the measurement catalogue records"
    low = _PLACES.get(pr.unit, "{:.3g}").format(pr.low)
    return (
        f"This value sits outside the {low} to {_fmt(pr.high, pr.unit)} range "
        f"reported by {source} for this {subject}. That range describes where "
        f"a cited sample fell, and a value outside it is a position relative "
        f"to that sample and nothing more."
    )


def _range_for(association: Association, spec_id: str) -> PublishedRange | None:
    """The association's own range, or the catalogue's, or neither.

    The association's range comes from the paper actually cited here, so it
    wins. Falling back to the catalogue keeps a measurement whose range is
    declared once, in ``registry.py``, from needing it repeated.
    """
    if association.published_range is not None:
        return association.published_range
    spec = BY_ID.get(spec_id)
    if spec is None or spec.reference_range is None:
        return None
    lo, hi, source = spec.reference_range
    return PublishedRange(low=lo, high=hi, unit=spec.unit.value, source=source)


def _note_for_measured(m: Measured) -> LiteratureNote | None:
    association = association_for(m.spec_id, "measurement")
    if association is None:
        return None
    pr = _range_for(association, m.spec_id)
    range_note = _range_note(pr, m.value, "measurement")
    if pr is not None and range_note is None:
        # Inside the range the literature reports. The range sentence in
        # ``prose.py`` has already said everything there is to say.
        return None
    return LiteratureNote(
        key=association.key,
        target="measurement",
        label=m.label,
        topic=association.topic,
        summary=association.summary,
        citations=association.citations,
        caveat=association.caveat,
        range_note=range_note,
    )


def _note_for_finding(f: Finding) -> LiteratureNote | None:
    association = association_for(f.kind.value, "derm_finding")
    if association is None:
        return None
    range_note = _range_note(association.published_range, f.magnitude, "finding")
    if association.published_range is not None and range_note is None:
        return None
    return LiteratureNote(
        key=association.key,
        target="derm_finding",
        label=f.label,
        topic=association.topic,
        summary=association.summary,
        citations=association.citations,
        caveat=association.caveat,
        range_note=range_note,
    )


def literature_for(
    measurements: Sequence[Measured] = (),
    findings: Sequence[Finding] = (),
    *,
    enabled: bool = False,
) -> tuple[LiteratureNote, ...]:
    """Associated literature for results that were actually reported.

    ``enabled`` defaults to ``False`` and the function returns an empty tuple
    until a caller passes ``True``. That is the gate, and it is a parameter
    rather than a module-level setting so that no import order, environment
    variable or partially-constructed config object can turn the layer on by
    accident.

    A measurement is skipped unless :attr:`Measured.shown`; a finding unless
    :attr:`Finding.reportable`. A measurement whose value falls inside the
    range its cited source publishes is skipped as well.
    """
    if not enabled:
        return ()
    notes: list[LiteratureNote] = []
    for m in measurements:
        if not m.shown:
            continue
        note = _note_for_measured(m)
        if note is not None:
            notes.append(note)
    for f in findings:
        if not f.reportable:
            continue
        note = _note_for_finding(f)
        if note is not None:
            notes.append(note)
    return tuple(notes)


def literature_text(notes: Sequence[LiteratureNote]) -> str:
    """The section as plain text, for the CLI and the text report.

    Empty input yields an empty string rather than an empty heading, so a
    caller that renders unconditionally does not print a section title over
    nothing.
    """
    if not notes:
        return ""
    lines: list[str] = [
        "Associated literature",
        "---------------------",
        WHY_OPT_IN,
        "",
    ]
    for note in notes:
        lines.extend(note.sentences())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ASSOCIATIONS",
    "BY_KEY",
    "DISCLAIMER",
    "WHY_OPT_IN",
    "Association",
    "Citation",
    "LiteratureNote",
    "PublishedRange",
    "association_for",
    "audit",
    "contains_prescription",
    "contains_second_person",
    "literature_for",
    "literature_text",
]
