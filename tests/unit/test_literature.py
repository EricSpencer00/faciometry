"""The informational-versus-prescriptive line, asserted over the whole table.

These tests iterate every association rather than a sample. A prescriptive
phrase that reaches a reader does so through one entry, and an entry that was
not in the sample is exactly the one that will carry it.
"""

from __future__ import annotations

import json
import re

import pytest

from faciometry.core.spec import Reportability, Unit, Verdict
from faciometry.derm.findings import Finding, FindingKind, contains_advice
from faciometry.measure.evaluate import Measured
from faciometry.measure.registry import BY_ID
from faciometry.report import literature as lit
from faciometry.report.prose import PRESCRIPTIVE_TERMS

# ---------------------------------------------------------------------------
# Fixtures: the smallest Measured and Finding the module reads
# ---------------------------------------------------------------------------


def make_measured(
    spec_id: str,
    value: float,
    *,
    unit: Unit = Unit.DEGREES,
    reportability: Reportability = Reportability.REPORT,
    label: str | None = None,
) -> Measured:
    spec = BY_ID.get(spec_id)
    return Measured(
        spec_id=spec_id,
        label=label or (spec.label if spec else spec_id),
        unit=spec.unit if spec else unit,
        value=value,
        ci_low=value - 1.0,
        ci_high=value + 1.0,
        sd=0.5,
        verdict=Verdict(reportability, ()),
        discriminability=None,
        formula_fingerprint="test",
        landmarks_used=(),
        n_samples=16,
        n_valid=16,
    )


def make_finding(
    kind: FindingKind,
    magnitude: float,
    *,
    reportability: Reportability = Reportability.REPORT,
) -> Finding:
    return Finding(
        kind=kind,
        label=f"{kind.value} under test",
        region=None,
        magnitude=magnitude,
        unit="a*",
        uncertainty=0.5,
        ci_low=magnitude - 1.0,
        ci_high=magnitude + 1.0,
        verdict=Verdict(reportability, ()),
        method="test",
    )


def outside_value(association: lit.Association, spec_id: str) -> float:
    """A value guaranteed to fall outside whatever range applies."""
    pr = lit._range_for(association, spec_id)
    return pr.high + abs(pr.high) + 10.0 if pr else 1.0


def inside_value(association: lit.Association, spec_id: str) -> float | None:
    pr = lit._range_for(association, spec_id)
    return (pr.low + pr.high) / 2.0 if pr else None


ALL = lit.ASSOCIATIONS
MEASUREMENT_ASSOCIATIONS = [a for a in ALL if a.target == "measurement"]
DERM_ASSOCIATIONS = [a for a in ALL if a.target == "derm_finding"]


# ---------------------------------------------------------------------------
# The whole table, string by string
# ---------------------------------------------------------------------------


def test_table_is_not_empty():
    assert len(ALL) >= 15
    assert MEASUREMENT_ASSOCIATIONS and DERM_ASSOCIATIONS


@pytest.mark.parametrize("association", ALL, ids=lambda a: f"{a.target}:{a.key}")
def test_no_prescriptive_term_anywhere_in_the_table(association):
    for text in association.strings():
        offender = lit.contains_prescription(text)
        assert offender is None, f"{association.key}: {offender!r} in {text!r}"


@pytest.mark.parametrize("association", ALL, ids=lambda a: f"{a.target}:{a.key}")
def test_no_advice_anywhere_in_the_table(association):
    for text in association.strings():
        offender = contains_advice(text)
        assert offender is None, f"{association.key}: {offender!r} in {text!r}"


@pytest.mark.parametrize("association", ALL, ids=lambda a: f"{a.target}:{a.key}")
def test_no_second_person_anywhere_in_the_table(association):
    for text in association.strings():
        offender = lit.contains_second_person(text)
        assert offender is None, f"{association.key}: {offender!r} in {text!r}"


def test_every_string_in_the_raw_json_passes_too():
    """Not just the fields the dataclass exposes: the file as shipped.

    ``Association.strings()`` could be edited to stop returning a field. This
    walks the JSON itself, so a new field cannot arrive unchecked.
    """
    raw = json.loads(lit._DATA.read_text(encoding="utf-8"))

    def walk(node, path="$"):
        if isinstance(node, str):
            offender = lit.audit(node)
            assert offender is None, f"{path}: {offender!r} in {node!r}"
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(k, f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(raw)


def test_disclaimer_passes_the_same_checks():
    """Including the disclaimer, which is why it avoids one particular word.

    ``contains_advice`` forbids the substring "diagnos" outright and cannot
    tell a claim of one from a denial of one. The disclaimer therefore says
    "not a medical assessment" and "no clinical determination". If a future
    edit reintroduces the word, this fails rather than the layer shipping a
    string the derm gate would refuse.
    """
    assert lit.audit(lit.DISCLAIMER) is None
    assert lit.audit(lit.WHY_OPT_IN) is None


def test_disclaimer_says_what_it_is_not():
    lowered = lit.DISCLAIMER.lower()
    assert "not a medical assessment" in lowered
    assert "no clinical determination" in lowered
    assert "not a medical device" in lowered


def test_prescriptive_terms_are_shared_not_redefined():
    """The list comes from ``prose``; a private copy here would drift."""
    assert lit.contains_prescription("this value is ideal") == "ideal"
    assert "ideal" in PRESCRIPTIVE_TERMS


@pytest.mark.parametrize(
    "phrase",
    [
        "You should consider rhinoplasty.",
        "This would be improved by a filler.",
        "Recommended: blepharoplasty.",
        "Your face deviates from the ideal.",
        "We suggest you consult a dermatologist.",
    ],
)
def test_the_audit_catches_the_phrases_this_module_exists_to_exclude(phrase):
    assert lit.audit(phrase) is not None


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


_DOI = re.compile(r"^10\.\d{4,9}/\S+$")
_PMC = re.compile(r"^PMC\d+$")
_ISBN = re.compile(r"^[\d-]{10,17}$")


@pytest.mark.parametrize("association", ALL, ids=lambda a: f"{a.target}:{a.key}")
def test_every_citation_has_a_resolvable_identifier(association):
    assert association.citations, f"{association.key} has no citation"
    for c in association.citations:
        assert c.identifier, f"{association.key}: {c.authors} {c.year} has no identifier"
        if c.doi:
            assert _DOI.match(c.doi), f"{association.key}: malformed DOI {c.doi!r}"
        elif c.pmcid:
            assert _PMC.match(c.pmcid), f"{association.key}: malformed PMC id {c.pmcid!r}"
        else:
            assert _ISBN.match(c.isbn or ""), f"{association.key}: malformed ISBN {c.isbn!r}"
        assert 1900 < c.year <= 2030
        assert c.authors and c.journal


def test_a_citation_without_an_identifier_is_refused():
    with pytest.raises(ValueError, match="resolvable identifier"):
        lit.Citation(authors="Nobody", year=2020, journal="Nowhere")


def test_an_association_without_a_citation_is_refused():
    with pytest.raises(ValueError, match="opinion"):
        lit.Association(
            key="x", target="measurement", topic="t", summary="s", citations=()
        )


def test_a_prescriptive_association_is_refused_at_construction():
    with pytest.raises(ValueError, match="literature layer"):
        lit.Association(
            key="x",
            target="measurement",
            topic="t",
            summary="You should consider surgery.",
            citations=(lit.Citation("A", 2020, "J", doi="10.1000/abc"),),
        )


@pytest.mark.parametrize("association", ALL, ids=lambda a: f"{a.target}:{a.key}")
def test_citation_urls_resolve_to_a_known_host(association):
    for c in association.citations:
        url = c.url
        if url is None:
            assert c.isbn, f"{association.key}: no URL and no ISBN"
            continue
        assert url.startswith(("https://doi.org/", "https://pmc.ncbi.nlm.nih.gov/"))


# ---------------------------------------------------------------------------
# Keys line up with the rest of the system
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "association", MEASUREMENT_ASSOCIATIONS, ids=lambda a: a.key
)
def test_measurement_keys_exist_in_the_catalogue(association):
    assert association.key in BY_ID, f"{association.key} is not a catalogue measurement"


@pytest.mark.parametrize("association", DERM_ASSOCIATIONS, ids=lambda a: a.key)
def test_derm_keys_are_finding_kinds(association):
    assert association.key in {k.value for k in FindingKind}


def test_the_brief_measurements_are_all_covered():
    covered = {a.key for a in MEASUREMENT_ASSOCIATIONS}
    for key in (
        "nasofrontal_angle",
        "nasolabial_angle",
        "e_line_upper_lip",
        "e_line_lower_lip",
        "gonial_angle_l",
        "gonial_angle_r",
        "mentocervical_angle",
        "canthal_tilt_l",
        "canthal_tilt_r",
        "bizygomatic_width",
        "facial_thirds_ratio",
        "lip_vermilion_ratio",
        "nasal_tip_projection_ratio",
    ):
        assert key in covered, f"{key} has no association"
    for kind in (
        FindingKind.ACNE_SEVERITY,
        FindingKind.ERYTHEMA,
        FindingKind.PERIORBITAL_PIGMENTATION,
    ):
        assert lit.association_for(kind.value, "derm_finding") is not None


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_disabled_by_default_over_the_entire_table():
    """No argument at all, every association's own key: nothing comes out."""
    for a in MEASUREMENT_ASSOCIATIONS:
        m = make_measured(a.key, outside_value(a, a.key))
        assert lit.literature_for([m]) == ()
    for a in DERM_ASSOCIATIONS:
        f = make_finding(FindingKind(a.key), 99.0)
        assert lit.literature_for(findings=[f]) == ()


def test_explicitly_disabled_is_empty():
    a = MEASUREMENT_ASSOCIATIONS[0]
    m = make_measured(a.key, outside_value(a, a.key))
    assert lit.literature_for([m], enabled=False) == ()


def test_enabled_is_the_only_way_to_get_anything():
    a = MEASUREMENT_ASSOCIATIONS[0]
    m = make_measured(a.key, outside_value(a, a.key))
    assert lit.literature_for([m], enabled=True)


def test_the_flag_is_keyword_only():
    a = MEASUREMENT_ASSOCIATIONS[0]
    m = make_measured(a.key, outside_value(a, a.key))
    with pytest.raises(TypeError):
        lit.literature_for([m], [], True)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Nothing for a withheld measurement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "association", MEASUREMENT_ASSOCIATIONS, ids=lambda a: a.key
)
def test_nothing_is_emitted_for_a_withheld_measurement(association):
    """The whole table, at a value that would otherwise speak."""
    value = outside_value(association, association.key)
    withheld = make_measured(association.key, value, reportability=Reportability.WITHHOLD)
    assert withheld.shown is False
    assert lit.literature_for([withheld], enabled=True) == ()
    shown = make_measured(association.key, value)
    assert lit.literature_for([shown], enabled=True), "control: it speaks when shown"


@pytest.mark.parametrize("association", DERM_ASSOCIATIONS, ids=lambda a: a.key)
def test_nothing_is_emitted_for_a_withheld_finding(association):
    kind = FindingKind(association.key)
    withheld = make_finding(kind, 42.0, reportability=Reportability.WITHHOLD)
    assert withheld.reportable is False
    assert lit.literature_for(findings=[withheld], enabled=True) == ()
    shown = make_finding(kind, 42.0)
    assert lit.literature_for(findings=[shown], enabled=True), "control: it speaks when shown"


def test_a_caveated_measurement_still_counts_as_reported():
    a = MEASUREMENT_ASSOCIATIONS[0]
    m = make_measured(
        a.key, outside_value(a, a.key), reportability=Reportability.CAVEAT
    )
    assert lit.literature_for([m], enabled=True)


# ---------------------------------------------------------------------------
# Only outside the published range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "association",
    [a for a in MEASUREMENT_ASSOCIATIONS if lit._range_for(a, a.key) is not None],
    ids=lambda a: a.key,
)
def test_silent_inside_the_published_range(association):
    mid = inside_value(association, association.key)
    assert mid is not None
    m = make_measured(association.key, mid)
    assert lit.literature_for([m], enabled=True) == ()


@pytest.mark.parametrize(
    "association",
    [a for a in MEASUREMENT_ASSOCIATIONS if lit._range_for(a, a.key) is not None],
    ids=lambda a: a.key,
)
def test_outside_the_published_range_says_so_neutrally(association):
    m = make_measured(association.key, outside_value(association, association.key))
    (note,) = lit.literature_for([m], enabled=True)
    assert note.range_note is not None
    assert "sits outside" in note.range_note
    for banned in ("abnormal", "deviate", "should", "normal range"):
        assert banned not in note.range_note.lower()


def test_a_measurement_without_a_range_speaks_at_any_value():
    without = [
        a
        for a in MEASUREMENT_ASSOCIATIONS
        if lit._range_for(a, a.key) is None
    ]
    assert without, "expected at least one association with no published range"
    for a in without:
        m = make_measured(a.key, 1.0)
        (note,) = lit.literature_for([m], enabled=True)
        assert note.range_note is None


def test_the_catalogue_range_is_used_when_the_association_declares_none():
    a = lit.association_for("facial_thirds_ratio")
    assert a is not None and a.published_range is None
    spec = BY_ID["facial_thirds_ratio"]
    assert spec.reference_range is None, (
        "this test assumes the catalogue publishes no range here; if that "
        "changed, the fallback path needs a different example"
    )


def test_the_association_range_wins_over_the_catalogue():
    a = lit.association_for("nasofrontal_angle")
    assert a is not None and a.published_range is not None
    spec = BY_ID["nasofrontal_angle"]
    assert spec.reference_range is not None
    assert lit._range_for(a, "nasofrontal_angle") is a.published_range


def test_an_unknown_measurement_gets_nothing():
    m = make_measured("not_a_measurement", 42.0, unit=Unit.DEGREES, label="Unknown")
    assert lit.literature_for([m], enabled=True) == ()


# ---------------------------------------------------------------------------
# What comes out
# ---------------------------------------------------------------------------


def test_every_emitted_sentence_passes_the_audit():
    """Generated sentences, not just stored ones, over the whole table."""
    measured = [
        make_measured(a.key, outside_value(a, a.key)) for a in MEASUREMENT_ASSOCIATIONS
    ]
    findings = [make_finding(FindingKind(a.key), 99.0) for a in DERM_ASSOCIATIONS]
    notes = lit.literature_for(measured, findings, enabled=True)
    assert len(notes) == len(ALL)
    for note in notes:
        for sentence in note.sentences():
            offender = lit.audit(sentence)
            assert offender is None, f"{note.key}: {offender!r} in {sentence!r}"
        assert lit.audit(note.format()) is None


def test_every_note_carries_the_disclaimer():
    measured = [
        make_measured(a.key, outside_value(a, a.key)) for a in MEASUREMENT_ASSOCIATIONS
    ]
    notes = lit.literature_for(measured, enabled=True)
    for note in notes:
        assert note.disclaimer == lit.DISCLAIMER
        assert lit.DISCLAIMER in note.sentences()


def test_every_note_names_its_sources():
    measured = [
        make_measured(a.key, outside_value(a, a.key)) for a in MEASUREMENT_ASSOCIATIONS
    ]
    for note in lit.literature_for(measured, enabled=True):
        joined = note.format()
        for c in note.citations:
            assert c.identifier in joined


def test_no_aggregate_score_in_the_rendered_section():
    """The project's standing rule, checked here too.

    ``tests/unit/test_no_aggregate_score.py`` covers the report; this covers
    the one section that names surgical topics, which is where a summary score
    would be most tempting and most harmful.
    """
    measured = [
        make_measured(a.key, outside_value(a, a.key)) for a in MEASUREMENT_ASSOCIATIONS
    ]
    text = lit.literature_text(lit.literature_for(measured, enabled=True))
    lowered = text.lower()
    for banned in ("overall score", "total score", "out of 10", "composite score", "rating"):
        assert banned not in lowered


def test_the_text_section_is_empty_when_nothing_was_emitted():
    assert lit.literature_text(()) == ""
    assert lit.literature_text(lit.literature_for([], [])) == ""


def test_the_text_section_states_why_it_is_opt_in():
    a = MEASUREMENT_ASSOCIATIONS[0]
    m = make_measured(a.key, outside_value(a, a.key))
    text = lit.literature_text(lit.literature_for([m], enabled=True))
    assert "18.6 percent" in text
    assert "2 of 43" in text


def test_notes_arrive_in_input_order():
    a, b = MEASUREMENT_ASSOCIATIONS[0], MEASUREMENT_ASSOCIATIONS[1]
    notes = lit.literature_for(
        [
            make_measured(b.key, outside_value(b, b.key)),
            make_measured(a.key, outside_value(a, a.key)),
        ],
        enabled=True,
    )
    assert [n.key for n in notes] == [b.key, a.key]


def test_mixed_input_keeps_measurements_before_findings():
    a = MEASUREMENT_ASSOCIATIONS[0]
    d = DERM_ASSOCIATIONS[0]
    notes = lit.literature_for(
        [make_measured(a.key, outside_value(a, a.key))],
        [make_finding(FindingKind(d.key), 99.0)],
        enabled=True,
    )
    assert [n.target for n in notes] == ["measurement", "derm_finding"]


def test_the_table_has_no_duplicate_keys():
    keys = [(a.target, a.key) for a in ALL]
    assert len(keys) == len(set(keys))
