"""The findings layer: intervals, refusals, provenance, and no advice.

Two sources of evidence reach one type here, so the tests check the arithmetic
of each (a Poisson interval on a count, a propagated interval on a contrast) and
then the rules that apply to both: a quantity below its own uncertainty is
withheld with a reason, a value is never printed without its interval, and no
note anywhere is allowed to turn into a recommendation.

Nothing loads weights. The detector is exercised through a stub, which is the
only way to test the crop-and-attribute logic without an AGPL-3.0 dependency
and a network fetch.
"""

from __future__ import annotations

import math
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from test_regions import face_points

from faciometry.core.spec import Reportability, Verdict
from faciometry.derm.colorimetry import (
    RegionColour,
    erythema,
    monk_tone,
    periorbital_pigmentation,
)
from faciometry.derm.detect_yolo import (
    DETECTION_REGIONS,
    ESTIMATED_HALF_FACE_NOTE,
    AcneDetection,
    AcneDetector,
    DetectorUnavailable,
    Lesion,
    RegionLesions,
    Severity,
    hayashi_severity,
    iou,
    merge_duplicates,
    tiles_for_region,
)
from faciometry.derm.findings import (
    DISCLAIMER,
    Finding,
    FindingKind,
    FindingSet,
    collect,
    contains_advice,
    from_acne_severity,
    from_erythema,
    from_pigmentation,
    from_region_lesions,
    from_tone,
)
from faciometry.derm.regions import Region, build_regions
from faciometry.models.licensing import YOLO_DERM_SEG, LicenseViolation, Tier


def colour(lab, region=None, se=(0.5, 0.5, 0.5), calibrated=False) -> RegionColour:
    return RegionColour(
        region=region,
        n_pixels=1200,
        n_blocks=18,
        mean=tuple(float(v) for v in lab),
        sd=(1.0, 1.0, 1.0),
        se=tuple(float(v) for v in se),
        calibrated=calibrated,
    )


# ---------------------------------------------------------------------------
# Hayashi grading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, Severity.CLEAR),
        (1, Severity.MILD),
        (5, Severity.MILD),
        (6, Severity.MODERATE),
        (20, Severity.MODERATE),
        (21, Severity.SEVERE),
        (50, Severity.SEVERE),
        (51, Severity.VERY_SEVERE),
        (200, Severity.VERY_SEVERE),
    ],
)
def test_hayashi_boundaries(count, expected):
    assert hayashi_severity(count) is expected


def test_a_negative_count_is_not_a_grade():
    with pytest.raises(ValueError):
        hayashi_severity(-1)


# ---------------------------------------------------------------------------
# Tiling and deduplication
# ---------------------------------------------------------------------------


def test_iou_is_closed_form():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Half overlap: intersection 50, union 150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50.0 / 150.0)


def test_a_small_region_gets_one_tile_at_the_trained_input_scale():
    tiles = tiles_for_region(Region.MALAR_R, (400.0, 400.0, 440.0, 430.0), 900, 800, tile_px=640)
    assert len(tiles) == 1
    assert tiles[0].width == 640 and tiles[0].height == 640
    assert tiles[0].x0 >= 0 and tiles[0].x1 <= 800


def test_a_large_region_is_covered_by_overlapping_tiles():
    tiles = tiles_for_region(
        Region.FOREHEAD, (0.0, 0.0, 800.0, 800.0), 2000, 2000, tile_px=256, overlap=0.25
    )
    assert len(tiles) > 1
    # Overlap means consecutive origins step by less than a full tile.
    xs = sorted({t.x0 for t in tiles})
    assert min(b - a for a, b in zip(xs, xs[1:])) < 256


def test_tiles_outside_the_image_are_empty_rather_than_negative():
    assert tiles_for_region(Region.MALAR_R, (900.0, 900.0, 950.0, 950.0), 100, 100) == ()


def test_merge_duplicates_keeps_the_most_confident_instance():
    a = Lesion(Region.MALAR_R, (10, 10, 20, 20), 0.9)
    b = Lesion(Region.MALAR_R, (11, 11, 21, 21), 0.5)  # the same papule on the next tile
    c = Lesion(Region.MALAR_R, (60, 60, 70, 70), 0.4)
    kept = merge_duplicates([b, a, c])
    assert [d.confidence for d in kept] == [0.9, 0.4]
    assert kept[0] is a


def test_lesion_geometry():
    lesion = Lesion(Region.MALAR_R, (10.0, 10.0, 20.0, 20.0), 0.8)
    assert lesion.centre == (15.0, 15.0)
    assert lesion.box_area == pytest.approx(100.0)
    assert lesion.equivalent_diameter == pytest.approx(2.0 * math.sqrt(100.0 / math.pi))


# ---------------------------------------------------------------------------
# The detector, without the detector
# ---------------------------------------------------------------------------


def test_the_licence_is_checked_before_anything_is_loaded():
    detector = AcneDetector(weights="/definitely/not/here.pt", allowed_tier=Tier.PERMISSIVE)
    with pytest.raises(LicenseViolation, match="copyleft"):
        detector.load()
    assert not detector.loaded


def test_detect_before_load_is_a_named_failure():
    detector = AcneDetector(weights="x.pt", allowed_tier=Tier.COPYLEFT)
    with pytest.raises(DetectorUnavailable):
        detector.detect(np.zeros((10, 10, 3), np.uint8), build_regions(face_points()))


class StubDetector(AcneDetector):
    """A detector whose model is a fixed list of boxes in tile coordinates."""

    def __init__(self, boxes, **kwargs):
        super().__init__(weights="stub.pt", allowed_tier=Tier.COPYLEFT, **kwargs)
        self._boxes = boxes
        self._model = object()

    def _predict_tile(self, crop):
        return list(self._boxes)


def test_detections_are_mapped_back_and_attributed_to_their_region():
    regions = build_regions(face_points())
    image = np.zeros((900, 800, 3), dtype=np.uint8)
    poly = regions.get(Region.MALAR_R)
    cx, cy = poly.centroid  # canonical frame, +y up
    row = 899 - cy
    # Tiles are laid out in array coordinates, so the polygon's vertical bounds
    # flip first. Getting this wrong is exactly the bug the y_up flag exists to
    # make visible, so the test reproduces the conversion rather than guessing.
    x_lo, y_lo, x_hi, y_hi = poly.bounds
    tiles = tiles_for_region(
        Region.MALAR_R, (x_lo, 899 - y_hi, x_hi, 899 - y_lo), 900, 800, tile_px=64
    )
    tile = tiles[0]
    inside = (cx - tile.x0 - 2, row - tile.y0 - 2, cx - tile.x0 + 2, row - tile.y0 + 2)
    outside = (0.0, 0.0, 3.0, 3.0)
    detector = StubDetector(
        [(inside, 0.8, "inflammatory", None), (outside, 0.7, "inflammatory", None)],
        tile_px=64,
    )
    result = detector.detect(image, regions, only=[Region.MALAR_R])

    group = result.per_region[Region.MALAR_R]
    assert group.count == 1, "a box inside the tile but outside the polygon is not a lesion"
    lesion = group.lesions[0]
    assert lesion.region is Region.MALAR_R
    assert lesion.centre[0] == pytest.approx(cx, abs=1.0)
    assert group.area_px == pytest.approx(poly.area)
    assert group.density_per_kpx == pytest.approx(1000.0 / poly.area)
    assert result.provenance is YOLO_DERM_SEG
    assert any("native resolution" in n for n in result.notes)


def test_detection_regions_exclude_the_eyelids():
    assert Region.PERIORBITAL_L not in DETECTION_REGIONS
    assert Region.INFRAORBITAL_R not in DETECTION_REGIONS
    assert Region.MALAR_L in DETECTION_REGIONS


def _detection(counts: dict[Region, list[float]]) -> AcneDetection:
    """Build a detection whose lesions sit at the given x coordinates."""
    per_region = {}
    for region, xs in counts.items():
        lesions = tuple(
            Lesion(region, (x - 2, 100.0, x + 2, 104.0), 0.9) for x in xs
        )
        per_region[region] = RegionLesions(region, lesions, tiles=1, area_px=1000.0)
    return AcneDetection(per_region=per_region, provenance=YOLO_DERM_SEG, confidence_threshold=0.25)


def test_half_face_counts_split_midline_regions_by_lesion_position():
    detection = _detection(
        {
            Region.FOREHEAD: [350.0, 360.0, 450.0],  # two left of the midline, one right
            Region.MALAR_R: [460.0, 470.0],
        }
    )
    assert detection.total_count == 5
    assert detection.count_on("l", 400.0) == 2
    assert detection.count_on("r", 400.0) == 3
    assert detection.hayashi_count(400.0) == 3
    with pytest.raises(ValueError):
        detection.count_on("left", 400.0)


def test_severity_without_a_midline_halves_the_total_and_rounds_up():
    detection = _detection({Region.FOREHEAD: [float(i) for i in range(13)]})
    severity, verdict = detection.severity()
    assert severity is hayashi_severity(7)
    assert any("Hayashi" in r for r in verdict.reasons)


def test_severity_without_a_midline_caveats_the_estimated_count():
    detection = _detection({Region.FOREHEAD: [float(i) for i in range(13)]})
    _, verdict = detection.severity()
    assert verdict.reportability is Reportability.CAVEAT
    assert ESTIMATED_HALF_FACE_NOTE in verdict.caveats


def test_severity_from_a_real_half_face_count_carries_no_estimate_caveat():
    detection = _detection({Region.FOREHEAD: [float(i) for i in range(13)]})
    _, verdict = detection.severity(half_face_count=7)
    assert verdict.reportability is Reportability.CAVEAT
    assert ESTIMATED_HALF_FACE_NOTE not in verdict.caveats


# ---------------------------------------------------------------------------
# Findings: the rules that apply to every source of evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "we recommend a gentle cleanser",
        "You should consult a dermatologist",
        "this responds well to treatment",
        "a daily skincare regimen helps",
        "consistent with a diagnosis of rosacea",
    ],
)
def test_advice_is_detected(text):
    assert contains_advice(text) is not None


def test_measurement_language_is_not_advice():
    for text in (
        "the infraorbital region is 4.2 L* units darker than the cheek below it",
        "withheld: the contrast is smaller than its own uncertainty",
        "detected 7 lesions over 3 crops",
    ):
        assert contains_advice(text) is None


def test_a_finding_refuses_to_carry_advice():
    with pytest.raises(ValueError, match="not advice"):
        Finding(
            kind=FindingKind.ERYTHEMA,
            label="erythema",
            region=Region.MALAR_R,
            magnitude=3.0,
            unit="a*",
            uncertainty=0.7,
            ci_low=1.6,
            ci_high=4.4,
            verdict=Verdict(Reportability.REPORT, ()),
            method="test",
            notes=("we recommend sunscreen",),
        )


def test_a_finding_refuses_an_inverted_interval():
    with pytest.raises(ValueError, match="inverted"):
        Finding(
            kind=FindingKind.ERYTHEMA,
            label="erythema",
            region=None,
            magnitude=3.0,
            unit="a*",
            uncertainty=0.7,
            ci_low=4.4,
            ci_high=1.6,
            verdict=Verdict(Reportability.REPORT, ()),
            method="test",
        )


def test_no_finding_field_is_an_aggregate_score():
    names = {f.name for f in fields(Finding)} | {f.name for f in fields(FindingSet)}
    for banned in ("score", "overall", "rating", "grade", "harmony", "total"):
        assert not any(banned in n for n in names), f"{banned} appeared in {names}"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def test_an_erythema_finding_carries_its_interval_and_its_caveat():
    contrast = erythema(
        colour((57.0, 19.0, 17.0), Region.MALAR_R), colour((59.0, 12.0, 17.0), Region.LATERAL_CHEEK_R)
    )
    finding = from_erythema(contrast)
    assert finding.kind is FindingKind.ERYTHEMA
    assert finding.magnitude == pytest.approx(7.0)
    assert finding.reference_region is Region.LATERAL_CHEEK_R
    assert finding.ci_low < finding.magnitude < finding.ci_high
    assert finding.reportable
    assert "no learned model" in finding.method
    assert any("uncalibrated" in r for r in finding.verdict.reasons)
    assert "7.00 a*" in finding.format() and "95% CI" in finding.format()


def test_an_erythema_below_its_uncertainty_is_withheld_with_a_reason():
    contrast = erythema(
        colour((58.0, 12.3, 17.0), Region.MALAR_R), colour((58.0, 12.0, 17.0), Region.LATERAL_CHEEK_R)
    )
    finding = from_erythema(contrast)
    assert not finding.reportable
    assert finding.verdict.reportability is Reportability.WITHHOLD
    assert finding.verdict.reasons
    line = finding.format()
    assert "withheld" in line
    # The reason bounds the contrast; it does not restate it.
    assert "0.30" not in line and "+0.3" not in line
    assert "below its own standard error" in line


def test_a_calibrated_contrast_needs_no_white_balance_caveat():
    contrast = erythema(
        colour((57.0, 19.0, 17.0), Region.MALAR_R, calibrated=True),
        colour((59.0, 12.0, 17.0), Region.LATERAL_CHEEK_R, calibrated=True),
    )
    finding = from_erythema(contrast)
    assert finding.calibrated
    assert finding.verdict.reportability is Reportability.REPORT
    assert finding.verdict.reasons == ()


def test_a_pigmentation_finding_reports_both_channels():
    contrast = periorbital_pigmentation(
        colour((50.0, 12.0, 21.0), Region.INFRAORBITAL_R), colour((56.0, 12.0, 17.0), Region.MALAR_R)
    )
    finding = from_pigmentation(contrast)
    assert finding.magnitude == pytest.approx(math.hypot(6.0, 4.0))
    assert finding.reportable
    assert any("L* deficit" in n for n in finding.notes)
    assert any("b* elevation" in n for n in finding.notes)
    assert any("no annotated public dataset" in n for n in finding.notes)
    assert any("darker and yellower" in r for r in finding.verdict.reasons)


def test_a_flat_infraorbital_region_is_withheld():
    contrast = periorbital_pigmentation(
        colour((56.1, 12.0, 17.0), Region.INFRAORBITAL_R), colour((56.0, 12.0, 17.0), Region.MALAR_R)
    )
    finding = from_pigmentation(contrast)
    assert not finding.reportable
    assert "smaller than the uncertainty" in finding.verdict.reasons[0]


def test_a_tone_finding_is_caveated_when_uncalibrated_and_clean_when_not():
    c = colour((55.14, 7.79, 26.74), Region.MALAR_R)  # Monk swatch 6
    uncal = from_tone(monk_tone(c), c)
    assert uncal.magnitude == 6.0
    assert uncal.verdict.reportability is Reportability.CAVEAT
    assert any("Fitzpatrick" in n for n in uncal.notes)
    assert any("ITA" in n for n in uncal.notes)

    c2 = colour((55.14, 7.79, 26.74), Region.MALAR_R, calibrated=True)
    cal = from_tone(monk_tone(c2), c2)
    assert cal.verdict.reportability is Reportability.REPORT


def test_a_lesion_count_carries_a_poisson_interval():
    group = RegionLesions(
        Region.MALAR_R,
        tuple(Lesion(Region.MALAR_R, (float(i), 0.0, float(i) + 2, 2.0), 0.9) for i in range(0, 40, 10)),
        tiles=2,
        area_px=2000.0,
    )
    finding = from_region_lesions(group, YOLO_DERM_SEG)
    assert finding.count == 4
    assert finding.uncertainty == pytest.approx(2.0)
    assert finding.ci_low == pytest.approx(4 - 1.96 * 2.0)
    assert finding.ci_high == pytest.approx(4 + 1.96 * 2.0)
    assert finding.provenance is YOLO_DERM_SEG
    assert any("counting noise only" in r for r in finding.verdict.reasons)
    assert "4 lesions" in finding.format()


def test_zero_lesions_is_a_statement_about_the_detector():
    group = RegionLesions(Region.MALAR_L, (), tiles=1, area_px=1000.0)
    finding = from_region_lesions(group, YOLO_DERM_SEG)
    assert finding.count == 0
    assert finding.ci_low == 0.0
    assert any("not about the skin" in r for r in finding.verdict.reasons)


def test_severity_finding_states_the_grading_convention():
    detection = _detection({Region.FOREHEAD: [350.0, 360.0, 370.0, 450.0, 460.0, 470.0, 480.0]})
    finding = from_acne_severity(detection, midline_x=400.0)
    assert finding.severity is Severity.MILD  # four on the right half
    assert finding.count == 4
    assert any("Hayashi" in r for r in finding.verdict.reasons)
    assert any("facial midline" in r for r in finding.verdict.reasons)
    assert finding.verdict.reportability is Reportability.CAVEAT
    assert "Hayashi grade mild" in finding.format()


def test_severity_without_a_midline_says_the_count_is_an_estimate():
    detection = _detection({Region.FOREHEAD: [float(i) for i in range(13)]})
    finding = from_acne_severity(detection)
    assert finding.count == 7
    assert any("no facial midline was supplied" in r for r in finding.verdict.reasons)


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------


def test_collect_assembles_everything_and_keeps_the_disclaimer():
    regions = build_regions(face_points())
    ery = erythema(
        colour((57.0, 19.0, 17.0), Region.MALAR_R), colour((59.0, 12.0, 17.0), Region.LATERAL_CHEEK_R)
    )
    flat = erythema(
        colour((58.0, 12.1, 17.0), Region.MALAR_L), colour((58.0, 12.0, 17.0), Region.LATERAL_CHEEK_L)
    )
    pig = periorbital_pigmentation(
        colour((50.0, 12.0, 21.0), Region.INFRAORBITAL_R), colour((56.0, 12.0, 17.0), Region.MALAR_R)
    )
    tone_colour = colour((55.14, 7.79, 26.74), Region.MALAR_R)
    detection = _detection({Region.FOREHEAD: [350.0, 450.0, 460.0]})

    result = collect(
        erythema_contrasts=[ery, flat],
        pigmentation=[pig],
        tone=(monk_tone(tone_colour), tone_colour),
        detection=detection,
        midline_x=regions.frame.origin[0],
        unavailable={Region.PERIORAL: ("cheilion_l",)},
    )
    assert isinstance(result, FindingSet)
    assert len(result) == 6
    assert len(result.withheld) == 1
    assert result.withheld[0].kind is FindingKind.ERYTHEMA
    assert {f.kind for f in result.reportable} >= {
        FindingKind.ERYTHEMA,
        FindingKind.PERIORBITAL_PIGMENTATION,
        FindingKind.SKIN_TONE,
        FindingKind.ACNE_LESION_COUNT,
        FindingKind.ACNE_SEVERITY,
    }
    assert result.of_kind(FindingKind.ACNE_SEVERITY)[0].severity is Severity.MILD
    assert any("cheilion_l" in n for n in result.notes)

    text = result.format()
    assert DISCLAIMER in text
    assert "not a medical" in text
    assert "withheld" in text
    assert contains_advice(text.replace(DISCLAIMER, "")) is None


def test_an_empty_set_still_carries_the_disclaimer():
    empty = collect()
    assert len(empty) == 0
    assert DISCLAIMER in empty.format()


def test_no_finding_ever_prints_a_bare_number():
    result = collect(
        erythema_contrasts=[
            erythema(
                colour((57.0, 19.0, 17.0), Region.MALAR_R),
                colour((59.0, 12.0, 17.0), Region.LATERAL_CHEEK_R),
            )
        ],
        detection=_detection({Region.FOREHEAD: [350.0, 450.0]}),
    )
    for finding in result.reportable:
        assert "95% CI" in finding.format()


def test_reasons_are_attributed_to_their_own_severity():
    """The blocking reason and the caveats that would have applied anyway differ."""
    withheld = from_erythema(
        erythema(
            colour((58.0, 12.3, 17.0), Region.MALAR_R),
            colour((58.0, 12.0, 17.0), Region.LATERAL_CHEEK_R),
        )
    )
    assert len(withheld.verdict.blocking) == 1
    assert "below its own standard error" in withheld.verdict.blocking[0]
    # The white balance was already a caveat and did not cause the withholding.
    assert any("uncalibrated" in c for c in withheld.verdict.caveats)
    assert withheld.verdict.reasons[0] == withheld.verdict.blocking[0]

    shown = from_erythema(
        erythema(
            colour((57.0, 19.0, 17.0), Region.MALAR_R),
            colour((59.0, 12.0, 17.0), Region.LATERAL_CHEEK_R),
        )
    )
    assert shown.verdict.blocking == ()
    assert shown.verdict.caveats


def test_a_withheld_pigmentation_finding_names_only_the_blocking_reason():
    finding = from_pigmentation(
        periorbital_pigmentation(
            colour((56.1, 12.0, 17.0), Region.INFRAORBITAL_R),
            colour((56.0, 12.0, 17.0), Region.MALAR_R),
        )
    )
    assert len(finding.verdict.blocking) == 1
    assert finding.verdict.caveats == ()
    assert "withheld" in finding.format()
