"""What the renderer is handed.

The report layer takes one frozen input and turns it into pixels and prose. It
deliberately does not reach back into the pipeline: everything it needs arrives
in :class:`ReportInput`, so a report renders from measurements constructed by
hand. That is how the tests build one, and it is what keeps the renderer
testable without a model, a GPU, or a photograph.

Grouping is by anatomical region rather than by evidence tier, because the
person reading the report is looking at a face and not at a taxonomy. The tier
travels on every row instead, where it cannot be skipped past.

There is no field on anything in this module that combines the *values* of two
different measurements. Counts of outcomes are not that -- how many
measurements were withheld is a fact about the photograph, not a rating of the
face -- and the distinction is asserted in
``tests/unit/test_no_aggregate_score.py``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Mapping

from ..core.spec import Reportability
from ..measure.evaluate import Measured, Unavailable
from ..norms import niosh


@dataclass(frozen=True)
class Region:
    """One anatomical grouping of measurements, and one overlay."""

    key: str
    title: str
    blurb: str


REGIONS: tuple[Region, ...] = (
    Region(
        "periocular",
        "Eye region",
        "Spacing, aperture and inclination around the orbits.",
    ),
    Region("nose", "Nose", "Nasal width, height and profile angles."),
    Region("mouth", "Mouth and philtrum", "Lip and philtral dimensions."),
    Region(
        "jaw",
        "Jaw and chin",
        "Mandibular width and angle. Both endpoints of a jaw width sit on a "
        "surface that turns away from the camera.",
    ),
    Region(
        "proportion",
        "Whole-face proportion",
        "Ratios between regions, and the vertical divisions of the face.",
    ),
    Region(
        "symmetry",
        "Side-to-side differences",
        "Differences between paired features, reported per feature. A common "
        "rotation of the head adds the same offset to both sides and cancels "
        "in the difference, which is what makes these the most pose-robust "
        "quantities here.",
    ),
    Region("profile", "Profile outline", "Angles and offsets read from the side view."),
    Region("other", "Other", "Measurements outside the named regions."),
)

BY_KEY: dict[str, Region] = {r.key: r for r in REGIONS}

#: Region of every measurement in the catalogue, by id. Explicit rather than
#: inferred from the name, because ``nasolabial_angle`` reads the mouth and
#: ``eye_spacing_ratio`` is a whole-face proportion, and a substring rule gets
#: both of those wrong.
REGION_OF: dict[str, str] = {
    "interpupillary_distance": "periocular",
    "intercanthal_width": "periocular",
    "biocular_width": "periocular",
    "palpebral_fissure_width_l": "periocular",
    "palpebral_fissure_width_r": "periocular",
    "palpebral_fissure_height_l": "periocular",
    "palpebral_fissure_height_r": "periocular",
    "canthal_tilt_l": "periocular",
    "canthal_tilt_r": "periocular",
    "eye_aspect_ratio_l": "periocular",
    "eye_aspect_ratio_r": "periocular",
    "intercanthal_biocular_ratio": "periocular",
    "nose_breadth": "nose",
    "nose_height": "nose",
    "nasofrontal_angle": "nose",
    "nasal_tip_projection_ratio": "nose",
    "mouth_width": "mouth",
    "philtrum_width": "mouth",
    "philtrum_length": "mouth",
    "lip_vermilion_ratio": "mouth",
    "nasolabial_angle": "mouth",
    "e_line_upper_lip": "mouth",
    "e_line_lower_lip": "mouth",
    "bigonial_width": "jaw",
    "bizygomatic_width": "jaw",
    "jaw_cheekbone_ratio": "jaw",
    "gonial_angle_l": "jaw",
    "gonial_angle_r": "jaw",
    "mentolabial_angle": "jaw",
    "chin_projection": "jaw",
    "submental_length": "jaw",
    "face_height_sellion_menton": "proportion",
    "upper_face_height": "proportion",
    "middle_third_height": "proportion",
    "lower_third_height": "proportion",
    "facial_thirds_ratio": "proportion",
    "nose_mouth_width_ratio": "proportion",
    "eye_spacing_ratio": "proportion",
    "facial_width_height_ratio": "proportion",
    "ocular_height_asymmetry": "symmetry",
    "mouth_corner_asymmetry": "symmetry",
    "canthal_tilt_asymmetry": "symmetry",
    "facial_convexity_angle": "profile",
    "mentocervical_angle": "profile",
    "submental_cervical_angle": "profile",
    "brow_apex_lateral_offset_l": "periocular",
    "brow_apex_lateral_offset_r": "periocular",
    "margin_reflex_distance_1_l": "periocular",
    "margin_reflex_distance_1_r": "periocular",
    "margin_reflex_distance_2_l": "periocular",
    "margin_reflex_distance_2_r": "periocular",
    "medial_canthal_angle_l": "periocular",
    "medial_canthal_angle_r": "periocular",
    "upper_vermilion_height": "mouth",
    "lower_vermilion_height": "mouth",
    "cupids_bow_peak_height_l": "mouth",
    "cupids_bow_peak_height_r": "mouth",
    "commissure_height_l": "mouth",
    "commissure_height_r": "mouth",
    "upper_lip_projection": "mouth",
    "labiomental_sulcus_depth": "mouth",
    "nasal_dorsal_deviation": "nose",
    "alar_base_intercanthal_ratio": "nose",
    "nasal_tip_rotation": "nose",
    "chin_height": "jaw",
    "ramus_body_ratio_l": "jaw",
    "ramus_body_ratio_r": "jaw",
    "midface_projection": "proportion",
}


def region_of(measurement_id: str) -> Region:
    return BY_KEY[REGION_OF.get(measurement_id, "other")]


@dataclass(frozen=True)
class NormativeStratum:
    """The reference sample one measurement is placed against.

    Every field here exists so the report can name the sample instead of
    implying a general population. A stratum with an ``n`` of 589 respirator
    users is a different claim from "the population", and the difference is the
    reader's to judge, not the renderer's to hide.
    """

    measurement_id: str
    #: Human phrase for the stratum, e.g. "589 Black female respirator users".
    label: str
    n: int
    mean: float
    sd: float
    unit: str
    method: str
    source: str
    license: str = ""
    caveat: str = ""


_ANCESTRY_WORDS = {"pooled": "", "Other": "other-ancestry"}


def _stratum_label(key: str, n: int) -> str:
    sex, _, ancestry = key.partition("|")
    words = []
    if ancestry not in ("pooled", ""):
        words.append(_ANCESTRY_WORDS.get(ancestry, ancestry))
    if sex in ("male", "female"):
        words.append(sex)
    who = " ".join(w for w in words if w)
    who = f"{who} respirator users" if who else "respirator users of both sexes"
    tail = "" if ancestry not in ("pooled", "") else ", pooled over ancestry"
    return f"{n:,} {who}{tail}"


def niosh_stratum(
    measurement_id: str, *, sex: str | None = None, ancestry: str | None = None
) -> NormativeStratum | None:
    """The narrowest NIOSH stratum for a subject who declared these attributes.

    ``sex`` and ``ancestry`` are whatever the subject said they are. Vitruve
    does not infer either one, so an undeclared subject gets the pooled cell
    and the report says so.
    """
    cell = niosh.stratum(measurement_id, sex=sex, ancestry=ancestry)
    if cell is None:
        return None
    meta = niosh.metadata()
    return NormativeStratum(
        measurement_id=measurement_id,
        label=_stratum_label(cell["stratum"], int(cell["n"])),
        n=int(cell["n"]),
        mean=float(cell["mean"]),
        sd=float(cell["sd"]),
        unit="mm" if not measurement_id.endswith("ratio") else "",
        method=str(meta.get("method", "")),
        source=str(meta.get("source", "")),
        license=str(meta.get("license", "")),
        caveat=str(meta.get("caveat", "")),
    )


@dataclass(frozen=True)
class QualityIssue:
    """One thing about the photograph that bears on the measurements."""

    code: str
    detail: str
    #: ``note``, ``caveat`` or ``blocking``.
    severity: str = "note"
    #: The number behind the issue, already formatted, e.g. ``yaw 7.2 deg``.
    reading: str = ""


@dataclass(frozen=True)
class OverlayImage:
    """A rendered annotation, carried as PNG bytes so the report is one file."""

    region: str
    title: str
    caption: str
    png: bytes

    @property
    def data_uri(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")


@dataclass(frozen=True)
class MeasurementGroup:
    """The measurements of one region, with the overlay that illustrates them."""

    region: Region
    measurements: tuple[Measured, ...]
    unavailable: tuple[Unavailable, ...]
    overlay: OverlayImage | None = None

    @property
    def n_shown(self) -> int:
        return sum(1 for m in self.measurements if m.shown)


@dataclass(frozen=True)
class ReportInput:
    """Everything the renderer needs, and nothing it does not.

    The pipeline assembles this; the tests build it by hand. Keeping the two
    paths identical is the point, because a renderer that can only be exercised
    through a torch model is a renderer nobody exercises.
    """

    measurements: tuple[Measured, ...] = ()
    unavailable: tuple[Unavailable, ...] = ()
    quality: tuple[QualityIssue, ...] = ()
    manifest: Mapping[str, object] = field(default_factory=dict)
    overlays: tuple[OverlayImage, ...] = ()
    #: Reference sample per measurement id, where one exists.
    strata: Mapping[str, NormativeStratum] = field(default_factory=dict)
    #: License obligations of the backend tier that actually ran, verbatim.
    obligations: tuple[str, ...] = ()
    #: Literature behind the measurement definitions, deduplicated.
    references: tuple[str, ...] = ()
    subject_label: str = "Unidentified subject"
    declared_sex: str | None = None
    declared_ancestry: str | None = None
    generated_at: str = ""
    #: Specs the pipeline set out to evaluate. Defaults to what came back,
    #: which is right when nothing was dropped before evaluation.
    n_attempted_override: int | None = None

    @property
    def n_attempted(self) -> int:
        if self.n_attempted_override is not None:
            return self.n_attempted_override
        return len(self.measurements) + len(self.unavailable)

    @property
    def n_reported(self) -> int:
        return sum(
            1
            for m in self.measurements
            if m.verdict.reportability is Reportability.REPORT
        )

    @property
    def n_caveated(self) -> int:
        return sum(
            1
            for m in self.measurements
            if m.verdict.reportability is Reportability.CAVEAT
        )

    @property
    def n_withheld(self) -> int:
        return sum(
            1
            for m in self.measurements
            if m.verdict.reportability is Reportability.WITHHOLD
        )

    @property
    def n_unavailable(self) -> int:
        return len(self.unavailable)

    @property
    def n_shown(self) -> int:
        """Reportable measurements, caveated ones included.

        A caveated measurement is printed with its caveat, so it counts as
        shown. The summary prints the two figures separately as well.
        """
        return self.n_reported + self.n_caveated

    @property
    def shown(self) -> tuple[Measured, ...]:
        return tuple(m for m in self.measurements if m.shown)

    @property
    def withheld(self) -> tuple[Measured, ...]:
        return tuple(m for m in self.measurements if not m.shown)

    def overlay_for(self, region_key: str) -> OverlayImage | None:
        for o in self.overlays:
            if o.region == region_key:
                return o
        return None

    def groups(self) -> tuple[MeasurementGroup, ...]:
        """Measurements by region, in the order of :data:`REGIONS`.

        Empty regions are dropped. Withheld measurements keep their place in
        their region rather than being swept into an appendix, so a reader
        scanning the jaw section sees the refusals next to the numbers.
        """
        out: list[MeasurementGroup] = []
        for region in REGIONS:
            ms = tuple(m for m in self.measurements if region_of(m.spec_id) is region)
            us = tuple(u for u in self.unavailable if region_of(u.spec_id) is region)
            if not ms and not us:
                continue
            out.append(
                MeasurementGroup(
                    region=region,
                    measurements=ms,
                    unavailable=us,
                    overlay=self.overlay_for(region.key),
                )
            )
        return tuple(out)
