"""The measurement catalogue.

Everything Vitruve can measure is declared here, once, as data. Adding a metric
means adding a `MeasurementSpec`; it never means touching the pipeline, the
report, or the uncertainty machinery.

The catalogue is organised by view and tagged by evidence tier. Read the tiers
before reading the formulas -- they are the honest part. A little under half of
this catalogue is `CONVENTIONAL`, meaning the facial-aesthetics literature uses
the measurement and publishes a reference range for it, but nobody has ever
shown that it can be recovered from a photograph in agreement with a caliper.
Those numbers still get computed. They are labelled.
"""

from __future__ import annotations

from ..core.formula import (
    Abs,
    AngleAt,
    AngleBetween,
    Axis,
    Diff,
    Dist,
    LineOffset,
    Mid,
    ProjLength,
    Ratio,
    SignedTilt,
    Pt,
    Vec,
)
from ..core.landmarks import Landmark as L
from ..core.spec import Evidence, MeasurementSpec, Unit, View

# Literature shorthands, expanded in docs/references.md
FARKAS = "Farkas 1994, Anthropometry of the Head and Face, 2nd ed."
LIM2022 = "Lim et al. 2022, Front Public Health 9:813058 (2D photogrammetry vs direct, n=96)"
RICKETTS = "Ricketts 1960, Am J Orthod 46:330 (E-line)"
POWELL = "Powell & Humphreys 1984, Proportions of the Aesthetic Face"
NAINI = "Naini 2011, Facial Aesthetics: Concepts and Clinical Diagnosis"
WEN2015 = "Wen et al. 2015, photogrammetric meta-analysis of facial norms by ancestry"
DODGSON = "Dodgson 2004, SPIE 5291 (ANSUR interpupillary distance, n=3976)"


def _spec(**kw) -> MeasurementSpec:
    return MeasurementSpec(**kw)


# ---------------------------------------------------------------------------
# Frontal: transverse lengths
# ---------------------------------------------------------------------------

_FRONTAL_LENGTHS = [
    _spec(
        id="interpupillary_distance",
        label="Interpupillary distance",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.VALIDATED_2D,
        formula=Dist(Pt(L.PUPIL_L), Pt(L.PUPIL_R)),
        description="Distance between pupil centres.",
        references=(LIM2022, DODGSON),
        reference_range=(52.0, 78.0, DODGSON),
        pose_tolerance_deg=8.0,
    ),
    _spec(
        id="intercanthal_width",
        label="Intercanthal width (en-en)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.VALIDATED_2D,
        formula=Dist(Pt(L.ENDOCANTHION_L), Pt(L.ENDOCANTHION_R)),
        description="Distance between the inner eye corners.",
        references=(FARKAS, WEN2015),
        pose_tolerance_deg=8.0,
    ),
    _spec(
        id="biocular_width",
        label="Biocular width (ex-ex)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.VALIDATED_2D,
        formula=Dist(Pt(L.EXOCANTHION_L), Pt(L.EXOCANTHION_R)),
        description="Distance between the outer eye corners.",
        references=(FARKAS, WEN2015),
        pose_tolerance_deg=8.0,
    ),
    _spec(
        id="nose_breadth",
        label="Nose breadth (al-al)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.VALIDATED_2D,
        formula=Dist(Pt(L.ALARE_L), Pt(L.ALARE_R)),
        description="Widest points of the nasal alae.",
        references=(LIM2022, FARKAS),
        pose_tolerance_deg=8.0,
    ),
    _spec(
        id="mouth_width",
        label="Mouth width (ch-ch)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.VALIDATED_2D,
        formula=Dist(Pt(L.CHEILION_L), Pt(L.CHEILION_R)),
        description="Distance between the mouth corners.",
        references=(FARKAS,),
        pose_tolerance_deg=8.0,
    ),
    _spec(
        id="philtrum_width",
        label="Philtrum width",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.CONVENTIONAL,
        formula=Dist(Pt(L.CRISTA_PHILTRI_L), Pt(L.CRISTA_PHILTRI_R)),
        references=(FARKAS,),
        pose_tolerance_deg=8.0,
    ),
    _spec(
        id="bizygomatic_width",
        label="Bizygomatic width (zy-zy)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.REQUIRES_3D,
        formula=Dist(Pt(L.ZYGION_L), Pt(L.ZYGION_R)),
        description=(
            "Widest points of the cheekbones. The endpoints sit on a laterally "
            "curved, self-occluding surface, so in a 2D photograph they are a "
            "silhouette artifact rather than the anatomical point."
        ),
        references=(LIM2022, FARKAS),
        reference_range=(120.0, 150.0, FARKAS),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="bigonial_width",
        label="Bigonial width (go-go)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.REQUIRES_3D,
        formula=Dist(Pt(L.GONION_L), Pt(L.GONION_R)),
        description=(
            "Jaw width at the mandibular angles. The worst-performing "
            "measurement in published 2D photogrammetry: mean difference from "
            "direct measurement 9.3 mm, limits of agreement -0.9 to 19.6 mm."
        ),
        references=(LIM2022, FARKAS),
        pose_tolerance_deg=5.0,
    ),
]

# ---------------------------------------------------------------------------
# Frontal: vertical lengths
# ---------------------------------------------------------------------------

_FRONTAL_VERTICALS = [
    _spec(
        id="nose_height",
        label="Nose height (se-sn)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.VALIDATED_2D,
        formula=Dist(Pt(L.SELLION), Pt(L.SUBNASALE)),
        description="Sellion to subnasale; agrees with direct measurement to ~0.3 mm.",
        references=(LIM2022, FARKAS),
        pose_tolerance_deg=6.0,
    ),
    _spec(
        id="face_height_sellion_menton",
        label="Face height (se-me)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.VALIDATED_2D,
        formula=Dist(Pt(L.SELLION), Pt(L.MENTON)),
        references=(LIM2022, FARKAS),
        pose_tolerance_deg=6.0,
    ),
    _spec(
        id="philtrum_length",
        label="Philtrum length (sn-ls)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.CONVENTIONAL,
        formula=Dist(Pt(L.SUBNASALE), Pt(L.LABIALE_SUPERIUS)),
        references=(FARKAS, NAINI),
        pose_tolerance_deg=6.0,
    ),
    _spec(
        id="upper_face_height",
        label="Upper face height (glabella to labiale superius)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.CONVENTIONAL,
        formula=Dist(Pt(L.GLABELLA), Pt(L.LABIALE_SUPERIUS)),
        description="Denominator of the facial width-to-height ratio.",
        references=(FARKAS,),
        pose_tolerance_deg=6.0,
    ),
    _spec(
        id="middle_third_height",
        label="Middle facial third (glabella to subnasale)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.CONVENTIONAL,
        formula=Dist(Pt(L.GLABELLA), Pt(L.SUBNASALE)),
        references=(FARKAS, NAINI),
        pose_tolerance_deg=6.0,
    ),
    _spec(
        id="lower_third_height",
        label="Lower facial third (subnasale to menton)",
        view=View.FRONTAL,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.CONVENTIONAL,
        formula=Dist(Pt(L.SUBNASALE), Pt(L.MENTON)),
        references=(FARKAS, NAINI),
        pose_tolerance_deg=6.0,
    ),
]

# ---------------------------------------------------------------------------
# Frontal: periocular
# ---------------------------------------------------------------------------


def _periocular(
    side: str, en: L, ex: L, sup: L, inf: L, near_pupil: L, far_pupil: L
) -> list[MeasurementSpec]:
    """The four periocular measurements for one side.

    The tilt axis runs from the contralateral pupil to the ipsilateral one, so
    it points away from the midline on this side. That gives both eyes a
    consistent sign, positive when the outer corner sits above the inner, and
    it makes the measurement roll-invariant: the axis rotates with the head, so
    a tilted camera cancels out of the signed angle rather than adding to it.
    """
    up = side.upper()
    return [
        _spec(
            id=f"palpebral_fissure_width_{side}",
            label=f"Palpebral fissure width ({up})",
            view=View.FRONTAL,
            unit=Unit.MILLIMETRES,
            evidence=Evidence.CONVENTIONAL,
            formula=Dist(Pt(en), Pt(ex)),
            references=(FARKAS,),
            pose_tolerance_deg=6.0,
        ),
        _spec(
            id=f"palpebral_fissure_height_{side}",
            label=f"Palpebral fissure height ({up})",
            view=View.FRONTAL,
            unit=Unit.MILLIMETRES,
            evidence=Evidence.CONVENTIONAL,
            formula=Dist(Pt(sup), Pt(inf)),
            references=(FARKAS,),
            pose_tolerance_deg=6.0,
        ),
        _spec(
            id=f"canthal_tilt_{side}",
            label=f"Canthal tilt ({up})",
            view=View.FRONTAL,
            unit=Unit.DEGREES,
            evidence=Evidence.POSE_CRITICAL,
            formula=SignedTilt(Pt(en), Pt(ex), Vec(Pt(far_pupil), Pt(near_pupil))),
            description=(
                "Inclination of the intercanthal axis, positive when the outer "
                "corner sits above the inner, measured against the "
                "interpupillary line rather than against the image horizon. "
                "The interpupillary line rotates with the head, so image roll "
                "cancels exactly instead of entering one-for-one. Measuring "
                "against the horizon reports the photographer's camera angle as "
                "if it were the subject's anatomy; pitch still enters at about "
                "0.27 degrees per degree (Vaca et al. 2022)."
            ),
            references=(NAINI,),
            reference_range=(0.0, 8.0, NAINI),
            pose_tolerance_deg=3.0,
        ),
        _spec(
            id=f"eye_aspect_ratio_{side}",
            label=f"Eye aspect ratio ({up})",
            view=View.FRONTAL,
            unit=Unit.RATIO,
            evidence=Evidence.POSE_INVARIANT_RATIO,
            formula=Ratio(Dist(Pt(sup), Pt(inf)), Dist(Pt(en), Pt(ex))),
            description="Fissure height over width; both terms lie in the same plane.",
            references=(FARKAS,),
            pose_tolerance_deg=8.0,
        ),
    ]


_PERIOCULAR: list[MeasurementSpec] = [
    spec
    for args in (
        ("l", L.ENDOCANTHION_L, L.EXOCANTHION_L, L.PALPEBRALE_SUP_L,
         L.PALPEBRALE_INF_L, L.PUPIL_L, L.PUPIL_R),
        ("r", L.ENDOCANTHION_R, L.EXOCANTHION_R, L.PALPEBRALE_SUP_R,
         L.PALPEBRALE_INF_R, L.PUPIL_R, L.PUPIL_L),
    )
    for spec in _periocular(*args)
]

# ---------------------------------------------------------------------------
# Frontal: ratios. The most trustworthy outputs -- scale cancels, and to first
# order so does pose when both terms lie in the same plane.
# ---------------------------------------------------------------------------

_FRONTAL_RATIOS = [
    _spec(
        id="intercanthal_biocular_ratio",
        label="Intercanthal : biocular width",
        view=View.FRONTAL,
        unit=Unit.RATIO,
        evidence=Evidence.POSE_INVARIANT_RATIO,
        formula=Ratio(
            Dist(Pt(L.ENDOCANTHION_L), Pt(L.ENDOCANTHION_R)),
            Dist(Pt(L.EXOCANTHION_L), Pt(L.EXOCANTHION_R)),
        ),
        description="Both terms are transverse, so yaw scales them equally and cancels.",
        references=(FARKAS,),
        pose_tolerance_deg=12.0,
    ),
    _spec(
        id="nose_mouth_width_ratio",
        label="Nose breadth : mouth width",
        view=View.FRONTAL,
        unit=Unit.RATIO,
        evidence=Evidence.POSE_INVARIANT_RATIO,
        formula=Ratio(
            Dist(Pt(L.ALARE_L), Pt(L.ALARE_R)), Dist(Pt(L.CHEILION_L), Pt(L.CHEILION_R))
        ),
        references=(FARKAS,),
        pose_tolerance_deg=12.0,
    ),
    _spec(
        id="eye_spacing_ratio",
        label="Intercanthal width : nose breadth",
        view=View.FRONTAL,
        unit=Unit.RATIO,
        evidence=Evidence.POSE_INVARIANT_RATIO,
        formula=Ratio(
            Dist(Pt(L.ENDOCANTHION_L), Pt(L.ENDOCANTHION_R)),
            Dist(Pt(L.ALARE_L), Pt(L.ALARE_R)),
        ),
        description="The classical 'alae align with the inner canthi' check, as a number.",
        references=(FARKAS, POWELL),
        pose_tolerance_deg=12.0,
    ),
    _spec(
        id="facial_thirds_ratio",
        label="Middle : lower facial third",
        view=View.FRONTAL,
        unit=Unit.RATIO,
        evidence=Evidence.CONVENTIONAL,
        formula=Ratio(
            Dist(Pt(L.GLABELLA), Pt(L.SUBNASALE)), Dist(Pt(L.SUBNASALE), Pt(L.MENTON))
        ),
        description=(
            "Both terms are vertical, so pitch cancels but yaw does not affect "
            "either. The upper third needs trichion, which no landmark model "
            "supplies, so only two of the three thirds are computed."
        ),
        references=(FARKAS, NAINI),
        pose_tolerance_deg=10.0,
    ),
    _spec(
        id="lip_vermilion_ratio",
        label="Upper : lower vermilion height",
        view=View.FRONTAL,
        unit=Unit.RATIO,
        evidence=Evidence.CONVENTIONAL,
        formula=Ratio(
            Dist(Pt(L.LABIALE_SUPERIUS), Pt(L.STOMION)),
            Dist(Pt(L.STOMION), Pt(L.LABIALE_INFERIUS)),
        ),
        references=(NAINI,),
        reference_range=(0.4, 0.7, NAINI),
        pose_tolerance_deg=10.0,
    ),
    _spec(
        id="facial_width_height_ratio",
        label="Facial width-to-height ratio (fWHR)",
        view=View.FRONTAL,
        unit=Unit.RATIO,
        evidence=Evidence.REQUIRES_3D,
        formula=Ratio(
            Dist(Pt(L.ZYGION_L), Pt(L.ZYGION_R)),
            Dist(Pt(L.GLABELLA), Pt(L.LABIALE_SUPERIUS)),
        ),
        description=(
            "Bizygomatic width over upper face height. Widely reported and "
            "widely wrong: the numerator is one of the two measurements that 2D "
            "photogrammetry demonstrably fails to reproduce."
        ),
        references=(LIM2022,),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="jaw_cheekbone_ratio",
        label="Bigonial : bizygomatic width",
        view=View.FRONTAL,
        unit=Unit.RATIO,
        evidence=Evidence.REQUIRES_3D,
        formula=Ratio(
            Dist(Pt(L.GONION_L), Pt(L.GONION_R)), Dist(Pt(L.ZYGION_L), Pt(L.ZYGION_R))
        ),
        description=(
            "The taper of the lower face. Both terms are transverse so the ratio "
            "is pose-robust, but both endpoints are self-occluding, so the "
            "problem is landmark localisation rather than projection."
        ),
        references=(LIM2022, FARKAS),
        pose_tolerance_deg=8.0,
    ),
]

# ---------------------------------------------------------------------------
# Frontal: symmetry. Reported as an asymmetry magnitude per paired feature,
# never pooled into a single "symmetry score".
# ---------------------------------------------------------------------------

#: The interpupillary line, which rotates with the head. Every asymmetry below
#: is measured against it rather than against the image horizon.
#:
#: The earlier version measured against world vertical and claimed that a
#: common rotation "adds the same offset to both sides and cancels in the
#: difference". That was the opposite of true. Because each side was measured
#: against its own lateral axis, roll entered the two sides with opposite sign
#: and therefore *added* in the difference: on a perfectly symmetric synthetic
#: face, two degrees of roll manufactured four degrees of canthal tilt
#: asymmetry. The pose sweep in evals/ measured the roll slope at 2.0 per
#: degree against a declared 0.002, a thousandfold error in the direction that
#: makes a photograph look like a finding.
_IPD_AXIS = Vec(Pt(L.PUPIL_R), Pt(L.PUPIL_L))

_SYMMETRY = [
    _spec(
        id="ocular_height_asymmetry",
        label="Ocular height asymmetry",
        view=View.FRONTAL,
        unit=Unit.DEGREES,
        evidence=Evidence.POSE_INVARIANT_RATIO,
        formula=Abs(SignedTilt(Pt(L.EXOCANTHION_R), Pt(L.EXOCANTHION_L), _IPD_AXIS)),
        description=(
            "Inclination of the line joining the outer eye corners, measured "
            "against the interpupillary line. Both rotate with the head, so "
            "image roll cancels exactly."
        ),
        references=(FARKAS,),
        pose_tolerance_deg=10.0,
    ),
    _spec(
        id="mouth_corner_asymmetry",
        label="Mouth corner asymmetry",
        view=View.FRONTAL,
        unit=Unit.DEGREES,
        evidence=Evidence.POSE_INVARIANT_RATIO,
        formula=Abs(SignedTilt(Pt(L.CHEILION_R), Pt(L.CHEILION_L), _IPD_AXIS)),
        description=(
            "Inclination of the mouth axis against the interpupillary line."
        ),
        references=(FARKAS,),
        pose_tolerance_deg=10.0,
    ),
    _spec(
        id="canthal_tilt_asymmetry",
        label="Canthal tilt asymmetry",
        view=View.FRONTAL,
        unit=Unit.DEGREES,
        evidence=Evidence.POSE_INVARIANT_RATIO,
        formula=Abs(
            Diff(
                SignedTilt(Pt(L.ENDOCANTHION_L), Pt(L.EXOCANTHION_L),
                           Vec(Pt(L.PUPIL_R), Pt(L.PUPIL_L))),
                SignedTilt(Pt(L.ENDOCANTHION_R), Pt(L.EXOCANTHION_R),
                           Vec(Pt(L.PUPIL_L), Pt(L.PUPIL_R))),
            )
        ),
        description=(
            "The difference between the two canthal tilts. Now genuinely "
            "roll-invariant, because each tilt is measured against an axis that "
            "rotates with the head."
        ),
        references=(NAINI,),
        pose_tolerance_deg=10.0,
    ),
]

# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

_PROFILE = [
    _spec(
        id="nasofrontal_angle",
        label="Nasofrontal angle",
        view=View.PROFILE,
        unit=Unit.DEGREES,
        evidence=Evidence.CONVENTIONAL,
        formula=AngleAt(Pt(L.GLABELLA), Pt(L.NASION), Pt(L.PRONASALE)),
        references=(POWELL, NAINI),
        reference_range=(115.0, 135.0, POWELL),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="nasolabial_angle",
        label="Nasolabial angle",
        view=View.PROFILE,
        unit=Unit.DEGREES,
        evidence=Evidence.CONVENTIONAL,
        formula=AngleAt(Pt(L.COLUMELLA), Pt(L.SUBNASALE), Pt(L.LABIALE_SUPERIUS)),
        references=(POWELL, NAINI),
        reference_range=(90.0, 120.0, POWELL),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="facial_convexity_angle",
        label="Facial convexity angle",
        view=View.PROFILE,
        unit=Unit.DEGREES,
        evidence=Evidence.CONVENTIONAL,
        formula=AngleAt(Pt(L.GLABELLA), Pt(L.SUBNASALE), Pt(L.POGONION)),
        references=(NAINI,),
        reference_range=(165.0, 175.0, NAINI),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="mentolabial_angle",
        label="Mentolabial angle",
        view=View.PROFILE,
        unit=Unit.DEGREES,
        evidence=Evidence.CONVENTIONAL,
        formula=AngleAt(Pt(L.LABIALE_INFERIUS), Pt(L.SUBLABIALE), Pt(L.POGONION)),
        references=(NAINI,),
        reference_range=(110.0, 130.0, NAINI),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="mentocervical_angle",
        label="Mentocervical angle",
        view=View.PROFILE,
        unit=Unit.DEGREES,
        evidence=Evidence.CONVENTIONAL,
        formula=AngleBetween(
            Vec(Pt(L.GLABELLA), Pt(L.POGONION)), Vec(Pt(L.MENTON), Pt(L.CERVICALE))
        ),
        references=(POWELL, NAINI),
        reference_range=(80.0, 95.0, POWELL),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="submental_cervical_angle",
        label="Submental-cervical angle",
        view=View.PROFILE,
        unit=Unit.DEGREES,
        evidence=Evidence.CONVENTIONAL,
        formula=AngleAt(Pt(L.POGONION), Pt(L.MENTON), Pt(L.CERVICALE)),
        references=(POWELL,),
        reference_range=(90.0, 120.0, POWELL),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="gonial_angle_l",
        label="Gonial angle (L)",
        view=View.PROFILE,
        unit=Unit.DEGREES,
        evidence=Evidence.POSE_CRITICAL,
        formula=AngleAt(Pt(L.TRAGION_L), Pt(L.GONION_L), Pt(L.GNATHION)),
        description=(
            "Ramus-to-body angle of the mandible, approximated from soft tissue. "
            "A projection artifact of about 3 degrees appears under 20 degrees of "
            "mandibular yaw, and gonion is a self-occluding landmark."
        ),
        references=(NAINI,),
        reference_range=(100.0, 140.0, NAINI),
        pose_tolerance_deg=4.0,
    ),
    _spec(
        id="gonial_angle_r",
        label="Gonial angle (R)",
        view=View.PROFILE,
        unit=Unit.DEGREES,
        evidence=Evidence.POSE_CRITICAL,
        formula=AngleAt(Pt(L.TRAGION_R), Pt(L.GONION_R), Pt(L.GNATHION)),
        references=(NAINI,),
        reference_range=(100.0, 140.0, NAINI),
        pose_tolerance_deg=4.0,
    ),
    _spec(
        id="e_line_upper_lip",
        label="Upper lip to E-line",
        view=View.PROFILE,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.CONVENTIONAL,
        formula=LineOffset(
            Pt(L.LABIALE_SUPERIUS), Pt(L.PRONASALE), Pt(L.POGONION), Axis("z")
        ),
        description="Signed offset from the Ricketts line; negative means behind it.",
        references=(RICKETTS,),
        reference_range=(-6.0, -2.0, RICKETTS),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="e_line_lower_lip",
        label="Lower lip to E-line",
        view=View.PROFILE,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.CONVENTIONAL,
        formula=LineOffset(
            Pt(L.LABIALE_INFERIUS), Pt(L.PRONASALE), Pt(L.POGONION), Axis("z")
        ),
        references=(RICKETTS,),
        reference_range=(-4.0, 0.0, RICKETTS),
        pose_tolerance_deg=5.0,
    ),
    _spec(
        id="nasal_tip_projection_ratio",
        label="Nasal tip projection (Goode ratio)",
        view=View.PROFILE,
        unit=Unit.RATIO,
        evidence=Evidence.CONVENTIONAL,
        formula=Ratio(
            Abs(ProjLength(Pt(L.ALARE_L), Pt(L.PRONASALE), Axis("z"))),
            Dist(Pt(L.NASION), Pt(L.PRONASALE)),
        ),
        references=(POWELL,),
        reference_range=(0.55, 0.60, POWELL),
        pose_tolerance_deg=6.0,
    ),
    _spec(
        id="submental_length",
        label="Submental length",
        view=View.PROFILE,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.CONVENTIONAL,
        formula=Dist(Pt(L.MENTON), Pt(L.CERVICALE)),
        references=(NAINI,),
        reference_range=(50.0, 75.0, NAINI),
        pose_tolerance_deg=6.0,
    ),
    _spec(
        id="chin_projection",
        label="Chin projection past the subnasale vertical",
        view=View.PROFILE,
        unit=Unit.MILLIMETRES,
        evidence=Evidence.CONVENTIONAL,
        formula=ProjLength(Pt(L.SUBNASALE), Pt(L.POGONION), Axis("z")),
        references=(NAINI,),
        pose_tolerance_deg=5.0,
    ),
]


CATALOGUE: tuple[MeasurementSpec, ...] = tuple(
    _FRONTAL_LENGTHS + _FRONTAL_VERTICALS + _PERIOCULAR + _FRONTAL_RATIOS + _SYMMETRY + _PROFILE
)

BY_ID: dict[str, MeasurementSpec] = {s.id: s for s in CATALOGUE}
if len(BY_ID) != len(CATALOGUE):
    raise RuntimeError("duplicate measurement ids in the catalogue")


def for_view(view: View) -> tuple[MeasurementSpec, ...]:
    return tuple(s for s in CATALOGUE if s.view is view or s.view is View.EITHER)


def satisfiable(available: frozenset[L]) -> tuple[MeasurementSpec, ...]:
    """Specs every landmark of which the backend can supply."""
    return tuple(s for s in CATALOGUE if s.landmarks <= available)


# ---------------------------------------------------------------------------
# Enrichment: attach pose sensitivity and published between-subject spread.
#
# Kept as a table rather than as arguments on each spec so that the two things
# a reader most wants to compare -- how much a measurement moves with pose, and
# how much it varies between people -- sit side by side.
# ---------------------------------------------------------------------------

from ..core import sensitivity as sens  # noqa: E402
from ..norms.published import DEFAULT_LINEAR_RSD, representative_spread  # noqa: E402

_SENSITIVITY: dict[str, "sens.PoseSensitivity"] = {
    # Transverse widths carry the full cos(yaw).
    "interpupillary_distance": sens.TRANSVERSE_WIDTH,
    "intercanthal_width": sens.TRANSVERSE_WIDTH,
    "biocular_width": sens.TRANSVERSE_WIDTH,
    "nose_breadth": sens.TRANSVERSE_WIDTH,
    "mouth_width": sens.TRANSVERSE_WIDTH,
    "philtrum_width": sens.TRANSVERSE_WIDTH,
    "bizygomatic_width": sens.KLEINBERG_WORST,
    "bigonial_width": sens.KLEINBERG_WORST,
    "palpebral_fissure_width_l": sens.TRANSVERSE_WIDTH,
    "palpebral_fissure_width_r": sens.TRANSVERSE_WIDTH,
    # Verticals carry cos(pitch).
    "nose_height": sens.VERTICAL_DISTANCE,
    "face_height_sellion_menton": sens.VERTICAL_DISTANCE,
    "philtrum_length": sens.VERTICAL_DISTANCE,
    "upper_face_height": sens.VERTICAL_DISTANCE,
    "middle_third_height": sens.VERTICAL_DISTANCE,
    "lower_third_height": sens.VERTICAL_DISTANCE,
    "palpebral_fissure_height_l": sens.VERTICAL_DISTANCE,
    "palpebral_fissure_height_r": sens.VERTICAL_DISTANCE,
    # Same-plane ratios: the cosine cancels.
    "intercanthal_biocular_ratio": sens.TRANSVERSE_RATIO,
    "nose_mouth_width_ratio": sens.TRANSVERSE_RATIO,
    "eye_spacing_ratio": sens.TRANSVERSE_RATIO,
    "eye_aspect_ratio_l": sens.TRANSVERSE_RATIO,
    "eye_aspect_ratio_r": sens.TRANSVERSE_RATIO,
    "facial_thirds_ratio": sens.TRANSVERSE_RATIO,
    "lip_vermilion_ratio": sens.TRANSVERSE_RATIO,
    "jaw_cheekbone_ratio": sens.TRANSVERSE_RATIO,
    # Width over height: cosine on the numerator only, so nothing cancels.
    "facial_width_height_ratio": sens.WIDTH_OVER_HEIGHT,
    # Angles with measured sensitivities.
    "canthal_tilt_l": sens.CANTHAL_TILT,
    "canthal_tilt_r": sens.CANTHAL_TILT,
    "gonial_angle_l": sens.GONIAL_ANGLE,
    "gonial_angle_r": sens.GONIAL_ANGLE,
}

#: Differences between paired features. Roll and pitch add the same offset to
#: both sides, so the difference cancels them -- these are the most pose-robust
#: quantities the pipeline produces, and the only ones that survive a handheld
#: photograph.
_CANCELLING = ("canthal_tilt_asymmetry", "ocular_height_asymmetry", "mouth_corner_asymmetry")

#: Within-person, between-photograph spread where somebody has measured it.
#: These override the derived pose model. There are only two entries because
#: only two measurements in this catalogue have ever been through a variance
#: decomposition -- which is itself the finding.
_MEASURED_WITHIN_PERSON: dict[str, tuple[float, str]] = {
    "facial_width_height_ratio": (
        0.12,
        "Kramer 2016 (PeerJ 4:e1801) decomposed fWHR variance and found posed "
        "expression accounting for more of it than identity did (eta-squared "
        "0.58 against 0.31), and the rank ordering of individuals changing with "
        "which photograph was used; Kramer et al. 2012 measured the same 66 men "
        "at 2.01 from photographs and 1.83 from 3D scans",
    ),
    "jaw_cheekbone_ratio": (
        0.08,
        "inherited from the bigonial and bizygomatic agreement failures in Lim "
        "et al. 2022; no direct within-person study exists for this ratio",
    ),
}


def _sensitivity_for(spec: MeasurementSpec) -> "sens.PoseSensitivity":
    if spec.id in _SENSITIVITY:
        return _SENSITIVITY[spec.id]
    if spec.id in _CANCELLING:
        return sens.PoseSensitivity(
            yaw=0.01,
            pitch=0.01,
            roll=0.01,
            source="measured against the interpupillary axis, so roll cancels exactly; "
            "residual is landmark noise in the pupils (evals/ pose sweep)",
        )
    if spec.unit is Unit.DEGREES:
        return sens.PoseSensitivity(
            yaw=0.15, pitch=0.15, roll=0.05, source="unmeasured angle, conservative default"
        )
    return sens.KLEINBERG_WORST


def _rsd_for(spec: MeasurementSpec) -> float | None:
    published = representative_spread(spec.id, spec.unit.value)
    if published is not None:
        return published
    if spec.unit is Unit.MILLIMETRES:
        return DEFAULT_LINEAR_RSD
    # Angles and ratios without a published spread stay None on purpose. The
    # report says "unknown whether this distinguishes individuals", which is a
    # different and more honest statement than a guessed number.
    return None


def _enriched(spec: MeasurementSpec) -> MeasurementSpec:
    return MeasurementSpec(
        id=spec.id,
        label=spec.label,
        view=spec.view,
        unit=spec.unit,
        evidence=spec.evidence,
        formula=spec.formula,
        description=spec.description,
        references=spec.references,
        reference_range=spec.reference_range,
        normalised_by=spec.normalised_by,
        pose_tolerance_deg=spec.pose_tolerance_deg,
        sensitivity=_sensitivity_for(spec),
        between_subject_rsd=_rsd_for(spec),
        measured_within_person_rsd=_MEASURED_WITHIN_PERSON.get(spec.id, (None, ""))[0],
        within_person_source=_MEASURED_WITHIN_PERSON.get(spec.id, (None, ""))[1],
    )


CATALOGUE = tuple(_enriched(s) for s in CATALOGUE)
BY_ID = {s.id: s for s in CATALOGUE}
