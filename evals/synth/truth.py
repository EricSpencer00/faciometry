"""Ground truth for the synthetic face, computed a second time and differently.

Nothing in this file imports ``faciometry.core.geometry``, ``faciometry.core.formula``
or ``faciometry.measure.registry``. Every value is recomputed in pure Python
``math`` from the *anatomical definition* of the measurement -- the one written
in the spec's label and description -- rather than from the expression tree the
registry happens to use. That is what makes arm 1 a check rather than a
tautology: agreement means two independent readings of the same definition
landed on the same number.

Where a value is fixed by how the face was built (a width between two points
placed at the same height and depth is a coordinate difference), it is marked
``exact=True`` and the expected value is the construction parameter itself. A
disagreement there is unambiguous.

Three measurements are returned with more than one candidate truth, because
the literature contains more than one convention and the registry picks one:
see ``E_LINE_CONVENTIONS``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from faciometry.core.landmarks import Landmark as L

from . import face as F


@dataclass(frozen=True)
class Truth:
    value: float
    exact: bool = False
    note: str = ""
    alternatives: dict[str, float] = field(default_factory=dict)


P = {name: tuple(map(float, xyz)) for name, xyz in F.FACE.items()}


def _d(a: L, b: L) -> float:
    return math.dist(P[a], P[b])


def _sub(a: L, b: L) -> tuple[float, float, float]:
    """b - a."""
    return (P[b][0] - P[a][0], P[b][1] - P[a][1], P[b][2] - P[a][2])


def _dot(u, v) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _norm(u) -> float:
    return math.sqrt(_dot(u, u))


def _ang_at(a: L, vertex: L, c: L) -> float:
    """Interior angle in degrees at ``vertex``, by arccos of the normalised dot.

    ``geometry.angle_at`` deliberately uses the atan2 form for precision near
    0 and 180 degrees. Using arccos here is the point: two different numerical
    routes to the same definition.
    """
    u, v = _sub(vertex, a), _sub(vertex, c)
    cos = _dot(u, v) / (_norm(u) * _norm(v))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _ang_between(a0: L, a1: L, b0: L, b1: L) -> float:
    """Undirected angle between two lines, folded into [0, 90]."""
    u, v = _sub(a0, a1), _sub(b0, b1)
    cos = _dot(u, v) / (_norm(u) * _norm(v))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return 180.0 - ang if ang > 90.0 else ang


def _line_frame(a: L, b: L):
    d = _sub(a, b)
    n = _norm(d)
    return (d[0] / n, d[1] / n, d[2] / n)


def _perp_from_line(p: L, a: L, b: L):
    """Component of p - a perpendicular to the line ab."""
    d = _line_frame(a, b)
    w = _sub(a, p)
    t = _dot(w, d)
    return (w[0] - t * d[0], w[1] - t * d[1], w[2] - t * d[2])


def _offset_conventions(p: L, a: L, b: L) -> dict[str, float]:
    """The three quantities a reader could mean by "point X mm from line AB".

    * ``perpendicular`` -- signed shortest distance to the line, the
      cephalometric convention and the one the Ricketts reference range is
      quoted in;
    * ``anteroposterior`` -- horizontal distance along z at the lip's own
      height, the convention used when the measurement is read off a photograph
      against a true horizontal;
    * ``z_component_of_perpendicular`` -- what
      ``formula.LineOffset(..., normal=Axis("z"))`` computes.

    The three coincide only when the reference line is exactly vertical. The
    ratio of the first to the third is ``cos`` of the line's inclination, so
    the size of the disagreement is a property of the line, not of the point:
    the Ricketts E-line is inclined 24.5 degrees on this face and Burstone's
    subnasale-pogonion line only 5.2, which is why the same coding decision
    costs 9 percent on one measurement and 0.4 percent on another.
    """
    perp = _perp_from_line(p, a, b)
    d = _line_frame(a, b)
    sign = 1.0 if perp[2] >= 0 else -1.0
    # Horizontal offset: walk the line to the lip's own height, compare z.
    w = _sub(a, p)
    t = w[1] / d[1]
    z_on_line = P[a][2] + t * d[2]
    return {
        "perpendicular": sign * math.hypot(perp[1], perp[2]),
        "anteroposterior": P[p][2] - z_on_line,
        "z_component_of_perpendicular": perp[2],
    }


_TILT_R = F.CANTHAL_TILT_R_DEG
_TILT_L = F.CANTHAL_TILT_L_DEG
_BIOCULAR = math.hypot(F.BIOCULAR_X_MM, P[L.EXOCANTHION_L][1] - P[L.EXOCANTHION_R][1])
_MOUTH_W = math.hypot(F.MOUTH_WIDTH_MM, F.MOUTH_CORNER_DROP_MM)
_FISSURE_W_R = _d(L.ENDOCANTHION_R, L.EXOCANTHION_R)
_FISSURE_W_L = _d(L.ENDOCANTHION_L, L.EXOCANTHION_L)

_E_UPPER = _offset_conventions(L.LABIALE_SUPERIUS, L.PRONASALE, L.POGONION)
_E_LOWER = _offset_conventions(L.LABIALE_INFERIUS, L.PRONASALE, L.POGONION)
#: Burstone's line: subnasale to pogonion, which unlike the E-line does not
#: pass through the nasal tip.
_BURSTONE_UPPER = _offset_conventions(L.LABIALE_SUPERIUS, L.SUBNASALE, L.POGONION)
#: The labiomental sulcus against the lower-lip-to-chin line.
_SULCUS = _offset_conventions(L.SUBLABIALE, L.LABIALE_INFERIUS, L.POGONION)

#: Every measurement in the catalogue built on ``LineOffset(..., Axis("z"))``,
#: with all three conventions a reader could mean. Four, not two: the same
#: coding decision the E-line pair makes is made again by
#: ``upper_lip_projection`` and ``labiomental_sulcus_depth``.
LINE_OFFSET_CONVENTIONS = {
    "e_line_upper_lip": _E_UPPER,
    "e_line_lower_lip": _E_LOWER,
    "upper_lip_projection": _BURSTONE_UPPER,
    "labiomental_sulcus_depth": _SULCUS,
}

#: Retained under the old name because it is the key arm 1 has always written.
E_LINE_CONVENTIONS = {"e_line_upper_lip": _E_UPPER, "e_line_lower_lip": _E_LOWER}


TRUTH: dict[str, Truth] = {
    # -- transverse lengths, all exact by construction
    "interpupillary_distance": Truth(F.IPD_MM, True, "pupils placed at +/- IPD/2, equal y and z"),
    "intercanthal_width": Truth(F.INTERCANTHAL_MM, True, "equal y and z"),
    "biocular_width": Truth(_BIOCULAR, False, "outer canthi differ in height by the tilt asymmetry"),
    "nose_breadth": Truth(F.NOSE_BREADTH_MM, True, "equal y and z"),
    "mouth_width": Truth(_MOUTH_W, True, "hypotenuse of the 52 mm span and the 0.8 mm corner drop"),
    "philtrum_width": Truth(F.PHILTRUM_WIDTH_MM, True, "equal y and z"),
    "bizygomatic_width": Truth(F.BIZYGOMATIC_MM, True, "equal y and z"),
    "bigonial_width": Truth(F.BIGONIAL_MM, True, "equal y and z"),
    # -- vertical lengths
    "nose_height": Truth(_d(L.SELLION, L.SUBNASALE)),
    "face_height_sellion_menton": Truth(_d(L.SELLION, L.MENTON)),
    "philtrum_length": Truth(_d(L.SUBNASALE, L.LABIALE_SUPERIUS)),
    "upper_face_height": Truth(_d(L.GLABELLA, L.LABIALE_SUPERIUS)),
    "middle_third_height": Truth(_d(L.GLABELLA, L.SUBNASALE)),
    "lower_third_height": Truth(_d(L.SUBNASALE, L.MENTON)),
    # -- periocular
    "palpebral_fissure_width_l": Truth(_FISSURE_W_L),
    "palpebral_fissure_width_r": Truth(_FISSURE_W_R),
    "palpebral_fissure_height_l": Truth(F.FISSURE_HEIGHT_L_MM, True, "equal x and z"),
    "palpebral_fissure_height_r": Truth(F.FISSURE_HEIGHT_R_MM, True, "equal x and z"),
    "canthal_tilt_l": Truth(_TILT_L, True, "outer canthus placed by dy = dx tan(theta)"),
    "canthal_tilt_r": Truth(_TILT_R, True, "outer canthus placed by dy = dx tan(theta)"),
    "eye_aspect_ratio_l": Truth(F.FISSURE_HEIGHT_L_MM / _FISSURE_W_L),
    "eye_aspect_ratio_r": Truth(F.FISSURE_HEIGHT_R_MM / _FISSURE_W_R),
    # -- ratios
    "intercanthal_biocular_ratio": Truth(F.INTERCANTHAL_MM / _BIOCULAR),
    "nose_mouth_width_ratio": Truth(F.NOSE_BREADTH_MM / _MOUTH_W),
    "eye_spacing_ratio": Truth(F.INTERCANTHAL_MM / F.NOSE_BREADTH_MM, True, "32 / 34"),
    "facial_thirds_ratio": Truth(_d(L.GLABELLA, L.SUBNASALE) / _d(L.SUBNASALE, L.MENTON)),
    "lip_vermilion_ratio": Truth(
        _d(L.LABIALE_SUPERIUS, L.STOMION) / _d(L.STOMION, L.LABIALE_INFERIUS)
    ),
    "facial_width_height_ratio": Truth(F.BIZYGOMATIC_MM / _d(L.GLABELLA, L.LABIALE_SUPERIUS)),
    "jaw_cheekbone_ratio": Truth(F.BIGONIAL_MM / F.BIZYGOMATIC_MM, True, "117 / 141"),
    # -- symmetry
    #
    # These three changed definition after the roll-cancellation defect: they
    # used to be dimensionless height differences read against the image
    # horizon and are now inclinations in degrees read against the
    # interpupillary line. The truth changes with them. Because both pupils sit
    # at the same y and z by construction, that line is exactly the x axis, so
    # each is the arctangent of a height difference over a lateral run -- and
    # the run is the *projection onto the interpupillary axis*, 90 mm and
    # 52 mm, not the point-to-point distance.
    "ocular_height_asymmetry": Truth(
        math.degrees(math.atan2(
            abs(P[L.EXOCANTHION_L][1] - P[L.EXOCANTHION_R][1]), F.BIOCULAR_X_MM)),
        False,
        "inclination of the outer-canthal line against the interpupillary axis: "
        "atan(29*(tan 6 - tan 5) / 90)",
    ),
    "mouth_corner_asymmetry": Truth(
        math.degrees(math.atan2(F.MOUTH_CORNER_DROP_MM, F.MOUTH_WIDTH_MM)),
        True,
        "atan(0.8 / 52), the 0.8 mm corner drop over the 52 mm lateral run",
    ),
    "canthal_tilt_asymmetry": Truth(abs(_TILT_L - _TILT_R), True, "6.0 - 5.0"),
    # -- profile angles
    "nasofrontal_angle": Truth(_ang_at(L.GLABELLA, L.NASION, L.PRONASALE)),
    "nasolabial_angle": Truth(_ang_at(L.COLUMELLA, L.SUBNASALE, L.LABIALE_SUPERIUS)),
    "facial_convexity_angle": Truth(_ang_at(L.GLABELLA, L.SUBNASALE, L.POGONION)),
    "mentolabial_angle": Truth(_ang_at(L.LABIALE_INFERIUS, L.SUBLABIALE, L.POGONION)),
    "mentocervical_angle": Truth(_ang_between(L.GLABELLA, L.POGONION, L.MENTON, L.CERVICALE)),
    "submental_cervical_angle": Truth(_ang_at(L.POGONION, L.MENTON, L.CERVICALE)),
    "gonial_angle_l": Truth(_ang_at(L.TRAGION_L, L.GONION_L, L.GNATHION)),
    "gonial_angle_r": Truth(_ang_at(L.TRAGION_R, L.GONION_R, L.GNATHION)),
    # -- profile lengths and offsets
    "e_line_upper_lip": Truth(
        _E_UPPER["z_component_of_perpendicular"],
        False,
        "registry computes the z component of the perpendicular; the Ricketts "
        "reference range is a perpendicular distance",
        dict(_E_UPPER),
    ),
    "e_line_lower_lip": Truth(
        _E_LOWER["z_component_of_perpendicular"],
        False,
        "registry computes the z component of the perpendicular; the Ricketts "
        "reference range is a perpendicular distance",
        dict(_E_LOWER),
    ),
    "nasal_tip_projection_ratio": Truth(
        abs(P[L.PRONASALE][2] - P[L.ALARE_L][2]) / _d(L.NASION, L.PRONASALE)
    ),
    "submental_length": Truth(_d(L.MENTON, L.CERVICALE)),
    "chin_projection": Truth(P[L.POGONION][2] - P[L.SUBNASALE][2], True, "26 - 30"),
}


# ---------------------------------------------------------------------------
# The twenty-three measurements added after the first harness run.
#
# Same rule as above: every value is written from the anatomical definition in
# the spec's label and description, in pure-Python ``math``, importing nothing
# from ``faciometry.core.formula`` or ``faciometry.measure.registry``. Where the
# construction makes the answer a coordinate difference it is marked exact and
# the expected value is the construction parameter.
#
# Two facts about the face do a lot of work here and are worth stating once.
# Both pupils sit at the same y and the same z, so the interpupillary *line* is
# exactly the x axis of the canonical frame and the interpupillary *axis* is
# exactly +/- x. That makes every perpendicular offset from that line a plain y
# difference, and every projection onto that axis a plain x difference -- which
# is the point: a formula that let the 16 mm brow depth or the 10 mm canthal
# depth leak into either would be caught here.
# ---------------------------------------------------------------------------

#: Lateral run from the outer canthus to the brow apex, measured along the
#: interpupillary axis. Negative on both sides means the apex sits medial to
#: the outer canthus, which is where Westmore's convention puts it.
_BROW_RUN_L = P[L.EXOCANTHION_L][0] - P[L.SUPERCILIARE_L][0]   # axis points -x
_BROW_RUN_R = P[L.SUPERCILIARE_R][0] - P[L.EXOCANTHION_R][0]   # axis points +x

#: Columellar rise and run in the sagittal plane, for the tip rotation.
_COLUMELLA_RISE = P[L.COLUMELLA][1] - P[L.SUBNASALE][1]
_COLUMELLA_RUN = P[L.COLUMELLA][2] - P[L.SUBNASALE][2]

_RAMUS_L = _d(L.TRAGION_L, L.GONION_L)
_RAMUS_R = _d(L.TRAGION_R, L.GONION_R)
_BODY_L = _d(L.GONION_L, L.GNATHION)
_BODY_R = _d(L.GONION_R, L.GNATHION)

TRUTH.update({
    # -- orbital, second set. Every vertical here is a perpendicular offset
    # -- from the interpupillary line, which is the x axis, so it is a y
    # -- difference and the eye's 10 mm of depth must not enter.
    "brow_apex_lateral_offset_l": Truth(
        _BROW_RUN_L / _FISSURE_W_L,
        False,
        "superciliare sits 20 mm medial to exocanthion and 16 mm in front of "
        "it; only the 20 mm may appear, over the fissure width",
    ),
    "brow_apex_lateral_offset_r": Truth(
        _BROW_RUN_R / _FISSURE_W_R,
        False,
        "as the left side, mirrored; the two differ only through the fissure "
        "widths, which differ only through the canthal tilts",
    ),
    "margin_reflex_distance_1_l": Truth(
        F.FISSURE_HEIGHT_L_MM / 2.0, True,
        "the upper lid margin is placed half a fissure height above the pupil, "
        "and the pupil is on the interpupillary line",
    ),
    "margin_reflex_distance_1_r": Truth(F.FISSURE_HEIGHT_R_MM / 2.0, True, "as the left side"),
    "margin_reflex_distance_2_l": Truth(
        F.FISSURE_HEIGHT_L_MM / 2.0, True,
        "the lower lid margin, the same distance below; the -y normal makes a "
        "lid below the line report positive",
    ),
    "margin_reflex_distance_2_r": Truth(F.FISSURE_HEIGHT_R_MM / 2.0, True, "as the left side"),
    "medial_canthal_angle_l": Truth(
        _ang_at(L.PALPEBRALE_SUP_L, L.ENDOCANTHION_L, L.PALPEBRALE_INF_L),
        False,
        "angle the two lid margins subtend at the inner corner; three points "
        "and no frame, so it carries no axis convention at all",
    ),
    "medial_canthal_angle_r": Truth(
        _ang_at(L.PALPEBRALE_SUP_R, L.ENDOCANTHION_R, L.PALPEBRALE_INF_R),
        False,
        "as the left side; the two differ through the fissure heights",
    ),

    # -- lips
    "upper_vermilion_height": Truth(
        _d(L.LABIALE_SUPERIUS, L.STOMION), False,
        "labiale superius to stomion; sqrt(4^2 + 1^2) in the sagittal plane",
    ),
    "lower_vermilion_height": Truth(
        _d(L.STOMION, L.LABIALE_INFERIUS), False,
        "stomion to labiale inferius; sqrt(8^2 + 2^2)",
    ),
    "cupids_bow_peak_height_l": Truth(
        P[L.CRISTA_PHILTRI_L][1] - P[L.LABIALE_SUPERIUS][1], True,
        "both terms are offsets from the same interpupillary line, so the "
        "line drops out and the answer is the 1.5 mm the crista philtri sits "
        "above labiale superius",
    ),
    "cupids_bow_peak_height_r": Truth(
        P[L.CRISTA_PHILTRI_R][1] - P[L.LABIALE_SUPERIUS][1], True, "as the left side",
    ),
    "commissure_height_l": Truth(
        (P[L.CHEILION_L][1] - P[L.STOMION][1]) / _MOUTH_W, True,
        "the left corner is the 0.8 mm dropped one, so this is -0.8 / mouth "
        "width; negative is a downturned corner",
    ),
    "commissure_height_r": Truth(
        (P[L.CHEILION_R][1] - P[L.STOMION][1]) / _MOUTH_W, True,
        "the right corner is level with the stomion by construction, so this "
        "is exactly zero -- and a formula that leaked the interpupillary line "
        "into it would not be",
    ),
    "upper_lip_projection": Truth(
        _BURSTONE_UPPER["z_component_of_perpendicular"],
        False,
        "registry computes the z component of the perpendicular to Burstone's "
        "line; the clinical convention is the perpendicular distance. The "
        "line is inclined only 5.2 degrees here, so the two differ by 0.4 "
        "percent rather than the E-line's 9 percent",
        dict(_BURSTONE_UPPER),
    ),

    # -- nose
    "nasal_dorsal_deviation": Truth(
        0.0, True,
        "sellion and pronasale are both on the midline, so the dorsal line is "
        "exactly perpendicular to the interpupillary axis and the departure "
        "from perpendicular is exactly zero",
    ),
    "alar_base_intercanthal_ratio": Truth(
        (P[L.SUBALARE_R][0] - P[L.SUBALARE_L][0]) / F.INTERCANTHAL_MM, True,
        "28 / 32; both pairs are placed at equal y and z, so both terms are "
        "coordinate differences",
    ),
    "nasal_tip_rotation": Truth(
        math.degrees(math.atan2(_COLUMELLA_RISE, _COLUMELLA_RUN)), True,
        "atan(7 / 11): the columella rises 7 mm and runs 11 mm forward from "
        "the subnasale, and the axis it is read against is the frame's z",
    ),

    # -- mandible, chin and midface
    "midface_projection": Truth(
        P[L.SUBNASALE][2] - P[L.NASION][2], True,
        "subnasale sits 8 mm in front of nasion; a pure z difference",
    ),
    "chin_height": Truth(
        _d(L.SUBLABIALE, L.GNATHION), False, "sublabiale to gnathion; sqrt(27^2 + 2^2)",
    ),
    "labiomental_sulcus_depth": Truth(
        _SULCUS["z_component_of_perpendicular"],
        False,
        "registry computes the z component of the perpendicular to the "
        "labiale inferius - pogonion line; that line is inclined 22.4 degrees "
        "here, so the z component reads 7.5 percent shallower than the "
        "perpendicular distance the sulcus is normally quoted as",
        dict(_SULCUS),
    ),
    "ramus_body_ratio_l": Truth(
        _RAMUS_L / _BODY_L, False,
        "tragion-gonion over gonion-gnathion; the jaw is mirror-symmetric by "
        "construction, so the two sides must agree exactly",
    ),
    "ramus_body_ratio_r": Truth(_RAMUS_R / _BODY_R, False, "as the left side, exactly"),
})
