"""What a measurement *is*, and whether it may be reported.

A `MeasurementSpec` is a declaration, not code: an id, a view, a unit, a
formula, and -- the part that matters most -- an honest statement of how well
that measurement is known to survive being taken from a photograph.

The evidence tier is the spine of this project. The published agreement data is
blunt: 2D photogrammetry reproduces midline sagittal measurements to within a
millimetre, and fails outright on the lateral ones. Lim et al. (2022, n=96)
measured bigonial breadth against direct anthropometry at a mean difference of
9.3 mm with limits of agreement spanning -0.9 to 19.6 mm, and bizygomatic
breadth at 3.3 mm with limits of -7.5 to 14.2 mm. Those two are the numerator
and denominator of the jaw-to-cheekbone ratio that every consumer face-analysis
tool prints to two decimal places.

Vitruve refuses to print them from a 2D image. Tagging the tier in the type is
what makes that refusal automatic rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .formula import ScalarExpr
from .landmarks import Landmark
from .sensitivity import (
    PoseSensitivity,
    Discriminability,
    discriminability,
    gated_pose,
)


class View(str, Enum):
    """Which photograph a measurement can be taken from."""

    FRONTAL = "frontal"
    PROFILE = "profile"
    EITHER = "either"


class Unit(str, Enum):
    MILLIMETRES = "mm"
    PIXELS = "px"
    DEGREES = "deg"
    RATIO = "ratio"

    @property
    def is_dimensionless(self) -> bool:
        return self in (Unit.RATIO, Unit.DEGREES)


class Evidence(str, Enum):
    """How well this measurement is known to survive photogrammetry.

    Ordered from most to least trustworthy. The tier is a property of the
    *measurement definition*, fixed by the literature; it does not depend on
    the particular photograph. Per-image quality is handled separately by the
    gate.
    """

    #: Published Bland-Altman agreement with direct anthropometry in 2D
    #: photogrammetry, mean difference under ~1 mm. Report freely.
    VALIDATED_2D = "validated_2d"

    #: Dimensionless ratio of two measurements taken in the same plane. Scale
    #: cancels, and to first order so does pose, because both terms shrink by
    #: the same cosine. The most robust thing this pipeline produces.
    POSE_INVARIANT_RATIO = "pose_invariant_ratio"

    #: 2D photogrammetry demonstrably fails: the endpoints sit on a laterally
    #: curved, self-occluding surface (zygion, gonion) whose apparent position
    #: is a silhouette artifact rather than the anatomical point. Emitted only
    #: from a 3D fit, always flagged.
    REQUIRES_3D = "requires_3d"

    #: Absorbs a pose error close to one-for-one against a small normal range.
    #: Canthal tilt is defined against the horizon, so image roll transfers
    #: into it directly -- 3 degrees of roll on a quantity whose normal range
    #: spans roughly 4 to 8 degrees.
    POSE_CRITICAL = "pose_critical"

    #: Standard vocabulary in the facial-aesthetics literature, with cited
    #: reference ranges, but no published agreement study against direct
    #: measurement. Reported with that stated.
    CONVENTIONAL = "conventional"


class Reportability(str, Enum):
    REPORT = "report"
    CAVEAT = "caveat"
    WITHHOLD = "withhold"


@dataclass(frozen=True)
class MeasurementSpec:
    """A declarative measurement definition."""

    id: str
    label: str
    view: View
    unit: Unit
    evidence: Evidence
    formula: ScalarExpr
    description: str = ""
    #: Literature the definition and any reference range come from.
    references: tuple[str, ...] = ()
    #: Optional published reference range, as (low, high, source). Displayed as
    #: context, never as a target and never as a score.
    reference_range: tuple[float, float, str] | None = None
    #: Measurement id whose value normalises this one into a ratio, if any.
    normalised_by: str | None = None
    #: Per-axis pose tolerance in degrees; exceeded means caveat or withhold.
    #: Defaults come from the literature and are overwritten by the empirical
    #: sweep in evals/arms/pose_sweep.py once it has run.
    pose_tolerance_deg: float = 8.0
    #: How far this measurement moves per degree of head rotation.
    sensitivity: PoseSensitivity = field(default_factory=PoseSensitivity)
    #: Between-subject spread, as a fraction of the measurement's own value for
    #: lengths and ratios, or in degrees for angles. This is the numerator of
    #: the discriminability ratio, and without it the measurement cannot be
    #: judged informative or not -- so ``None`` is itself reported, as "spread
    #: between people is unknown for this measurement".
    between_subject_rsd: float | None = None
    #: Measured within-person, between-photograph spread, where a study has
    #: actually quantified it. When present this *overrides* the derived pose
    #: model rather than adding to it.
    #:
    #: The derived model only knows about rigid head rotation. It cannot see
    #: expression, landmark-definition drift, or the difference between two
    #: cameras. Kramer (2016) decomposed the variance of the facial
    #: width-to-height ratio and found expression accounting for more of it
    #: than identity did -- something no projection model predicts. Where the
    #: literature has measured the real within-person spread, that number wins.
    measured_within_person_rsd: float | None = None
    #: Where ``measured_within_person_rsd`` came from.
    within_person_source: str = ""

    def __post_init__(self) -> None:
        if not self.id or " " in self.id:
            raise ValueError(f"measurement id must be a non-empty slug, got {self.id!r}")
        if self.unit is Unit.RATIO and self.reference_range is None and self.normalised_by:
            pass  # ratios need no reference range
        lo_hi = self.reference_range
        if lo_hi is not None and not lo_hi[0] < lo_hi[1]:
            raise ValueError(f"{self.id}: reference_range must be increasing, got {lo_hi}")

    @property
    def landmarks(self) -> frozenset[Landmark]:
        return self.formula.landmarks()

    @property
    def fingerprint(self) -> str:
        return self.formula.fingerprint

    @property
    def needs_metric_scale(self) -> bool:
        """True when the value is a length in millimetres.

        No permissively-licensed metric-scale 3D face model exists, so every
        millimetre value in Vitruve descends from a scale assumption -- an
        interpupillary-distance prior or a user-supplied ruler. Angles and
        ratios do not, which is why they are the preferred output.
        """
        return self.unit is Unit.MILLIMETRES


@dataclass(frozen=True)
class Verdict:
    """The decision about whether a measured value may be shown, and why."""

    reportability: Reportability
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def shown(self) -> bool:
        return self.reportability is not Reportability.WITHHOLD


def assess_discriminability(
    spec: MeasurementSpec,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    relative_landmark_error: float,
    relative_scale_error: float = 0.0,
    inflate_for_pose_uncertainty: bool = True,
) -> Discriminability | None:
    """Between-person spread over total measurement error for this photograph.

    Returns ``None`` when the between-subject spread is unknown, which is a
    different statement from "the ratio is low" and is reported differently.

    Pose magnitudes are inflated by the pose estimator's own uncertainty before
    the sensitivity is applied. Skipping that step is how a five-degree gate
    passes a face whose pose is only known to plus or minus five degrees.
    """
    if spec.between_subject_rsd is None:
        return None
    if inflate_for_pose_uncertainty:
        yaw_deg, pitch_deg, roll_deg = (gated_pose(a) for a in (yaw_deg, pitch_deg, roll_deg))
    derived_pose_error = spec.sensitivity.error_at(yaw_deg, pitch_deg, roll_deg)
    if spec.measured_within_person_rsd is not None:
        # A measured within-person spread already contains the pose term along
        # with everything the projection model cannot see, so it replaces the
        # derived value instead of adding to it. Taking the maximum keeps the
        # derived term honest at extreme pose, where a study conducted on
        # near-frontal photographs has nothing to say.
        pose_error = max(spec.measured_within_person_rsd, derived_pose_error)
    else:
        pose_error = derived_pose_error
    return discriminability(
        between_subject_sd=spec.between_subject_rsd,
        pose_error=pose_error,
        landmark_error=relative_landmark_error,
        scale_error=relative_scale_error if spec.needs_metric_scale else 0.0,
    )


def decide_reportability(
    spec: MeasurementSpec,
    *,
    max_pose_error_deg: float,
    roll_deg: float,
    have_3d: bool,
    subject_distance_m: float | None,
    relative_ci_width: float | None,
    repeatability_cv: float | None = None,
    disc: Discriminability | None = None,
) -> Verdict:
    """Decide whether a measured value is fit to print.

    Every rule here traces to a specific finding, and every refusal names it.
    A withheld measurement is a correct output, not a missing one.
    """
    reasons: list[str] = []
    worst = Reportability.REPORT

    def escalate(level: Reportability, why: str) -> None:
        nonlocal worst
        reasons.append(why)
        if level is Reportability.WITHHOLD or worst is Reportability.WITHHOLD:
            worst = Reportability.WITHHOLD
        else:
            worst = Reportability.CAVEAT

    # The primary gate. Kleinberg and Vanezis (2007) found facial indices
    # moving 8 to 19 percent at ten degrees of yaw against a between-subject
    # spread of 1.2 percent, so a measurement that cannot beat its own
    # photograph is withheld before any other consideration applies.
    if disc is not None and not disc.informative:
        escalate(
            Reportability.WITHHOLD,
            f"{disc.verdict} (between-person spread {disc.between_subject_sd:.3g} against "
            f"measurement error {disc.total_error_sd:.3g}; pose contributes "
            f"{disc.pose_component:.3g}, landmarks {disc.landmark_component:.3g})",
        )
    elif disc is not None and disc.ratio < 1.5:
        escalate(Reportability.CAVEAT, disc.verdict)
    elif disc is None:
        escalate(
            Reportability.CAVEAT,
            "no published between-person spread for this measurement, so whether "
            "it distinguishes individuals at all is unknown",
        )

    if spec.evidence is Evidence.REQUIRES_3D and not have_3d:
        escalate(
            Reportability.WITHHOLD,
            "endpoints lie on a self-occluding lateral surface; 2D photogrammetry "
            "does not reproduce this measurement (Lim et al. 2022: bigonial breadth "
            "mean difference 9.3 mm, limits of agreement -0.9 to 19.6 mm)",
        )

    if spec.evidence is Evidence.POSE_CRITICAL and abs(roll_deg) > 2.0:
        escalate(
            Reportability.WITHHOLD if abs(roll_deg) > 5.0 else Reportability.CAVEAT,
            f"image roll {roll_deg:+.1f} deg transfers almost one-for-one into this "
            "measurement, whose normal range spans only a few degrees",
        )

    if max_pose_error_deg > spec.pose_tolerance_deg:
        escalate(
            Reportability.WITHHOLD if max_pose_error_deg > 2 * spec.pose_tolerance_deg
            else Reportability.CAVEAT,
            f"head pose {max_pose_error_deg:.1f} deg exceeds the {spec.pose_tolerance_deg:.0f} deg "
            "tolerance for this measurement",
        )

    if spec.needs_metric_scale:
        escalate(
            Reportability.CAVEAT,
            "millimetre value rests on a scale assumption; interpupillary-distance "
            "priors carry roughly 5.5% error at one standard deviation "
            "(Dodgson 2004, ANSUR: mean 63 mm, SD ~3.5 mm)",
        )

    if subject_distance_m is not None and subject_distance_m < 1.0:
        # ICAO magnification distortion K = 50 mm / camera-subject distance.
        k = 0.05 / subject_distance_m
        escalate(
            Reportability.WITHHOLD if subject_distance_m < 0.6 else Reportability.CAVEAT,
            f"camera about {subject_distance_m:.2f} m from the subject implies roughly "
            f"{k * 100:.1f}% perspective magnification between the eye and nose planes "
            "(ICAO portrait-quality guidance asks for 1.0-2.5 m)",
        )

    if relative_ci_width is not None and relative_ci_width > 0.15:
        escalate(
            Reportability.WITHHOLD if relative_ci_width > 0.35 else Reportability.CAVEAT,
            f"95% interval spans {relative_ci_width * 100:.0f}% of the value; landmark "
            "uncertainty dominates the measurement",
        )

    if repeatability_cv is not None and repeatability_cv > 0.10:
        escalate(
            Reportability.WITHHOLD,
            f"measured test-retest coefficient of variation {repeatability_cv * 100:.0f}% "
            "exceeds the 10% reportability threshold",
        )

    if spec.evidence is Evidence.CONVENTIONAL:
        escalate(
            Reportability.CAVEAT,
            "standard in the facial-aesthetics literature, but with no published "
            "agreement study against direct anthropometric measurement",
        )

    return Verdict(worst, tuple(reasons))
