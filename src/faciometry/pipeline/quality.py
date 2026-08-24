"""Is this photograph fit to measure, and if not, what should the user change?

The quality gate is advisory by construction. It does not decide whether a
number may be printed; :func:`faciometry.measure.evaluate.evaluate` already does
that, per measurement, against that measurement's own tolerance and its own
between-person spread. A global gate that also withheld things would either
duplicate that logic or contradict it. So this module produces evidence -- pose
and its uncertainty, blur, exposure, face size, subject distance -- and hands it
to the measurement layer, and reserves refusal for the one case where there is
nothing to hand over: an image so degraded that no measurement in the catalogue
could survive it.

Two design points worth stating.

**Pose is measured twice on purpose.** The landmark model reports a pose and an
independent head-pose network reports another, and the interesting quantity is
the difference. Published head-pose estimators sit around 4 degrees mean
absolute error on AFLW2000-3D, so two sources that disagree by fifteen are not
both right, and neither the mean nor either one deserves the 4-degree interval
that a single estimator's published figure would imply. Disagreement therefore
inflates the pose standard deviation, which propagates into every
discriminability ratio, which is where it belongs. Reporting the agreement is
not a diagnostic nicety; it is the only observable that catches a pose estimate
that has failed rather than merely been imprecise.

**Every issue carries a remedy.** A quality report that says "blurred" is a
verdict on the user. One that says "hold the camera still or brace it, and
increase the light so the shutter can be shorter" is an instruction they can
follow, and the photograph is the one thing in this pipeline they can change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ..core.scale import magnification_distortion
from ..core.sensitivity import POSE_ESTIMATOR_SD_DEG, gated_pose
from ..core.spec import View
from .ingest import SubjectDistance
from .ports import HeadPose

# ---------------------------------------------------------------------------
# Thresholds. Every one of these is a judgement call, so every one of them is
# named, given a number, and given the reason for the number.
# ---------------------------------------------------------------------------

#: ICAO 9303 portrait guidance asks for at least 90 pixels between the eyes and
#: prefers 120. Below 60 the landmark model is interpolating rather than
#: localising and no catalogue measurement survives.
ICAO_INTEROCULAR_PREFERRED_PX = 120.0
ICAO_INTEROCULAR_MINIMUM_PX = 90.0
INTEROCULAR_UNUSABLE_PX = 60.0

#: A profile photograph has no interocular axis, and ICAO's minimum is a frontal
#: standard, so face size there is checked against the detection box height
#: instead. Adult interocular distance runs close to three tenths of head
#: height, which puts ICAO's 90 pixels at roughly 300 pixels of head. It is a
#: proxy and the report says so.
FACE_HEIGHT_MINIMUM_PX = 300.0
FACE_HEIGHT_UNUSABLE_PX = 200.0

#: Variance of the Laplacian on the 512-pixel canonical crop, 0-255 grayscale.
#: The classic Pech-Pacheco focus measure. The threshold only means anything at
#: a fixed image size, which is why it is computed on the canonical crop rather
#: than on the original: a 4000-pixel photograph and a 400-pixel one of the same
#: face have wildly different Laplacian variances and identical sharpness.
BLUR_WARN_VARIANCE = 100.0
BLUR_FAIL_VARIANCE = 20.0

#: Fraction of face pixels pinned at either end of the range. Clipping destroys
#: the gradient a heatmap landmark model localises on, and it does so silently.
CLIP_WARN_FRACTION = 0.02
CLIP_FAIL_FRACTION = 0.10
_CLIP_DARK_LEVEL = 2
_CLIP_BRIGHT_LEVEL = 253

#: Pose thresholds are applied to the *gated* magnitude, meaning the estimate
#: inflated by the estimator's own uncertainty. Gating on the point estimate is
#: how a 5-degree tolerance passes a face whose pose is known to plus or minus
#: 5 degrees.
POSE_INFO_DEG = 8.0
POSE_WARN_DEG = 15.0
POSE_FAIL_DEG = 25.0
ROLL_WARN_DEG = 12.0
ROLL_FAIL_DEG = 25.0

#: Disagreement between the two pose sources, worst axis.
AGREEMENT_INFO_DEG = 4.0
AGREEMENT_WARN_DEG = 8.0
AGREEMENT_FAIL_DEG = 15.0

#: ICAO portrait-quality guidance asks for 1.0 to 2.5 m. Below 0.6 m the
#: perspective magnification between the eye and nose planes exceeds 8%, which
#: is larger than the between-person spread of most of the catalogue.
DISTANCE_FAIL_M = 0.6
DISTANCE_WARN_M = 1.0
DISTANCE_FAR_M = 2.5

#: Standard deviation assigned to a rotation axis nobody estimated. An
#: uncontrolled portrait's yaw spreads over roughly plus or minus 25 degrees,
#: and the standard deviation of a uniform distribution that wide is about 15.
#: Large on purpose: an unknown pose should widen every interval it touches.
UNKNOWN_POSE_SD_DEG = 15.0


def nominal_yaw_deg(view: View, yaw_deg: float) -> float:
    """The yaw this view is *supposed* to have.

    A profile photograph at ninety degrees of yaw is not an off-axis frontal
    photograph. It is a correct profile photograph, and gating it against zero
    would reject every profile the clinical protocol asks for and then withhold
    every profile measurement for a pose the user was told to adopt. The sign
    follows whichever side the subject actually turned to, which is read from
    the estimate rather than declared.
    """
    if view is View.PROFILE:
        return 90.0 if yaw_deg >= 0 else -90.0
    return 0.0


def yaw_deviation(view: View, yaw_deg: float) -> float:
    """Signed departure from the yaw this view was meant to be taken at."""
    return yaw_deg - nominal_yaw_deg(view, yaw_deg)


class Severity(str, Enum):
    """How much a quality finding matters.

    ``FAIL`` is reserved for conditions under which no measurement in the
    catalogue could be trusted, because a failing report short-circuits the run.
    Anything a per-measurement gate can handle on its own is at most ``WARN``.
    """

    INFO = "info"
    WARN = "warn"
    FAIL = "fail"

    @property
    def rank(self) -> int:
        return {"info": 0, "warn": 1, "fail": 2}[self.value]


@dataclass(frozen=True)
class QualityIssue:
    """One thing wrong with the photograph, and what to do about it."""

    code: str
    severity: Severity
    message: str
    remedy: str

    def format(self) -> str:
        return f"[{self.severity.value}] {self.message} Remedy: {self.remedy}"


@dataclass(frozen=True)
class PoseAgreement:
    """How far apart two independent pose estimates are, per axis."""

    sources: tuple[str, str]
    yaw_delta_deg: float
    pitch_delta_deg: float
    roll_delta_deg: float

    @property
    def max_delta_deg(self) -> float:
        return max(
            abs(self.yaw_delta_deg), abs(self.pitch_delta_deg), abs(self.roll_delta_deg)
        )


@dataclass(frozen=True)
class PoseEstimate:
    """Head pose with a per-axis standard deviation that reflects the evidence.

    The standard deviation is never smaller than the published single-estimator
    figure. Averaging two estimates that read the same landmarks off the same
    image does not buy the ``1/sqrt(2)`` an independence assumption would claim,
    and pretending otherwise would tighten every downstream interval on the
    strength of a correlation that is not there.
    """

    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    yaw_sd_deg: float = POSE_ESTIMATOR_SD_DEG
    pitch_sd_deg: float = POSE_ESTIMATOR_SD_DEG
    roll_sd_deg: float = POSE_ESTIMATOR_SD_DEG
    sources: tuple[str, ...] = ()
    agreement: PoseAgreement | None = None
    notes: tuple[str, ...] = ()

    @staticmethod
    def _gated(value: float, sd: float) -> float:
        """Magnitude inflated by uncertainty, floored at the published figure.

        :func:`~faciometry.core.sensitivity.gated_pose` is the project-wide rule
        and it uses the published estimator spread. Where this run has evidence
        that its own pose is worse than published -- two sources disagreeing,
        or an axis nobody estimated -- the larger number wins.
        """
        return max(gated_pose(value), abs(value) + sd)

    @property
    def gated_yaw_deg(self) -> float:
        return self._gated(self.yaw_deg, self.yaw_sd_deg)

    @property
    def gated_pitch_deg(self) -> float:
        return self._gated(self.pitch_deg, self.pitch_sd_deg)

    @property
    def gated_roll_deg(self) -> float:
        return self._gated(self.roll_deg, self.roll_sd_deg)

    @property
    def max_gated_deg(self) -> float:
        return max(self.gated_yaw_deg, self.gated_pitch_deg, self.gated_roll_deg)


def reconcile_pose(
    landmarker: HeadPose | None = None,
    independent: HeadPose | None = None,
    *,
    geometric_roll_deg: float | None = None,
    landmarker_name: str = "landmarker",
    independent_name: str = "head-pose network",
) -> PoseEstimate:
    """Combine the pose sources available into one estimate plus its spread.

    ``geometric_roll_deg`` is the roll read straight off the interocular axis of
    the located landmarks. It is not a substitute for a pose network -- it says
    nothing about yaw or pitch -- but roll is the one axis that plane geometry
    determines exactly, so it is used when no network reported roll, and it is
    always available as a third opinion.
    """
    triples: list[tuple[str, float, float, float]] = []
    if landmarker is not None and _has_pose(landmarker):
        triples.append(
            (
                landmarker_name,
                float(landmarker.yaw_deg or 0.0),
                float(landmarker.pitch_deg or 0.0),
                float(landmarker.roll_deg or 0.0),
            )
        )
    if independent is not None and _has_pose(independent):
        triples.append(
            (
                independent_name,
                float(independent.yaw_deg),
                float(independent.pitch_deg),
                float(independent.roll_deg),
            )
        )

    notes: list[str] = []

    if len(triples) >= 2:
        (na, ya, pa, ra), (nb, yb, pb, rb) = triples[0], triples[1]
        agreement = PoseAgreement((na, nb), ya - yb, pa - pb, ra - rb)
        yaw, pitch, roll = (ya + yb) / 2, (pa + pb) / 2, (ra + rb) / 2
        # Half the disagreement is the smallest spread consistent with two
        # estimates that far apart. Below that, the published single-estimator
        # figure is the floor.
        yaw_sd = max(POSE_ESTIMATOR_SD_DEG, abs(agreement.yaw_delta_deg) / 2)
        pitch_sd = max(POSE_ESTIMATOR_SD_DEG, abs(agreement.pitch_delta_deg) / 2)
        roll_sd = max(POSE_ESTIMATOR_SD_DEG, abs(agreement.roll_delta_deg) / 2)
        sources = (na, nb)
    elif len(triples) == 1:
        name, yaw, pitch, roll = triples[0]
        agreement = None
        yaw_sd = pitch_sd = roll_sd = POSE_ESTIMATOR_SD_DEG
        sources = (name,)
        notes.append(
            f"only one pose source ({name}) was available, so the estimate carries "
            "the published single-estimator spread and nothing cross-checks it"
        )
    else:
        yaw = pitch = roll = 0.0
        agreement = None
        yaw_sd = pitch_sd = roll_sd = UNKNOWN_POSE_SD_DEG
        sources = ()
        notes.append(
            "no pose source reported, so yaw and pitch are treated as unknown "
            "rather than as zero and every interval widens accordingly"
        )

    if geometric_roll_deg is not None:
        if not sources:
            roll, roll_sd = float(geometric_roll_deg), POSE_ESTIMATOR_SD_DEG
            sources = ("interocular axis",)
            notes.append(
                "roll was read from the interocular axis of the located landmarks, "
                "which plane geometry fixes exactly up to the landmarks themselves"
            )
        else:
            drift = abs(float(geometric_roll_deg) - roll)
            notes.append(
                f"the interocular axis reads roll as {geometric_roll_deg:+.1f} deg, "
                f"{drift:.1f} deg from the reported estimate"
            )
            roll_sd = max(roll_sd, drift / 2)

    return PoseEstimate(
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        yaw_sd_deg=yaw_sd,
        pitch_sd_deg=pitch_sd,
        roll_sd_deg=roll_sd,
        sources=tuple(sources),
        agreement=agreement,
        notes=tuple(notes),
    )


def _has_pose(obj: object) -> bool:
    return all(
        getattr(obj, axis, None) is not None for axis in ("yaw_deg", "pitch_deg", "roll_deg")
    )


# ---------------------------------------------------------------------------
# Image-quality measures
# ---------------------------------------------------------------------------


def to_grayscale(pixels: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Rec. 601 luma. Any fixed weighting would do; this one is conventional."""
    arr = np.asarray(pixels, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def laplacian_variance(gray: NDArray[np.float64]) -> float:
    """Variance of the 4-neighbour Laplacian, the standard focus measure.

    Convolved by slicing rather than through scipy: the whole point of the
    ``core`` and ``pipeline`` split is that nothing below the model layer
    reaches for a heavy dependency, and this is nine lines.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
        - 4.0 * g[1:-1, 1:-1]
    )
    return float(np.var(lap))


def clipped_fractions(gray: NDArray[np.float64]) -> tuple[float, float]:
    """Fraction of pixels pinned near black and near white, in that order."""
    g = np.asarray(gray, dtype=np.float64)
    n = g.size
    if n == 0:
        return 0.0, 0.0
    return (
        float(np.count_nonzero(g <= _CLIP_DARK_LEVEL) / n),
        float(np.count_nonzero(g >= _CLIP_BRIGHT_LEVEL) / n),
    )


@dataclass(frozen=True)
class QualityReport:
    """Everything known about the photograph as a photograph.

    This is evidence, not a verdict. The one exception is :attr:`failed`, which
    means the run should stop, and it is set only by conditions under which no
    catalogue measurement could be trusted.
    """

    view: View
    pose: PoseEstimate
    blur_laplacian_variance: float
    clipped_dark_fraction: float
    clipped_bright_fraction: float
    interocular_px: float | None
    face_height_px: float | None = None
    subject_distance: SubjectDistance | None = None
    magnification_distortion: float | None = None
    detection_score: float | None = None
    issues: tuple[QualityIssue, ...] = field(default_factory=tuple)

    @property
    def worst_severity(self) -> Severity | None:
        if not self.issues:
            return None
        return max((i.severity for i in self.issues), key=lambda s: s.rank)

    @property
    def failed(self) -> bool:
        return any(i.severity is Severity.FAIL for i in self.issues)

    @property
    def subject_distance_m(self) -> float | None:
        """The value :func:`~faciometry.measure.evaluate.evaluate` expects."""
        return self.subject_distance.metres if self.subject_distance else None

    def issues_at(self, severity: Severity) -> tuple[QualityIssue, ...]:
        return tuple(i for i in self.issues if i.severity is severity)

    def failure_message(self) -> str:
        """One paragraph a user can act on, or an empty string if nothing failed."""
        fails = self.issues_at(Severity.FAIL)
        if not fails:
            return ""
        head = (
            f"The {self.view.value} photograph cannot be measured: "
            + " ".join(i.message for i in fails)
        )
        return head + " " + " ".join(f"Remedy: {i.remedy}" for i in fails)


def _face_height_issues(face_height_px: float | None) -> list[QualityIssue]:
    """Face-size findings for a view with no interocular axis to measure.

    Kept separate because the threshold is a derived proxy rather than a
    standard, and folding it into the ICAO branch would let a reader mistake it
    for one.
    """
    if face_height_px is None:
        return [
            QualityIssue(
                "face_size_unknown",
                Severity.INFO,
                "this view has no interocular axis and no face box, so face size "
                "was not checked at all.",
                "none available. It is a note about what could not be measured.",
            )
        ]
    if face_height_px < FACE_HEIGHT_UNUSABLE_PX:
        return [
            QualityIssue(
                "face_too_small",
                Severity.FAIL,
                f"the detected face is only {face_height_px:.0f} pixels tall, below "
                f"the {FACE_HEIGHT_UNUSABLE_PX:.0f} at which a landmark model is "
                "interpolating rather than localising.",
                "move closer or use a longer lens so the head fills at least "
                f"{FACE_HEIGHT_MINIMUM_PX:.0f} pixels of height.",
            )
        ]
    if face_height_px < FACE_HEIGHT_MINIMUM_PX:
        return [
            QualityIssue(
                "face_below_size_proxy",
                Severity.WARN,
                f"the detected face is {face_height_px:.0f} pixels tall, below the "
                f"{FACE_HEIGHT_MINIMUM_PX:.0f} that corresponds to ICAO's frontal "
                "minimum. This view has no interocular axis, so the head box is a "
                "proxy rather than the standard's own quantity.",
                "fill more of the frame with the head, without moving the camera "
                "closer than a metre.",
            )
        ]
    return []


def assess(
    *,
    view: View,
    aligned_pixels: NDArray[np.uint8],
    interocular_px: float | None,
    pose: PoseEstimate,
    face_height_px: float | None = None,
    subject_distance: SubjectDistance | None = None,
    detection_score: float | None = None,
) -> QualityReport:
    """Score one photograph and say what the photographer should change.

    ``aligned_pixels`` is the canonical crop, not the original image. Blur and
    exposure are properties of the face, and measuring them over a 4000-pixel
    frame that is nine-tenths background answers a different question.
    ``interocular_px`` is in *original* image pixels, because that is the
    quantity ICAO's minimum is defined against. It is ``None`` on a profile
    view, which has no interocular axis at all; ``face_height_px`` then stands
    in, and the report names it as the proxy it is.
    """
    gray = to_grayscale(aligned_pixels)
    blur = laplacian_variance(gray)
    dark, bright = clipped_fractions(gray)
    magnification = (
        magnification_distortion(subject_distance.metres) if subject_distance else None
    )

    issues: list[QualityIssue] = []

    # -- face size ---------------------------------------------------------
    if interocular_px is None:
        issues.extend(_face_height_issues(face_height_px))
    elif interocular_px < INTEROCULAR_UNUSABLE_PX:
        issues.append(
            QualityIssue(
                "face_too_small",
                Severity.FAIL,
                f"only {interocular_px:.0f} pixels between the eyes, below the "
                f"{INTEROCULAR_UNUSABLE_PX:.0f} at which a landmark model is "
                "interpolating rather than localising.",
                "move closer or use a longer lens so the eyes are at least "
                f"{ICAO_INTEROCULAR_MINIMUM_PX:.0f} pixels apart, ideally "
                f"{ICAO_INTEROCULAR_PREFERRED_PX:.0f}.",
            )
        )
    elif interocular_px < ICAO_INTEROCULAR_MINIMUM_PX:
        issues.append(
            QualityIssue(
                "face_below_icao_minimum",
                Severity.WARN,
                f"{interocular_px:.0f} pixels between the eyes, below the ICAO "
                f"minimum of {ICAO_INTEROCULAR_MINIMUM_PX:.0f}.",
                "fill more of the frame with the face, without moving the camera "
                "closer than a metre.",
            )
        )
    elif interocular_px < ICAO_INTEROCULAR_PREFERRED_PX:
        issues.append(
            QualityIssue(
                "face_below_icao_preferred",
                Severity.INFO,
                f"{interocular_px:.0f} pixels between the eyes, above the ICAO "
                f"minimum but below the preferred {ICAO_INTEROCULAR_PREFERRED_PX:.0f}.",
                "a longer lens from the same distance would tighten every interval "
                "without adding perspective distortion.",
            )
        )

    # -- focus -------------------------------------------------------------
    if blur < BLUR_FAIL_VARIANCE:
        issues.append(
            QualityIssue(
                "severe_blur",
                Severity.FAIL,
                f"focus measure {blur:.0f} on the canonical crop, far below the "
                f"{BLUR_WARN_VARIANCE:.0f} a sharp portrait reaches.",
                "brace the camera or use a tripod, add light so the shutter can be "
                "shorter, and confirm the focus point is on the eyes.",
            )
        )
    elif blur < BLUR_WARN_VARIANCE:
        issues.append(
            QualityIssue(
                "soft_focus",
                Severity.WARN,
                f"focus measure {blur:.0f}, below the {BLUR_WARN_VARIANCE:.0f} "
                "expected of a sharp portrait; landmark scatter will be larger "
                "than the model's published figure.",
                "focus on the nearer eye and add light so the shutter can be shorter.",
            )
        )

    # -- exposure ----------------------------------------------------------
    for name, fraction, direction, remedy in (
        (
            "clipped_highlights",
            bright,
            "blown out",
            "reduce exposure or move the light off-axis; a face lit flat from the "
            "camera position clips on the forehead and nose first.",
        ),
        (
            "clipped_shadows",
            dark,
            "crushed to black",
            "add fill light on the shadow side, or turn the subject toward the "
            "existing light.",
        ),
    ):
        if fraction > CLIP_FAIL_FRACTION:
            issues.append(
                QualityIssue(
                    name,
                    Severity.FAIL,
                    f"{fraction * 100:.0f}% of the face is {direction}, so the "
                    "gradients a landmark model localises on are gone there.",
                    remedy,
                )
            )
        elif fraction > CLIP_WARN_FRACTION:
            issues.append(
                QualityIssue(
                    name,
                    Severity.WARN,
                    f"{fraction * 100:.0f}% of the face is {direction}.",
                    remedy,
                )
            )

    # -- pose --------------------------------------------------------------
    # Measured against the yaw this view was meant to have, not against zero.
    yaw_off = yaw_deviation(view, pose.yaw_deg)
    gated_yaw_off = max(gated_pose(yaw_off), abs(yaw_off) + pose.yaw_sd_deg)
    worst_axis = max(gated_yaw_off, pose.gated_pitch_deg)
    against = (
        "a true profile" if view is View.PROFILE else "square to the camera"
    )
    squarely = (
        "turn until one ear, the nose tip and the chin all sit on the outline, with "
        "the far eye just hidden."
        if view is View.PROFILE
        else "face the camera squarely, with the eyes level and the gaze on the lens."
    )
    if worst_axis > POSE_FAIL_DEG:
        issues.append(
            QualityIssue(
                "extreme_pose",
                Severity.FAIL,
                f"head pose is {worst_axis:.0f} deg away from {against} once the "
                "estimator's own uncertainty is included, beyond the tolerance of "
                "every measurement in the catalogue.",
                squarely,
            )
        )
    elif worst_axis > POSE_WARN_DEG:
        issues.append(
            QualityIssue(
                "off_axis_pose",
                Severity.WARN,
                f"head pose is {worst_axis:.0f} deg away from {against} including "
                "estimator uncertainty; Kleinberg and Vanezis (2007) measured facial "
                "indices moving 8 to 19 percent at ten degrees of yaw.",
                squarely,
            )
        )
    elif worst_axis > POSE_INFO_DEG:
        issues.append(
            QualityIssue(
                "slightly_off_axis",
                Severity.INFO,
                f"head pose is {worst_axis:.0f} deg away from {against} including "
                "estimator uncertainty.",
                "a pose closer to the protocol would let the pose-sensitive "
                "measurements report rather than caveat.",
            )
        )

    if pose.gated_roll_deg > ROLL_FAIL_DEG:
        issues.append(
            QualityIssue(
                "extreme_roll",
                Severity.FAIL,
                f"image roll reaches {pose.gated_roll_deg:.0f} deg including "
                "estimator uncertainty, which no amount of correction recovers from.",
                "hold the camera level and keep the eyes level in the frame.",
            )
        )
    elif pose.gated_roll_deg > ROLL_WARN_DEG:
        issues.append(
            QualityIssue(
                "image_roll",
                Severity.WARN,
                f"image roll reaches {pose.gated_roll_deg:.0f} deg including "
                "estimator uncertainty; roll is corrected in the point estimate but "
                "the correction's own error stays in every canthal measurement.",
                "hold the camera level and keep the eyes level in the frame.",
            )
        )

    # -- pose agreement ----------------------------------------------------
    if pose.agreement is None:
        issues.append(
            QualityIssue(
                "pose_uncrosschecked",
                Severity.INFO,
                "only one pose source was available, so nothing cross-checks it; a "
                "failed pose estimate looks exactly like a good one from here.",
                "no photographic change helps this one. It is a note about the "
                "pipeline's configuration, not about the photograph.",
            )
        )
    else:
        delta = pose.agreement.max_delta_deg
        a, b = pose.agreement.sources
        if delta > AGREEMENT_FAIL_DEG:
            issues.append(
                QualityIssue(
                    "pose_sources_disagree",
                    Severity.FAIL,
                    f"the {a} and the {b} disagree by {delta:.0f} deg on head pose. "
                    "Published estimators sit near 4 deg mean absolute error, so a "
                    "gap this size means one of them has failed rather than been "
                    "imprecise, and there is no way to tell which.",
                    "retake with the whole face visible, evenly lit, and unobstructed "
                    "by hair, glasses or a hand.",
                )
            )
        elif delta > AGREEMENT_WARN_DEG:
            issues.append(
                QualityIssue(
                    "pose_sources_diverge",
                    Severity.WARN,
                    f"the {a} and the {b} disagree by {delta:.0f} deg on head pose, "
                    "so the pose interval is widened to match and every "
                    "pose-sensitive measurement inherits that.",
                    "retake with the whole face visible and evenly lit.",
                )
            )
        elif delta > AGREEMENT_INFO_DEG:
            issues.append(
                QualityIssue(
                    "pose_sources_differ",
                    Severity.INFO,
                    f"the {a} and the {b} differ by {delta:.0f} deg on head pose, "
                    "within what their published error bars allow.",
                    "none needed.",
                )
            )

    # -- camera distance ---------------------------------------------------
    if subject_distance is not None and magnification is not None:
        d = subject_distance.metres
        if d < DISTANCE_FAIL_M:
            issues.append(
                QualityIssue(
                    "camera_too_close",
                    Severity.FAIL,
                    f"camera about {d:.2f} m away, implying {magnification * 100:.0f}% "
                    "magnification between the eye and nose planes. Every ratio whose "
                    "terms straddle those planes carries that as bias.",
                    "step back to between 1.0 and 2.5 m and use a longer lens to keep "
                    "the face the same size in frame.",
                )
            )
        elif d < DISTANCE_WARN_M:
            issues.append(
                QualityIssue(
                    "camera_close",
                    Severity.WARN,
                    f"camera about {d:.2f} m away, implying {magnification * 100:.1f}% "
                    "perspective magnification; ICAO portrait guidance asks for "
                    f"{DISTANCE_WARN_M:.1f} to {DISTANCE_FAR_M:.1f} m.",
                    "step back and zoom in rather than moving closer.",
                )
            )
        elif d > DISTANCE_FAR_M:
            issues.append(
                QualityIssue(
                    "camera_far",
                    Severity.INFO,
                    f"camera about {d:.2f} m away, beyond the "
                    f"{DISTANCE_FAR_M:.1f} m ICAO guidance suggests; perspective "
                    "distortion is negligible but face pixels are being spent on air.",
                    "none needed if the face is large enough in frame.",
                )
            )
    else:
        issues.append(
            QualityIssue(
                "distance_unknown",
                Severity.INFO,
                "camera-to-subject distance could not be determined, so the "
                "perspective-magnification bias is unquantified rather than small.",
                "photographs from a camera that records a 35mm-equivalent focal "
                "length let this be estimated; most phones do.",
            )
        )

    return QualityReport(
        view=view,
        pose=pose,
        blur_laplacian_variance=blur,
        clipped_dark_fraction=dark,
        clipped_bright_fraction=bright,
        interocular_px=None if interocular_px is None else float(interocular_px),
        face_height_px=None if face_height_px is None else float(face_height_px),
        subject_distance=subject_distance,
        magnification_distortion=magnification,
        detection_score=detection_score,
        issues=tuple(issues),
    )
