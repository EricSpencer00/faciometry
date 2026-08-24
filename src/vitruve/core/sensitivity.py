"""Pose sensitivity, and the question that actually matters.

The forensic literature settled this and the aesthetics literature has not
caught up. Kleinberg and Vanezis (2007) photographed subjects in ten-degree
steps and measured how far each facial index moved. At **ten degrees of yaw**,
their indices shifted by 8 to 19 percent -- against a between-subject relative
standard deviation at zero degrees of **1.2 percent** for the tightest index.
The pose artifact was larger than the entire spread between different people.
FISWG's 2026 guidance (V2.1, section 6.4.1) prohibits photo-anthropometry for
identification, citing precisely this.

That result reframes the whole problem. The useful question is not "how
accurate is this measurement" but:

    Does this measurement vary more between people than it does between
    photographs of the same person?

If not, the number is noise wearing a decimal point, and Vitruve declines to
print it. :func:`discriminability` computes that ratio and it is the primary
gate -- pose tolerances, confidence intervals and evidence tiers all feed into
it rather than standing beside it.

A second consequence: the pose estimate is itself uncertain. Published head
pose estimators sit at roughly 3.5 to 4 degrees mean absolute error on
AFLW2000-3D, with an effective label-noise floor around 2.5 to 3 degrees. A
five-degree gate is therefore at the edge of what can be resolved, so Vitruve
gates on ``|pose| + k * pose_sd`` rather than on the point estimate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

#: Mean absolute error of the head-pose estimator, degrees. 6DRepNet reports
#: 3.97 on AFLW2000-3D; near-frontal performance is better (about 2.8) and
#: extreme pose far worse (about 13.3), so this is a frontal-regime figure.
POSE_ESTIMATOR_MAE_DEG = 3.97

#: Converting mean absolute error to a standard deviation under a normal
#: assumption: sd = mae * sqrt(pi / 2).
POSE_ESTIMATOR_SD_DEG = POSE_ESTIMATOR_MAE_DEG * math.sqrt(math.pi / 2)


@dataclass(frozen=True)
class PoseSensitivity:
    """How much a measurement moves per degree of head pose.

    Values are *fractional* change per degree for ratios and lengths, and
    *degrees* of change per degree for angular measurements. Defaults are the
    orthographic first-order prediction; the empirical sweep in
    ``evals/arms/pose_sweep.py`` overwrites them with measured values, and the
    literature notes below say where the first-order model is known to be
    optimistic.
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    source: str = "first-order orthographic projection"

    def error_at(self, yaw: float, pitch: float, roll: float) -> float:
        """Expected magnitude of pose-induced error at a given head pose.

        Contributions add in quadrature: the three rotations are close to
        independent, and summing them linearly would overstate the total for
        the common case of small errors on all three axes.
        """
        return math.sqrt(
            (self.yaw * yaw) ** 2 + (self.pitch * pitch) ** 2 + (self.roll * roll) ** 2
        )


def cosine_yaw_sensitivity(yaw_deg: float = 10.0) -> float:
    """Fractional shrinkage per degree of a transverse width under yaw.

    A frontal-plane width scales as cos(yaw), so the fractional error is
    ``1 - cos(yaw)``: 0.4% at 5 degrees, 1.5% at 10, 6.0% at 20. Divided by the
    angle, this gives a per-degree slope for the small-angle regime.

    This is a *lower bound*. Kleinberg's measured curves have turning points
    away from zero, which proves the landmarks do not lie in a single plane;
    depth offsets and perspective add asymmetric bias on top of the cosine.
    """
    return (1.0 - math.cos(math.radians(yaw_deg))) / yaw_deg


#: A transverse width under yaw, first-order.
TRANSVERSE_WIDTH = PoseSensitivity(yaw=cosine_yaw_sensitivity(), source="cos(yaw), first order")

#: A vertical distance under pitch, first-order.
VERTICAL_DISTANCE = PoseSensitivity(
    pitch=cosine_yaw_sensitivity(), source="cos(pitch), first order"
)

#: A ratio of two transverse widths: both terms carry the same cosine, so it
#: cancels to first order. Not exactly zero, because the two widths sit at
#: different depths.
TRANSVERSE_RATIO = PoseSensitivity(
    yaw=0.1 * cosine_yaw_sensitivity(), source="cosine cancels between same-plane terms"
)

#: A width over a height, which carries the full cosine on the numerator only.
WIDTH_OVER_HEIGHT = PoseSensitivity(
    yaw=cosine_yaw_sensitivity(),
    pitch=cosine_yaw_sensitivity(),
    source="cos(yaw) on the width, 1/cos(pitch) on the height",
)

#: Canthal tilt. Roll enters one-for-one because the measurement is defined
#: against the horizon. Pitch enters at about 0.27 degrees per degree: Vaca et
#: al. (2022) swept the Frankfort plane over 30 degrees and watched
#: intercanthal height collapse from 4.39 mm to 0.128 mm, which is 8.3 degrees
#: of apparent tilt to 0.2.
CANTHAL_TILT = PoseSensitivity(
    yaw=0.05, pitch=0.27, roll=1.0, source="Vaca et al. 2022 pitch sweep; roll by definition"
)

#: Measured on the mandible: about 3 degrees of apparent change over 20 degrees
#: of yaw.
GONIAL_ANGLE = PoseSensitivity(yaw=0.15, pitch=0.10, roll=0.05, source="mandibular yaw study")

#: Kleinberg's worst measured indices, as an empirical upper bound for any
#: index whose endpoints span different depth planes.
KLEINBERG_WORST = PoseSensitivity(
    yaw=0.019, pitch=0.010, roll=0.005, source="Kleinberg & Vanezis 2007, AB/BB' index at 10 deg"
)


@dataclass(frozen=True)
class SignalOverError:
    """Is the quantity of interest larger than the uncertainty around it?

    The general form of the question this project keeps asking. For a
    morphometric measurement the signal is the between-person spread, and the
    specialised view is :class:`Discriminability`. For a within-face colour
    contrast the signal is the contrast itself, which is not a between-person
    spread at all -- so it gets the same rule without a misleading label.
    """

    signal: float
    error: float

    @property
    def ratio(self) -> float:
        return self.signal / self.error if self.error > 0 else math.inf

    @property
    def informative(self) -> bool:
        """The threshold is 1.0 rather than something more generous because at
        exactly 1.0 the quantity is already as much noise as it is signal."""
        return self.ratio > 1.0


def signal_over_error(*, signal: float, errors: Iterable[float]) -> SignalOverError:
    """Combine independent error sources in quadrature and compare to a signal."""
    if signal < 0:
        raise ValueError("signal cannot be negative")
    total = math.sqrt(sum(e**2 for e in errors))
    return SignalOverError(signal=signal, error=total)


@dataclass(frozen=True)
class Discriminability:
    """Whether a measurement separates people better than it separates photos.

    A :class:`SignalOverError` whose signal is specifically the between-person
    spread, carrying the error breakdown the report needs to name a cause.
    """

    ratio: float
    between_subject_sd: float
    total_error_sd: float
    pose_component: float
    landmark_component: float
    scale_component: float

    @property
    def informative(self) -> bool:
        """True when between-person spread exceeds measurement error.

        The threshold is 1.0 rather than something more generous because at
        exactly 1.0 the measurement already carries as much photograph in it as
        it carries person.
        """
        return self.ratio > 1.0

    @property
    def verdict(self) -> str:
        if self.ratio > 3.0:
            return "separates individuals well"
        if self.ratio > 1.5:
            return "separates individuals"
        if self.ratio > 1.0:
            return "marginal: measurement error approaches between-person spread"
        return "uninformative: this photograph contributes more variance than this person does"


def discriminability(
    *,
    between_subject_sd: float,
    pose_error: float,
    landmark_error: float,
    scale_error: float = 0.0,
) -> Discriminability:
    """Between-person spread over total measurement error.

    All four arguments must be in the same units -- either all absolute, or all
    expressed as a fraction of the measurement's own value. Errors combine in
    quadrature as independent sources.
    """
    if between_subject_sd < 0:
        raise ValueError("between-subject spread cannot be negative")
    total = math.sqrt(pose_error**2 + landmark_error**2 + scale_error**2)
    ratio = between_subject_sd / total if total > 0 else math.inf
    return Discriminability(
        ratio=ratio,
        between_subject_sd=between_subject_sd,
        total_error_sd=total,
        pose_component=pose_error,
        landmark_component=landmark_error,
        scale_component=scale_error,
    )


def gated_pose(estimate_deg: float, k: float = 1.0) -> float:
    """Pose magnitude inflated by the estimator's own uncertainty.

    Gating on the point estimate would let a face that the estimator called
    4 degrees, with a 5-degree standard deviation, through a 5-degree gate.
    """
    return abs(estimate_deg) + k * POSE_ESTIMATOR_SD_DEG
