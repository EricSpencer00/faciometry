"""Combining several photographs of the same face.

Landmark placement is the term that most often decides whether a measurement
can be reported, and it is the only large term a person can reduce without
buying anything. Photograph the same face N times and the independent part of
that error falls as the square root of N.

The word doing the work is *independent*. Two captures taken a second apart, at
the same distance, in the same light, with the same expression, give the
landmark model very nearly the same input, so it makes very nearly the same
mistake. Averaging those does not reduce a systematic bias; it just reduces the
part that was already random. Treating N captures as N independent samples is
the same error as fusing two scale cues read by one model from one image and
calling them independent, which this codebase already refuses to do in
`core.scale.fuse`.

So the reduction here is capped by a correlation floor. With a shared fraction
`rho` of the error variance, averaging N captures leaves

    var = rho + (1 - rho) / N

times the single-capture variance, which tends to `rho` rather than to zero.
The default floor is deliberately pessimistic. Deliberate variation between
captures, which the capture guidance asks for, is what lowers it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.landmarks import Landmark, PointSet
from .evaluate import LandmarkUncertainty

#: Fraction of landmark error variance shared across captures of one face.
#:
#: A landmark model's error on a given face is largely a property of that face
#: and that model: an ambiguous jaw contour is ambiguous in every photograph.
#: Published test-retest work puts within-person variation across re-captures
#: well above the within-image repeatability, which is where this number comes
#: from. It is an assumption, it is stated, and it is adjustable.
DEFAULT_SHARED_FRACTION = 0.35

#: Captures whose landmark positions sit further than this many robust
#: deviations from the group are dropped. A blink, a swallow or a missed
#: detection should not be averaged in.
OUTLIER_MAD_LIMIT = 3.5


@dataclass(frozen=True)
class Combined:
    """The pooled estimate, and an honest account of what pooling achieved."""

    points: PointSet
    uncertainty: LandmarkUncertainty
    n_used: int
    n_supplied: int
    dropped: tuple[int, ...]
    shared_fraction: float

    @property
    def variance_factor(self) -> float:
        """What the pooled variance is, as a fraction of a single capture's."""
        rho, n = self.shared_fraction, max(self.n_used, 1)
        return rho + (1.0 - rho) / n

    @property
    def error_factor(self) -> float:
        """What the pooled *error* is, as a fraction of a single capture's."""
        return math.sqrt(self.variance_factor)

    @property
    def effective_n(self) -> float:
        """The number of truly independent captures this is worth.

        Printing "nine photographs" next to a reduction that only reached the
        value of four would overstate what averaging bought.
        """
        f = self.variance_factor
        return 1.0 / f if f > 0 else float("inf")

    def note(self) -> str:
        if self.n_used <= 1:
            return "a single photograph, so no averaging was possible"
        dropped = (
            f", {len(self.dropped)} discarded as inconsistent with the rest"
            if self.dropped
            else ""
        )
        return (
            f"{self.n_used} of {self.n_supplied} photographs averaged{dropped}. "
            f"Landmark error falls to {self.error_factor:.2f} of a single capture, "
            f"which is worth {self.effective_n:.1f} independent captures rather than "
            f"{self.n_used} because a shared fraction of {self.shared_fraction:.0%} of "
            "the error does not average away"
        )


def _robust_centre(stack: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.median(stack, axis=0)


def combine(
    captures: list[tuple[PointSet, LandmarkUncertainty]],
    *,
    shared_fraction: float = DEFAULT_SHARED_FRACTION,
    reject_outliers: bool = True,
) -> Combined:
    """Pool several captures of one face into a single point set.

    Every capture must already be in the same canonical frame, so that pooling
    combines measurements of one face rather than of several poses. Landmarks
    absent from a capture are simply not pooled for that capture; a model that
    could not see gonion in one photograph does not get to vote on it.
    """
    if not captures:
        raise ValueError("combine() needs at least one capture")
    if not 0.0 <= shared_fraction < 1.0:
        raise ValueError("shared_fraction must lie in [0, 1)")
    if len(captures) == 1:
        ps, unc = captures[0]
        return Combined(ps, unc, 1, 1, (), shared_fraction)

    dim = captures[0][0].dim
    if any(ps.dim != dim for ps, _ in captures):
        raise ValueError("captures must share a dimensionality")

    common = set(captures[0][0].available)
    for ps, _ in captures[1:]:
        common &= set(ps.available)
    if not common:
        raise ValueError("the captures share no landmark in common")
    names = sorted(common, key=lambda m: m.value)

    stack = np.stack([np.stack([ps.get(n) for n in names]) for ps, _ in captures])

    dropped: list[int] = []
    keep = list(range(len(captures)))
    if reject_outliers and len(captures) >= 3:
        centre = _robust_centre(stack)
        # Distance of each capture from the group, summed over landmarks, then
        # compared against the median absolute deviation of those distances.
        d = np.linalg.norm(stack - centre, axis=-1).mean(axis=1)
        med = np.median(d)
        mad = np.median(np.abs(d - med))
        if mad > 1e-9:
            score = 0.6745 * (d - med) / mad
            keep = [i for i in range(len(captures)) if score[i] <= OUTLIER_MAD_LIMIT]
            dropped = [i for i in range(len(captures)) if i not in keep]
        if len(keep) < 2:  # the rejection was too aggressive to be useful
            keep, dropped = list(range(len(captures))), []

    used = stack[keep]
    pooled = used.mean(axis=0)
    n = len(keep)

    # Pool the covariances the same way the points were pooled, then apply the
    # correlated-average factor. Averaging N estimates divides the independent
    # part of the variance by N; the shared part survives untouched.
    covs = np.stack(
        [
            np.stack([unc.covariances[unc.index[nm]] for nm in names])
            for i, (_, unc) in enumerate(captures)
            if i in keep
        ]
    )
    mean_cov = covs.mean(axis=0)
    factor = shared_fraction + (1.0 - shared_fraction) / n

    index = {nm: i for i, nm in enumerate(names)}
    return Combined(
        points=PointSet(index=index, coords=pooled),
        uncertainty=LandmarkUncertainty(index=index, covariances=mean_cov * factor),
        n_used=n,
        n_supplied=len(captures),
        dropped=tuple(dropped),
        shared_fraction=shared_fraction,
    )
