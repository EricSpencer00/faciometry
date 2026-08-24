"""Evaluating a measurement, with its uncertainty.

A landmark is not a point. A heatmap landmark model produces a probability
field over pixels, and collapsing that to an argmax throws away exactly the
information needed to say how much a derived measurement can be trusted.

So Vitruve keeps the covariance. Each landmark carries a 2x2 (or 3x3) matrix,
the measurement's formula is evaluated over a seeded Monte-Carlo ensemble drawn
from those covariances, and what comes out is a distribution rather than a
number. The report prints a median and a 95% interval, and a measurement whose
interval spans two population deciles says so instead of printing three decimal
places.

Monte Carlo rather than the delta method because several of these formulas --
signed line offsets, ratios that can approach a zero denominator, angles near
the degenerate collinear case -- are badly behaved under linearisation exactly
where the uncertainty matters most.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ..core.landmarks import Landmark, MissingLandmarkError, PointSet
from ..core.scale import ScaleEstimate
from ..core.spec import (
    Discriminability,
    MeasurementSpec,
    Reportability,
    Unit,
    Verdict,
    assess_discriminability,
    decide_reportability,
)

DEFAULT_SAMPLES = 2048


@dataclass(frozen=True)
class LandmarkUncertainty:
    """Per-landmark positional covariance, in the same units as the points.

    ``covariances`` is ``(n_landmarks, dim, dim)``. An isotropic model is a
    legitimate fallback, but real heatmaps are anisotropic -- a point on a jaw
    contour is far better localised across the contour than along it -- and
    that anisotropy is what makes a contour-derived measurement's interval
    honest.
    """

    index: dict[Landmark, int]
    covariances: NDArray[np.float64]

    @classmethod
    def isotropic(cls, ps: PointSet, sd: float) -> "LandmarkUncertainty":
        n, dim = ps.coords.shape[-2], ps.dim
        eye = np.broadcast_to(np.eye(dim) * sd**2, (n, dim, dim)).copy()
        return cls(index=dict(ps.index), covariances=eye)

    def sample(
        self, ps: PointSet, n_samples: int, rng: np.random.Generator
    ) -> PointSet:
        """Draw an ensemble of perturbed point sets."""
        base = ps.coords
        if base.ndim != 2:
            raise ValueError("sample() expects a single point set, not a batch")
        n, dim = base.shape
        out = np.empty((n_samples, n, dim))
        for name, i in ps.index.items():
            j = self.index.get(name)
            cov = self.covariances[j] if j is not None else np.zeros((dim, dim))
            out[:, i, :] = rng.multivariate_normal(base[i], cov, size=n_samples, method="cholesky")
        return PointSet(index=ps.index, coords=out)

    def positional_sd_for(self, names: Iterable[Landmark]) -> float:
        """Root-mean-square positional standard deviation over ``names`` only.

        Scoped to the landmarks a measurement actually reads. Pooling across the
        whole point set would let a handful of poorly localised contour points
        inflate the error term of every measurement in the catalogue -- the
        interpupillary distance would inherit the jawline's uncertainty and be
        withheld for a reason that has nothing to do with it.
        """
        idx = [self.index[n] for n in names if n in self.index]
        if not idx:
            return 0.0
        traces = np.trace(self.covariances[idx], axis1=-2, axis2=-1)
        return float(np.sqrt(np.mean(traces / self.covariances.shape[-1])))

    @property
    def mean_positional_sd(self) -> float:
        """Pooled across every landmark. Use :meth:`positional_sd_for` instead
        when evaluating a measurement; this is for whole-image quality summaries."""
        traces = np.trace(self.covariances, axis1=-2, axis2=-1)
        return float(np.sqrt(np.mean(traces / self.covariances.shape[-1])))


@dataclass(frozen=True)
class Measured:
    """One evaluated measurement, with everything needed to audit it."""

    spec_id: str
    label: str
    unit: Unit
    value: float
    ci_low: float
    ci_high: float
    sd: float
    verdict: Verdict
    discriminability: Discriminability | None
    formula_fingerprint: str
    landmarks_used: tuple[str, ...]
    n_samples: int
    n_valid: int
    scale_source: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def shown(self) -> bool:
        return self.verdict.shown

    @property
    def relative_ci_width(self) -> float:
        span = self.ci_high - self.ci_low
        denom = abs(self.value)
        return span / denom if denom > 1e-12 else float("inf")

    def format(self) -> str:
        """A one-line rendering that never hides the interval."""
        if self.verdict.reportability is Reportability.WITHHOLD:
            return f"{self.label}: withheld"
        u = "" if self.unit is Unit.RATIO else f" {self.unit.value}"
        flag = " (caveat)" if self.verdict.reportability is Reportability.CAVEAT else ""
        return (
            f"{self.label}: {self.value:.3g}{u} "
            f"[{self.ci_low:.3g} to {self.ci_high:.3g}]{flag}"
        )


class Unavailability(str, Enum):
    """Why a measurement could not be attempted at all."""

    MISSING_LANDMARKS = "missing_landmarks"
    NO_SCALE = "no_scale"
    DEGENERATE = "degenerate"


@dataclass(frozen=True)
class Unavailable:
    """A measurement that could not be attempted, and precisely why.

    Distinct from a withheld measurement: this says the input was not there,
    not that the number would be meaningless. The renderer writes a different
    sentence for each, so the cause is a typed field rather than free text
    smuggled through a list of landmark names.
    """

    spec_id: str
    label: str
    #: Kept third, where it has always been, so positional callers keep working.
    missing_landmarks: tuple[str, ...] = ()
    kind: Unavailability = Unavailability.MISSING_LANDMARKS
    detail: str = ""

    @property
    def reason(self) -> str:
        if self.kind is Unavailability.MISSING_LANDMARKS:
            return "the landmark model does not supply " + ", ".join(self.missing_landmarks)
        if self.kind is Unavailability.NO_SCALE:
            return "no scale reference is available in this image"
        return self.detail or "the formula is degenerate for this geometry"


def evaluate(
    spec: MeasurementSpec,
    ps: PointSet,
    uncertainty: LandmarkUncertainty,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    have_3d: bool,
    scale: ScaleEstimate | None = None,
    subject_distance_m: float | None = None,
    repeatability_cv: float | None = None,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> Measured | Unavailable:
    """Evaluate one measurement over a Monte-Carlo ensemble.

    Returns :class:`Unavailable` when the point set lacks a required landmark.
    That is a different outcome from a withheld measurement -- one says the
    model cannot see the landmark, the other says the number would not mean
    anything -- and the report distinguishes them.
    """
    missing = ps.missing(spec.landmarks)
    if missing:
        return Unavailable(spec.id, spec.label, tuple(m.value for m in missing))

    rng = np.random.default_rng(seed)
    ensemble = uncertainty.sample(ps, n_samples, rng)
    try:
        samples = np.asarray(spec.formula.eval(ensemble), dtype=float)
    except MissingLandmarkError as exc:  # pragma: no cover - guarded above
        return Unavailable(spec.id, spec.label, (exc.landmark.value,))

    finite = samples[np.isfinite(samples)]
    n_valid = int(finite.size)
    if n_valid < max(32, n_samples // 100):
        return Unavailable(
            spec.id,
            spec.label,
            kind=Unavailability.DEGENERATE,
            detail=f"only {n_valid} of {n_samples} Monte-Carlo samples were finite",
        )

    scale_notes: tuple[str, ...] = ()
    scale_source: str | None = None
    if spec.unit is Unit.MILLIMETRES:
        if scale is None:
            return Unavailable(spec.id, spec.label, kind=Unavailability.NO_SCALE)
        # Fold scale uncertainty in as a multiplicative factor rather than
        # applying the point estimate: the scale prior is often the dominant
        # error term for a millimetre value, and applying it as a constant
        # would make the interval look far tighter than it is.
        factors = rng.normal(scale.mm_per_px, scale.mm_per_px * scale.relative_sd, size=n_valid)
        finite = finite * factors
        scale_notes = scale.notes
        scale_source = scale.source.value

    value = float(np.median(finite))
    lo, hi = (float(x) for x in np.percentile(finite, [2.5, 97.5]))
    sd = float(np.std(finite))

    landmark_sd = uncertainty.positional_sd_for(spec.landmarks)
    relative_landmark_error = (
        landmark_sd / abs(value) if spec.unit is not Unit.DEGREES else
        # For an angle, express landmark scatter as the angle it subtends at a
        # typical facial baseline, so the units of the discriminability ratio
        # match the units of the published between-subject spread.
        float(np.degrees(np.arctan2(landmark_sd, 60.0)))
    )
    disc = assess_discriminability(
        spec,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        relative_landmark_error=abs(relative_landmark_error),
        relative_scale_error=scale.relative_sd if scale else 0.0,
    )
    verdict = decide_reportability(
        spec,
        max_pose_error_deg=max(abs(yaw_deg), abs(pitch_deg)),
        roll_deg=roll_deg,
        have_3d=have_3d,
        subject_distance_m=subject_distance_m,
        relative_ci_width=(hi - lo) / abs(value) if abs(value) > 1e-12 else float("inf"),
        repeatability_cv=repeatability_cv,
        disc=disc,
    )

    return Measured(
        spec_id=spec.id,
        label=spec.label,
        unit=spec.unit,
        value=value,
        ci_low=lo,
        ci_high=hi,
        sd=sd,
        verdict=verdict,
        discriminability=disc,
        formula_fingerprint=spec.fingerprint,
        landmarks_used=tuple(sorted(m.value for m in spec.landmarks)),
        n_samples=n_samples,
        n_valid=n_valid,
        scale_source=scale_source,
        notes=scale_notes,
    )
