"""Where a measurement's error comes from, and what would remove it.

A report that withholds a number and prints four reasons has told the reader
what happened without telling them what to do. This module answers the second
question, and it answers it per photograph rather than in general, because the
ranking changes: a close-range selfie is dominated by perspective, a distant
one by landmark noise, and an uncalibrated one by the scale prior no matter
what else is true.

The three error terms combine in quadrature, so their contributions to the
variance are what matter and a term at half the size of another contributes a
quarter as much. That is why "your head was turned" can be the loudest sentence
in a report and account for one percent of the problem.

The counterfactuals are exact rather than advisory. Each one substitutes a
single term and recomputes the same ratio the gate uses, so "four photographs
would report this" is a statement about arithmetic already performed and not an
encouragement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .multishot import DEFAULT_SHARED_FRACTION


class Lever(str, Enum):
    """Something a person can actually change about how they photograph."""

    RULER = "ruler"
    REPEATS = "repeats"
    RESOLUTION = "resolution"
    POSE = "pose"
    DISTANCE = "distance"
    DECLARE_SEX = "declare_sex"


#: What each lever does to which term, and what it costs the user.
LEVER_TEXT: dict[Lever, tuple[str, str]] = {
    Lever.RULER: (
        "photograph a ruler held in the plane of the face",
        "removes the scale prior entirely, because the millimetre becomes an "
        "observation rather than a population average",
    ),
    Lever.REPEATS: (
        "take several photographs and analyse them together",
        "the independent part of landmark noise averages away across captures. "
        "It does not fall as the square root of N, because a landmark model "
        "makes much the same mistake on much the same photograph, so a shared "
        "fraction survives no matter how many are taken. Varying the distance "
        "and lighting between shots is what lowers that fraction",
    ),
    Lever.RESOLUTION: (
        "fill more of the frame with the face, or use a longer lens",
        "landmark error scales with the face's pixel size, so doubling the "
        "inter-ocular span halves it",
    ),
    Lever.POSE: (
        "look straight at the lens and keep the camera at eye height",
        "removes the projection term, which matters most for widths and for any "
        "ratio whose two terms sit at different depths",
    ),
    Lever.DISTANCE: (
        "stand further back and zoom in",
        "perspective magnification between the eye and nose planes falls roughly "
        "as one over distance, from about 17 percent at 0.3 m to 3 percent at 1.5 m",
    ),
    Lever.DECLARE_SEX: (
        "state your sex so the scale prior can be conditioned",
        "narrows the interpupillary prior from the pooled 63.36 mm to a "
        "sex-specific one, which is a small gain and the only one on this list "
        "that costs a disclosure",
    ),
}


@dataclass(frozen=True)
class Term:
    """One contributor to a measurement's error."""

    name: str
    value: float
    lever: Lever | None

    def share_of(self, total: float) -> float:
        """Fraction of the *variance*, which is what quadrature makes relevant."""
        return (self.value**2) / (total**2) if total > 0 else 0.0


@dataclass(frozen=True)
class Counterfactual:
    """The gate's own ratio, recomputed with one term changed."""

    lever: Lever
    detail: str
    error: float
    ratio: float
    reports: bool

    @property
    def action(self) -> str:
        return LEVER_TEXT[self.lever][0]

    @property
    def because(self) -> str:
        return LEVER_TEXT[self.lever][1]


@dataclass(frozen=True)
class Budget:
    """The full decomposition for one measurement on one photograph."""

    spec_id: str
    label: str
    spread: float
    terms: tuple[Term, ...]
    counterfactuals: tuple[Counterfactual, ...]

    @property
    def total(self) -> float:
        return math.sqrt(sum(t.value**2 for t in self.terms))

    @property
    def ratio(self) -> float:
        return self.spread / self.total if self.total > 0 else math.inf

    @property
    def dominant(self) -> Term | None:
        return max(self.terms, key=lambda t: t.value, default=None)

    @property
    def sufficient(self) -> tuple[Counterfactual, ...]:
        """Only the changes that would actually carry this measurement over."""
        return tuple(c for c in self.counterfactuals if c.reports)


def _q(*values: float) -> float:
    return math.sqrt(sum(v * v for v in values))


def repeat_factor(n: int, *, shared_fraction: float = DEFAULT_SHARED_FRACTION) -> float:
    """What averaging ``n`` captures does to the landmark error.

    The same correlated-average model `multishot.combine` applies, imported
    rather than re-derived. Using the naive one-over-root-N here would have the
    report promise a reduction the feature cannot deliver: at nine captures the
    honest factor is about 0.65, not 0.33.
    """
    n = max(int(n), 1)
    return math.sqrt(shared_fraction + (1.0 - shared_fraction) / n)


def budget_for(
    *,
    spec_id: str,
    label: str,
    spread: float,
    pose_error: float,
    landmark_error: float,
    scale_error: float,
    scale_is_measured: bool,
    sex_declared: bool,
    repeats: int = 1,
    max_repeats: int = 9,
) -> Budget:
    """Decompose one measurement's error and price each available lever.

    ``repeats`` is how many captures already went into this measurement, so the
    repeat counterfactual proposes going beyond it rather than restating it.
    """
    terms = (
        Term("scale prior", scale_error, None if scale_is_measured else Lever.RULER),
        Term("landmark placement", landmark_error, Lever.REPEATS),
        Term("head pose", pose_error, Lever.POSE),
    )
    total = _q(*(t.value for t in terms))
    if total <= 0:
        return Budget(spec_id, label, spread, terms, ())

    out: list[Counterfactual] = []

    def offer(lever: Lever, detail: str, err: float) -> None:
        if err >= total - 1e-12:
            return  # would not improve anything; do not suggest it
        out.append(
            Counterfactual(
                lever=lever,
                detail=detail,
                error=err,
                ratio=spread / err if err > 0 else math.inf,
                reports=(spread / err if err > 0 else math.inf) > 1.0,
            )
        )

    if not scale_is_measured and scale_error > 0:
        offer(Lever.RULER, "a ruler in the plane of the face", _q(landmark_error, pose_error))
        if not sex_declared:
            # Sex conditioning narrows the interpupillary prior by roughly a
            # tenth. Offered last because it is the only lever that costs a
            # disclosure, and it is never enough on its own.
            offer(Lever.DECLARE_SEX, "sex declared", _q(scale_error * 0.9, landmark_error, pose_error))

    if pose_error > 0:
        offer(Lever.POSE, "square to the lens", _q(scale_error, landmark_error, pose_error * 0.2))

    for n in (4, max_repeats):
        if n > repeats:
            offer(
                Lever.REPEATS,
                f"{n} photographs analysed together",
                _q(scale_error, landmark_error * repeat_factor(n) / repeat_factor(repeats), pose_error),
            )

    # The combination a user would most plausibly reach for, offered once.
    best_landmark = landmark_error * repeat_factor(max_repeats) / repeat_factor(repeats)
    combined = _q(0.0 if not scale_is_measured else scale_error, best_landmark, pose_error * 0.2)
    if combined < total:
        out.append(
            Counterfactual(
                lever=Lever.REPEATS,
                detail=f"a ruler, square to the lens, and {max_repeats} photographs together",
                error=combined,
                ratio=spread / combined if combined > 0 else math.inf,
                reports=(spread / combined if combined > 0 else math.inf) > 1.0,
            )
        )

    out.sort(key=lambda c: c.error)
    return Budget(spec_id, label, spread, terms, tuple(out))


def ranked_levers(budgets: list[Budget]) -> list[tuple[Lever, int, str]]:
    """Across a whole report, which change would recover the most measurements.

    Answers the question a reader actually has after seeing a page of withheld
    measurements, which is not "why" but "what do I do differently".
    """
    gained: dict[Lever, int] = {}
    for b in budgets:
        if b.ratio > 1.0:
            continue  # already reporting; a lever cannot recover it
        for lever in {c.lever for c in b.sufficient}:
            gained[lever] = gained.get(lever, 0) + 1
    return sorted(
        ((lever, n, LEVER_TEXT[lever][0]) for lever, n in gained.items()),
        key=lambda row: -row[1],
    )
