"""The orchestrator: photographs in, measurements and a manifest out.

The stages are deliberately dull. Every interesting decision in Vitruve lives
somewhere else -- what a measurement is, in ``measure/registry``; whether a
number may be shown, in ``core/spec``; how much a pose costs, in
``core/sensitivity`` -- and this module's job is to run them in the right order
and to record what it did. When orchestration starts making judgements, the
judgements stop being reviewable, because they end up scattered across a
function nobody reads as policy.

Three things here are not dull, and each is here rather than downstream for a
reason.

**Frontal and profile are separate runs that meet once.** They have different
detections, different alignments, different quality, and different pose. Fusing
them earlier would mean inventing a joint geometry from two uncalibrated
cameras. They join at the measurement step, where the catalogue already knows
which view each measurement belongs to, and a missing profile costs exactly the
profile measurements and nothing else.

**The scale ladder is walked once, in one place.** A ruler wins outright; with
no ruler, the iris and the interpupillary priors are fused with their
correlation stated. Every millimetre in the report descends from this one
object, so there is one place to look when a millimetre value is surprising.

**Quality advises, it does not veto.** The per-measurement gate already knows
each measurement's tolerance and its between-person spread, and it is strictly
better informed than any global threshold. So quality feeds it -- pose, pose
uncertainty, subject distance -- and the only thing a global gate does on its
own is stop a run whose photograph could not support any measurement at all.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..core.landmarks import Landmark, PointSet
from ..core.scale import ScaleEstimate, from_interpupillary, from_iris, from_ruler, fuse
from ..core.spec import Reportability, View
from ..core.sensitivity import POSE_ESTIMATOR_SD_DEG
from ..measure import registry
from ..measure.evaluate import (
    DEFAULT_SAMPLES,
    LandmarkUncertainty,
    Measured,
    Unavailable,
    evaluate,
)
from ..measure.multishot import Combined, combine
from ..models.licensing import Tier
from .align import AlignedFace
from .align import align as align_face
from .canonical import CanonicalFrame, interocular_distance_px, roll_from_eyes, to_canonical
from .ingest import Ruler, SourceImage, SubjectDistance, estimate_subject_distance, load_image
from .manifest import ImageRecord, ModelRecord, RunManifest, StageTiming
from .ports import Backends, Detection, LandmarkSet, load_backends
from .quality import PoseEstimate, QualityIssue, QualityReport, Severity, reconcile_pose
from .quality import assess as assess_quality


#: How far apart the pose estimates of several captures may sit before they
#: stop describing one pose. Three times the estimator's own spread, so a
#: disagreement this large is not something the estimator's noise explains, and
#: pooling captures taken at genuinely different poses would average a face
#: photographed from two directions into a face that was never photographed.
CAPTURE_POSE_SPREAD_DEG = 3.0 * POSE_ESTIMATOR_SD_DEG

#: How far the captures may scatter, as a multiple of the landmark model's own
#: positional spread, before they stop looking like repeats of one face. The
#: pooling model in ``measure.multishot`` assumes the captures differ by
#: landmark noise; where they differ by much more than that, the assumption is
#: false and the reduction it promises is not there to be had.
CAPTURE_SCATTER_LIMIT = 2.0


class NoFaceFound(RuntimeError):
    """Raised when the detector returns nothing on an image it was given."""

    def __init__(self, view: View) -> None:
        super().__init__(
            f"no face was detected in the {view.value} photograph. Vitruve does not "
            "fall back to a whole-image crop: a measurement taken off a guessed face "
            "box is not a measurement."
        )
        self.view = view


class _Stages:
    """Wall-clock timing for each stage, in the order they ran."""

    def __init__(self) -> None:
        self._timings: list[StageTiming] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self._timings.append(StageTiming(name, time.perf_counter() - started))

    @property
    def timings(self) -> tuple[StageTiming, ...]:
        return tuple(self._timings)


@dataclass(frozen=True)
class ViewAnalysis:
    """Everything one photograph produced, before any measurement is taken."""

    view: View
    source: SourceImage
    detection_score: float
    aligned: AlignedFace
    landmarks_image: PointSet
    pose: PoseEstimate
    quality: QualityReport
    frame: CanonicalFrame
    subject_distance: SubjectDistance | None
    iris_diameter_px: float | None = None

    @property
    def interocular_px(self) -> float | None:
        return self.frame.interocular_px


@dataclass(frozen=True)
class AnalysisResult:
    """The outcome of one run.

    Note what is absent. There is no overall figure, no harmony index, no mean
    of the measurements and no rank. That is a design constraint of the project
    rather than an oversight, and it is asserted by a test: a single number is
    the part of this product class with no defensible measurement basis, and it
    is also the documented harm vector.
    """

    measured: tuple[Measured, ...]
    unavailable: tuple[Unavailable, ...]
    manifest: RunManifest
    frontal: ViewAnalysis | None = None
    profile: ViewAnalysis | None = None
    scale: ScaleEstimate | None = None
    failed: bool = False
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)
    #: Every frontal capture that produced landmarks, in the order supplied.
    #: ``frontal`` is the first of them and is the one whose pixels the scale
    #: ladder and the overlays are read from, so it is kept as its own field
    #: rather than left to be recovered by indexing.
    frontals: tuple[ViewAnalysis, ...] = field(default_factory=tuple)
    #: What pooling those captures achieved, or ``None`` when there was only
    #: one capture or when the captures disagreed too much to be pooled.
    pooled: Combined | None = None

    @property
    def n_captures(self) -> int:
        """Frontal photographs that went into the measured point set.

        The number *used*, not the number supplied: a capture the outlier
        rejection discarded did not contribute, and counting it would inflate
        every claim the report makes about what averaging bought.
        """
        return self.pooled.n_used if self.pooled is not None else 1

    @property
    def capture_note(self) -> str:
        return self.pooled.note() if self.pooled is not None else ""

    def by_id(self, spec_id: str) -> Measured | Unavailable | None:
        for m in self.measured:
            if m.spec_id == spec_id:
                return m
        for u in self.unavailable:
            if u.spec_id == spec_id:
                return u
        return None

    @property
    def reportable(self) -> tuple[Measured, ...]:
        """Measurements the gate is willing to show, caveats included."""
        return tuple(m for m in self.measured if m.verdict.shown)

    @property
    def withheld(self) -> tuple[Measured, ...]:
        """Evaluated and refused. A result in its own right, with its reasons."""
        return tuple(
            m for m in self.measured if m.verdict.reportability is Reportability.WITHHOLD
        )

    def quality_reports(self) -> tuple[QualityReport, ...]:
        return tuple(v.quality for v in (self.frontal, self.profile) if v is not None)


def analyze(
    frontal_path: str | Path | Sequence[str | Path],
    profile_path: str | Path | None = None,
    *,
    tier: Tier | str = Tier.PERMISSIVE,
    declared_sex: str | None = None,
    declared_ancestry: str | None = None,
    ruler_mm: Ruler | float | None = None,
    seed: int = 0,
    backends: Backends | None = None,
    n_samples: int = DEFAULT_SAMPLES,
    correct_roll: bool = True,
) -> AnalysisResult:
    """Run the whole pipeline over one or two photographs.

    ``frontal_path`` may be a sequence, in which case every capture is
    detected, landmarked and posed on its own and the results are pooled in the
    canonical frame before a single measurement pass. Pooling in the canonical
    frame rather than in image coordinates is the whole point: roll has been
    rotated out and the interocular span normalised by then, so what is
    averaged is one face rather than several framings of it. The captures must
    be of the same person in one session, and where their poses or their
    landmark positions disagree by more than the models' own noise explains,
    the run says so and measures the first capture alone.

    ``declared_sex`` and ``declared_ancestry`` are exactly that: declared. They
    select a normative stratum and narrow the interpupillary prior, they are
    optional, and Vitruve never infers them. An undeclared subject gets the
    pooled prior and a wider interval, which is the correct trade and not a
    degradation.

    ``ruler_mm`` collapses the scale assumption and accepts two forms. A
    :class:`~vitruve.pipeline.ingest.Ruler` is a physical reference of known
    length measured in the frame, which is what a clinical protocol asks for. A
    bare float is read as the subject's own interpupillary distance in
    millimetres, measured with a pupillometer or read off a spectacle
    prescription; it is applied against the pixel span the pipeline already
    measures between the pupils, and it removes the roughly 5.5% population
    prior that otherwise sits under every millimetre in the report.
    """
    tier = Tier[tier.upper()] if isinstance(tier, str) else tier
    stages = _Stages()

    if backends is None:
        with stages.stage("load_backends"):
            backends = load_backends(tier=tier, device=None)
    backends.check_licences(tier)

    manifest = RunManifest.begin(
        tier=tier,
        device=backends.device,
        seed=seed,
        n_samples=n_samples,
        declared_sex=declared_sex,
        declared_ancestry=declared_ancestry,
    ).with_(models=_model_records(backends))

    failure_reasons: list[str] = []

    frontal_paths = _frontal_paths(frontal_path)
    frontals: list[ViewAnalysis] = []
    for i, path in enumerate(frontal_paths):
        try:
            frontals.append(
                _analyse_view(
                    path,
                    View.FRONTAL,
                    backends,
                    stages,
                    correct_roll=correct_roll,
                    tag=f"frontal.{i}" if len(frontal_paths) > 1 else None,
                )
            )
        except NoFaceFound as exc:
            # One unusable capture out of several costs that capture. Failing
            # the run would throw away the photographs that did work over one
            # the user can simply drop.
            failure_reasons.append(str(exc))
    if not frontals:
        return _failed(manifest, stages, failure_reasons or [str(NoFaceFound(View.FRONTAL))])

    frontal, pooled = _pool_frontals(frontals, stages)

    profile: ViewAnalysis | None = None
    if profile_path is not None:
        try:
            profile = _analyse_view(
                profile_path, View.PROFILE, backends, stages, correct_roll=correct_roll
            )
        except NoFaceFound as exc:
            # A profile that cannot be used costs the profile measurements and
            # nothing else. Failing the whole run would throw away a perfectly
            # good frontal analysis over a photograph the user can simply retake.
            failure_reasons.append(str(exc))

    views = [v for v in (frontal, profile) if v is not None]
    manifest = manifest.with_(
        # Every frontal that produced landmarks is recorded, not just the one
        # the measurements were read against, so a reader can tie the pooled
        # point set back to each file that fed it.
        images=tuple(
            _image_record(v.source) for v in list(frontals) + ([profile] if profile else [])
        ),
        quality={v.view.value: _quality_dict(v) for v in views},
        captures=_captures_dict(frontals, pooled, frontal),
    )

    hard_fail = [v for v in views if v.quality.failed]
    if hard_fail:
        return _failed(
            manifest,
            stages,
            failure_reasons + [v.quality.failure_message() for v in hard_fail],
            frontal=frontal,
            profile=profile,
        )

    with stages.stage("scale"):
        scale = _build_scale(frontal, ruler_mm=ruler_mm, declared_sex=declared_sex)

    with stages.stage("measure"):
        measured, unavailable = _measure(
            frontal=frontal,
            profile=profile,
            scale=scale,
            seed=seed,
            n_samples=n_samples,
        )

    manifest = manifest.with_(
        stages=stages.timings,
        scale=_scale_dict(scale),
        measurements=RunManifest.records_for(measured, unavailable),
        failure_reasons=tuple(failure_reasons),
    )
    return AnalysisResult(
        measured=measured,
        unavailable=unavailable,
        manifest=manifest,
        frontal=frontal,
        profile=profile,
        scale=scale,
        failed=False,
        failure_reasons=tuple(failure_reasons),
        frontals=tuple(frontals),
        pooled=pooled,
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _analyse_view(
    path: str | Path,
    view: View,
    backends: Backends,
    stages: _Stages,
    *,
    correct_roll: bool,
    tag: str | None = None,
) -> ViewAnalysis:
    # ``tag`` only names the timings. Several frontal captures would otherwise
    # write several stages called "frontal.detect" and the manifest would say
    # how long detection took without saying which photograph it took it on.
    tag = tag or view.value

    with stages.stage(f"{tag}.ingest"):
        source = load_image(path, view=view)

    with stages.stage(f"{tag}.detect"):
        detections = list(backends.detector.detect(source.pixels))
        if not detections:
            raise NoFaceFound(view)
        detection = _pick_face(detections)

    with stages.stage(f"{tag}.align"):
        aligned = align_face(source.pixels, detection)

    with stages.stage(f"{tag}.landmark"):
        located: LandmarkSet = backends.landmarker.locate(source.pixels, detection)

    with stages.stage(f"{tag}.pose"):
        independent = (
            backends.pose_estimator.estimate(source.pixels, detection)
            if backends.pose_estimator is not None
            else None
        )
        pose = reconcile_pose(
            landmarker=located,
            independent=independent,
            geometric_roll_deg=(
                roll_from_eyes(located.points)
                if view is View.FRONTAL
                else None
            ),
            landmarker_name=backends.landmarker.provenance.name,
            independent_name=(
                backends.pose_estimator.provenance.name
                if backends.pose_estimator is not None
                else "head-pose network"
            ),
        )

    iris_px: float | None = None
    if backends.iris is not None:
        with stages.stage(f"{tag}.iris"):
            iris_px = _iris_diameter(backends, source.pixels, detection)

    interocular_px = interocular_distance_px(located.points)
    with stages.stage(f"{tag}.distance"):
        subject_distance = estimate_subject_distance(
            source.exif,
            feature_px=_pupil_distance_px(located.points),
            image_diagonal_px=source.diagonal_px,
        )

    with stages.stage(f"{tag}.quality"):
        report = assess_quality(
            view=view,
            aligned_pixels=aligned.pixels,
            # ICAO's minimum is stated against the original capture, so the
            # canonical crop's fixed 512 pixels would answer a question nobody
            # asked. This is the number in the photograph the user took, and it
            # is None on a profile view, which has no interocular axis.
            interocular_px=interocular_px,
            face_height_px=float(detection.bbox[3]),
            pose=pose,
            subject_distance=subject_distance,
            detection_score=float(detection.score),
        )

    with stages.stage(f"{tag}.canonical"):
        frame = to_canonical(
            view,
            located.points,
            np.asarray(located.covariances, dtype=float),
            pose,
            correct_roll=correct_roll,
        )

    return ViewAnalysis(
        view=view,
        source=source,
        detection_score=float(detection.score),
        aligned=aligned,
        landmarks_image=located.points,
        pose=pose,
        quality=report,
        frame=frame,
        subject_distance=subject_distance,
        iris_diameter_px=iris_px,
    )


# ---------------------------------------------------------------------------
# Several captures of one face
# ---------------------------------------------------------------------------


def _frontal_paths(given: str | Path | Sequence[str | Path]) -> tuple[str | Path, ...]:
    """One path or many, always as a tuple.

    ``str`` and ``Path`` are both sequences of a sort, so the check is by type
    rather than by ``isinstance(given, Sequence)``, which would turn a single
    filename into a list of its characters.
    """
    if isinstance(given, (str, Path)):
        return (given,)
    return tuple(given)


def _rescaled_frame(frame: CanonicalFrame, reference: CanonicalFrame) -> CanonicalFrame:
    """One capture's canonical frame in the reference capture's pixel units.

    Canonical conversion is a rigid motion, so two captures taken at different
    distances arrive with the same face at different pixel sizes and averaging
    them would average a large face with a small one. Normalising the
    interocular span is the one similarity term that closes that gap, and doing
    it against the *reference* capture rather than against a mean keeps the
    scale ladder honest: every millimetre in the report descends from the pupil
    span measured in one real photograph, not in a synthetic average of
    several.
    """
    here, there = frame.interocular_px, reference.interocular_px
    if not here or not there or here <= 0:
        return frame
    s = there / here
    return replace(
        frame,
        points=PointSet(
            index=dict(frame.points.index),
            coords=np.asarray(frame.points.coords, dtype=float) * s,
        ),
        uncertainty=LandmarkUncertainty(
            index=dict(frame.uncertainty.index),
            covariances=np.asarray(frame.uncertainty.covariances, dtype=float) * (s * s),
        ),
        interocular_px=there,
    )


def _pose_spread_deg(views: Sequence[ViewAnalysis]) -> float:
    """The widest disagreement between the captures on any pose axis."""
    axes = (
        [v.pose.yaw_deg for v in views],
        [v.pose.pitch_deg for v in views],
        [v.pose.roll_deg for v in views],
    )
    return max(float(max(a) - min(a)) for a in axes)


def _capture_scatter(
    frames: Sequence[CanonicalFrame], originals: Sequence[CanonicalFrame]
) -> tuple[float, float] | None:
    """How far the captures scatter, against how far the models said they would.

    The pooling model treats the captures as one face seen through independent
    landmark noise. If that is true, a landmark's spread across captures is the
    size of what the models themselves account for. If it is much larger, the
    captures are not repeats: they are different faces, different expressions,
    or a landmark model that failed on one of them, and averaging them would
    produce a point set describing none of them.

    Two things go into what the models account for, and leaving the second one
    out was enough to make this guard refuse every honest set of captures.

    The first is the landmark model's own positional spread. The second is the
    alignment that brought the captures into a common frame, which is read off
    the two eye landmarks and therefore carries their noise: the roll
    correction rotates each capture by a slightly different angle, and the
    interocular normalisation scales each one by a slightly different factor.
    With a positional spread of ``sd`` on each endpoint of an interocular span
    ``D``, both of those have a spread of about ``sd * sqrt(2) / D``, and both
    move a landmark in proportion to its distance ``r`` from the eye midpoint,
    so together they contribute about ``2 * r * sd / D``. On a face measured
    two hundred pixels out from the eyes that is larger than the landmark term
    it sits beside, and omitting it made this guard refuse every honest set of
    captures.

    Both terms are taken from the spread the landmark model *claims*, never
    from the spread the captures were *observed* to have. Deriving the second
    from the first would let a set of captures explain its own disagreement,
    and the guard would then pass anything.

    Returns ``(observed, accounted for)`` in pixels, or ``None`` when the
    captures share too few landmarks for the comparison to mean anything.
    """
    common = set(frames[0].points.available)
    for f in frames[1:]:
        common &= set(f.points.available)
    if len(common) < 3:
        return None
    names = sorted(common, key=lambda m: m.value)

    stack = np.stack([np.stack([f.points.get(n) for n in names]) for f in frames])
    # Standard deviation across captures, per landmark, pooled over coordinates.
    observed = np.sqrt((stack.std(axis=0, ddof=1) ** 2).mean(axis=-1))

    model_sd = []
    for f in frames:
        cov = np.asarray(f.uncertainty.covariances, dtype=float)
        rows = [f.uncertainty.index[n] for n in names]
        model_sd.append(np.sqrt(np.trace(cov[rows], axis1=-2, axis2=-1) / cov.shape[-1]))

    per_landmark = np.median(np.stack(model_sd), axis=0)
    spans = [f.interocular_px for f in originals if f.interocular_px]
    span = float(np.median(spans)) if spans else 0.0
    # One factor of sqrt(2) for the two endpoints the eye axis is read from,
    # another for the roll and the scale entering independently.
    alignment = (
        2.0 * float(np.median(per_landmark)) / span * np.linalg.norm(
            stack.mean(axis=0), axis=-1
        )
        if span > 0
        else 0.0
    )
    accounted = np.sqrt(per_landmark**2 + alignment**2)

    return float(np.median(observed)), float(np.median(accounted))


def _capture_issues(
    views: Sequence[ViewAnalysis], frames: Sequence[CanonicalFrame]
) -> tuple[tuple[QualityIssue, ...], bool]:
    """What the captures say about each other, and whether to pool them.

    Never ``Severity.FAIL``. A set of captures that cannot be pooled is not a
    photograph that cannot be measured: the first capture is still a perfectly
    good single-photograph analysis, and failing the run would take that away
    over a mistake the user made in *addition* to taking a usable picture.
    """
    issues: list[QualityIssue] = []
    agreed = True

    spread = _pose_spread_deg(views)
    if spread > CAPTURE_POSE_SPREAD_DEG:
        agreed = False
        issues.append(
            QualityIssue(
                code="captures.pose_spread",
                severity=Severity.WARN,
                message=(
                    f"The {len(views)} frontal captures disagree by {spread:.1f} "
                    f"degrees of head pose, further than the {CAPTURE_POSE_SPREAD_DEG:.1f} "
                    "degrees the pose estimator's own noise accounts for. They "
                    "were not photographs of one pose, so they were not pooled "
                    "and the first capture was measured on its own."
                ),
                remedy=(
                    "Captures pooled together have to be of one person holding "
                    "one pose in one session."
                ),
            )
        )
    else:
        issues.append(
            QualityIssue(
                code="captures.pose_spread",
                severity=Severity.INFO,
                message=(
                    f"The {len(views)} frontal captures agree on head pose to "
                    f"within {spread:.1f} degrees."
                ),
                remedy="",
            )
        )

    scatter = _capture_scatter(frames, [v.frame for v in views])
    if scatter is not None:
        observed, claimed = scatter
        ratio = observed / claimed if claimed > 0 else float("inf")
        if ratio > CAPTURE_SCATTER_LIMIT:
            agreed = False
            issues.append(
                QualityIssue(
                    code="captures.scatter",
                    severity=Severity.WARN,
                    message=(
                        f"Landmarks move {observed:.1f} px between the captures, "
                        f"{ratio:.1f} times the {claimed:.1f} px the landmark model "
                        "and the alignment between them account for. Captures that "
                        "differ by more than their own noise are not repeats of one "
                        "face, so they were not pooled and the first capture was "
                        "measured on its own."
                    ),
                    remedy=(
                        "Captures pooled together have to be of one person in one "
                        "session, with the same expression."
                    ),
                )
            )

    return tuple(issues), agreed


def _with_issues(
    report: QualityReport, issues: Sequence[QualityIssue]
) -> QualityReport:
    return replace(report, issues=tuple(report.issues) + tuple(issues))


def _pool_frontals(
    views: list[ViewAnalysis], stages: _Stages
) -> tuple[ViewAnalysis, Combined | None]:
    """Pool several frontal captures into one measurable frame.

    Returns the capture the rest of the run reads from, which is the first one
    with its canonical frame replaced by the pooled point set, plus the record
    of what pooling achieved. Pose for the pooled frame is the median across
    captures rather than the first capture's, because a median is the estimate
    that the outlier rejection downstream is already built around.
    """
    reference = views[0]
    if len(views) == 1:
        return reference, None

    with stages.stage("frontal.pool"):
        frames = [_rescaled_frame(v.frame, reference.frame) for v in views]
        issues, agreed = _capture_issues(views, frames)
        quality = _with_issues(reference.quality, issues)
        if not agreed:
            return replace(reference, quality=quality), None

        pooled = combine([(f.points, f.uncertainty) for f in frames])
        frame = replace(
            reference.frame,
            points=pooled.points,
            uncertainty=pooled.uncertainty,
            yaw_deg=float(np.median([f.yaw_deg for f in frames])),
            pitch_deg=float(np.median([f.pitch_deg for f in frames])),
            roll_deg=float(np.median([f.roll_deg for f in frames])),
            notes=(*reference.frame.notes, pooled.note()),
        )
        return replace(reference, frame=frame, quality=quality), pooled


def _captures_dict(
    frontals: Sequence[ViewAnalysis],
    pooled: Combined | None,
    reference: ViewAnalysis,
) -> dict:
    """What the manifest records about the frontal captures.

    Always written, including for a single photograph, because "one capture,
    no averaging possible" is a fact about the run and a reader should not have
    to infer it from a missing key.
    """
    if pooled is None:
        return {
            "view": "frontal",
            "n_supplied": len(frontals),
            "n_used": 1,
            "dropped": [],
            "shared_fraction": None,
            "pooled": False,
            "note": (
                "a single photograph, so no averaging was possible"
                if len(frontals) == 1
                else "the captures did not agree closely enough to be pooled, so "
                "the first was measured on its own"
            ),
            "pose_spread_deg": (
                None if len(frontals) < 2 else _pose_spread_deg(frontals)
            ),
            "interocular_px": [v.interocular_px for v in frontals],
        }
    return {
        "view": "frontal",
        "n_supplied": pooled.n_supplied,
        "n_used": pooled.n_used,
        "dropped": list(pooled.dropped),
        "shared_fraction": pooled.shared_fraction,
        "pooled": True,
        "note": pooled.note(),
        "error_factor": pooled.error_factor,
        "effective_n": pooled.effective_n,
        "pose_spread_deg": _pose_spread_deg(frontals),
        "interocular_px": [v.interocular_px for v in frontals],
        "reference_interocular_px": reference.interocular_px,
    }


def _pick_face(detections: Sequence[Detection]) -> Detection:
    """The largest confident face, which for a portrait is the subject.

    Score alone picks a small sharp bystander over a large slightly-turned
    subject. Area alone picks a blurred foreground shoulder. The product of the
    two is the usual compromise and it is stated here rather than assumed.
    """

    def weight(d: Detection) -> float:
        _, _, w, h = d.bbox
        return float(d.score) * float(w) * float(h)

    return max(detections, key=weight)


def _iris_diameter(backends: Backends, pixels, detection: Detection) -> float | None:
    """Mean visible iris diameter across the eyes that could be measured.

    A dense frontal mesh model cannot find a face in a true profile, and it
    signals that by raising rather than returning nothing. That is a normal
    outcome for the profile view, not a failure of the run: the profile
    contributes angles, which need no scale at all, and any millimetre value it
    does produce can take the scale recovered from the frontal photograph.

    Letting the exception escape aborted the entire two-photograph analysis,
    including the frontal view that had already succeeded, which is the one
    capture protocol a full report asks for.
    """
    assert backends.iris is not None
    try:
        measurement = backends.iris.measure_iris(pixels, detection)
    except Exception:
        # Deliberately broad: every dense-mesh backend signals "no face here"
        # differently, and none of those ways should end the run.
        return None
    if measurement is None:
        return None
    values = [
        v
        for v in (measurement.iris_diameter_px_l, measurement.iris_diameter_px_r)
        if v is not None and v > 0
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _pupil_distance_px(points: PointSet) -> float | None:
    """Pupil-to-pupil distance, or ``None``.

    Distinct from :func:`canonical.interocular_distance_px`, which falls back to
    the canthi when a backend has no pupil centres. That fallback is fine for
    judging face size and wrong for the interpupillary prior, whose population
    mean is about pupils and nothing else. Silently feeding a canthal span into
    an interpupillary prior would put a systematic error into every millimetre
    in the report.
    """
    if not points.has(Landmark.PUPIL_L, Landmark.PUPIL_R):
        return None
    return float(np.linalg.norm(points.get(Landmark.PUPIL_L) - points.get(Landmark.PUPIL_R)))


def _build_scale(
    frontal: ViewAnalysis,
    *,
    ruler_mm: Ruler | float | None,
    declared_sex: str | None,
) -> ScaleEstimate | None:
    """Walk the scale ladder from most to least trusted cue.

    Returns ``None`` when nothing in the frame fixes a size. Every
    millimetre-valued measurement then comes back as unavailable with "no scale
    reference in the image", which is the honest outcome, and the ratios and
    angles -- the outputs this project prefers anyway -- are unaffected.
    """
    pupil_px = _pupil_distance_px(frontal.landmarks_image)

    if isinstance(ruler_mm, Ruler):
        return from_ruler(
            px_per_known_mm=ruler_mm.pixel_span,
            known_mm=ruler_mm.known_mm,
            reading_error_mm=ruler_mm.reading_error_mm,
        )
    if ruler_mm is not None:
        if pupil_px is None:
            return None
        return from_ruler(
            px_per_known_mm=pupil_px,
            known_mm=float(ruler_mm),
            # A pupillometer reading or a spectacle prescription is quoted to
            # the nearest half millimetre, which is a far tighter reference than
            # a ruler held up beside the face and read off a photograph.
            reading_error_mm=0.5,
        )

    cues: list[ScaleEstimate] = []
    if frontal.iris_diameter_px:
        cues.append(from_iris(frontal.iris_diameter_px))
    if pupil_px:
        cues.append(
            from_interpupillary(
                pupil_px,
                declared_sex=declared_sex,
                subject_distance_m=(
                    frontal.subject_distance.metres if frontal.subject_distance else None
                ),
            )
        )
    if not cues:
        return None
    return fuse(*cues)


def _measure(
    *,
    frontal: ViewAnalysis | None,
    profile: ViewAnalysis | None,
    scale: ScaleEstimate | None,
    seed: int,
    n_samples: int,
) -> tuple[tuple[Measured, ...], tuple[Unavailable, ...]]:
    """Evaluate every catalogue spec against whichever view owns it.

    All specs share one seed, which means one consistent Monte-Carlo realisation
    of the landmark uncertainty across the whole catalogue: two measurements
    that read the same landmark see the same perturbation of it. That is what
    makes their intervals comparable, and it costs nothing, since each spec
    draws its own ensemble from the same generator state.
    """
    measured: list[Measured] = []
    unavailable: list[Unavailable] = []
    done: set[str] = set()

    for view, analysis in ((View.FRONTAL, frontal), (View.PROFILE, profile)):
        for spec in registry.for_view(view):
            if spec.id in done:
                # A View.EITHER spec belongs to whichever view ran first, which
                # is the frontal one. Evaluating it twice would put two rows
                # with the same id and different numbers into one report.
                continue
            done.add(spec.id)

            if analysis is None:
                unavailable.append(
                    # Keyword form on purpose. ``Unavailable`` has gained
                    # fields between the middle two positions before, and a
                    # positional call silently lands a tuple of landmark names
                    # in whatever field is third that week.
                    Unavailable(
                        spec.id,
                        spec.label,
                        missing_landmarks=(
                            f"any {view.value} landmark, because no {view.value} "
                            "photograph was supplied",
                        ),
                    )
                )
                continue

            frame = analysis.frame
            if not spec.landmarks <= frame.points.available:
                unavailable.append(
                    Unavailable(
                        spec.id,
                        spec.label,
                        missing_landmarks=tuple(
                            m.value for m in frame.points.missing(spec.landmarks)
                        ),
                    )
                )
                continue

            outcome = evaluate(
                spec,
                frame.points,
                frame.uncertainty,
                yaw_deg=frame.yaw_deg,
                pitch_deg=frame.pitch_deg,
                roll_deg=frame.roll_deg,
                have_3d=frame.have_3d,
                scale=scale,
                subject_distance_m=analysis.quality.subject_distance_m,
                # No test-retest data exists for a single photograph, and the
                # catalogue's own within-person figures already stand in for it
                # where a study has measured one.
                repeatability_cv=None,
                n_samples=n_samples,
                seed=seed,
            )
            if isinstance(outcome, Unavailable):
                unavailable.append(outcome)
            else:
                measured.append(outcome)

    measured.sort(key=lambda m: m.spec_id)
    unavailable.sort(key=lambda u: u.spec_id)
    return tuple(measured), tuple(unavailable)


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def _model_records(backends: Backends) -> tuple[ModelRecord, ...]:
    return tuple(
        ModelRecord.from_provenance(
            role, backend.provenance, getattr(backend, "weights_sha256", None)
        )
        for role, backend in backends.all()
    )


def _image_record(source: SourceImage) -> ImageRecord:
    return ImageRecord(
        view=source.view.value,
        sha256=source.sha256,
        width=source.width,
        height=source.height,
        filename=source.path.name if source.path else None,
        exif_focal_length_35mm=source.exif.focal_length_35mm,
        exif_subject_distance_m=source.exif.subject_distance_m,
        exif_tags_present=source.exif.tag_names_seen,
        exif_stripped=True,
    )


def _quality_dict(v: ViewAnalysis) -> dict:
    """The quality report as JSON-native values, for the manifest.

    Plain types only, no tuples and no enums, because the manifest has to
    survive a round trip through JSON and come back equal to what went in.
    """
    q = v.quality
    return {
        "detection_score": q.detection_score,
        "interocular_px": q.interocular_px,
        "face_height_px": q.face_height_px,
        "blur_laplacian_variance": q.blur_laplacian_variance,
        "clipped_dark_fraction": q.clipped_dark_fraction,
        "clipped_bright_fraction": q.clipped_bright_fraction,
        "subject_distance_m": q.subject_distance_m,
        "subject_distance_source": (
            q.subject_distance.source if q.subject_distance else None
        ),
        "magnification_distortion": q.magnification_distortion,
        "pose": {
            "yaw_deg": q.pose.yaw_deg,
            "pitch_deg": q.pose.pitch_deg,
            "roll_deg": q.pose.roll_deg,
            "yaw_sd_deg": q.pose.yaw_sd_deg,
            "pitch_sd_deg": q.pose.pitch_sd_deg,
            "roll_sd_deg": q.pose.roll_sd_deg,
            "gated_yaw_deg": q.pose.gated_yaw_deg,
            "gated_pitch_deg": q.pose.gated_pitch_deg,
            "gated_roll_deg": q.pose.gated_roll_deg,
            "sources": list(q.pose.sources),
            "notes": list(q.pose.notes),
        },
        "pose_agreement": (
            {
                "sources": list(q.pose.agreement.sources),
                "yaw_delta_deg": q.pose.agreement.yaw_delta_deg,
                "pitch_delta_deg": q.pose.agreement.pitch_delta_deg,
                "roll_delta_deg": q.pose.agreement.roll_delta_deg,
                "max_delta_deg": q.pose.agreement.max_delta_deg,
            }
            if q.pose.agreement is not None
            else None
        ),
        "applied_roll_deg": v.frame.applied_roll_deg,
        "residual_roll_deg": v.frame.roll_deg,
        "have_3d": v.frame.have_3d,
        "issues": [
            {
                "code": i.code,
                "severity": i.severity.value,
                "message": i.message,
                "remedy": i.remedy,
            }
            for i in q.issues
        ],
        "worst_severity": q.worst_severity.value if q.worst_severity else None,
        "frame_notes": list(v.frame.notes),
    }


def _scale_dict(scale: ScaleEstimate | None) -> dict:
    if scale is None:
        return {
            "source": "none",
            "mm_per_px": None,
            "relative_sd": None,
            "notes": [
                "nothing in the frame fixed a physical size, so no millimetre "
                "value was produced"
            ],
        }
    return {
        "source": scale.source.value,
        "mm_per_px": scale.mm_per_px,
        "relative_sd": scale.relative_sd,
        "relative_ci95": scale.relative_ci95,
        "notes": list(scale.notes),
    }


def _failed(
    manifest: RunManifest,
    stages: _Stages,
    reasons: Sequence[str],
    *,
    frontal: ViewAnalysis | None = None,
    profile: ViewAnalysis | None = None,
) -> AnalysisResult:
    """Stop the run, saying what went wrong and what to change.

    The manifest is still written. A run that produced no measurements is
    exactly the case where a reader most needs to know what was attempted, with
    what models, on which pixels.
    """
    reasons = tuple(r for r in reasons if r)
    return AnalysisResult(
        measured=(),
        unavailable=(),
        manifest=manifest.with_(stages=stages.timings, failure_reasons=reasons),
        frontal=frontal,
        profile=profile,
        scale=None,
        failed=True,
        failure_reasons=reasons,
    )


__all__ = [
    "AnalysisResult",
    "NoFaceFound",
    "ViewAnalysis",
    "analyze",
]
