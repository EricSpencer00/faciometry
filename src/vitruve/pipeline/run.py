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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.landmarks import Landmark, PointSet
from ..core.scale import ScaleEstimate, from_interpupillary, from_iris, from_ruler, fuse
from ..core.spec import Reportability, View
from ..measure import registry
from ..measure.evaluate import DEFAULT_SAMPLES, Measured, Unavailable, evaluate
from ..models.licensing import Tier
from .align import AlignedFace
from .align import align as align_face
from .canonical import CanonicalFrame, interocular_distance_px, roll_from_eyes, to_canonical
from .ingest import Ruler, SourceImage, SubjectDistance, estimate_subject_distance, load_image
from .manifest import ImageRecord, ModelRecord, RunManifest, StageTiming
from .ports import Backends, Detection, LandmarkSet, load_backends
from .quality import PoseEstimate, QualityReport, reconcile_pose
from .quality import assess as assess_quality


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
    frontal_path: str | Path,
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

    try:
        frontal = _analyse_view(
            frontal_path, View.FRONTAL, backends, stages, correct_roll=correct_roll
        )
    except NoFaceFound as exc:
        return _failed(manifest, stages, [str(exc)])

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
        images=tuple(_image_record(v.source) for v in views),
        quality={v.view.value: _quality_dict(v) for v in views},
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
) -> ViewAnalysis:
    tag = view.value

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
    """Mean visible iris diameter across the eyes that could be measured."""
    assert backends.iris is not None
    measurement = backends.iris.measure_iris(pixels, detection)
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
