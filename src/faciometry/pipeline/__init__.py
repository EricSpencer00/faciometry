"""Stage orchestration: photographs in, gated measurements and a manifest out.

The pipeline layer owns *sequence and accountability*, not policy. What a
measurement is, whether it may be shown, and how much a degree of yaw costs are
all decided in ``core`` and ``measure``; this package runs the stages in order,
records what each one did, and hands the measurement layer the evidence it needs
to make those decisions.

Nothing here imports torch. The model backends sit behind the narrow protocols
in :mod:`faciometry.pipeline.ports` and are constructed by the caller or loaded
lazily, so the whole orchestration layer -- including its end-to-end test -- runs
against fake implementations with no weights and no network.
"""

from __future__ import annotations

from .align import AlignedFace, Similarity, align, estimate_similarity
from .canonical import CanonicalFrame, roll_from_eyes, to_canonical
from .ingest import ExifFacts, Ruler, SourceImage, SubjectDistance, load_image, strip_exif
from .manifest import MeasurementRecord, ModelRecord, RunManifest
from .ports import Backends, Detection, HeadPose, LandmarkSet
from .quality import PoseEstimate, QualityIssue, QualityReport, Severity, assess
from .run import AnalysisResult, NoFaceFound, ViewAnalysis, analyze

__all__ = [
    "AlignedFace",
    "AnalysisResult",
    "Backends",
    "CanonicalFrame",
    "Detection",
    "ExifFacts",
    "HeadPose",
    "LandmarkSet",
    "MeasurementRecord",
    "ModelRecord",
    "NoFaceFound",
    "PoseEstimate",
    "QualityIssue",
    "QualityReport",
    "Ruler",
    "RunManifest",
    "Severity",
    "Similarity",
    "SourceImage",
    "SubjectDistance",
    "ViewAnalysis",
    "align",
    "analyze",
    "assess",
    "estimate_similarity",
    "load_image",
    "roll_from_eyes",
    "strip_exif",
    "to_canonical",
]
