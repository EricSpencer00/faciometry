"""The accountability artifact: why every number appeared, or did not.

Every run writes a ``run.json``. It is not a log and it is not telemetry. It is
the record a second person needs in order to work out, months later and without
the original machine, why a particular measurement carried the value it did, or
why it carried no value at all.

Which fixes what has to be in it. The formula fingerprint, because a definition
can change and a number attributed to the wrong definition is worse than no
number. The landmarks each measurement consumed, because that is the shortest
path from a suspicious value to the point that caused it. The verdict *and its
reasons*, because "withheld" without the reason is indistinguishable from a
crash. The weight hashes, because a silently substituted checkpoint is the
classic irreproducibility. The licence tier and the obligations it carries,
because a user who ran at ``copyleft`` has taken on a duty and should be able to
find out that they did. The seed, because the intervals are Monte-Carlo.

One rule shapes the schema more than any other. A withheld measurement's value
is **not** recorded. Rule three of the core contract says to print the reason and
never the number, and a manifest that stored the number would leave that rule one
`jq` invocation from being broken. The record says what was asked, what was used,
what was decided, and why. It does not say what the answer would have been.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.spec import Reportability, Unit, View
from ..measure.evaluate import Measured, Unavailable
from ..measure.registry import BY_ID
from ..models.licensing import Provenance, Tier, obligations_at

#: Bumped whenever a field changes meaning. A reader that does not recognise the
#: version should refuse to interpret the file rather than guess at it.
SCHEMA_VERSION = 1

_UNSET = "unknown"


def _git_sha(start: Path | None = None) -> str | None:
    """Commit of the working tree this code was loaded from, if it is a checkout.

    Best effort and silent on failure: an installed wheel has no git directory,
    and that is a normal way to run. ``None`` is recorded rather than a
    placeholder so a reader can tell "not a checkout" from "checkout at HEAD".
    """
    root = Path(start or Path(__file__).resolve().parent)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("faciometry")
    except Exception:
        # Running from a source tree that was never installed. Saying so beats
        # inventing a version number that no artifact will ever match.
        return "0+source"


@dataclass(frozen=True)
class ModelRecord:
    """One backend, with everything needed to obtain the same one again."""

    role: str
    name: str
    license_id: str
    tier: str
    source_url: str
    weights_sha256: str | None = None
    inherited_from: tuple[str, ...] = ()
    note: str = ""

    @classmethod
    def from_provenance(
        cls, role: str, prov: Provenance, weights_sha256: str | None = None
    ) -> ModelRecord:
        return cls(
            role=role,
            name=prov.name,
            license_id=prov.license_id,
            tier=prov.tier.name.lower(),
            source_url=prov.source_url,
            weights_sha256=weights_sha256,
            inherited_from=tuple(prov.inherited_from),
            note=prov.note,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "license_id": self.license_id,
            "tier": self.tier,
            "source_url": self.source_url,
            "weights_sha256": self.weights_sha256,
            "inherited_from": list(self.inherited_from),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ModelRecord:
        return cls(
            role=d["role"],
            name=d["name"],
            license_id=d["license_id"],
            tier=d["tier"],
            source_url=d["source_url"],
            weights_sha256=d.get("weights_sha256"),
            inherited_from=tuple(d.get("inherited_from", ())),
            note=d.get("note", ""),
        )


@dataclass(frozen=True)
class ImageRecord:
    """One input photograph, identified by its pixels rather than by its file."""

    view: str
    sha256: str
    width: int
    height: int
    filename: str | None = None
    exif_focal_length_35mm: float | None = None
    exif_subject_distance_m: float | None = None
    exif_tags_present: tuple[str, ...] = ()
    exif_stripped: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "filename": self.filename,
            "exif_focal_length_35mm": self.exif_focal_length_35mm,
            "exif_subject_distance_m": self.exif_subject_distance_m,
            # Tag *names* only. A manifest that reproduced the values would be a
            # copy of the metadata the ingest stage exists to discard.
            "exif_tags_present": list(self.exif_tags_present),
            "exif_stripped": self.exif_stripped,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ImageRecord:
        return cls(
            view=d["view"],
            sha256=d["sha256"],
            width=int(d["width"]),
            height=int(d["height"]),
            filename=d.get("filename"),
            exif_focal_length_35mm=d.get("exif_focal_length_35mm"),
            exif_subject_distance_m=d.get("exif_subject_distance_m"),
            exif_tags_present=tuple(d.get("exif_tags_present", ())),
            exif_stripped=bool(d.get("exif_stripped", True)),
        )


@dataclass(frozen=True)
class StageTiming:
    """Wall time for one pipeline stage. Wall, not CPU: the user waits on wall.

    Rounded to a microsecond at construction rather than at serialisation, so
    that a manifest read back off disk compares equal to the one that was
    written. Rounding on the way out is the classic way to build a record that
    fails its own round-trip test.
    """

    name: str
    seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "seconds", round(float(self.seconds), 6))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seconds": self.seconds}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StageTiming:
        return cls(name=d["name"], seconds=float(d["seconds"]))


@dataclass(frozen=True)
class MeasurementRecord:
    """The full provenance of one measurement, reported or not.

    ``outcome`` is three-valued and the distinction is the whole point:

    * ``measured`` -- evaluated and fit to report, possibly with caveats.
    * ``withheld`` -- evaluated, and the gate decided the number would not mean
      anything. ``value`` is null here *by design*, not because it is missing.
    * ``unavailable`` -- never evaluated, because a landmark it needs was not
      supplied. ``missing`` names them.
    """

    spec_id: str
    label: str
    view: str
    unit: str
    evidence: str
    outcome: str
    formula_fingerprint: str
    landmarks_used: tuple[str, ...] = ()
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    sd: float | None = None
    reportability: str | None = None
    reasons: tuple[str, ...] = ()
    discriminability_ratio: float | None = None
    between_subject_sd: float | None = None
    total_error_sd: float | None = None
    pose_component: float | None = None
    landmark_component: float | None = None
    scale_component: float | None = None
    scale_source: str | None = None
    missing: tuple[str, ...] = ()
    n_valid: int | None = None

    @classmethod
    def from_measured(cls, m: Measured) -> MeasurementRecord:
        spec = BY_ID.get(m.spec_id)
        shown = m.verdict.reportability is not Reportability.WITHHOLD
        disc = m.discriminability
        return cls(
            spec_id=m.spec_id,
            label=m.label,
            view=spec.view.value if spec else _UNSET,
            unit=m.unit.value,
            evidence=spec.evidence.value if spec else _UNSET,
            outcome="measured" if shown else "withheld",
            formula_fingerprint=m.formula_fingerprint,
            landmarks_used=tuple(m.landmarks_used),
            # Withheld means withheld everywhere, including here. Recording the
            # value would put it one field access from a renderer that prints it.
            value=m.value if shown else None,
            ci_low=m.ci_low if shown else None,
            ci_high=m.ci_high if shown else None,
            sd=m.sd if shown else None,
            reportability=m.verdict.reportability.value,
            reasons=tuple(m.verdict.reasons),
            discriminability_ratio=disc.ratio if disc else None,
            between_subject_sd=disc.between_subject_sd if disc else None,
            total_error_sd=disc.total_error_sd if disc else None,
            pose_component=disc.pose_component if disc else None,
            landmark_component=disc.landmark_component if disc else None,
            scale_component=disc.scale_component if disc else None,
            scale_source=m.scale_source,
            n_valid=m.n_valid,
        )

    @classmethod
    def from_unavailable(cls, u: Unavailable) -> MeasurementRecord:
        spec = BY_ID.get(u.spec_id)
        return cls(
            spec_id=u.spec_id,
            label=u.label,
            view=spec.view.value if spec else _UNSET,
            unit=spec.unit.value if spec else _UNSET,
            evidence=spec.evidence.value if spec else _UNSET,
            outcome="unavailable",
            formula_fingerprint=spec.fingerprint if spec else _UNSET,
            landmarks_used=tuple(sorted(x.value for x in spec.landmarks)) if spec else (),
            reasons=(u.reason,),
            missing=tuple(u.missing_landmarks),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "label": self.label,
            "view": self.view,
            "unit": self.unit,
            "evidence": self.evidence,
            "outcome": self.outcome,
            "formula_fingerprint": self.formula_fingerprint,
            "landmarks_used": list(self.landmarks_used),
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "sd": self.sd,
            "reportability": self.reportability,
            "reasons": list(self.reasons),
            "discriminability": {
                "ratio": self.discriminability_ratio,
                "between_subject_sd": self.between_subject_sd,
                "total_error_sd": self.total_error_sd,
                "pose_component": self.pose_component,
                "landmark_component": self.landmark_component,
                "scale_component": self.scale_component,
            },
            "scale_source": self.scale_source,
            "missing": list(self.missing),
            "n_valid": self.n_valid,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MeasurementRecord:
        disc = d.get("discriminability") or {}
        return cls(
            spec_id=d["spec_id"],
            label=d["label"],
            view=d["view"],
            unit=d["unit"],
            evidence=d["evidence"],
            outcome=d["outcome"],
            formula_fingerprint=d["formula_fingerprint"],
            landmarks_used=tuple(d.get("landmarks_used", ())),
            value=d.get("value"),
            ci_low=d.get("ci_low"),
            ci_high=d.get("ci_high"),
            sd=d.get("sd"),
            reportability=d.get("reportability"),
            reasons=tuple(d.get("reasons", ())),
            discriminability_ratio=disc.get("ratio"),
            between_subject_sd=disc.get("between_subject_sd"),
            total_error_sd=disc.get("total_error_sd"),
            pose_component=disc.get("pose_component"),
            landmark_component=disc.get("landmark_component"),
            scale_component=disc.get("scale_component"),
            scale_source=d.get("scale_source"),
            missing=tuple(d.get("missing", ())),
            n_valid=d.get("n_valid"),
        )


@dataclass(frozen=True)
class RunManifest:
    """Everything about one run, in a form that survives the machine it ran on."""

    schema_version: int = SCHEMA_VERSION
    created_utc: str = ""
    git_sha: str | None = None
    package_version: str = _UNSET
    python_version: str = _UNSET
    platform: str = _UNSET
    device: str = _UNSET
    seed: int = 0
    n_samples: int = 0
    license_tier: str = Tier.PERMISSIVE.name.lower()
    obligations: tuple[str, ...] = ()
    declared_sex: str | None = None
    declared_ancestry: str | None = None
    models: tuple[ModelRecord, ...] = ()
    images: tuple[ImageRecord, ...] = ()
    stages: tuple[StageTiming, ...] = ()
    measurements: tuple[MeasurementRecord, ...] = ()
    scale: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    #: What the frontal captures were and what pooling them achieved:
    #: ``n_supplied``, ``n_used``, ``dropped``, ``shared_fraction`` and the
    #: sentence ``multishot.Combined.note()`` produced. Recorded even for a
    #: single photograph, because "one capture, no averaging" is a fact about
    #: the run and not the absence of one.
    captures: dict[str, Any] = field(default_factory=dict)
    failure_reasons: tuple[str, ...] = ()

    # -- construction ------------------------------------------------------

    @classmethod
    def begin(
        cls,
        *,
        tier: Tier,
        device: str,
        seed: int,
        n_samples: int,
        declared_sex: str | None = None,
        declared_ancestry: str | None = None,
    ) -> RunManifest:
        """Start a manifest with everything knowable before any stage runs."""
        return cls(
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            git_sha=_git_sha(),
            package_version=_package_version(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            device=device,
            seed=seed,
            n_samples=n_samples,
            license_tier=tier.name.lower(),
            # Not the tier alone: the obligations the tier actually entails,
            # spelled out, so a reader does not have to reconstruct them from a
            # version of the licence catalogue they may not have.
            obligations=obligations_at(tier),
            declared_sex=declared_sex,
            declared_ancestry=declared_ancestry,
        )

    def with_(self, **changes: Any) -> RunManifest:
        """A copy with fields replaced. The manifest is frozen; runs are not."""
        from dataclasses import replace

        return replace(self, **changes)

    @staticmethod
    def records_for(
        measured: Sequence[Measured], unavailable: Sequence[Unavailable]
    ) -> tuple[MeasurementRecord, ...]:
        records = [MeasurementRecord.from_measured(m) for m in measured]
        records += [MeasurementRecord.from_unavailable(u) for u in unavailable]
        return tuple(sorted(records, key=lambda r: r.spec_id))

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "git_sha": self.git_sha,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "device": self.device,
            "seed": self.seed,
            "n_samples": self.n_samples,
            "license_tier": self.license_tier,
            "obligations": list(self.obligations),
            "declared_sex": self.declared_sex,
            "declared_ancestry": self.declared_ancestry,
            "models": [m.to_dict() for m in self.models],
            "images": [i.to_dict() for i in self.images],
            "stages": [s.to_dict() for s in self.stages],
            "measurements": [m.to_dict() for m in self.measurements],
            "scale": dict(self.scale),
            "quality": dict(self.quality),
            "captures": dict(self.captures),
            "failure_reasons": list(self.failure_reasons),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunManifest:
        version = int(d.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"run manifest schema version {version} cannot be read by this "
                f"build, which writes and understands version {SCHEMA_VERSION}"
            )
        return cls(
            schema_version=version,
            created_utc=d.get("created_utc", ""),
            git_sha=d.get("git_sha"),
            package_version=d.get("package_version", _UNSET),
            python_version=d.get("python_version", _UNSET),
            platform=d.get("platform", _UNSET),
            device=d.get("device", _UNSET),
            seed=int(d.get("seed", 0)),
            n_samples=int(d.get("n_samples", 0)),
            license_tier=d.get("license_tier", Tier.PERMISSIVE.name.lower()),
            obligations=tuple(d.get("obligations", ())),
            declared_sex=d.get("declared_sex"),
            declared_ancestry=d.get("declared_ancestry"),
            models=tuple(ModelRecord.from_dict(m) for m in d.get("models", ())),
            images=tuple(ImageRecord.from_dict(i) for i in d.get("images", ())),
            stages=tuple(StageTiming.from_dict(s) for s in d.get("stages", ())),
            measurements=tuple(
                MeasurementRecord.from_dict(m) for m in d.get("measurements", ())
            ),
            scale=dict(d.get("scale", {})),
            quality=dict(d.get("quality", {})),
            captures=dict(d.get("captures", {})),
            failure_reasons=tuple(d.get("failure_reasons", ())),
        )

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        return cls.from_dict(json.loads(text))

    def write(self, directory: str | Path, *, filename: str = "run.json") -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    # -- reading ------------------------------------------------------------

    def record(self, spec_id: str) -> MeasurementRecord | None:
        for m in self.measurements:
            if m.spec_id == spec_id:
                return m
        return None

    def outcomes(self) -> dict[str, int]:
        counts = {"measured": 0, "withheld": 0, "unavailable": 0}
        for m in self.measurements:
            counts[m.outcome] = counts.get(m.outcome, 0) + 1
        return counts


def unit_of(record: MeasurementRecord) -> Unit | None:
    """The record's unit as the enum, or ``None`` for a spec no longer in the catalogue."""
    try:
        return Unit(record.unit)
    except ValueError:
        return None


def view_of(record: MeasurementRecord) -> View | None:
    try:
        return View(record.view)
    except ValueError:
        return None
