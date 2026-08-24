"""Name to backend, with the licence check in front of the import.

The registry exists so that a licence refusal costs nothing. Each entry pairs a
name with a `Provenance` and a *lazy* factory: `require()` runs against the
declared tier before the factory is called, so asking for an AGPL backend at
the permissive tier raises `LicenseViolation` without importing torch, without
reading a checkpoint, and without touching the cache. A registry that imported
first and checked afterwards would still be correct, and would still have
loaded two hundred megabytes of weights the user is not permitted to use.

The entries are declared against `licensing.CATALOGUE` rather than repeating
its facts, so a backend cannot drift out of agreement with the obligations
recorded for it. Backends that Faciometry knows about but cannot load are listed
too, with `factory=None`: `build_*` then raises a message naming what is
missing, which is a better answer than `KeyError` for a name the user read in
the licensing catalogue.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from . import licensing
from .licensing import LicenseViolation, Provenance, Tier

Factory = Callable[..., Any]


class UnknownBackend(KeyError):
    """No backend registered under that name."""


class BackendUnavailable(RuntimeError):
    """A known backend that this build cannot construct."""


@dataclass(frozen=True)
class Entry:
    """One registered backend."""

    name: str
    role: str
    provenance: Provenance
    #: Imported on first construction, not at registry definition time, so
    #: `import faciometry.models.registry` stays free of torch and mediapipe.
    module: str | None = None
    attribute: str = "build"
    unavailable_reason: str = ""

    @property
    def tier(self) -> Tier:
        return self.provenance.tier

    @property
    def loadable(self) -> bool:
        return self.module is not None

    def factory(self) -> Factory:
        if self.module is None:
            raise BackendUnavailable(
                f"{self.name} is known to Faciometry but not implemented here: "
                f"{self.unavailable_reason}"
            )
        import importlib

        return getattr(importlib.import_module(self.module, package=__package__), self.attribute)


DETECTORS: dict[str, Entry] = {
    "yunet": Entry(
        name="yunet",
        role="detector",
        provenance=licensing.YUNET,
        module=".detect_yunet",
    ),
    "yolo_face": Entry(
        name="yolo_face",
        role="detector",
        provenance=licensing.YOLO_FACE,
        unavailable_reason=(
            "no Ultralytics-lineage detector is vendored. It is registered so that "
            "requesting it at the permissive tier fails with a licence message "
            "rather than a name error, which is the question a user asking for it "
            "is actually asking"
        ),
    ),
}

LANDMARKERS: dict[str, Entry] = {
    "spiga": Entry(
        name="spiga",
        role="landmarker",
        provenance=licensing.SPIGA,
        module=".landmark_spiga",
    ),
    "star": Entry(
        name="star",
        role="landmarker",
        provenance=licensing.STAR_LOSS,
        unavailable_reason=(
            "the best published landmark accuracy comes with no licence file at "
            "all, so all rights are reserved and there is no tier at which Faciometry "
            "will load it"
        ),
    ),
}

DENSE: dict[str, Entry] = {
    "mediapipe": Entry(
        name="mediapipe",
        role="dense landmarker",
        provenance=licensing.MEDIAPIPE_FACE_LANDMARKER,
        module=".dense_mediapipe",
    ),
    "threeddfa_v2": Entry(
        name="threeddfa_v2",
        role="dense landmarker",
        provenance=licensing.THREEDDFA_V2,
        unavailable_reason=(
            "the code is MIT but the Basel Face Model basis it needs is "
            "non-commercial and cannot be redistributed, so the user must supply "
            "the basis file themselves"
        ),
    ),
    "mica": Entry(
        name="mica",
        role="dense landmarker",
        provenance=licensing.MICA,
        unavailable_reason=(
            "the only genuinely metric face model in existence, and also the most "
            "encumbered; it needs FLAME 2020 and InsightFace weights that cannot "
            "be shipped"
        ),
    ),
}

POSE: dict[str, Entry] = {
    "sixdrepnet": Entry(
        name="sixdrepnet",
        role="pose estimator",
        provenance=licensing.SIXDREPNET,
        module=".pose_sixdrepnet",
    ),
}

BY_ROLE: dict[str, dict[str, Entry]] = {
    "detector": DETECTORS,
    "landmarker": LANDMARKERS,
    "dense": DENSE,
    "pose": POSE,
}

#: What `build_*` picks when the caller does not name a backend.
DEFAULTS: dict[str, str] = {
    "detector": "yunet",
    "landmarker": "spiga",
    "dense": "mediapipe",
    "pose": "sixdrepnet",
}


def _lookup(role: str, name: str) -> Entry:
    table = BY_ROLE[role]
    try:
        return table[name]
    except KeyError as exc:
        raise UnknownBackend(
            f"no {role} named {name!r}; registered {role}s are {sorted(table)}"
        ) from exc


def _build(role: str, name: str | None, tier: Tier, kwargs: Mapping[str, Any]) -> Any:
    entry = _lookup(role, name or DEFAULTS[role])
    # The refusal happens here, before Entry.factory() imports anything. The
    # backend constructor calls require() a second time; that repetition is
    # deliberate, because a backend constructed directly rather than through
    # the registry must not escape the check.
    licensing.require(entry.provenance, tier)
    return entry.factory()(allowed_tier=tier, **kwargs)


def build_detector(name: str | None = None, *, tier: Tier = Tier.PERMISSIVE, **kwargs: Any) -> Any:
    """A `FaceDetector`. Raises `LicenseViolation` above ``tier``."""
    return _build("detector", name, tier, kwargs)


def build_landmarker(name: str | None = None, *, tier: Tier = Tier.PERMISSIVE, **kwargs: Any) -> Any:
    """A `Landmarker` with per-point covariances."""
    return _build("landmarker", name, tier, kwargs)


def build_dense(name: str | None = None, *, tier: Tier = Tier.PERMISSIVE, **kwargs: Any) -> Any:
    """A `DenseLandmarker` with a separated iris ring."""
    return _build("dense", name, tier, kwargs)


def build_pose(name: str | None = None, *, tier: Tier = Tier.PERMISSIVE, **kwargs: Any) -> Any:
    """A `PoseEstimator` independent of the landmarker."""
    return _build("pose", name, tier, kwargs)


def default_backends(*, tier: Tier = Tier.PERMISSIVE, device: str | None = None) -> Any:
    """The bundle `pipeline.run.analyze` consumes: detector, landmarker, pose, iris.

    `pipeline.ports.Backends` is imported inside the function rather than at
    module scope. The dependency runs pipeline -> models, and importing the
    pipeline's dataclass at the top of the registry would close the loop; doing
    it here keeps the arrow pointing one way at import time while still handing
    back the exact type the pipeline annotates.

    The dense landmarker is registered twice in the bundle, once as the iris
    measurer, because it is one model and one inference. Constructing two would
    run the 478-vertex mesh over the same photograph a second time to read ten
    of its points.
    """
    from ..pipeline.ports import Backends

    resolved = _resolve_device(device)
    dense = build_dense(tier=tier)
    return Backends(
        detector=build_detector(tier=tier),
        landmarker=build_landmarker(tier=tier, device=resolved),
        pose_estimator=build_pose(tier=tier, device=resolved),
        iris=dense,
        device=str(resolved),
    )


def _resolve_device(device: str | None) -> Any:
    """Turn the pipeline's device string into a `device.Device`.

    Imported lazily for the same reason the backends are: `device` reaches for
    torch to answer "is MPS available", and a caller listing the registry should
    not pay for that.
    """
    from .device import resolve as resolve_device

    return resolve_device(device)


def available(tier: Tier = Tier.PERMISSIVE, *, role: str | None = None) -> tuple[Entry, ...]:
    """Every entry loadable at ``tier``, optionally filtered to one role."""
    roles = [role] if role is not None else sorted(BY_ROLE)
    out: list[Entry] = []
    for r in roles:
        out.extend(e for e in BY_ROLE[r].values() if e.tier <= tier and e.loadable)
    return tuple(out)


def describe(tier: Tier = Tier.PERMISSIVE) -> tuple[str, ...]:
    """One line per registered backend, saying whether it is usable and why not.

    Written for `faciometry backends`, so a user comparing tiers can see what
    raising one buys and what it costs before doing it.
    """
    lines: list[str] = []
    for role in sorted(BY_ROLE):
        for entry in BY_ROLE[role].values():
            if entry.tier > tier:
                status = f"refused at tier {tier.name.lower()}"
            elif not entry.loadable:
                status = f"not implemented: {entry.unavailable_reason}"
            else:
                status = "available"
            lines.append(f"{entry.role}/{entry.name}: {status} -- {entry.provenance.describe()}")
    return tuple(lines)


def selection_manifest(
    *,
    detector: str | None = None,
    landmarker: str | None = None,
    dense: str | None = None,
    pose: str | None = None,
    tier: Tier = Tier.PERMISSIVE,
) -> dict[str, str]:
    """The backend half of `run.json`.

    Records the resolved names and their obligations, so a report can be
    audited later without re-reading the code that produced it.
    """
    chosen = {
        "detector": detector or DEFAULTS["detector"],
        "landmarker": landmarker or DEFAULTS["landmarker"],
        "dense": dense or DEFAULTS["dense"],
        "pose": pose or DEFAULTS["pose"],
    }
    manifest = {"license_tier": tier.name.lower()}
    for role, name in chosen.items():
        entry = _lookup(role, name)
        manifest[role] = name
        manifest[f"{role}_provenance"] = entry.provenance.describe()
    obligations = licensing.obligations_at(tier)
    manifest["obligations"] = "; ".join(obligations) if obligations else "none"
    return manifest


__all__ = [
    "DEFAULTS",
    "BackendUnavailable",
    "Entry",
    "LicenseViolation",
    "UnknownBackend",
    "available",
    "build_dense",
    "build_detector",
    "build_landmarker",
    "build_pose",
    "default_backends",
    "describe",
    "selection_manifest",
]
