"""What this machine can actually do, checked rather than assumed.

`doctor` answers three questions a user asks in this order: is the
measurement layer intact, can a model be loaded here, and are the weights the
ones the manifest pins. It resolves the device the same way the pipeline does
so that the answer it prints is the answer the pipeline will get.

The check that matters most is the digest check. A weights file that has been
replaced upstream changes every number Vitruve prints and leaves no trace in
the output, so `doctor` verifies each cached artifact against the manifest
instead of checking that the file exists.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import sys
from dataclasses import dataclass

from ..models.licensing import Tier
from . import weights as weights_mod
from .runner import analysis_available

OK, WARN, FAIL = "ok", "warn", "FAIL"


@dataclass(frozen=True)
class Check:
    level: str
    name: str
    detail: str


def _version_of(module_name: str) -> str | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - a broken install
        return f"present but will not import ({exc})"
    return str(getattr(mod, "__version__", "unknown version"))


def resolve_device() -> tuple[str, str]:
    """``(device, why)``, decided the way the pipeline decides it.

    Deferred to :mod:`vitruve.models.device` when that layer is installed, so
    the device `doctor` prints is the device an analysis will actually get,
    including a ``VITRUVE_DEVICE`` override. Falling back to a local guess
    would let `doctor` report `mps` on a machine where the run resolves to
    `cpu`, which is the one thing this check exists to prevent.
    """
    try:
        from ..models import device as model_device

        chosen = model_device.resolve()
        return chosen.kind, chosen.description
    except ImportError:
        pass
    except Exception as exc:
        return "unknown", f"device selection failed: {exc}"

    if importlib.util.find_spec("torch") is None:
        return "cpu", "torch is not installed, so only ONNX and MediaPipe paths can run"
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return "cpu", f"torch present but will not import ({exc})"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", f"Apple GPU via torch {torch.__version__}"
    if torch.cuda.is_available():
        return "cuda", f"{torch.cuda.get_device_name(0)} via torch {torch.__version__}"
    return "cpu", f"torch {torch.__version__} with no GPU backend available"


def collect(tier: Tier = Tier.PERMISSIVE) -> list[Check]:
    from .. import __version__

    checks: list[Check] = [
        Check(OK, "vitruve", __version__),
        Check(
            OK if sys.version_info[:2] in ((3, 11), (3, 12)) else WARN,
            "python",
            f"{platform.python_version()} on {platform.machine()} {platform.system()}",
        ),
    ]

    try:
        from ..measure.registry import CATALOGUE

        fingerprints = {s.fingerprint for s in CATALOGUE}
        level = OK if len(fingerprints) == len(CATALOGUE) else WARN
        detail = f"{len(CATALOGUE)} measurements, {len(fingerprints)} distinct formulas"
        if level is WARN:
            detail += " (two measurements share a formula hash)"
        checks.append(Check(level, "catalogue", detail))
    except Exception as exc:
        checks.append(Check(FAIL, "catalogue", f"the measurement registry will not import: {exc}"))

    device, why = resolve_device()
    checks.append(Check(OK, "device", f"{device}: {why}"))

    backends = (
        ("cv2", "opencv-python", "YuNet detector", Tier.PERMISSIVE),
        ("mediapipe", "mediapipe", "dense landmarks and iris", Tier.PERMISSIVE),
        ("onnxruntime", "onnxruntime", "ONNX backends", Tier.PERMISSIVE),
        ("ultralytics", "ultralytics", "YOLO detection and derm segmentation", Tier.COPYLEFT),
    )
    for module_name, dist, what, needs in backends:
        version = _version_of(module_name)
        if version is None:
            level = WARN if needs <= tier else OK
            note = "not installed"
            if needs > tier:
                note += f"; only used at the {needs.name.lower()} tier"
            checks.append(Check(level, dist, f"{note} ({what})"))
        else:
            level = OK if needs <= tier else WARN
            note = f"{version} ({what})"
            if needs > tier:
                note += (
                    f"; installed but the selected tier is {tier.name.lower()}, so it "
                    "will not be loaded"
                )
            checks.append(Check(level, dist, note))

    for module_name, dist in (("fastapi", "fastapi"), ("uvicorn", "uvicorn")):
        version = _version_of(module_name)
        checks.append(
            Check(OK, dist, version)
            if version
            else Check(WARN, dist, "not installed; `vitruve serve` needs vitruve[api]")
        )

    checks.append(
        Check(OK, "pipeline", "importable")
        if analysis_available()
        else Check(WARN, "pipeline", "no vitruve.pipeline in this build, so analysis will not run")
    )

    checks.extend(_weight_checks(tier))
    return checks


def _weight_checks(tier: Tier) -> list[Check]:
    """Every permitted artifact, verified against its pin rather than counted.

    Existence is not the check. A weights file replaced upstream changes every
    number Vitruve prints and leaves nothing in the output to show it, so the
    digest is what `doctor` reports.
    """
    try:
        _permitted, refused = weights_mod.split_by_tier(tier)
        results = weights_mod.verify(tier)
    except weights_mod.WeightsLayerMissing as exc:
        return [Check(WARN, "weights", str(exc))]
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return [Check(FAIL, "weights", f"the weight lock will not load: {exc}")]

    out = [Check(OK, "weights cache", str(weights_mod.cache_dir()))]
    for pin in refused:
        out.append(
            Check(OK, pin.key, f"skipped: {pin.license_id} needs the {pin.tier.name.lower()} tier")
        )
    for result in results:
        if result.ok:
            out.append(Check(OK, result.key, f"{result.path.name} verified"))
        elif not result.present:
            out.append(Check(WARN, result.key, "not fetched; run `vitruve fetch-weights`"))
        else:
            out.append(
                Check(
                    FAIL,
                    result.key,
                    f"sha256 does not match the pin: cached bytes hash to "
                    f"{result.actual_sha256}. Delete it and re-fetch; do not run an "
                    "analysis against it.",
                )
            )
    return out


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"[{c.level:>4}] {c.name:<{width}}  {c.detail}" for c in checks]
    fails = sum(1 for c in checks if c.level == FAIL)
    warns = sum(1 for c in checks if c.level == WARN)
    lines.append("")
    if fails:
        lines.append(f"{fails} failure(s) and {warns} warning(s). Analysis will not be correct.")
    elif warns:
        lines.append(f"{warns} warning(s). Everything a warning names is optional or not yet fetched.")
    else:
        lines.append("All checks passed.")
    return "\n".join(lines)


def run(tier: Tier = Tier.PERMISSIVE) -> int:
    checks = collect(tier)
    print(render(checks))
    return 1 if any(c.level == FAIL for c in checks) else 0
