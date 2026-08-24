"""The seam between a transport and the pipeline.

The CLI and the HTTP API both need the same four things: decode an image
without letting its metadata through, hand it to the pipeline, map whatever
the pipeline raises onto a status a caller can branch on, and render the
result without ever printing a number that has lost its interval. That work
lives here once rather than twice.

Two decisions in this module are load-bearing.

**EXIF never survives the boundary.** A photograph of a face carries a capture
timestamp and, on a phone, a GPS fix. Faciometry decodes to a pixel array and
drops everything else at the point of ingest, so nothing downstream has the
option of writing it out. :class:`LoadedImage` records which tags were dropped
so the privacy claim is inspectable rather than promised.

**The pipeline is discovered, not imported.** ``faciometry.pipeline`` and
``faciometry.report`` are built separately from the transports, and a transport
that hard-imports a module that does not exist yet fails with a traceback
about an import instead of a sentence about what is missing. So the import is
attempted at call time and its failure is a value.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ..models.licensing import LicenseViolation, Tier

#: Where the pipeline entry point is looked for. ``analyze`` takes paths; a
#: future ``analyze_images`` taking decoded arrays would let the HTTP API keep
#: an upload entirely in memory, so it is preferred when present.
ARRAY_ENTRY_POINTS: tuple[str, ...] = ("analyze_images", "analyse_images")
PATH_ENTRY_POINTS: tuple[str, ...] = ("analyze", "analyse")

#: Maximum decoded pixels. A 100-megapixel upload is a denial of service, not
#: a portrait.
MAX_PIXELS = 80_000_000


class BadImage(ValueError):
    """The bytes are not a decodable image, or are too large to decode."""


class PipelineUnavailable(RuntimeError):
    """This build has no analysis pipeline installed.

    Carries a message naming what was looked for, because the fix is either
    installing an extra or waiting for the module to land, and those are
    different actions.
    """


class QualityGateFailed(RuntimeError):
    """The photograph did not clear the gate, so nothing was measured."""


class Status(str, Enum):
    OK = "ok"
    BAD_INPUT = "bad_input"
    QUALITY_GATE = "quality_gate"
    LICENSE = "license"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class LoadedImage:
    """Pixels, and a record of what was thrown away to get them."""

    pixels: np.ndarray
    width: int
    height: int
    #: sha256 of the bytes as they arrived, for the run manifest. The digest is
    #: of the original file, so a report can be tied back to its source without
    #: the source being kept.
    sha256: str
    #: Human-readable names of the metadata tags dropped at ingest.
    exif_tags_dropped: tuple[str, ...] = ()
    had_gps: bool = False
    source: str = "upload"
    #: Set when the image came off disk. A pipeline that opens files can then
    #: be handed the original path instead of a temporary copy.
    path: Path | None = None

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    def to_png_bytes(self) -> bytes:
        """Re-encode from the pixel array, which carries no metadata.

        Used only when the user asked for the image to be stored. Encoding
        from the array rather than copying the file is what makes the stored
        copy metadata-free by construction.
        """
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(self.pixels).save(buf, format="PNG")
        return buf.getvalue()


def load_image(data: bytes, *, source: str = "upload") -> LoadedImage:
    """Decode to RGB pixels and strip everything else."""
    from PIL import ExifTags, Image, UnidentifiedImageError

    digest = hashlib.sha256(data).hexdigest()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError as exc:
        raise BadImage(f"{source}: not a decodable image") from exc
    except Image.DecompressionBombError as exc:
        raise BadImage(f"{source}: image is implausibly large") from exc
    except Exception as exc:
        # Decoding untrusted bytes fails in many ways, and `PIL.Image.open` is
        # not necessarily PIL's: ultralytics replaces it with a wrapper that
        # falls back to a HEIF plugin and raises ModuleNotFoundError when that
        # plugin is absent. Anything raised out of a decode means these bytes
        # are not a usable image, which is the caller's answer either way.
        raise BadImage(f"{source}: could not decode this image ({exc})") from exc

    if img.width * img.height > MAX_PIXELS:
        raise BadImage(
            f"{source}: {img.width}x{img.height} exceeds the {MAX_PIXELS // 1_000_000} "
            "megapixel ceiling"
        )

    dropped: list[str] = []
    had_gps = False
    try:
        exif = img.getexif()
    except Exception:  # pragma: no cover - Pillow raises variously on odd files
        exif = None
    if exif:
        for tag_id in exif:
            name = ExifTags.TAGS.get(tag_id, f"tag_{tag_id}")
            dropped.append(name)
            if name == "GPSInfo":
                had_gps = True
        if not had_gps:
            try:
                had_gps = bool(exif.get_ifd(0x8825))
            except Exception:  # pragma: no cover
                had_gps = False
            if had_gps and "GPSInfo" not in dropped:
                dropped.append("GPSInfo")

    pixels = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return LoadedImage(
        pixels=pixels,
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        sha256=digest,
        exif_tags_dropped=tuple(sorted(set(dropped))),
        had_gps=had_gps,
        source=source,
    )


def load_image_file(path: str | Path) -> LoadedImage:
    p = Path(path)
    if not p.exists():
        raise BadImage(f"{p}: no such file")
    if p.is_dir():
        raise BadImage(f"{p}: is a directory")
    loaded = load_image(p.read_bytes(), source=str(p))
    return replace(loaded, path=p)


@dataclass(frozen=True)
class AnalysisRequest:
    """Everything the pipeline needs, and nothing it should infer.

    ``declared_sex`` and ``declared_ancestry`` select a normative stratum.
    They are declared by the subject or left empty; Faciometry never predicts
    them, and an empty value widens an interval rather than triggering a
    guess.
    """

    frontal: LoadedImage
    profile: LoadedImage | None = None
    #: Further captures of the same face in the same session, pooled with
    #: ``frontal`` before anything is measured. ``frontal`` stays a field of
    #: its own because it is the capture the scale ladder and the overlays are
    #: read from, and a tuple with a privileged first element is a worse way to
    #: say that than a field is.
    extra_frontals: tuple[LoadedImage, ...] = ()
    license_tier: Tier = Tier.PERMISSIVE
    declared_sex: str | None = None
    declared_ancestry: str | None = None
    ruler_mm: float | None = None
    seed: int = 0


@dataclass(frozen=True)
class AnalysisOutcome:
    """What a transport turns into an exit code or an HTTP status."""

    status: Status
    message: str = ""
    report: Any = None
    request: AnalysisRequest | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status is Status.OK


# ---------------------------------------------------------------------------
# Finding and calling the pipeline.
# ---------------------------------------------------------------------------


def _pipeline_module() -> Any:
    try:
        return importlib.import_module("faciometry.pipeline")
    except ImportError as exc:
        raise PipelineUnavailable(
            "no analysis pipeline in this build: `import faciometry.pipeline` failed "
            f"({exc}). The measurement layer is independent of it, so "
            "`faciometry catalogue`, `faciometry licenses` and `faciometry doctor` still work."
        ) from exc


def _entry_point() -> tuple[str, Any]:
    """``(kind, callable)`` where kind is ``"array"`` or ``"path"``."""
    module = _pipeline_module()
    for name in ARRAY_ENTRY_POINTS:
        fn = getattr(module, name, None)
        if callable(fn):
            return "array", fn
    for name in PATH_ENTRY_POINTS:
        fn = getattr(module, name, None)
        if callable(fn):
            return "path", fn
    raise PipelineUnavailable(
        "faciometry.pipeline imported but exposes none of "
        f"{', '.join(ARRAY_ENTRY_POINTS + PATH_ENTRY_POINTS)}."
    )


def analysis_available() -> bool:
    """Whether an analysis would get as far as the pipeline."""
    try:
        _entry_point()
    except PipelineUnavailable:
        return False
    return True


def run_analysis(request: AnalysisRequest) -> AnalysisOutcome:
    """Run the pipeline and classify whatever comes back.

    Never raises for an expected condition. A missing pipeline, a rejected
    photograph and a license refusal are all outcomes with a status, because
    both transports have to turn them into a code rather than a traceback.
    """
    try:
        kind, entry = _entry_point()
    except PipelineUnavailable as exc:
        return AnalysisOutcome(Status.UNAVAILABLE, str(exc), request=request)

    kwargs = {
        "tier": request.license_tier,
        "declared_sex": request.declared_sex,
        "declared_ancestry": request.declared_ancestry,
        "ruler_mm": request.ruler_mm,
        "seed": request.seed,
    }
    try:
        with _materialised(request, kind) as args:
            result = entry(*args, **kwargs)
    except LicenseViolation as exc:
        return AnalysisOutcome(Status.LICENSE, str(exc), request=request)
    except (BadImage, ValueError) as exc:
        return AnalysisOutcome(Status.BAD_INPUT, str(exc), request=request)
    except RuntimeError as exc:
        # NoFaceFound and "no backends installed" both land here. Neither is a
        # bug in the caller and neither is a measurement.
        if type(exc).__name__ == "NoFaceFound":
            return AnalysisOutcome(
                Status.QUALITY_GATE, str(exc), request=request, reasons=(str(exc),)
            )
        return AnalysisOutcome(Status.UNAVAILABLE, str(exc), request=request)
    except (ImportError, AttributeError) as exc:
        # An incomplete install: a model backend whose optional dependency is
        # absent, or a partly built model registry. Reported verbatim with the
        # unavailable status rather than as a traceback, because the message
        # names the missing piece and a traceback names a line number.
        return AnalysisOutcome(
            Status.UNAVAILABLE,
            f"the model layer is incomplete in this build: {exc}",
            request=request,
        )

    if getattr(result, "failed", False):
        return AnalysisOutcome(
            Status.QUALITY_GATE,
            "the photograph did not clear the quality gate, so nothing was measured",
            report=result,
            request=request,
            reasons=tuple(getattr(result, "failure_reasons", ()) or ()),
        )
    return AnalysisOutcome(Status.OK, report=result, request=request)


@contextmanager
def _materialised(request: AnalysisRequest, kind: str) -> Iterator[tuple[Any, ...]]:
    """Present the request's images in the form the entry point accepts.

    An array entry point gets the decoded pixels and the upload never leaves
    memory. A path entry point is the current pipeline, and it opens files, so
    an image that arrived over HTTP has to exist somewhere for the length of
    the call. It is written into a private directory created with mode 0700,
    re-encoded from the pixel array so it carries no metadata, and unlinked in
    a ``finally``. `docs/PRIVACY.md` says so in those words; when the pipeline
    grows an array entry point this branch stops being reachable.
    """
    fronts = (request.frontal, *request.extra_frontals)

    if kind == "array":
        yield (
            [f.pixels for f in fronts] if len(fronts) > 1 else request.frontal.pixels,
            request.profile.pixels if request.profile else None,
        )
        return

    if all(f.path is not None for f in fronts) and (
        request.profile is None or request.profile.path is not None
    ):
        yield (
            [f.path for f in fronts] if len(fronts) > 1 else request.frontal.path,
            request.profile.path if request.profile else None,
        )
        return

    with tempfile.TemporaryDirectory(prefix="faciometry-") as tmp:
        root = Path(tmp)
        root.chmod(0o700)
        written: list[Path] = []
        for i, image in enumerate(fronts):
            path = root / f"frontal-{i}.png"
            path.write_bytes(image.to_png_bytes())
            written.append(path)
        profile: Path | None = None
        if request.profile is not None:
            profile = root / "profile.png"
            profile.write_bytes(request.profile.to_png_bytes())
            written.append(profile)
        try:
            yield (
                written[: len(fronts)] if len(fronts) > 1 else written[0],
                profile,
            )
        finally:
            for path in written:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Rendering. The report layer owns the prose and the HTML; this section owns
# only the assembly, which is the one job neither the pipeline nor the report
# package can do, because it is the point where a run becomes a document.
# ---------------------------------------------------------------------------

_SEVERITY_WORDS = {"info": "note", "warn": "caveat", "fail": "blocking"}


def build_report_input(outcome: AnalysisOutcome) -> Any:
    """Turn an ``AnalysisResult`` into the report layer's ``ReportInput``.

    Overlays are rendered in two passes because a group's overlay needs the
    groups, and the groups come from the report input. Landmark covariances
    are not carried on the result, so the ellipses are omitted rather than
    drawn at a guessed radius.
    """
    from dataclasses import replace

    from ..measure.registry import BY_ID
    from ..models.licensing import obligations_at
    from ..report import model as report_model
    from ..report import overlay as report_overlay

    result = outcome.report
    request = outcome.request

    issues: list[Any] = []
    for quality in _quality_reports(result):
        for issue in getattr(quality, "issues", ()):
            issues.append(
                report_model.QualityIssue(
                    code=f"{quality.view.value}.{issue.code}",
                    detail=f"{issue.message} {issue.remedy}".strip(),
                    severity=_SEVERITY_WORDS.get(issue.severity.value, "note"),
                    reading=quality.view.value,
                )
            )

    measurements = tuple(getattr(result, "measured", ()) or ())
    unavailable = tuple(getattr(result, "unavailable", ()) or ())

    strata = {}
    for m in measurements:
        stratum = report_model.niosh_stratum(
            m.spec_id,
            sex=request.declared_sex if request else None,
            ancestry=request.declared_ancestry if request else None,
        )
        if stratum is not None:
            strata[m.spec_id] = stratum

    references: list[str] = []
    for m in list(measurements) + list(unavailable):
        spec = BY_ID.get(m.spec_id)
        for ref in spec.references if spec else ():
            if ref not in references:
                references.append(ref)

    tier = request.license_tier if request else Tier.PERMISSIVE
    manifest = getattr(result, "manifest", None)
    report_input = report_model.ReportInput(
        measurements=measurements,
        unavailable=unavailable,
        quality=tuple(issues),
        manifest=manifest.to_dict() if manifest is not None else {},
        strata=strata,
        obligations=obligations_at(tier),
        references=tuple(references),
        declared_sex=request.declared_sex if request else None,
        declared_ancestry=request.declared_ancestry if request else None,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_captures=int(getattr(result, "n_captures", 1) or 1),
        capture_note=str(getattr(result, "capture_note", "") or ""),
        scale_is_measured=_scale_is_measured(result),
    )

    overlays = _overlays(result, report_input, report_overlay)
    return replace(report_input, overlays=overlays) if overlays else report_input


def _scale_is_measured(result: Any) -> bool:
    """Whether the millimetres rest on something measured in the frame.

    It decides whether a ruler is a lever the error budget may offer at all,
    and offering a ruler to a run that already used one would be the report
    pricing a change the user has already made. Read off the scale estimate's
    own source rather than off a measurement's ``scale_source`` string, because
    a run with no millimetre measurements at all has no such string to read.
    """
    scale = getattr(result, "scale", None)
    source = getattr(scale, "source", None)
    return str(getattr(source, "value", source or "")) == "ruler"


def _quality_reports(result: Any) -> tuple[Any, ...]:
    fn = getattr(result, "quality_reports", None)
    if callable(fn):
        return tuple(fn())
    return ()


def _overlays(result: Any, report_input: Any, report_overlay: Any) -> tuple[Any, ...]:
    """Annotated crops, when the run kept enough to draw them."""
    view = getattr(result, "frontal", None)
    if view is None:
        return ()
    try:
        from PIL import Image

        points = view.landmarks_image
        coords = {
            name.value: tuple(float(x) for x in points.get(name)[:2])
            for name in points.index
        }
        image = Image.fromarray(view.source.pixels)
        plotted = report_overlay.landmarks_from(coords)
        return report_overlay.overlays_for_groups(
            image,
            plotted,
            report_input.groups(),
            roll_deg=float(getattr(view.pose, "roll_deg", 0.0) or 0.0),
        )
    except Exception:
        # An overlay is an illustration. Losing it must not lose the numbers.
        return ()


def render_text(outcome: AnalysisOutcome) -> str:
    """The report as plain text, from the report layer's prose."""
    try:
        from ..report import report_text

        return report_text(build_report_input(outcome))
    except ImportError:
        return _fallback_text(outcome)


def render_html(outcome: AnalysisOutcome) -> str | None:
    """One self-contained HTML file, or ``None`` if the renderer is absent."""
    try:
        from ..report import render

        return render(build_report_input(outcome))
    except ImportError:
        return None


def to_dict(outcome: AnalysisOutcome) -> dict[str, Any]:
    """JSON-shaped result.

    There is no aggregate field here and there is not going to be one. Every
    entry is a named measurement carrying its own interval, and the run
    manifest carries the provenance of the lot.
    """
    from .. import __version__

    result = outcome.report
    measured = tuple(getattr(result, "measured", ()) or ())
    unavailable = tuple(getattr(result, "unavailable", ()) or ())
    shown = [m for m in measured if m.verdict.shown]
    withheld = [m for m in measured if not m.verdict.shown]

    manifest = getattr(result, "manifest", None)
    out: dict[str, Any] = {
        "faciometry_version": __version__,
        "status": outcome.status.value,
        "message": outcome.message,
        "measurements": [_measured_dict(m) for m in shown],
        "withheld": [
            {"id": m.spec_id, "label": m.label, "reasons": list(m.verdict.reasons)}
            for m in withheld
        ],
        "unavailable": [
            {"id": u.spec_id, "label": u.label, "reason": u.reason} for u in unavailable
        ],
        "quality": [
            {
                "view": q.view.value,
                "code": i.code,
                "severity": i.severity.value,
                "message": i.message,
                "remedy": i.remedy,
            }
            for q in _quality_reports(result)
            for i in getattr(q, "issues", ())
        ],
        "manifest": manifest.to_dict() if manifest is not None else {},
    }

    request = outcome.request
    if request is not None:
        out["run"] = {
            "license_tier": request.license_tier.name.lower(),
            "seed": request.seed,
            "declared_sex": request.declared_sex,
            "declared_ancestry": request.declared_ancestry,
            "ruler_mm": request.ruler_mm,
            "frontal": _image_dict(request.frontal),
            "extra_frontals": [_image_dict(f) for f in request.extra_frontals],
            "profile": _image_dict(request.profile) if request.profile else None,
        }
    return out


def _image_dict(image: LoadedImage) -> dict[str, Any]:
    return {
        "sha256": image.sha256,
        "width": image.width,
        "height": image.height,
        "exif_tags_dropped": list(image.exif_tags_dropped),
        "had_gps": image.had_gps,
    }


def _measured_dict(m: Any) -> dict[str, Any]:
    disc = getattr(m, "discriminability", None)
    return {
        "id": m.spec_id,
        "label": m.label,
        "unit": m.unit.value,
        "value": m.value,
        "ci_low": m.ci_low,
        "ci_high": m.ci_high,
        "sd": m.sd,
        "reportability": m.verdict.reportability.value,
        "reasons": list(m.verdict.reasons),
        "discriminability": None if disc is None else disc.ratio,
        "formula_fingerprint": m.formula_fingerprint,
        "landmarks_used": list(m.landmarks_used),
        "scale_source": m.scale_source,
        "notes": list(m.notes),
    }


def _fallback_text(outcome: AnalysisOutcome) -> str:
    """Used only when the report package is absent. Same rules apply."""
    result = outcome.report
    measured = tuple(getattr(result, "measured", ()) or ())
    unavailable = tuple(getattr(result, "unavailable", ()) or ())
    shown = [m for m in measured if m.verdict.shown]
    withheld = [m for m in measured if not m.verdict.shown]

    lines = ["Faciometry report", "=" * 60, "", f"REPORTED ({len(shown)})"]
    lines += ["  " + m.format() for m in shown] or ["  none"]
    lines += ["", f"WITHHELD ({len(withheld)})"]
    for m in withheld:
        lines.append(f"  {m.label}: withheld")
        lines += [f"      {r}" for r in m.verdict.reasons]
    if not withheld:
        lines.append("  none")
    lines += ["", f"UNAVAILABLE ({len(unavailable)})"]
    lines += [f"  {u.label}: {u.reason}" for u in unavailable] or ["  none"]
    return "\n".join(lines)
