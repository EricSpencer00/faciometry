"""The local HTTP API.

This service receives photographs of faces, so two defaults are chosen against
convenience.

**It binds loopback.** ``vitruve serve`` refuses a non-loopback address unless
the operator passes ``--allow-remote``, and prints what that means before it
binds. A face-analysis service that defaults to ``0.0.0.0`` is one
``docker run -p`` away from being a public face-analysis service.

**Uploads stay in memory.** Starlette spools any multipart file over a
megabyte to a temporary file on disk, which would quietly break the promise
that nothing is written. :func:`create_app` raises that threshold above the
upload ceiling so the spool never triggers, and writes an image out only when
the operator asked for ``--store``.

Endpoints:

===============  ==============================================================
GET  /           the web UI
GET  /health     version, device, whether weights and the pipeline are present
GET  /catalogue  the 45 measurement specs, same data the CLI prints
GET  /licenses   obligations for a tier
POST /analyze    multipart: frontal, optional profile, declared attributes
===============  ==============================================================

Statuses `/analyze` returns, and why each one: 400 for an image that will not
decode, 422 when the photograph did not clear the quality gate (the request
was well formed, the photograph was not), 451 when a backend exceeds the
permitted license tier, and 503 when this build has no pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartParser

from .. import __version__
from ..cli import catalogue as catalogue_cmd
from ..cli import doctor as doctor_cmd
from ..cli import licenses as licenses_cmd
from ..cli.runner import (
    AnalysisRequest,
    BadImage,
    Status,
    analysis_available,
    load_image,
    run_analysis,
    to_dict,
)
from ..core.spec import Evidence, View
from ..models.licensing import Tier

#: Largest upload accepted, per image. A portrait that clears the quality gate
#: is a few megabytes; this is generous enough for a raw phone capture and
#: small enough that holding it in memory is not a denial of service.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

STATUS_HTTP = {
    Status.BAD_INPUT: 400,
    Status.QUALITY_GATE: 422,
    Status.LICENSE: 451,
    Status.UNAVAILABLE: 503,
}


def web_root() -> Path | None:
    """Where the single-page UI lives, in a wheel or in a checkout."""
    override = os.environ.get("VITRUVE_WEB")
    if override:
        p = Path(override)
        return p if (p / "index.html").exists() else None
    here = Path(__file__).resolve()
    for candidate in (here.parents[1] / "web", here.parents[3] / "web"):
        if (candidate / "index.html").exists():
            return candidate
    return None


def create_app(
    *,
    store: bool = False,
    store_dir: Path | None = None,
    license_tier: Tier = Tier.PERMISSIVE,
) -> FastAPI:
    # Class attributes, so this applies to every multipart parse in the
    # process. That is the point: there is no code path where an uploaded face
    # lands in /tmp because a request took a different route.
    MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES + (1 << 20)

    app = FastAPI(
        title="Vitruve",
        version=__version__,
        description=(
            "Facial morphometrics with intervals. Runs locally, downloads nothing "
            "during analysis, and emits no aggregate score."
        ),
    )
    app.state.store = store
    app.state.store_dir = Path(store_dir) if store_dir else Path("vitruve-store")
    app.state.license_tier = license_tier

    @app.get("/health")
    def health() -> dict:
        device, why = doctor_cmd.resolve_device()
        return {
            "status": "ok",
            "version": __version__,
            "device": device,
            "device_detail": why,
            "pipeline_available": analysis_available(),
            "license_tier": app.state.license_tier.name.lower(),
            "storing_uploads": bool(app.state.store),
            "n_measurements": len(catalogue_cmd.CATALOGUE),
        }

    @app.get("/catalogue")
    def catalogue(view: str | None = None, evidence: str | None = None) -> dict:
        try:
            rows = catalogue_cmd.rows(
                view=View(view) if view else None,
                evidence=Evidence(evidence) if evidence else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "quoted_pose_deg": catalogue_cmd.QUOTED_POSE_DEG,
            "n_total": len(catalogue_cmd.CATALOGUE),
            "evidence_meaning": {
                e.value: text for e, text in catalogue_cmd.EVIDENCE_MEANING.items()
            },
            "measurements": [r.__dict__ for r in rows],
        }

    @app.get("/licenses", response_class=PlainTextResponse)
    def licenses(tier: str = "permissive") -> str:
        try:
            parsed = licenses_cmd.tier_from_string(tier)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return licenses_cmd.render(parsed)

    @app.post("/analyze")
    async def analyze(
        frontal: Annotated[UploadFile, File(description="frontal photograph")],
        profile: Annotated[UploadFile | None, File()] = None,
        declared_sex: Annotated[str | None, Form()] = None,
        declared_ancestry: Annotated[str | None, Form()] = None,
        ruler_mm: Annotated[float | None, Form()] = None,
        seed: Annotated[int, Form()] = 0,
    ) -> JSONResponse:
        try:
            frontal_image = load_image(
                await _read_capped(frontal), source=frontal.filename or "frontal"
            )
            profile_image = (
                load_image(await _read_capped(profile), source=profile.filename or "profile")
                if profile is not None and profile.filename
                else None
            )
        except BadImage as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if app.state.store:
            _store(app.state.store_dir, frontal_image)
            if profile_image is not None:
                _store(app.state.store_dir, profile_image)

        outcome = run_analysis(
            AnalysisRequest(
                frontal=frontal_image,
                profile=profile_image,
                license_tier=app.state.license_tier,
                declared_sex=declared_sex or None,
                declared_ancestry=declared_ancestry or None,
                ruler_mm=ruler_mm,
                seed=seed,
            )
        )
        if outcome.status is Status.OK:
            return JSONResponse(to_dict(outcome))
        return JSONResponse(
            status_code=STATUS_HTTP[outcome.status],
            content={
                "status": outcome.status.value,
                "detail": outcome.message,
                "reasons": list(outcome.reasons),
            },
        )

    root = web_root()
    if root is not None:
        app.mount("/static", StaticFiles(directory=root), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(root / "index.html")

    else:  # pragma: no cover - only when the wheel was built without web/

        @app.get("/", response_class=PlainTextResponse)
        def index_missing() -> str:
            return (
                "The web UI is not present in this install. The API is up: try "
                "/health, /catalogue, /licenses, and POST /analyze.\n"
            )

    return app


async def _read_capped(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise BadImage(
            f"{upload.filename or 'upload'}: {len(data) // (1 << 20)} MB exceeds the "
            f"{MAX_UPLOAD_BYTES // (1 << 20)} MB ceiling"
        )
    if not data:
        raise BadImage(f"{upload.filename or 'upload'}: empty upload")
    return data


def _store(directory: Path, image) -> Path:
    """Write a metadata-free copy, and only when asked.

    Re-encoded from the pixel array rather than copied, so the stored file
    cannot carry the capture timestamp or the GPS fix that came in with it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{image.sha256[:16]}.png"
    path.write_bytes(image.to_png_bytes())
    return path
