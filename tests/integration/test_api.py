"""The local HTTP API.

Three properties are worth a test and the rest is plumbing: the server refuses
to leave loopback by accident, an upload is not written to disk unless the
operator asked for it, and every failure comes back as a status a client can
branch on rather than a 500.

The multipart spool is the subtle one. Starlette writes any uploaded file over
a megabyte into a temporary file, so "held in memory" is a claim about a
configuration value, not about the code being obviously correct. It is
asserted here against a file large enough to have triggered the old default.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from faciometry.api.app import MAX_UPLOAD_BYTES, create_app
from faciometry.cli.runner import analysis_available
from faciometry.measure.registry import CATALOGUE


def jpeg_bytes(size=(640, 800), colour=(150, 130, 120), quality=90) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def noisy_jpeg_bytes(size=(1600, 2000)) -> bytes:
    """A large, incompressible image, so the multipart part is over a megabyte."""
    import numpy as np

    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def storing_client(tmp_path: Path) -> tuple[TestClient, Path]:
    store = tmp_path / "store"
    return TestClient(create_app(store=True, store_dir=store)), store


# -------------------------------------------------------------------- health


def test_health_says_what_this_build_can_do(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_measurements"] == len(CATALOGUE)
    assert body["device"] in ("cpu", "mps", "cuda")
    assert body["pipeline_available"] == analysis_available()
    assert body["storing_uploads"] is False


# ----------------------------------------------------------------- catalogue


def test_catalogue_is_the_same_catalogue_the_cli_prints(client):
    body = client.get("/catalogue").json()
    assert body["n_total"] == len(CATALOGUE)
    ids = {m["id"] for m in body["measurements"]}
    assert ids == {s.id for s in CATALOGUE}
    assert set(body["evidence_meaning"]) == {s.evidence.value for s in CATALOGUE}


def test_catalogue_carries_no_aggregate_field(client):
    body = client.get("/catalogue").json()
    banned = ("score", "overall", "harmony", "attractiveness", "rating", "rank")
    text = json.dumps(body).lower()
    for word in banned:
        assert word not in text, f"the catalogue payload mentions {word!r}"


def test_catalogue_filters(client):
    body = client.get("/catalogue", params={"view": "profile"}).json()
    assert body["measurements"] and all(m["view"] == "profile" for m in body["measurements"])
    assert client.get("/catalogue", params={"view": "sideways"}).status_code == 400


# ------------------------------------------------------------------ licenses


def test_licenses_endpoint_states_the_agpl_consequence(client):
    res = client.get("/licenses", params={"tier": "copyleft"})
    assert res.status_code == 200
    assert "becomes AGPL-3.0" in " ".join(res.text.split())


def test_licenses_endpoint_rejects_an_unknown_tier(client):
    assert client.get("/licenses", params={"tier": "gratis"}).status_code == 400


# ------------------------------------------------------------------- analyze


def test_analyze_requires_a_frontal_image(client):
    assert client.post("/analyze").status_code == 422


def test_analyze_rejects_bytes_that_are_not_an_image(client):
    res = client.post("/analyze", files={"frontal": ("f.jpg", b"nope", "image/jpeg")})
    assert res.status_code == 400
    assert "decod" in res.json()["detail"]


def test_analyze_rejects_an_empty_upload(client):
    res = client.post("/analyze", files={"frontal": ("f.jpg", b"", "image/jpeg")})
    assert res.status_code == 400


def test_analyze_returns_a_status_a_client_can_branch_on(client):
    res = client.post("/analyze", files={"frontal": ("f.jpg", jpeg_bytes(), "image/jpeg")})
    # 200 with measurements once backends are installed; 422 when the gate
    # refuses the photograph; 503 when this build has no pipeline. Never a 500.
    assert res.status_code in (200, 422, 503)
    body = res.json()
    if res.status_code == 200:
        assert "measurements" in body and "withheld" in body
        assert "score" not in json.dumps(body).lower()
    else:
        assert body["detail"]


# --------------------------------------------------------------------- store


def test_nothing_is_written_to_disk_by_default(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client.post("/analyze", files={"frontal": ("f.jpg", jpeg_bytes(), "image/jpeg")})
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written == [], f"an upload reached disk: {written}"


def test_store_writes_a_metadata_free_copy(storing_client):
    client, store = storing_client
    exif = Image.Exif()
    exif[0x010F] = "TestCamera"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (41.0, 52.0, 43.0)
    buf = io.BytesIO()
    Image.new("RGB", (320, 400), (90, 80, 70)).save(buf, format="JPEG", exif=exif)

    client.post("/analyze", files={"frontal": ("f.jpg", buf.getvalue(), "image/jpeg")})

    written = sorted(store.glob("*.png"))
    assert len(written) == 1
    assert not dict(Image.open(written[0]).getexif())


def test_a_large_upload_does_not_spool_to_a_temporary_file(client, monkeypatch, tmp_path):
    """The 'held in memory' claim, against a part over the old 1 MB spool."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    payload = noisy_jpeg_bytes()
    assert len(payload) > 1024 * 1024, "the fixture is too small to test the spool"

    client.post("/analyze", files={"frontal": ("big.jpg", payload, "image/jpeg")})

    spooled = [p for p in tmp_path.rglob("*") if p.is_file() and p.stat().st_size > 1_000_000]
    assert spooled == [], f"an upload was spooled to disk: {spooled}"


def test_an_upload_over_the_ceiling_is_refused(client):
    oversized = b"\xff\xd8\xff" + b"\x00" * (MAX_UPLOAD_BYTES + 1)
    res = client.post("/analyze", files={"frontal": ("big.jpg", oversized, "image/jpeg")})
    assert res.status_code == 400
    assert "ceiling" in res.json()["detail"]


# ----------------------------------------------------------------------- ui


def test_the_web_ui_is_served_from_the_same_origin(client):
    """No third-party origin anywhere in the page.

    A CDN request for a font or a stylesheet would announce to somebody else
    that a face-analysis tool was opened, which is why there is no build step
    and no webfont.
    """
    import re

    index = client.get("/")
    assert index.status_code == 200
    assert "Faciometry" in index.text

    remote_ref = re.compile(
        r"""(?:src|href)\s*=\s*["']https?://"""
        r"""|from\s+["']https?://"""
        r"""|url\(\s*["']?https?://"""
        r"""|@import\s+["']?https?://""",
        re.IGNORECASE,
    )
    assert not remote_ref.search(index.text)

    for asset in ("faciometry.css", "app.js", "capture.js", "report.js"):
        res = client.get(f"/static/{asset}")
        assert res.status_code == 200, asset
        assert not remote_ref.search(res.text), asset
