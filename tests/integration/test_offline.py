"""Analysis does not touch the network, asserted rather than promised.

The claim is easy to make and easy to break: one library that phones home for
a model file on first use, and a tool that reads faces is talking to a server.
`insightface` does exactly that, which is why the licensing module names it.

So the guarantee is enforced here by replacing ``socket.socket``,
``socket.create_connection`` and ``socket.getaddrinfo`` with functions that
raise, and then running the measurement path underneath. The first test in the
file is the control: it proves the block actually blocks. Without that, every
other test in the file would pass just as well against a fixture that does
nothing.
"""

from __future__ import annotations

import io
import socket
import urllib.error
import urllib.request

import numpy as np
import pytest

from faciometry.cli.main import main
from faciometry.core.landmarks import Landmark as L
from faciometry.core.landmarks import PointSet
from faciometry.core.scale import from_interpupillary
from faciometry.core.spec import Reportability
from faciometry.measure.evaluate import LandmarkUncertainty, Measured, Unavailable, evaluate
from faciometry.measure.registry import CATALOGUE


class NetworkBlocked(OSError):
    """Raised in place of any socket operation while a test holds the block."""


@pytest.fixture
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise NetworkBlocked("outbound network is blocked by this test")

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    return deny


# A synthetic adult face in millimetres, origin at nasion, +x the subject's
# right, +y up, +z toward the viewer. Coordinates are approximate Farkas-order
# anthropometry: they exist so the formulas have non-degenerate geometry to
# evaluate, not to be anyone's face.
FACE_MM: dict[L, tuple[float, float, float]] = {
    L.TRICHION: (0, 95, -10),
    L.GLABELLA: (0, 20, 8),
    L.NASION: (0, 0, 0),
    L.SELLION: (0, -5, 2),
    L.ENDOCANTHION_L: (16, -4, -2),
    L.ENDOCANTHION_R: (-16, -4, -2),
    L.EXOCANTHION_L: (46, -2, -14),
    L.EXOCANTHION_R: (-46, -2, -14),
    L.PUPIL_L: (31, -3, -6),
    L.PUPIL_R: (-31, -3, -6),
    L.PALPEBRALE_SUP_L: (31, 4, -6),
    L.PALPEBRALE_SUP_R: (-31, 4, -6),
    L.PALPEBRALE_INF_L: (31, -9, -6),
    L.PALPEBRALE_INF_R: (-31, -9, -6),
    L.ORBITALE_L: (33, -14, -12),
    L.ORBITALE_R: (-33, -14, -12),
    L.SUPERCILIARE_L: (28, 14, -4),
    L.SUPERCILIARE_R: (-28, 14, -4),
    L.PRONASALE: (0, -46, 28),
    L.SUBNASALE: (0, -55, 12),
    L.ALARE_L: (17, -50, 8),
    L.ALARE_R: (-17, -50, 8),
    L.SUBALARE_L: (13, -54, 9),
    L.SUBALARE_R: (-13, -54, 9),
    L.COLUMELLA: (0, -50, 20),
    L.LABIALE_SUPERIUS: (0, -68, 14),
    L.LABIALE_INFERIUS: (0, -80, 12),
    L.STOMION: (0, -73, 13),
    L.CHEILION_L: (25, -73, 2),
    L.CHEILION_R: (-25, -73, 2),
    L.CRISTA_PHILTRI_L: (6, -66, 14),
    L.CRISTA_PHILTRI_R: (-6, -66, 14),
    L.SUBLABIALE: (0, -90, 6),
    L.POGONION: (0, -105, 12),
    L.GNATHION: (0, -113, 6),
    L.MENTON: (0, -117, 0),
    L.GONION_L: (62, -85, -40),
    L.GONION_R: (-62, -85, -40),
    L.ZYGION_L: (70, -20, -30),
    L.ZYGION_R: (-70, -20, -30),
    L.TRAGION_L: (75, -5, -75),
    L.TRAGION_R: (-75, -5, -75),
    L.PORION_L: (75, 0, -78),
    L.PORION_R: (-75, 0, -78),
    L.CERVICALE: (0, -140, -50),
}


def synthetic_face() -> PointSet:
    return PointSet.from_mapping(
        {name: np.array(xyz, dtype=float) for name, xyz in FACE_MM.items()}
    )


def measure_everything(seed: int = 0) -> list[Measured | Unavailable]:
    """Run every spec in the catalogue over the synthetic face."""
    points = synthetic_face()
    uncertainty = LandmarkUncertainty.isotropic(points, sd=0.8)
    scale = from_interpupillary(ipd_px=62.0, declared_sex="female")
    return [
        evaluate(
            spec,
            points,
            uncertainty,
            yaw_deg=1.5,
            pitch_deg=2.0,
            roll_deg=0.5,
            have_3d=True,
            scale=scale,
            subject_distance_m=1.5,
            n_samples=256,
            seed=seed,
        )
        for spec in CATALOGUE
    ]


def test_the_block_actually_blocks(no_network):
    """Control. If this passes trivially, nothing else in the file means anything."""
    with pytest.raises((NetworkBlocked, urllib.error.URLError, OSError)):
        urllib.request.urlopen("http://127.0.0.1:9/never", timeout=1)

    with pytest.raises(NetworkBlocked):
        socket.socket()


def test_measurement_runs_with_the_network_blocked(no_network):
    results = measure_everything()

    # Not a hardcoded count: the catalogue grows, and pinning it here means a
    # measurement added elsewhere fails an offline test for no reason. What
    # matters is that every catalogue entry was evaluated with no network.
    assert len(results) == len(CATALOGUE)
    assert len(CATALOGUE) >= 45
    measured = [r for r in results if isinstance(r, Measured)]
    assert measured, "no measurement evaluated on a complete synthetic face"

    reported = [m for m in measured if m.verdict.reportability is not Reportability.WITHHOLD]
    assert reported, "every measurement was withheld on a clean synthetic face"

    # Rule 4 of the core contract: no bare point estimates. Anything shown
    # carries a two-sided interval around its value.
    for m in reported:
        assert m.ci_low <= m.value <= m.ci_high
        assert m.format().count("[") == 1

    # Rule 3: a withheld measurement is a result, and it has to say why.
    for m in measured:
        if m.verdict.reportability is Reportability.WITHHOLD:
            assert m.verdict.reasons, f"{m.spec_id} was withheld without a reason"


def test_measurement_is_deterministic_under_a_seed(no_network):
    a = [r.value for r in measure_everything(seed=7) if isinstance(r, Measured)]
    b = [r.value for r in measure_everything(seed=7) if isinstance(r, Measured)]
    assert a == b


def test_no_aggregate_score_anywhere_in_the_output(no_network):
    """Rule 1. Every result is a named measurement, and there is no total."""
    banned = ("overall", "harmony", "attractive", "score", "rating", "rank")
    for result in measure_everything():
        identifier = result.spec_id.lower()
        assert not any(word in identifier for word in banned)


def test_cli_commands_that_must_never_need_the_network(no_network, capsys):
    for argv in (["catalogue"], ["licenses", "--tier", "permissive"], ["doctor"]):
        assert main(argv) == 0, argv
    out = capsys.readouterr().out
    assert "measurement catalogue" in out


def test_analyze_reaches_the_pipeline_without_a_socket(no_network, tmp_path):
    """Whatever `analyze` does, it does not get there by opening a connection."""
    from PIL import Image

    from faciometry.cli.runner import AnalysisRequest, Status, load_image, run_analysis

    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (128, 120, 110)).save(buf, format="JPEG")
    request = AnalysisRequest(frontal=load_image(buf.getvalue(), source="synthetic"))

    outcome = run_analysis(request)

    # A build without a pipeline reports that, a build with one runs it, and a
    # grey rectangle has no face in it. Every one of those is a valid outcome.
    # What none of them is, is a network error: a socket opened anywhere under
    # this call would have raised NetworkBlocked straight out of it.
    assert outcome.status in (
        Status.OK,
        Status.UNAVAILABLE,
        Status.QUALITY_GATE,
        Status.BAD_INPUT,
    )
    assert "NetworkBlocked" not in outcome.message
    assert "blocked by this test" not in outcome.message


def test_ingest_drops_exif_with_the_network_blocked(no_network, tmp_path):
    from PIL import Image

    from faciometry.cli.runner import load_image

    exif = Image.Exif()
    exif[0x010F] = "TestCamera"
    exif[0x0132] = "2026:08:23 10:11:12"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (41.0, 52.0, 43.0)
    gps[3] = "W"
    gps[4] = (87.0, 37.0, 24.0)

    path = tmp_path / "with_gps.jpg"
    Image.new("RGB", (320, 240), (90, 90, 90)).save(path, format="JPEG", exif=exif)

    loaded = load_image(path.read_bytes(), source=str(path))

    assert loaded.had_gps, "the fixture did not actually carry a GPS tag"
    assert "GPSInfo" in loaded.exif_tags_dropped

    # The re-encoded copy is what `--store` writes, and it carries nothing.
    stored = Image.open(io.BytesIO(loaded.to_png_bytes()))
    assert not dict(stored.getexif())
