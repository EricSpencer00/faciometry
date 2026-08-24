"""The command line, its exit codes, and the two things it promises in print.

The tests that matter here are not the ones checking that a subcommand runs.
They are the ones checking that the license text the CLI prints agrees with
the license metadata a packager reads, and that the exit codes distinguish a
photograph the gate refused from a flag the user got wrong. Both are places
where a wrong answer is silent.
"""

from __future__ import annotations

import io
import json
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from vitruve.cli import catalogue as catalogue_cmd
from vitruve.cli import doctor as doctor_cmd
from vitruve.cli import licenses as licenses_cmd
from vitruve.cli import weights as weights_cmd
from vitruve.cli.exits import Exit
from vitruve.cli.main import main
from vitruve.cli.runner import BadImage, load_image, load_image_file
from vitruve.measure.registry import CATALOGUE
from vitruve.models.licensing import Tier

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def jpeg(tmp_path: Path) -> Path:
    path = tmp_path / "frontal.jpg"
    Image.new("RGB", (800, 1000), (140, 128, 118)).save(path, format="JPEG")
    return path


# --------------------------------------------------------------------- shape


def test_no_arguments_prints_help_and_succeeds(capsys):
    assert main([]) == Exit.OK
    assert "vitruve" in capsys.readouterr().out


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


# ---------------------------------------------------------------- catalogue


def test_catalogue_prints_every_measurement(capsys):
    assert main(["catalogue"]) == Exit.OK
    out = capsys.readouterr().out
    assert f"{len(CATALOGUE)} of {len(CATALOGUE)} measurements" in out
    for spec in CATALOGUE:
        assert spec.id in out


def test_catalogue_reports_the_ratio_that_decides_reportability(capsys):
    main(["catalogue"])
    out = capsys.readouterr().out
    # The column that matters: between-person spread over pose-induced movement.
    assert "ratio" in out
    assert "unknown" in out, "measurements with no published spread must say so"
    assert "Kleinberg and Vanezis (2007)" in out


def test_catalogue_json_is_machine_readable_and_has_no_aggregate(capsys):
    assert main(["catalogue", "--json"]) == Exit.OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_total"] == len(CATALOGUE)
    assert len(payload["measurements"]) == len(CATALOGUE)
    banned = ("overall", "harmony", "attractiveness", "total", "score")
    assert not any(k in payload for k in banned)


def test_catalogue_filters_by_view_and_evidence(capsys):
    main(["catalogue", "--view", "profile", "--json"])
    profile = json.loads(capsys.readouterr().out)["measurements"]
    assert profile and all(r["view"] == "profile" for r in profile)

    main(["catalogue", "--evidence", "requires_3d", "--json"])
    needs3d = json.loads(capsys.readouterr().out)["measurements"]
    assert needs3d and all(r["evidence"] == "requires_3d" for r in needs3d)


def test_catalogue_detail_names_its_formula_and_landmarks(capsys):
    assert main(["catalogue", "--id", "gonial_angle_l"]) == Exit.OK
    out = capsys.readouterr().out
    assert "gonion_l" in out
    assert "pose sensitivity" in out
    assert "references" in out


def test_catalogue_unknown_id_is_bad_input(capsys):
    assert main(["catalogue", "--id", "cheekbone_charisma"]) == Exit.BAD_INPUT


def test_between_subject_spread_is_never_invented():
    """A measurement with no published spread reports unknown, not a guess."""
    rows = catalogue_cmd.rows()
    unknown = [r for r in rows if r.between_subject_spread is None]
    assert unknown, "the catalogue used to contain measurements with no published spread"
    for row in unknown:
        assert row.discriminability_at_quoted_pose is None


# ----------------------------------------------------------------- licenses


def test_licenses_permissive_keeps_the_deployment_apache(capsys):
    assert main(["licenses", "--tier", "permissive"]) == Exit.OK
    out = " ".join(capsys.readouterr().out.split())
    assert "stays Apache-2.0" in out
    assert "BACKENDS THIS TIER REFUSES TO LOAD" in out


def test_licenses_copyleft_states_the_agpl_consequence(capsys):
    assert main(["licenses", "--tier", "copyleft"]) == Exit.OK
    out = " ".join(capsys.readouterr().out.split())
    assert "becomes AGPL-3.0" in out
    assert "network use as distribution" in out
    # The specific trap: a relabelled checkpoint does not launder the licence.
    assert "relabels" in out


def test_licenses_names_the_backend_it_will_not_load(capsys):
    main(["licenses", "--tier", "permissive"])
    out = " ".join(capsys.readouterr().out.split())
    assert "STAR loss landmark model" in out
    assert "no license file" in out


def test_extras_text_matches_pyproject():
    """The obligation a packager reads and the one the CLI prints are one text."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = data["tool"]["vitruve"]["extras"]
    assert declared == licenses_cmd.EXTRAS
    for name in data["project"]["optional-dependencies"]:
        assert name in declared, f"extra {name} has no stated obligation"
    assert "AGPL-3.0" in declared["copyleft"]


def test_tier_parsing_round_trips_and_rejects_nonsense():
    for tier in Tier:
        assert licenses_cmd.tier_from_string(tier.name.lower()) is tier
    with pytest.raises(ValueError, match="unknown license tier"):
        licenses_cmd.tier_from_string("free")


# ------------------------------------------------------------------- doctor


def test_doctor_reports_device_and_catalogue(capsys):
    assert main(["doctor"]) in (Exit.OK, Exit.ERROR)
    out = capsys.readouterr().out
    assert "device" in out
    assert f"{len(CATALOGUE)} measurements" in out


def _lock(tmp_path: Path, entries: dict) -> Path:
    lock = tmp_path / "weights.lock.json"
    lock.write_text(json.dumps({"weights": entries}, indent=2))
    return lock


def _artifact(tmp_path: Path, payload: bytes, *, provenance: str, sha: str | None = None) -> dict:
    import hashlib

    source = tmp_path / "artifact.bin"
    source.write_bytes(payload)
    return {
        "filename": "artifact.bin",
        "url": source.as_uri(),
        "sha256": sha or hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "provenance": provenance,
        "license_id": "MIT",
        "note": "a test fixture",
    }


@pytest.fixture
def sandboxed_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VITRUVE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache" / "weights"


def test_doctor_flags_a_corrupt_weight_file(tmp_path, monkeypatch, sandboxed_cache):
    lock = _lock(
        tmp_path,
        {"fake": _artifact(tmp_path, b"the pinned bytes", provenance="YuNet face detector")},
    )
    monkeypatch.setenv("VITRUVE_WEIGHTS_LOCK", str(lock))
    sandboxed_cache.mkdir(parents=True)
    (sandboxed_cache / "artifact.bin").write_bytes(b"not the pinned artifact")

    checks = doctor_cmd.collect(Tier.PERMISSIVE)
    corrupt = [c for c in checks if c.name == "fake"]
    assert corrupt and corrupt[0].level == doctor_cmd.FAIL
    assert "sha256 does not match" in corrupt[0].detail


# ------------------------------------------------------------ fetch-weights


def test_every_pinned_artifact_belongs_to_a_known_backend():
    """The lock file and the licensing catalogue have to agree on the tier.

    They are separate files maintained for separate reasons, and a pin whose
    provenance string does not match a `Provenance.name` would silently be
    treated as unlicensed and never fetched.
    """
    for pin in weights_cmd.pins():
        assert pin.tier is not Tier.UNLICENSED, (
            f"{pin.key} names provenance {pin.provenance!r}, which is not in "
            "vitruve.models.licensing.CATALOGUE"
        )


def test_the_default_stack_is_entirely_permissive():
    permitted, refused = weights_cmd.split_by_tier(Tier.PERMISSIVE)
    assert permitted, "the permissive tier has nothing pinned to it"
    assert all(p.tier is Tier.PERMISSIVE for p in permitted)
    assert {p.license_id for p in permitted} <= {"MIT", "BSD-3-Clause", "Apache-2.0"}
    assert not any(p.tier is Tier.PERMISSIVE for p in refused)


def test_an_unknown_provenance_is_treated_as_unlicensed():
    assert weights_cmd.tier_of("YuNet face detector") is Tier.PERMISSIVE
    assert weights_cmd.tier_of("YOLO face detector (Ultralytics lineage)") is Tier.COPYLEFT
    assert weights_cmd.tier_of("something nobody declared") is Tier.UNLICENSED


def test_fetch_writes_only_after_the_digest_matches(tmp_path, monkeypatch, sandboxed_cache):
    from vitruve.models import weights as models_weights

    payload = b"weights, notionally"
    lock = _lock(
        tmp_path, {"fake": _artifact(tmp_path, payload, provenance="YuNet face detector")}
    )
    monkeypatch.setenv("VITRUVE_WEIGHTS_LOCK", str(lock))

    assert main(["fetch-weights"]) == Exit.OK
    assert (sandboxed_cache / "artifact.bin").read_bytes() == payload
    assert models_weights.verify_only("fake").ok


def test_a_changed_artifact_is_a_hard_failure(tmp_path, monkeypatch, sandboxed_cache, capsys):
    from vitruve.models import weights as models_weights

    lock = _lock(
        tmp_path,
        {
            "fake": _artifact(
                tmp_path, b"served bytes", provenance="YuNet face detector", sha="f" * 64
            )
        },
    )
    monkeypatch.setenv("VITRUVE_WEIGHTS_LOCK", str(lock))

    with pytest.raises(models_weights.WeightHashMismatch):
        models_weights.download("fake")
    assert not (sandboxed_cache / "artifact.bin").exists()

    # And through the command, where it is a failure rather than an exception.
    assert main(["fetch-weights"]) == 1
    assert "FAILED" in capsys.readouterr().err


def test_fetch_refuses_an_artifact_above_the_tier(tmp_path, monkeypatch, sandboxed_cache, capsys):
    lock = _lock(
        tmp_path,
        {
            "yolo": _artifact(
                tmp_path,
                b"agpl weights",
                provenance="YOLO face detector (Ultralytics lineage)",
            )
        },
    )
    monkeypatch.setenv("VITRUVE_WEIGHTS_LOCK", str(lock))

    assert main(["fetch-weights"]) == Exit.OK
    out = capsys.readouterr().out
    assert "refused" in out
    assert "--tier copyleft" in out
    assert not (sandboxed_cache / "artifact.bin").exists()

    assert main(["fetch-weights", "--tier", "copyleft"]) == Exit.OK
    assert (sandboxed_cache / "artifact.bin").exists()


def test_fetch_without_a_lock_file_is_bad_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VITRUVE_WEIGHTS_LOCK", str(tmp_path / "absent.json"))
    assert main(["fetch-weights"]) == Exit.BAD_INPUT
    assert "no weight lock file" in capsys.readouterr().err


# ------------------------------------------------------------------ analyze


def test_analyze_missing_file_is_bad_input(capsys):
    assert main(["analyze", "/nonexistent/face.jpg"]) == Exit.BAD_INPUT
    assert "no such file" in capsys.readouterr().err


def test_analyze_undecodable_file_is_bad_input(tmp_path, capsys):
    bogus = tmp_path / "face.jpg"
    bogus.write_bytes(b"this is not a JPEG")
    assert main(["analyze", str(bogus)]) == Exit.BAD_INPUT
    err = capsys.readouterr().err
    assert "vitruve analyze:" in err
    assert "decod" in err


def test_analyze_rejects_an_unknown_license_tier(jpeg):
    with pytest.raises(SystemExit) as exc:
        main(["analyze", str(jpeg), "--license-tier", "whatever"])
    assert exc.value.code == 2


def test_analyze_warns_before_accepting_a_copyleft_tier(jpeg, capsys):
    main(["analyze", str(jpeg), "--license-tier", "copyleft"])
    err = capsys.readouterr().err
    assert "license tier copyleft selected" in err
    assert "vitruve licenses --tier copyleft" in err


def test_analyze_on_a_faceless_image_never_exits_zero(jpeg):
    """A grey rectangle has no face in it, and no measurement may come back."""
    assert main(["analyze", str(jpeg)]) != Exit.OK


def test_analyze_accepts_several_frontal_photographs():
    from vitruve.cli.main import build_parser

    args = build_parser().parse_args(["analyze", "a.jpg", "b.jpg", "c.jpg"])
    assert args.frontal == ["a.jpg", "b.jpg", "c.jpg"]
    assert args.profile is None


def test_one_frontal_photograph_is_still_a_list_of_one():
    from vitruve.cli.main import build_parser

    assert build_parser().parse_args(["analyze", "a.jpg"]).frontal == ["a.jpg"]


def test_the_help_says_the_captures_must_be_one_person_in_one_session(capsys):
    """Pooling photographs of two people would average a face nobody has."""
    with pytest.raises(SystemExit):
        main(["analyze", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "same person in the same session" in out


def test_a_missing_file_among_several_frontals_is_bad_input(capsys):
    assert main(["analyze", "/nonexistent/a.jpg", "/nonexistent/b.jpg"]) == Exit.BAD_INPUT
    assert "no such file" in capsys.readouterr().err


# -------------------------------------------------------------------- ingest


def test_load_image_strips_exif_and_notices_gps(tmp_path):
    exif = Image.Exif()
    exif[0x010F] = "TestCamera"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (41.0, 52.0, 43.0)

    path = tmp_path / "gps.jpg"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(path, format="JPEG", exif=exif)

    loaded = load_image_file(path)
    assert loaded.had_gps
    assert "Make" in loaded.exif_tags_dropped
    assert loaded.path == path

    stored = Image.open(io.BytesIO(loaded.to_png_bytes()))
    assert not dict(stored.getexif())


def test_load_image_rejects_a_directory(tmp_path):
    with pytest.raises(BadImage, match="is a directory"):
        load_image_file(tmp_path)


def test_load_image_rejects_empty_bytes():
    with pytest.raises(BadImage):
        load_image(b"")


# ---------------------------------------------------------------------- bind


def test_serve_refuses_a_public_bind_without_the_flag(capsys):
    from vitruve.api.serve import RemoteBindRefused, is_loopback, resolve_bind

    assert is_loopback("127.0.0.1")
    assert is_loopback("::1")
    assert not is_loopback("0.0.0.0")

    assert resolve_bind("127.0.0.1", allow_remote=False) == "127.0.0.1"
    with pytest.raises(RemoteBindRefused, match="no authentication"):
        resolve_bind("0.0.0.0", allow_remote=False)
    assert resolve_bind("0.0.0.0", allow_remote=True) == "0.0.0.0"

    assert main(["serve", "--host", "0.0.0.0"]) == Exit.BAD_INPUT
    assert "refusing to bind" in capsys.readouterr().err
