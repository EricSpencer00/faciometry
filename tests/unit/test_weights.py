"""The weight cache: pins hold, and analysis never reaches the network.

Two properties are load-bearing and both are asserted here rather than trusted.

**A hash mismatch is fatal.** Not a warning, not a re-download. The two ways a
pinned hash stops matching are a compromised mirror and an upstream quietly
republishing under the same URL, and in both cases every number the previous
weights produced is a different measurement.

**`resolve` cannot open a socket.** The test that proves it replaces
`urllib.request.urlopen` with something that raises, then calls the whole
read path. A comment saying "this does not download" is not evidence.

Every test here runs against a temporary cache and a fixture lock file, so
nothing depends on what happens to be in `~/.cache/faciometry` on the machine
running it.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import pytest

from faciometry.models import weights as W

PAYLOAD = b"a small artefact standing in for a checkpoint\n" * 8
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def lockfile(tmp_path):
    """A two-entry lock file, one of whose URLs is a local file."""
    source = tmp_path / "upstream" / "toy.bin"
    source.parent.mkdir()
    source.write_bytes(PAYLOAD)
    body = {
        "weights": {
            "toy": {
                "filename": "toy.bin",
                "url": source.as_uri(),
                "sha256": DIGEST,
                "size_bytes": len(PAYLOAD),
                "provenance": "Toy backend",
                "license_id": "MIT",
                "note": "a fixture, not a model",
            },
            "absent": {
                "filename": "absent.bin",
                "url": "https://example.invalid/absent.bin",
                "sha256": "0" * 64,
                "size_bytes": 1,
                "provenance": "Missing backend",
                "license_id": "MIT",
            },
        }
    }
    path = tmp_path / "weights.lock.json"
    path.write_text(json.dumps(body))
    return path


@pytest.fixture
def cache(tmp_path, lockfile, monkeypatch):
    monkeypatch.setenv(W.CACHE_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.setenv(W.LOCK_ENV_VAR, str(lockfile))
    return tmp_path / "cache" / "weights"


@pytest.fixture
def no_network(monkeypatch):
    """Any socket attempt becomes a loud failure."""

    def explode(*_args, **_kwargs):
        raise AssertionError("this code path opened a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", explode)


# ---------------------------------------------------------------------------
# The lock file itself
# ---------------------------------------------------------------------------


def test_the_shipped_lock_file_parses_and_every_pin_is_well_formed():
    """The real ``assets/weights.lock.json``, not a fixture.

    A malformed pin is only discovered at the moment somebody tries to fetch a
    model, which is the least convenient moment.
    """
    specs = W.load_lock()
    assert set(specs) == {"yunet", "spiga_wflw", "mediapipe_face_landmarker", "sixdrepnet_300wlp"}
    for key, spec in specs.items():
        assert spec.key == key
        assert len(spec.sha256) == 64
        assert spec.size_bytes > 0
        assert spec.url.startswith("https://")
        assert spec.note, f"{key} has no note saying where the pin came from"


def test_every_pinned_artefact_names_a_backend_in_the_licensing_catalogue():
    """A weight file whose obligations are not recorded is a weight file that
    can be loaded past the tier check."""
    from faciometry.models.licensing import BY_NAME

    for spec in W.load_lock().values():
        assert spec.provenance in BY_NAME, spec.provenance
        assert BY_NAME[spec.provenance].license_id.startswith(spec.license_id[:3])


def test_a_bad_hash_in_a_lock_file_is_rejected_at_parse_time():
    with pytest.raises(ValueError, match="64 lowercase hex"):
        W.WeightSpec(
            key="k",
            filename="f",
            url="https://example.invalid/f",
            sha256="NOTAHASH",
            size_bytes=1,
            provenance="p",
            license_id="MIT",
        )


def test_a_missing_lock_file_names_itself(tmp_path, monkeypatch):
    monkeypatch.setenv(W.LOCK_ENV_VAR, str(tmp_path / "nope.json"))
    with pytest.raises(W.WeightsError, match=r"nope\.json"):
        W.load_lock()


def test_an_unknown_key_lists_the_known_ones(cache):
    with pytest.raises(W.UnknownWeight, match=r"absent.*toy"):
        W.spec_for("nonexistent")


# ---------------------------------------------------------------------------
# resolve: offline, verifying, and fatal on mismatch
# ---------------------------------------------------------------------------


def test_resolve_never_touches_the_network(cache, no_network):
    """Design rule 5: no egress during analysis.

    Both branches are exercised -- the file present and the file absent -- so a
    lazy download hidden in the not-found path would be caught.
    """
    with pytest.raises(W.WeightsUnavailable):
        W.resolve("toy")
    cache.mkdir(parents=True)
    (cache / "toy.bin").write_bytes(PAYLOAD)
    assert W.resolve("toy").read_bytes() == PAYLOAD


def test_an_absent_artefact_tells_the_user_how_to_fetch_it(cache, no_network):
    with pytest.raises(W.WeightsUnavailable) as exc:
        W.resolve("toy")
    message = str(exc.value)
    assert "weights fetch toy" in message
    assert "Analysis does not download" in message
    assert "Toy backend" in message


def test_a_hash_mismatch_is_fatal_and_explains_why(cache, no_network):
    cache.mkdir(parents=True)
    (cache / "toy.bin").write_bytes(PAYLOAD + b"tampered")
    with pytest.raises(W.WeightHashMismatch) as exc:
        W.resolve("toy")
    message = str(exc.value)
    assert DIGEST in message
    assert "different measurement" in message


def test_a_truncated_file_is_caught_rather_than_handed_to_torch(cache, no_network):
    """An interrupted fetch otherwise surfaces as an unpickling error."""
    cache.mkdir(parents=True)
    (cache / "toy.bin").write_bytes(PAYLOAD[: len(PAYLOAD) // 2])
    with pytest.raises(W.WeightHashMismatch):
        W.resolve("toy")


# ---------------------------------------------------------------------------
# verify_only: reports rather than raising
# ---------------------------------------------------------------------------


def test_verify_only_reports_instead_of_raising(cache, no_network):
    absent = W.verify_only("toy")
    assert not absent.present and not absent.ok
    assert "absent from the cache" in absent.describe()

    cache.mkdir(parents=True)
    (cache / "toy.bin").write_bytes(PAYLOAD)
    good = W.verify_only("toy")
    assert good.ok and good.actual_sha256 == DIGEST
    assert good.describe().endswith("verified")

    (cache / "toy.bin").write_bytes(b"different")
    bad = W.verify_only("toy")
    assert bad.present and not bad.matches
    assert "HASH MISMATCH" in bad.describe()


def test_verify_all_covers_every_pin_rather_than_stopping_at_the_first_bad_one(cache, no_network):
    cache.mkdir(parents=True)
    (cache / "toy.bin").write_bytes(b"wrong")
    results = W.verify_all()
    assert [r.key for r in results] == ["absent", "toy"]
    assert not any(r.ok for r in results)


def test_describe_cache_is_manifest_shaped(cache, no_network):
    described = W.describe_cache()
    assert described["cache_dir"].endswith("weights")
    assert set(described) == {"cache_dir", "toy", "absent"}


def test_the_cache_directory_honours_its_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(W.CACHE_ENV_VAR, str(tmp_path / "elsewhere"))
    assert W.cache_dir() == tmp_path / "elsewhere" / "weights"
    monkeypatch.delenv(W.CACHE_ENV_VAR)
    assert W.cache_dir() == W.DEFAULT_CACHE / "weights"


def test_path_for_says_where_without_saying_whether(cache, no_network):
    assert W.path_for("toy").name == "toy.bin"
    assert not W.path_for("toy").exists()


# ---------------------------------------------------------------------------
# download: the only half that opens a socket
# ---------------------------------------------------------------------------


def test_download_verifies_before_publishing(cache, lockfile):
    """A file:// URL exercises the real streaming path without a server."""
    seen: list[tuple[str, int, int]] = []
    path = W.download("toy", progress=lambda k, d, t: seen.append((k, d, t)))
    assert path.read_bytes() == PAYLOAD
    assert seen and seen[-1][1] == len(PAYLOAD)
    # A second call is a no-op rather than a refetch.
    assert W.download("toy") == path


def test_a_download_whose_bytes_do_not_match_leaves_nothing_behind(cache, lockfile, tmp_path):
    """The bad bytes must not be published, and neither must a partial file."""
    body = json.loads(lockfile.read_text())
    body["weights"]["toy"]["sha256"] = "1" * 64
    lockfile.write_text(json.dumps(body))
    with pytest.raises(W.WeightHashMismatch, match="must not be edited to match"):
        W.download("toy")
    assert not (cache / "toy.bin").exists()
    leftovers = [p for p in cache.iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_download_refuses_to_overwrite_a_mismatched_cache_entry_by_default(cache, lockfile):
    cache.mkdir(parents=True)
    (cache / "toy.bin").write_bytes(b"stale")
    with pytest.raises(W.WeightHashMismatch, match="force=True"):
        W.download("toy")
    assert W.download("toy", force=True).read_bytes() == PAYLOAD


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("FACIOMETRY_TEST_NETWORK") != "1",
    reason="opt in with FACIOMETRY_TEST_NETWORK=1",
)
def test_the_real_pins_still_match_upstream(tmp_path, monkeypatch):
    """The one test that reaches the internet, and the only one that can catch
    an upstream republish. Fetches the smallest artefact only."""
    monkeypatch.setenv(W.CACHE_ENV_VAR, str(tmp_path))
    path = W.download("yunet")
    assert W.sha256_of(path) == W.spec_for("yunet").sha256
