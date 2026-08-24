"""The weight cache, and the wall between fetching and inferring.

Design rule 5 in `docs/CORE_API.md` says there is no network egress during
analysis. That is only true if the code makes it structurally true, so this
module has two entry points and they do not call each other:

* :func:`resolve` looks a weight up in the cache, verifies its sha256, and
  returns a path. It never opens a socket. If the file is not there it raises
  `WeightsUnavailable` naming the exact fetch command, because a lazy download
  hidden inside the first inference call is precisely the egress the rule
  exists to forbid.
* :func:`download` fetches. It is called by `vitruve weights fetch` and by
  tests marked `network`, and by nothing else.

Every artefact is pinned by sha256 in ``assets/weights.lock.json``. A mismatch
is a hard failure, never a warning and never a re-download: the two ways a hash
moves are a compromised mirror and an upstream silently republishing different
weights under the same URL, and both of them invalidate every number the
previous weights produced.

Some notes on the pins actually recorded, since they were resolved by fetching
the files rather than by copying a README:

* **SPIGA** upstream distributes its weights through Google Drive, whose
  ``uc?export=download`` endpoint returns an HTML virus-scan interstitial for
  files this size rather than the file. The pin therefore points at the
  author's own Hugging Face mirror, which was checked byte-for-byte against the
  Drive copy (identical sha256, 254,871,697 bytes).
* **YuNet** is pinned to a specific opencv_zoo commit rather than to `main`,
  because `main` is a moving target and the ONNX has been republished before.
* **6DRepNet** is served from the author's institutional cloud share. There is
  no version in the URL, so the sha256 is the only thing standing between a
  silent upstream replacement and a changed pose number.

.. warning::

   On macOS, any freshly pip-installed package with compiled extensions must be
   ad-hoc codesigned before first import, or XProtect deep-scans every ``.so``
   and the import hangs for minutes::

       find .venv \\( -name "*.so" -o -name "*.dylib" \\) -print0 \\
           | xargs -0 -P 8 -n 20 codesign -s - -f

   This is unrelated to weights but it is where anyone debugging a "model load
   takes four minutes" report will look first.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

#: Where downloaded artefacts live. Overridable so a test, or a machine with a
#: small home volume, can put them elsewhere without a code change.
CACHE_ENV_VAR = "VITRUVE_CACHE_DIR"
DEFAULT_CACHE = Path.home() / ".cache" / "vitruve"

#: Where the pins live. Overridable mainly so tests can pin against a fixture.
LOCK_ENV_VAR = "VITRUVE_WEIGHTS_LOCK"

#: Read size for hashing and streaming. One megabyte, because the largest
#: artefact is a quarter of a gigabyte and per-chunk overhead is not the cost.
CHUNK = 1 << 20


class WeightsError(RuntimeError):
    """Base for every failure in this module."""


class WeightsUnavailable(WeightsError):
    """The artefact is not in the cache and analysis may not fetch it."""


class WeightHashMismatch(WeightsError):
    """The cached bytes are not the pinned bytes. Always fatal."""


class UnknownWeight(WeightsError, KeyError):
    """No such key in the lock file."""


@dataclass(frozen=True)
class WeightSpec:
    """One pinned artefact."""

    key: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    #: The `licensing.Provenance.name` this artefact belongs to, so a tier
    #: refusal can be raised before anything is fetched or read.
    provenance: str
    license_id: str
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or not all(c in "0123456789abcdef" for c in self.sha256):
            raise ValueError(f"{self.key}: sha256 must be 64 lowercase hex characters")
        if self.size_bytes <= 0:
            raise ValueError(f"{self.key}: size_bytes must be positive")


@dataclass(frozen=True)
class VerifyResult:
    """The outcome of an offline check on one artefact."""

    key: str
    present: bool
    matches: bool
    path: Path
    actual_sha256: str | None = None

    @property
    def ok(self) -> bool:
        return self.present and self.matches

    def describe(self) -> str:
        if not self.present:
            return f"{self.key}: absent from the cache"
        if self.matches:
            return f"{self.key}: verified"
        return (
            f"{self.key}: HASH MISMATCH, cached bytes hash to {self.actual_sha256} "
            "which is not the pinned value"
        )


def lock_path() -> Path:
    """Location of ``assets/weights.lock.json``."""
    override = os.environ.get(LOCK_ENV_VAR)
    if override:
        return Path(override).expanduser()
    # src/vitruve/models/weights.py -> models -> vitruve -> src -> repo root
    return Path(__file__).resolve().parents[3] / "assets" / "weights.lock.json"


def cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV_VAR)
    return (Path(override).expanduser() if override else DEFAULT_CACHE) / "weights"


def load_lock(path: Path | None = None) -> dict[str, WeightSpec]:
    """Parse the lock file into specs.

    Not cached at module level: a test that points `VITRUVE_WEIGHTS_LOCK` at a
    fixture in the middle of a session must actually get the fixture.
    """
    p = path if path is not None else lock_path()
    if not p.exists():
        raise WeightsError(
            f"no weight lock file at {p}. It is checked into the repository at "
            "assets/weights.lock.json; a missing one means the checkout is incomplete."
        )
    raw = json.loads(p.read_text())
    entries = raw["weights"] if isinstance(raw, dict) and "weights" in raw else raw
    specs: dict[str, WeightSpec] = {}
    for key, body in entries.items():
        specs[key] = WeightSpec(key=key, **body)
    return specs


def spec_for(key: str, *, lock: Mapping[str, WeightSpec] | None = None) -> WeightSpec:
    specs = lock if lock is not None else load_lock()
    try:
        return specs[key]
    except KeyError as exc:
        raise UnknownWeight(
            f"{key!r} is not pinned in {lock_path()}; known keys are {sorted(specs)}"
        ) from exc


def path_for(key: str, *, lock: Mapping[str, WeightSpec] | None = None) -> Path:
    """Where the artefact would live. Says nothing about whether it is there."""
    return cache_dir() / spec_for(key, lock=lock).filename


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_only(key: str, *, lock: Mapping[str, WeightSpec] | None = None) -> VerifyResult:
    """Check one cached artefact against its pin, offline, without loading it.

    Returns a result rather than raising, so `vitruve weights verify` can report
    on every artefact in one pass instead of stopping at the first bad one.
    """
    spec = spec_for(key, lock=lock)
    p = cache_dir() / spec.filename
    if not p.exists():
        return VerifyResult(key=key, present=False, matches=False, path=p)
    actual = sha256_of(p)
    return VerifyResult(
        key=key,
        present=True,
        matches=(actual == spec.sha256),
        path=p,
        actual_sha256=actual,
    )


def verify_all(
    keys: Iterable[str] | None = None, *, lock: Mapping[str, WeightSpec] | None = None
) -> tuple[VerifyResult, ...]:
    specs = lock if lock is not None else load_lock()
    wanted = list(keys) if keys is not None else sorted(specs)
    return tuple(verify_only(k, lock=specs) for k in wanted)


def resolve(key: str, *, lock: Mapping[str, WeightSpec] | None = None) -> Path:
    """Path to a verified cached artefact. Never touches the network.

    This is what backends call. The verification is not optional and not
    cached in memory: it costs about a second on the largest artefact and it is
    the only thing that catches a half-written file from an interrupted fetch,
    which otherwise surfaces as an unpickling error deep inside torch.
    """
    spec = spec_for(key, lock=lock)
    result = verify_only(key, lock={key: spec})
    if not result.present:
        raise WeightsUnavailable(
            f"{spec.provenance} needs {spec.filename}, which is not in {cache_dir()}. "
            f"Analysis does not download. Fetch it first:\n"
            f"    vitruve weights fetch {key}\n"
            f"    (or python -m vitruve.models.weights fetch {key})\n"
            f"Source: {spec.url}"
        )
    if not result.matches:
        raise WeightHashMismatch(
            f"{spec.filename} does not match its pin.\n"
            f"  expected {spec.sha256}\n"
            f"  actual   {result.actual_sha256}\n"
            f"This is fatal. Either the cached file is damaged, or {spec.url} is "
            "serving different bytes than the ones this pin was taken from. Delete "
            f"{result.path} and refetch, and if the hash still differs do not use it: "
            "every number produced by different weights is a different measurement."
        )
    return result.path


# ---------------------------------------------------------------------------
# The network half. Nothing above this line opens a socket.
# ---------------------------------------------------------------------------


def download(
    key: str,
    *,
    force: bool = False,
    lock: Mapping[str, WeightSpec] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    timeout: float = 60.0,
) -> Path:
    """Fetch one artefact into the cache and verify it before publishing it.

    The download goes to a temporary file in the cache directory and is renamed
    into place only after the hash matches, so an interrupted fetch can never
    leave something that `resolve` would accept. The rename is within one
    filesystem, so it is atomic.
    """
    import urllib.request

    spec = spec_for(key, lock=lock)
    dest = cache_dir() / spec.filename
    if dest.exists() and not force:
        existing = verify_only(key, lock={key: spec})
        if existing.ok:
            return dest
        raise WeightHashMismatch(
            f"{dest} exists but does not match its pin ({existing.actual_sha256}). "
            "Pass force=True to overwrite it, but read the mismatch message in "
            "resolve() first: a changed hash on an unchanged URL is a fact worth "
            "understanding before it is overwritten."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    written = 0
    tmp_fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{spec.filename}.")
    tmp = Path(tmp_name)
    try:
        request = urllib.request.Request(spec.url, headers={"User-Agent": "vitruve/weights"})
        with urllib.request.urlopen(request, timeout=timeout) as response, os.fdopen(
            tmp_fd, "wb"
        ) as out:
            total = int(response.headers.get("Content-Length") or spec.size_bytes)
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                written += len(chunk)
                if progress is not None:
                    progress(key, written, total)
        actual = h.hexdigest()
        if actual != spec.sha256:
            raise WeightHashMismatch(
                f"{spec.url} served bytes hashing to {actual}, but the pin for "
                f"{key} is {spec.sha256}. Nothing was written to the cache. "
                "Either the mirror is compromised or upstream republished; in both "
                "cases the pin is doing its job and must not be edited to match."
            )
        if written != spec.size_bytes:
            raise WeightHashMismatch(
                f"{key}: got {written} bytes, pin says {spec.size_bytes}. The hash "
                "matched, so this is a corrupt lock file rather than a corrupt download."
            )
        shutil.move(str(tmp), str(dest))
        return dest
    finally:
        if tmp.exists():
            tmp.unlink()


def download_all(
    keys: Iterable[str] | None = None,
    *,
    force: bool = False,
    lock: Mapping[str, WeightSpec] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[Path, ...]:
    specs = lock if lock is not None else load_lock()
    wanted = list(keys) if keys is not None else sorted(specs)
    return tuple(download(k, force=force, lock=specs, progress=progress) for k in wanted)


def describe_cache(lock: Mapping[str, WeightSpec] | None = None) -> dict[str, str]:
    """Cache state for the run manifest: what is present and what it hashes to."""
    specs = lock if lock is not None else load_lock()
    out = {"cache_dir": str(cache_dir())}
    for key in sorted(specs):
        out[key] = verify_only(key, lock=specs).describe()
    return out


def _main(argv: list[str] | None = None) -> int:
    """A tiny CLI so the fetch step is runnable before the real CLI exists."""
    import argparse

    parser = argparse.ArgumentParser(prog="vitruve.models.weights")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fetch = sub.add_parser("fetch", help="download pinned weights (uses the network)")
    fetch.add_argument("keys", nargs="*", help="default: all")
    fetch.add_argument("--force", action="store_true")
    check = sub.add_parser("verify", help="check the cache offline")
    check.add_argument("keys", nargs="*")
    sub.add_parser("list", help="show the pins")

    args = parser.parse_args(argv)
    specs = load_lock()
    if args.cmd == "list":
        for key, spec in sorted(specs.items()):
            print(f"{key}\n  {spec.filename}  {spec.size_bytes} bytes  {spec.license_id}")
            print(f"  {spec.url}\n  sha256 {spec.sha256}")
        return 0
    if args.cmd == "verify":
        results = verify_all(args.keys or None, lock=specs)
        for r in results:
            print(r.describe())
        return 0 if all(r.ok for r in results) else 1

    def show(key: str, done: int, total: int) -> None:
        pct = 100.0 * done / total if total else 0.0
        print(f"\r{key}: {done / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)

    for key in args.keys or sorted(specs):
        path = download(key, force=args.force, lock=specs, progress=show)
        print(f"\r{key}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
