"""`vitruve fetch-weights`: the only command that opens a socket.

The pinning, hashing and atomic-rename logic lives in
:mod:`vitruve.models.weights`, next to the backends that consume it. This
module is the command around it, and it adds exactly one thing the model layer
does not do: it maps each pinned artifact onto the license tier of the backend
it belongs to, and refuses to download anything above the tier the user
selected. Refusing before the download rather than at load time is the point,
because bandwidth spent on an AGPL checkpoint that will never be loaded is
bandwidth spent announcing an intention the user did not have.

Analysis never calls anything in here. `vitruve.models.weights.resolve` reads
the cache and raises if a file is absent; it does not fall back to fetching.
That separation is what makes the offline guarantee structural rather than
a matter of nobody having added a lazy download yet.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ..models.licensing import BY_NAME, Tier



def _progress(key: str, received: int, total: int) -> None:
    """Redraw a single line as bytes arrive.

    Without this the command prints one line and then appears hung: the largest
    pinned artefact is 254 MB, which is minutes on a domestic connection, and a
    silent process that long reads as a failure rather than as work.

    Written to stderr and redrawn in place, so piping the command's stdout
    somewhere still yields clean output.
    """
    if total <= 0:
        sys.stderr.write(f"\r    {_human(received)}")
    else:
        pct = 100.0 * received / total
        done = int(pct // 4)
        bar = "#" * done + "." * (25 - done)
        sys.stderr.write(f"\r    [{bar}] {pct:5.1f}%  {_human(received)} of {_human(total)}")
    sys.stderr.flush()


def _end_progress() -> None:
    """Clear the progress line so the next print starts at the left margin."""
    sys.stderr.write("\r" + " " * 72 + "\r")
    sys.stderr.flush()

class WeightsLayerMissing(RuntimeError):
    """This build has no model layer, so there is nothing pinned to fetch."""


@dataclass(frozen=True)
class Pinned:
    """One artifact, with the tier it inherits from its backend."""

    key: str
    filename: str
    provenance: str
    license_id: str
    tier: Tier
    size_bytes: int
    sha256: str


def _module():
    try:
        from ..models import weights as models_weights
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise WeightsLayerMissing(
            "no vitruve.models in this build, so no weights are pinned. The "
            "measurement layer runs without them; `vitruve catalogue` and "
            "`vitruve licenses` do not need a single byte of model."
        ) from exc
    return models_weights


def tier_of(provenance_name: str) -> Tier:
    """The tier an artifact inherits from the backend it belongs to.

    An artifact whose provenance is not in the licensing catalogue is treated
    as ``UNLICENSED``, which means it is never fetched by default. Defaulting
    the other way would make a typo in the lock file into a silent licence
    breach.
    """
    prov = BY_NAME.get(provenance_name)
    return prov.tier if prov is not None else Tier.UNLICENSED


def pins() -> tuple[Pinned, ...]:
    mod = _module()
    lock = mod.load_lock()
    return tuple(
        Pinned(
            key=key,
            filename=spec.filename,
            provenance=spec.provenance,
            license_id=spec.license_id,
            tier=tier_of(spec.provenance),
            size_bytes=spec.size_bytes,
            sha256=spec.sha256,
        )
        for key, spec in sorted(lock.items())
    )


def split_by_tier(tier: Tier) -> tuple[tuple[Pinned, ...], tuple[Pinned, ...]]:
    """``(permitted, refused)`` at ``tier``."""
    everything = pins()
    return (
        tuple(p for p in everything if p.tier <= tier),
        tuple(p for p in everything if p.tier > tier),
    )


def verify(tier: Tier = Tier.PERMISSIVE) -> tuple:
    """Offline check of every permitted artifact against its pin."""
    mod = _module()
    permitted, _ = split_by_tier(tier)
    return mod.verify_all([p.key for p in permitted])


def cache_dir():
    return _module().cache_dir()


def _human(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def run(tier: Tier = Tier.PERMISSIVE, *, force: bool = False) -> int:
    try:
        mod = _module()
        permitted, refused = split_by_tier(tier)
    except (WeightsLayerMissing, OSError, ValueError, KeyError, RuntimeError) as exc:
        # A missing model layer, a missing lock file and a lock file that will
        # not parse are all the same thing to a user: there is nothing to fetch
        # and the message says which.
        print(f"vitruve fetch-weights: {exc}", file=sys.stderr)
        return 2

    print(f"cache      {cache_dir()}")
    print(f"tier       {tier.name.lower()}")
    print(f"artifacts  {len(permitted)} permitted, {len(refused)} refused by tier")
    print("")

    for p in refused:
        print(
            f"  refused  {p.key}: {p.provenance} is {p.license_id} "
            f"({p.tier.name.lower()}). Re-run with --tier {p.tier.name.lower()} to "
            "accept that obligation."
        )

    failures = 0
    for p in permitted:
        state = mod.verify_only(p.key)
        if state.ok and not force:
            print(f"  cached   {p.key:<28} {p.filename}")
            continue
        print(f"  fetching {p.key:<28} {p.filename} ({_human(p.size_bytes)})", flush=True)
        try:
            mod.download(p.key, force=True, progress=_progress)
            _end_progress()
        except Exception as exc:
            print(f"  FAILED   {p.key}: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"  verified {p.key:<28} {p.sha256[:16]}")

    if failures:
        print("")
        print(f"{failures} artifact(s) failed. Analysis will not run without them.")
        return 1
    return 0
