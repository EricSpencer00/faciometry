"""Between-subject spreads measured with calipers, from NIOSH 2003.

This is the preferred source for every measurement it covers, for one reason:
it is *direct* anthropometry. The discriminability gate asks whether a
photographic measurement's error is small relative to how much the quantity
varies between people. Answering that with a spread that was itself derived
photographically would be circular -- the same errors would sit on both sides
of the ratio.

The data is a US Government work in the public domain, so unlike almost
everything else in this field it can simply be shipped.

Coverage is nine measurements. Where NIOSH is silent, the published CC-BY
tables in :mod:`faciometry.norms.published` fill in, and where both are silent the
report says the spread is unknown rather than inventing one.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).parent / "data" / "niosh2003.json"


@lru_cache(maxsize=1)
def _load() -> dict:
    if not _DATA.exists():  # pragma: no cover - built by scripts/build_niosh_norms.py
        return {"strata": {}}
    return json.loads(_DATA.read_text())


def metadata() -> dict:
    d = _load()
    return {k: v for k, v in d.items() if k != "strata"}


def covers(measurement_id: str) -> bool:
    return measurement_id in _load()["strata"]


def stratum(
    measurement_id: str, *, sex: str | None = None, ancestry: str | None = None
) -> dict | None:
    """The narrowest stratum available for this subject, with its own ``n``.

    Falls back from (sex, ancestry) to (sex, pooled) to (both, pooled). The
    returned dictionary always names which stratum it actually is, so a report
    can say "compared against 642 Black male respirator users" rather than
    implying a general population.
    """
    strata = _load()["strata"].get(measurement_id)
    if not strata:
        return None
    for key in (
        f"{sex}|{ancestry}" if sex and ancestry else None,
        f"{sex}|pooled" if sex else None,
        "both|pooled",
    ):
        if key and key in strata:
            return {**strata[key], "stratum": key}
    return None


def spread(measurement_id: str, *, sex: str | None = None, ancestry: str | None = None) -> float | None:
    """Relative between-subject standard deviation, or ``None`` if uncovered."""
    s = stratum(measurement_id, sex=sex, ancestry=ancestry)
    if not s or not s["mean"]:
        return None
    return s["sd"] / s["mean"]


def available() -> tuple[str, ...]:
    return tuple(sorted(_load()["strata"]))
