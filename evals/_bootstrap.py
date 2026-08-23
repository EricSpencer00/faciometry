"""Shared plumbing for the evaluation harness.

`src/` is not installed as a package (no pyproject yet, and the harness does
not own one), so the path is wired here rather than in every arm. Seeds live
here too: every arm draws from :func:`rng` so that a rerun of `run_all.py`
reproduces byte-identical JSON.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = Path(__file__).resolve().parent / "results"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: One master seed for the whole harness. Every arm derives a stream from it.
MASTER_SEED = 20260823


def rng(stream: int) -> np.random.Generator:
    """A named, reproducible generator. ``stream`` separates the arms."""
    return np.random.default_rng([MASTER_SEED, stream])


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def provenance() -> dict[str, Any]:
    import numpy
    return {
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "master_seed": MASTER_SEED,
    }


def write_json(name: str, payload: Mapping[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / f"{name}.json"
    dest.write_text(json.dumps({"provenance": provenance(), **payload},
                               indent=1, sort_keys=True, default=_default) + "\n")
    return dest


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not serialisable: {type(o)}")


def write_csv(name: str, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> Path | None:
    rows = list(rows)
    if not rows:
        return None
    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / f"{name}.csv"
    fields = fields or list(rows[0])
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return dest
